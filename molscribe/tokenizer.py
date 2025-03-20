import os
import json
import numpy as np
import random
from SmilesPE.pretokenizer import atomwise_tokenizer
from typing import Dict, List, Optional

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
MASK = '<mask>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4

SEP = "<sep>"
BRA = "<bra>"
KET = "<ket>"


class Tokenizer:

    def __init__(self, path=None):
        self.stoi = {}
        self.itos = {}
        if path:
            self.load(path)

    @property
    def output_constraint(self):
        return False

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.stoi, f)

    def load(self, path):
        with open(path) as f:
            self.stoi = json.load(f)
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions

    def sequence_to_smiles(self, sequence):
        return {'smiles': self.predict_caption(sequence)}


class NodeTokenizer(Tokenizer):

    def __init__(self, input_size=100, path=None, sep_xy=False, continuous_coords=False, debug=False):
        super().__init__(path)
        self.maxx = input_size  # height
        self.maxy = input_size  # width
        self.sep_xy = sep_xy
        self.special_tokens = [PAD, SOS, EOS, UNK, MASK]
        self.continuous_coords = continuous_coords
        self.debug = debug

        self.stoi_ext = {
            SEP: self.offset + self.maxx + self.maxy,
            BRA: self.offset + self.maxx + self.maxy + 1,
            KET: self.offset + self.maxx + self.maxy + 2
        }

    def __len__(self):
        assert self.sep_xy
        return self.offset + self.maxx + self.maxy + self.ext_size

    @property
    def offset(self):
        return len(self.stoi)

    @property
    def ext_size(self):
        return len(self.stoi_ext)

    @property
    def output_constraint(self):
        assert not self.continuous_coords
        return True

    def len_symbols(self):
        return len(self.stoi)

    def is_x(self, x):
        return self.offset <= x < self.offset + self.maxx

    def is_y(self, y):
        assert self.sep_xy
        return self.offset + self.maxx <= y < self.offset + self.maxx + self.maxy

    def is_symbol(self, s):
        return len(self.special_tokens) <= s < self.offset or s == UNK_ID

    def is_atom(self, id):
        # The logic here might be trickier (or maybe just process downstream)
        return self.is_symbol(id) and self.is_atom_token(self.itos[id])

    @staticmethod
    def is_atom_token(token):
        return token.isalpha() or token.startswith("[") or token == '*' or token == UNK

    def x_to_id(self, x):
        return self.offset + round(x * (self.maxx - 1))

    def y_to_id(self, y):
        assert self.sep_xy
        return self.offset + self.maxx + round(y * (self.maxy - 1))

    def id_to_x(self, id):
        return (id - self.offset) / (self.maxx - 1)

    def id_to_y(self, id):
        assert self.sep_xy
        return (id - self.offset - self.maxx) / (self.maxy - 1)

    def symbol_to_id(self, symbol):
        return self.stoi.get(symbol, UNK_ID)


class CharTokenizer(NodeTokenizer):

    def __init__(
        self,
        input_size=100,
        path=None,
        sep_xy=False,
        continuous_coords=False,
        debug=False
    ):
        super().__init__(input_size, path, sep_xy, continuous_coords, debug)

    def fit_atom_symbols(self, atoms):
        atoms = list(set(atoms))
        chars = []
        for atom in atoms:
            chars.extend(list(atom))
        vocab = self.special_tokens + chars
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        assert self.stoi[PAD] == PAD_ID
        assert self.stoi[SOS] == SOS_ID
        assert self.stoi[EOS] == EOS_ID
        assert self.stoi[UNK] == UNK_ID
        assert self.stoi[MASK] == MASK_ID
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def get_output_mask(self, id):
        """TO FIX"""
        mask = [False] * len(self)
        if self.is_x(id):
            return [True] * (self.offset + self.maxx) + [False] * self.maxy
        if self.is_y(id):
            return [False] * self.offset + [True] * (self.maxx + self.maxy)
        return mask

    def smiles_and_bracket_to_sequence(
        self,
        smiles: str,
        bracket_tokens: List[List[str]],
        node_coords: Optional[np.ndarray] = None,
        bracket_coords: Optional[np.ndarray] = None,
        mask_ratio: float = 0
    ):
        tokens = atomwise_tokenizer(smiles)
        labels = [SOS_ID]
        indices = []
        atom_idx = -1
        for token in tokens:
            for c in token:
                labels.append(self.stoi.get(c, UNK_ID))
            if self.is_atom_token(token):
                atom_idx += 1
                if mask_ratio > 0 and random.random() < mask_ratio:
                    labels.append(MASK_ID)
                    labels.append(MASK_ID)
                elif node_coords is not None:
                    if atom_idx < len(node_coords):
                        x, y = node_coords[atom_idx]
                        assert 0 <= x <= 1
                        assert 0 <= y <= 1
                    else:
                        x = random.random()
                        y = random.random()
                    labels.append(self.x_to_id(x))
                    labels.append(self.y_to_id(y))
                indices.append(len(labels) - 1)
        labels.append(self.stoi_ext[SEP])

        assert len(bracket_tokens) == len(bracket_coords)
        for tokens, coords in zip(bracket_tokens, bracket_coords):
            for token in tokens:
                labels.append(self.stoi_ext.get(token, self.stoi.get(token, UNK_ID)))
            x, y = coords
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            labels.append(self.x_to_id(x))
            labels.append(self.y_to_id(y))

        labels.append(EOS_ID)

        return labels, indices

    def sequence_to_smiles(self, sequence):
        # sequence = [1, 81, 101, 207, 57, 122, 186, 57, 143, 207, 81, 164, 186, 2]
        has_coords = not self.continuous_coords
        smiles = ''
        coords, symbols, indices = [], [], []
        i = 0
        while i < len(sequence):
            label = sequence[i]
            if label == EOS_ID or label == PAD_ID:
                break
            if self.is_x(label) or self.is_y(label):
                i += 1
                continue
            if not self.is_atom(label):
                smiles += self.itos[label]
                i += 1
                continue
            if self.itos[label] == '[':
                j = i + 1
                while j < len(sequence):
                    if not self.is_symbol(sequence[j]):
                        break
                    if self.itos[sequence[j]] == ']':
                        j += 1
                        break
                    j += 1
            else:
                if i+1 < len(sequence) and (self.itos[label] == 'C' and self.is_symbol(sequence[i+1]) and self.itos[sequence[i+1]] == 'l' \
                        or self.itos[label] == 'B' and self.is_symbol(sequence[i+1]) and self.itos[sequence[i+1]] == 'r'):
                    j = i+2
                else:
                    j = i+1
            token = ''.join(self.itos[sequence[k]] for k in range(i, j))
            smiles += token
            if has_coords:
                if j+2 < len(sequence) and self.is_x(sequence[j]) and self.is_y(sequence[j+1]):
                    x = self.id_to_x(sequence[j])
                    y = self.id_to_y(sequence[j+1])
                    coords.append([x, y])
                    symbols.append(token)
                    indices.append(j+2)
                    i = j+2
                else:
                    i = j
            else:
                if j < len(sequence):
                    symbols.append(token)
                    indices.append(j)
                i = j
        results = {'smiles': smiles, 'symbols': symbols, 'indices': indices}
        if has_coords:
            results['coords'] = coords
        return results


def get_tokenizer(args) -> Dict[str, Tokenizer]:
    args.vocab_file = os.path.join(os.path.dirname(__file__), 'vocab/vocab_chars.json')

    tokenizer = {
        "chartok_coords": CharTokenizer(
            args.coord_bins,
            args.vocab_file,
            args.sep_xy,
            continuous_coords=args.continuous_coords)
    }

    return tokenizer
