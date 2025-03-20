import albumentations as A
import cv2
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional
from .augment import SafeRotate, CropWhite, PadWhite, SaltAndPepperNoise
from .utils import FORMAT_INFO
from .tokenizer import PAD_ID
from .chemistry import normalize_nodes

cv2.setNumThreads(1)


def get_transforms(input_size, augment=True, rotate=True, debug=False) -> A.Compose:
    trans_list = []
    if augment and rotate:
        trans_list.append(SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    trans_list.append(CropWhite(pad=5))
    if augment:
        trans_list += [
            # NormalizedGridDistortion(num_steps=10, distort_limit=0.3),
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            PadWhite(pad_ratio=0.4, p=0.2),
            A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
            A.Blur(),
            A.GaussNoise(),
            SaltAndPepperNoise(num_dots=20, p=0.5)
        ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


class TrainDataset(Dataset):
    def __init__(self, args, df, tokenizer, split="train"):
        super().__init__()
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        if 'file_path' in df.columns:
            self.file_paths = df['file_path'].values
            if not self.file_paths[0].startswith(args.data_path):
                self.file_paths = [os.path.join(args.data_path, path) for path in df['file_path']]

        self.smiles = df['SMILES'].values if "SMILES" in df else None
        self.formats = args.formats
        self.labelled = (split in ["train", "val"])

        self.labels = {}
        self.transform = get_transforms(
            input_size=args.input_size,
            augment=(split == "train" and args.augment)
        )
        # self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        self.coords_df = df if self.labelled else None
        self.pseudo_coords = self.labelled

    def __len__(self):
        return len(self.df)

    def image_transform(self, image, coords=[], renormalize=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        augmented = self.transform(image=image, keypoints=coords)
        image = augmented['image']
        if len(coords) > 0:
            coords = np.array(augmented['keypoints'])
            if renormalize:
                coords = normalize_nodes(coords, flip_y=False)
            else:
                _, height, width = image.shape
                coords[:, 0] = coords[:, 0] / width
                coords[:, 1] = coords[:, 1] / height
            coords = np.array(coords).clip(0, 1)
            return image, coords
        return image

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            with open(os.path.join(self.args.save_path, f'error_dataset_{int(time.time())}.log'), 'w') as f:
                f.write(str(e))
            raise e

    def getitem(self, idx: int):
        ref = {}
        cond = (idx == 0)
        cond = False

        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        if image is None:
            image = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
            if cond: print(file_path, 'not found!')
        if self.coords_df is not None:
            if cond: print(f"idx: {idx}")
            h, w, _ = image.shape
            if cond: print(f"shape: h {h}, w {w}")

            bracket_tokens = eval(self.coords_df.loc[idx, 'bracket_tokens'])
            node_coords = np.array(eval(self.coords_df.loc[idx, 'node_coords']))
            bracket_coords = np.array(eval(self.coords_df.loc[idx, 'bracket_coords']))
            if cond: print(f"raw coords: {node_coords}")
            coords = np.concatenate((node_coords, bracket_coords), axis=0)

            if self.pseudo_coords:
                coords = normalize_nodes(coords)
                if cond: print(f"normalized: {coords}")
            coords[:, 0] = coords[:, 0] * w
            coords[:, 1] = coords[:, 1] * h
            if cond: print(f"rescaled: {coords}")

            image, coords = self.image_transform(image, coords, renormalize=self.pseudo_coords)
            node_coords = coords[:len(node_coords)]
            bracket_coords = coords[len(node_coords):]

            """
            # here we abuse the notation a bit to keep only the node coords
            coords = coords[:len(node_coords)]
            """
            if cond: exit(0)
        else:
            image = self.image_transform(image)
            bracket_tokens = []
            node_coords = None
            bracket_coords = None

        if self.labelled:
            smiles = self.smiles[idx]
            assert "chartok_coords" in self.formats
            mask_ratio = 1 if node_coords is None else 0
            self._process_chartok_coords(
                idx=idx,
                ref=ref,
                smiles=smiles,
                bracket_tokens=bracket_tokens,
                node_coords=node_coords,
                bracket_coords=bracket_coords,
                mask_ratio=mask_ratio
            )

        return idx, image, ref

    def _process_chartok_coords(
        self,
        idx: int,
        ref: Dict[str, Any],
        smiles: str,
        bracket_tokens: List[List[str]],
        node_coords: Optional[np.ndarray],
        bracket_coords: Optional[np.ndarray],
        mask_ratio=0
    ):
        max_len = FORMAT_INFO['chartok_coords']['max_len']
        tokenizer = self.tokenizer['chartok_coords']
        if smiles is None or type(smiles) is not str:
            smiles = ""
            raise ValueError(f"Invalid smiles: {smiles}")

        label, indices = tokenizer.smiles_and_bracket_to_sequence(
            smiles=smiles,
            bracket_tokens=bracket_tokens,
            node_coords=node_coords,
            bracket_coords=bracket_coords,
            mask_ratio=mask_ratio
        )
        ref['chartok_coords'] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref['atom_indices'] = torch.LongTensor(indices)

        assert not tokenizer.continuous_coords
        assert "edge" in self.df.columns
        edge_list = eval(self.df.loc[idx, 'edges'])
        n = len(indices)
        edges = torch.zeros((n, n), dtype=torch.long)
        for u, v, t in edge_list:
            if u < n and v < n:
                edges[u, v] = t
                if t <= 4:
                    edges[v, u] = t
                elif t == 5:
                    edges[v, u] = 6
                elif t == 6:
                    edges[v, u] = 5
                else:
                    raise ValueError(f"Invalid edge: {t}")
        ref['edges'] = edges

def pad_images(imgs):
    # B, C, H, W
    max_shape = [0, 0]
    for img in imgs:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[-1 - i])
    stack = []
    for img in imgs:
        pad = []
        for i in range(len(max_shape)):
            pad = pad + [0, max_shape[i] - img.shape[-1 - i]]
        stack.append(F.pad(img, pad, value=0))
    return torch.stack(stack)


def polymer_collate(batch):
    ids = []
    imgs = []
    batch = [ex for ex in batch if ex[1] is not None]
    formats = list(batch[0][2].keys())
    seq_formats = [k for k in formats if
                   k in ['atomtok', 'inchi', 'nodes', 'atomtok_coords', 'chartok_coords', 'atom_indices']]
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        ref = ex[2]
        for key in seq_formats:
            refs[key][0].append(ref[key])
            refs[key][1].append(torch.LongTensor([len(ref[key])]))
    # Sequence
    for key in seq_formats:
        # this padding should work for atomtok_with_coords too, each of which has shape (length, 4)
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    # Edges
    if 'edges' in formats:
        edges_list = [ex[2]['edges'] for ex in batch]
        max_len = max([len(edges) for edges in edges_list])
        refs['edges'] = torch.stack(
            [F.pad(edges, (0, max_len - len(edges), 0, max_len - len(edges)), value=-100)
             for edges in edges_list],
            dim=0
        )
    return ids, pad_images(imgs), refs
