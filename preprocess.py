import argparse
import traceback as tb
import csv
import glob
import json
import numpy as np
import os
from rdkit import Chem
from SmilesPE.pretokenizer import atomwise_tokenizer
from typing import Dict, List, Tuple


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_id", type=str, default=None, required=True)

    return parser.parse_args()


def parse_mol_file(mol_file: str) -> List[Tuple[int, int, int]]:
    # RDKit does not track BEGINWEDGE/BEGINDASH by default
    # need to manually parse this info from molfile, like in MolScribe
    with open(mol_file) as f:
        mol_data = f.read()
        lines = mol_data.split('\n')
        stereo_bonds = []

        for i, line in enumerate(lines):
            if line.endswith("V2000"):
                tokens = line.split()
                num_atoms = int(tokens[0])
                num_bonds = int(tokens[1])
                for atom_line in lines[i+1:i+1+num_atoms]:
                    # atom_tokens = atom_line.strip().split()
                    # coords.append([float(atom_tokens[0]), float(atom_tokens[1])])
                    pass
                for bond_line in lines[i+1+num_atoms:i+1+num_atoms+num_bonds]:
                    # bond_tokens = bond_line.strip().split()
                    # start, end, bond_type, stereo = [int(token) for token in bond_tokens[:4]]

                    bond_tokens = [bond_line[:3], bond_line[3:6], bond_line[6:9], bond_line[9:12]]
                    start, end, bond_type, stereo = [int(token) for token in bond_tokens]

                    if bond_type == 1:
                        if stereo == 0:
                            continue

                        if stereo == 1:
                            etype = 5
                        elif stereo == 6:
                            etype = 6
                        elif stereo == 4:
                            etype = 8
                        else:
                            raise ValueError(f"Unsupported stereo type: {stereo}, {mol_file}")
                        stereo_bonds.append((start - 1, end - 1, etype))
                break

    return stereo_bonds


def _get_edges(mol, mol_path: str, inverse_map: list) -> List[List]:
    stereo_bonds = parse_mol_file(mol_path)
    stereo_bonds = {
        (start, end): bond_type
        for start, end, bond_type in stereo_bonds
    }

    edges = []
    for bond_i, bond in enumerate(mol.GetBonds()):
        bond_type = bond.GetBondTypeAsDouble()
        try:
            assert bond_type.is_integer() or bond_type == 1.5
        except AssertionError:
            print(f"{bond_type}, {mol_path}")
            break

        if bond_type == 1.5:
            bond_type = 4
        else:
            bond_type = int(bond_type)

        if bond_type == 2:
            if bond.GetStereo() == Chem.BondStereo.STEREOANY:
                bond_type = 7

        begin_atom_i_in_mol = bond.GetBeginAtomIdx()
        end_atom_i_in_mol = bond.GetEndAtomIdx()
        begin_atom_i = inverse_map[bond.GetBeginAtomIdx()]
        end_atom_i = inverse_map[bond.GetEndAtomIdx()]
        # print(f"begin i: {bond.GetBeginAtomIdx()} -> {begin_atom_i}, "
        #       f"symbol: {mol.GetAtomWithIdx(begin_atom_i_in_mol).GetSymbol()} "
        #       f"end i: {bond.GetEndAtomIdx()} -> {end_atom_i}, "
        #       f"symbol: {mol.GetAtomWithIdx(end_atom_i_in_mol).GetSymbol()}")

        # Override with type 5 and 6, if in stereo_bonds
        bond_type = stereo_bonds.get(
            (begin_atom_i_in_mol, end_atom_i_in_mol),
            bond_type
        )

        edge = [begin_atom_i, end_atom_i, bond_type]
        edges.append(edge)

    return edges


def _get_row(png_fn: str) -> Dict[str, str]:
    png_path = png_fn
    mol_path = f"{png_fn[:-4]}.corrected.mol"

    assert os.path.exists(png_path), png_path
    assert os.path.exists(mol_path), mol_path

    # Disable sanitization to suppress aromatic labeling by rdkit.
    # The data provider should ensure the correctness of molfiles.
    mol = Chem.MolFromMolFile(mol_path, sanitize=False, removeHs=False)
    # Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    # Chem.Kekulize(mol)
    # raw_smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)
    try:
        raw_smi = Chem.MolToSmiles(mol, kekuleSmiles=False, canonical=False)
    except RuntimeError:
        print(f"Runtime error for {png_fn}")
        tb.print_exc()
        raw_smi = Chem.MolToSmiles(mol, kekuleSmiles=False, canonical=False, isomericSmiles=False)

    reordered_atom_i = eval(mol.GetProp("_smilesAtomOutputOrder"))
    inverse_map = np.argsort(reordered_atom_i).tolist()
    # print(reordered_atom_i)
    # print(inverse_map)

    smi = raw_smi
    superatoms = {}
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        alias = atom.GetPropsAsDict().get("molFileAlias")
        if alias:
            superatoms[inverse_map[atom_idx]] = alias

    # copied over from molscribe
    if superatoms:
        tokens = atomwise_tokenizer(smi)
        atom_idx = 0
        for i, t in enumerate(tokens):
            if t.isalpha() or t[0] == '[' or t == '*':
                if atom_idx in superatoms:
                    symbol = superatoms[atom_idx]
                    tokens[i] = f"[{symbol}]"
                atom_idx += 1
        smi = "".join(tokens)

    conf = mol.GetConformer()
    node_coords = []
    for atom_i in reordered_atom_i:
        coord = conf.GetAtomPosition(atom_i)
        node_coords.append([coord.x, coord.y])

    edges = _get_edges(mol=mol, mol_path=mol_path, inverse_map=inverse_map)

    bracket_tokens = []
    bracket_coords = []
    for i, sg in enumerate(Chem.GetMolSubstanceGroups(mol)):
        brackets = sg.GetBrackets()
        if len(brackets) > 2:
            print(f"{len(brackets)} brackets found for {mol_path}")

        properties = sg.GetPropsAsDict()
        SCN = properties.get("CONNECT", "")  # superscript, essentially
        SMT = properties.get("LABEL", "")  # subscript, essentially
        # print(brackets)
        # use for loop to cover images with >2 brackets
        for bracket in brackets[:-1]:
            bracket_tokens.append(["<bra>"])
            bracket_coords.append([bracket[0].x, bracket[0].y])
            bracket_tokens.append(["<ket>"])
            bracket_coords.append([bracket[1].x, bracket[1].y])

        # lastly, attaching CONNECT and LABEL with the last <ket>
        # just to keep a record and ensure length consistency.
        # These will be further processed downstream

        # bracket_tokens.append(["<bra>"] + [token for token in SCN])
        # bracket_coords.append([brackets[-1][0].x, brackets[-1][0].y])
        # bracket_tokens.append(["<ket>"] + [token for token in str(SMT)])
        # bracket_coords.append([brackets[-1][1].x, brackets[-1][1].y])

        bracket_tokens.append(["<bra>"])
        bracket_coords.append([brackets[-1][0].x, brackets[-1][0].y])
        bracket_tokens.append(
            ["<ket>"] +
            ["<scn>"] + [token for token in str(SCN)] +
            ["<smt>"] + [token for token in str(SMT)]
        )
        bracket_coords.append([brackets[-1][1].x, brackets[-1][1].y])

    row = {
        "file_path": png_path,
        "mol_path": mol_path,
        "raw_SMILES": raw_smi,
        "SMILES": smi,
        "node_coords": json.dumps(node_coords, separators=(",", ":")),
        "bracket_tokens": json.dumps(bracket_tokens, separators=(",", ":")),
        "bracket_coords": json.dumps(bracket_coords, separators=(",", ":")),
        "edges": json.dumps(edges, separators=(",", ":")),
    }

    return row


def aggregate_into_csv(args) -> None:
    fieldnames = [
        "file_path", "mol_path",
        "raw_SMILES", "SMILES", "node_coords",
        "bracket_tokens", "bracket_coords", "edges"
    ]

    for phase in ["train", "val", "test"]:
        fn = os.path.join("experiments", args.expt_id, f"{args.expt_id}_{phase}.filelist.txt"
        )
        if not os.path.exists(fn):
            continue

        ofn = os.path.join("experiments", args.expt_id, f"{args.expt_id}_{phase}.processed.csv")

        rows = []
        with open(fn, "r") as f:
            for line in f:
                if line.strip().endswith("/"):
                    png_fl = sorted(glob.glob(f"{line.strip()}/*.png"))
                    for png_fn in png_fl:
                        row = _get_row(png_fn=png_fn)
                        rows.append(row)
                else:
                    png_fn = line.strip()
                    assert png_fn.endswith(".png"), png_fn
                    row = _get_row(png_fn=png_fn)
                    rows.append(row)

        with open(ofn, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main(args):
    aggregate_into_csv(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
