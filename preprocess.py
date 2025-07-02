import argparse
import csv
import glob
import json
import numpy as np
import os
from rdkit import Chem
from typing import Dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_id", type=str, default=None, required=True)

    return parser.parse_args()


def _get_row(png_fn: str) -> Dict[str, str]:
    png_path = png_fn
    mol_path = f"{png_fn[:-4]}.corrected.mol"

    assert os.path.exists(png_path)
    assert os.path.exists(mol_path)

    # Disable sanitization to suppress aromatic labeling by rdkit.
    # The data provider should ensure the correctness of molfiles.
    mol = Chem.MolFromMolFile(mol_path, sanitize=False, removeHs=False)
    # Chem.Kekulize(mol)
    # raw_smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)
    raw_smi = Chem.MolToSmiles(mol, kekuleSmiles=False, canonical=False)
    reordered_atom_i = eval(mol.GetProp("_smilesAtomOutputOrder"))
    inverse_map = np.argsort(reordered_atom_i).tolist()
    # print(reordered_atom_i)
    # print(inverse_map)

    # for now SMILES is the same as raw_SMILES which suffices for the SI images
    # TODO: extension for R groups may be added later; we'll see
    smi = raw_smi
    conf = mol.GetConformer()

    node_coords = []
    for atom_i in reordered_atom_i:
        coord = conf.GetAtomPosition(atom_i)
        node_coords.append([coord.x, coord.y])

    edges = {}
    for bond_i, bond in enumerate(mol.GetBonds()):
        # the SI images have no chirality
        # TODO: extension for stereochemistry may be added later

        bond_type = bond.GetBondTypeAsDouble()
        try:
            assert bond_type.is_integer()
        except AssertionError:
            print(f"{bond_type}, {mol_path}")
            break
        bond_type = int(bond_type)

        # begin_atom_i = bond.GetBeginAtomIdx()
        # end_atom_i = bond.GetEndAtomIdx()

        begin_atom_i_in_mol = bond.GetBeginAtomIdx()
        end_atom_i_in_mol = bond.GetEndAtomIdx()
        begin_atom_i = inverse_map[bond.GetBeginAtomIdx()]
        end_atom_i = inverse_map[bond.GetEndAtomIdx()]
        # print(f"begin i: {bond.GetBeginAtomIdx()} -> {begin_atom_i}, "
        #       f"symbol: {mol.GetAtomWithIdx(begin_atom_i_in_mol).GetSymbol()} "
        #       f"end i: {bond.GetEndAtomIdx()} -> {end_atom_i}, "
        #       f"symbol: {mol.GetAtomWithIdx(end_atom_i_in_mol).GetSymbol()}")

        edge = {
            "begin_atom_i": begin_atom_i,
            "end_atom_i": end_atom_i,
            "bond_type": bond_type,
            "begin_coord_x": conf.GetAtomPosition(begin_atom_i_in_mol).x,
            "begin_coord_y": conf.GetAtomPosition(begin_atom_i_in_mol).y,
            "end_coord_x": conf.GetAtomPosition(end_atom_i_in_mol).x,
            "end_coord_y": conf.GetAtomPosition(end_atom_i_in_mol).y
        }
        edges[bond_i] = edge

    bracket_tokens = []
    bracket_coords = []

    for i, sg in enumerate(Chem.GetMolSubstanceGroups(mol)):
        brackets = sg.GetBrackets()
        if len(brackets) > 2:
            print(f"{len(brackets)} brackets found for {mol_path}")

        properties = sg.GetPropsAsDict()
        SCN = properties.get("CONNECT", "")  # superscript, essentially
        SMT = properties.get("LABEL", "")  # subscript, essentially

        # use for loop to cover images with >2 brackets
        for bracket in brackets[:-1]:
            bracket_tokens.append(["<bra>"])
            bracket_coords.append([bracket[0].x, bracket[0].y])
            bracket_tokens.append(["<ket>"])
            bracket_coords.append([bracket[1].x, bracket[1].y])

        # lastly, assuming CONNECT and LABEL are attached with the last bracket
        bracket_tokens.append(["<bra>"] + [token for token in SCN])
        bracket_coords.append([brackets[-1][0].x, brackets[-1][0].y])
        bracket_tokens.append(["<ket>"] + [token for token in SMT])
        bracket_coords.append([brackets[-1][1].x, brackets[-1][1].y])

    edges = [[
        edge["begin_atom_i"],
        edge["end_atom_i"],
        edge["bond_type"]
    ] for edge in edges.values()]

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

    for phase in ["train", "val"]:
        fn = os.path.join(
            "experiments",
            args.expt_id,
            f"{args.expt_id}_{phase}.filelist.txt"
        )
        if not os.path.exists(fn):
            continue

        ofn = os.path.join(
            "experiments",
            args.expt_id,
            f"{args.expt_id}_{phase}.processed.csv"
        )

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
                    assert png_fn.endswith(".png")
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
