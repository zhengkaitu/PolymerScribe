import cairosvg
import csv
import pandas as pd
import json
from rdkit import Chem
from typing import Any, Dict, List


def _label_bracket_bonds(edges: Dict[int, Any], bracket: Dict[str, Any]) -> int:
    # {0: {'begin_atom_i': 0, 'end_atom_i': 1, 'bond_type': 1,
    # 'begin_coord_x': 10.125, 'begin_coord_y': -6.975,
    # 'end_coord_x': 10.991, 'end_coord_y': -6.475} ...}

    # print(edges)

    def ccw(A, B, C):
        return (C["y"] - A["y"]) * (B["x"] - A["x"]) > (B["y"] - A["y"]) * (C["x"] - A["x"])

    # Return true if line segments AB and CD intersect
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    pA = {"x": bracket["begin_bracket_x"], "y": bracket["begin_bracket_y"]}
    pB = {"x": bracket["end_bracket_x"], "y": bracket["end_bracket_y"]}

    intersecting_bonds = []
    for bond_i, edge in edges.items():
        pC = {"x": edge["begin_coord_x"], "y": edge["begin_coord_y"]}
        pD = {"x": edge["end_coord_x"], "y": edge["end_coord_y"]}

        if intersect(pA, pB, pC, pD):
            intersecting_bonds.append(bond_i)

    assert len(intersecting_bonds) == 1
    # print(intersecting_bonds)

    intersecting_bond = edges[intersecting_bonds[0]]
    pC = {"x": intersecting_bond["begin_coord_x"], "y": intersecting_bond["begin_coord_y"]}
    if ccw(pA, pB, pC):
        # ccw means C is "outside" of the bracket
        bond_type = 7
    else:
        bond_type = 8
    intersecting_bond["bond_type"] = bond_type

    return bond_type


def extract_images_and_mol_blocks() -> None:
    fn = "data/archived_round_trip_translation.json"
    with open(fn, "r") as f:
        data = json.load(f)

    rows = []

    for d in data:
        i = d["index"]
        mol_block = d["input_molblock"]
        svg = d["input_structure_diagram"]

        fn_svg = f"data/si_mol/{i}.svg"
        with open(fn_svg, "w") as of:
            of.write(svg)

        fn_png = f"data/si_mol/{i}.png"
        cairosvg.svg2png(
            url=fn_svg,
            write_to=fn_png
        )

        fn_mol = f"data/si_mol/{i}.mol"
        with open(fn_mol, "w") as of:
            of.write(mol_block)

        row = {
            "file_path": fn_png,
            "mol_path": fn_mol
        }

        rows.append(row)


def aggregate_into_csv() -> None:
    fieldnames = [
        "file_path", "mol_path",
        "raw_SMILES", "SMILES", "node_coords",
        "bracket_tokens", "bracket_coords", "edges",
        "is_single_sg"
    ]
    rows = []
    single_sg_count = 0

    for file_i in range(1, 301):
        # if not file_i == 5:
        #     continue

        file_path = f"data/si_mol/{file_i}.png"
        mol_path = f"data/si_mol/{file_i}.mol"

        mol = Chem.MolFromMolFile(mol_path)
        Chem.Kekulize(mol)

        raw_smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)

        # for now SMILES is the same as raw_SMILES which suffices for the SI images
        # TODO: extension for R groups may be added later; we'll see
        smi = raw_smi
        conf = mol.GetConformer()

        node_coords = []
        for atom_i, atom in enumerate(mol.GetAtoms()):
            coord = conf.GetAtomPosition(atom_i)
            # print(atom.GetSymbol(), coord.x, coord.y, coord.z)
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

            begin_atom_i = bond.GetBeginAtomIdx()
            end_atom_i = bond.GetEndAtomIdx()
            edge = {
                "begin_atom_i": begin_atom_i,
                "end_atom_i": end_atom_i,
                "bond_type": bond_type,
                "begin_coord_x": conf.GetAtomPosition(begin_atom_i).x,
                "begin_coord_y": conf.GetAtomPosition(begin_atom_i).y,
                "end_coord_x": conf.GetAtomPosition(end_atom_i).x,
                "end_coord_y": conf.GetAtomPosition(end_atom_i).y
            }
            edges[bond_i] = edge

        """
        bracket_coords = []
        for sg in Chem.GetMolSubstanceGroups(mol):
            brackets = sg.GetBrackets()
            bracket_coords.append([brackets[0][0].x, brackets[0][0].y])
            bracket_coords.append([brackets[0][1].x, brackets[0][1].y])
            bracket_coords.append([brackets[1][0].x, brackets[1][0].y])
            bracket_coords.append([brackets[1][1].x, brackets[1][1].y])

        if len(Chem.GetMolSubstanceGroups(mol)) == 1:
            is_single_bracket_pair = 1

            sg = Chem.GetMolSubstanceGroupWithIdx(mol, 0)
            brackets = sg.GetBrackets()
            # x1, y1, x2, y2, in order of appearance in the mol block
            bracket_1 = {
                "begin_bracket_x": brackets[0][0].x,
                "begin_bracket_y": brackets[0][0].y,
                "end_bracket_x": brackets[0][1].x,
                "end_bracket_y": brackets[0][1].y
            }
            bracket_2 = {
                "begin_bracket_x": brackets[1][0].x,
                "begin_bracket_y": brackets[1][0].y,
                "end_bracket_x": brackets[1][1].x,
                "end_bracket_y": brackets[1][1].y
            }
            # print(bracket_1)
            # print(bracket_2)
            try:
                bond_type_1 = _label_bracket_bonds(edges, bracket_1)
                bond_type_2 = _label_bracket_bonds(edges, bracket_2)
                if not bond_type_1 or not bond_type_2:
                    print(f"Missing label for file_i: {file_i}, "
                          f"bond_type_1: {bond_type_1}, "
                          f"bond_type_2: {bond_type_2}")

                single_pair_count += 1

            except AssertionError:
                # FIXME: it's still possible to successfully label bracket_1 but not bracket_2
                print(f"Multiple intersections detected for file_i: {file_i}")
                is_single_bracket_pair = 0

        else:
            # FIXME: no further processing of non-single-pair images
            # as they're ignored
            is_single_bracket_pair = 0
        
        """

        bracket_tokens = []
        bracket_coords = []
        is_single_sg = 0
        for i, sg in enumerate(Chem.GetMolSubstanceGroups(mol)):
            if i == 0:
                is_single_sg = 1
            else:
                is_single_sg = 0

            brackets = sg.GetBrackets()
            if len(brackets) > 2:
                print(f"{len(brackets)} brackets found for {mol_path}")

            properties = sg.GetPropsAsDict()
            SCN = properties.get("CONNECT", "")         # superscript, essentially
            SMT = properties.get("LABEL", "")           # subscript, essentially

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

        single_sg_count += is_single_sg
        edges = [[
            edge["begin_atom_i"],
            edge["end_atom_i"],
            edge["bond_type"]
        ] for edge in edges.values()]

        row = {
            "file_path": file_path,
            "mol_path": mol_path,
            "raw_SMILES": raw_smi,
            "SMILES": smi,
            "node_coords": json.dumps(node_coords, separators=(",", ":")),
            "bracket_tokens": json.dumps(bracket_tokens, separators=(",", ":")),
            "bracket_coords": json.dumps(bracket_coords, separators=(",", ":")),
            "edges": json.dumps(edges, separators=(",", ":")),
            "is_single_sg": is_single_sg
        }
        rows.append(row)

    print(f"Number of images with a single Sgroup: {single_sg_count}")

    with open("data/si_mol/si_all.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_csv_single_bracket_pair() -> None:
    df = pd.read_csv("data/si_mol/si_all.csv")
    df_single_bracket = df.loc[df["is_single_bracket_pair"] == 1]
    df_train = df_single_bracket[:150]
    df_val = df_single_bracket[150:]

    df_train.to_csv("data/si_mol/single_bracket_train.csv", index=False, header=True)
    df_val.to_csv("data/si_mol/single_bracket_val.csv", index=False, header=True)


def split_csv() -> None:
    df = pd.read_csv("data/si_mol/si_all.csv")
    df_train = df[:200]
    df_val = df[200:]

    df_train.to_csv("data/si_mol/si_all_train.csv", index=False, header=True)
    df_val.to_csv("data/si_mol/si_all_val.csv", index=False, header=True)


def main():
    # extract_images_and_mol_blocks()
    aggregate_into_csv()
    # split_csv_single_bracket_pair()
    split_csv()


if __name__ == "__main__":
    main()

# raw_SMILES,SMILES,node_coords,edges
# C.C[N+](C)(C)C,[X-].[R1][N+]([R2])([R4])[R3],"[[2.3474,0.0],[-2.3474,0.0],[-0.8451,0.0],[-0.8451,1.5023],[-0.8451,-1.5023],[0.6573,0.0]]","[[1,2,1],[2,3,1],[2,4,1],[2,5,1]]"
