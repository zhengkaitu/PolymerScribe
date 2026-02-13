import argparse
import numpy as np
import os
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from typing import Any

sgroup_cost_threshold = 1.0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_filelist", type=str, default=None, required=True)
    parser.add_argument("--pred_root_path", type=str, default=None, required=True)

    return parser.parse_args()


def normalize_nodes(
    nodes,
    flip_y=True,
    bbox=None
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x, y = nodes[:, 0], nodes[:, 1]
    if bbox is None:
        minx, maxx = min(x), max(x)
        miny, maxy = min(y), max(y)
    else:
        minx, maxx = bbox[0], bbox[1]
        miny, maxy = bbox[2], bbox[3]

    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)

    return np.stack([x, y], axis=1), (minx, maxx, miny, maxy)


def parse_molblock(molblock: str) -> dict[tuple[int, int], Any]:
    lines = molblock.split('\n')
    stereo_bonds = {}

    for i, line in enumerate(lines):
        if line.endswith("V2000"):
            tokens = line.split()
            num_atoms = int(tokens[0])
            num_bonds = int(tokens[1])
            for bond_line in lines[i + 1 + num_atoms:i + 1 + num_atoms + num_bonds]:
                # bond_tokens = bond_line.strip().split()
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
                        raise ValueError(f"Unsupported stereo type: {stereo}")
                    stereo_bonds[(start - 1, end - 1)] = etype
            break
    return stereo_bonds

def _get_norm_coords(mol) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    conf= mol.GetConformer()
    coords = []
    for i, a in enumerate(mol.GetAtoms()):
        coord = conf.GetAtomPosition(i)
        coords.append([coord.x, coord.y])
    coords = np.array(coords, dtype=np.float32)
    coords, bbox = normalize_nodes(coords)

    return coords, bbox


def _atom_equal(a_pred, a_gt) -> bool:
    symbol_pred = a_pred.GetPropsAsDict().get("molFileAlias", a_pred.GetSymbol())
    symbol_gt = a_gt.GetPropsAsDict().get("molFileAlias", a_gt.GetSymbol())

    if not symbol_pred.lower() == symbol_gt.lower():
        return False
    if not a_pred.GetFormalCharge() == a_gt.GetFormalCharge():
        return False
    if not a_pred.GetNumRadicalElectrons() == a_gt.GetNumRadicalElectrons():
        return False

    return True


def _get_bond_type(b, stereo_bond_override: int) -> float:
    if not b:
        return 0.0

    bond_type = b.GetBondTypeAsDouble()
    if bond_type == 1.5:
        bond_type = 4

    if bond_type == 2:
        if b.GetStereo() == Chem.BondStereo.STEREOANY:
            bond_type = 7

    assert stereo_bond_override in [0, 5, 6, 8]
    if stereo_bond_override:
        bond_type = stereo_bond_override

    return bond_type

def _get_bracket_coords(brackets) -> np.ndarray:
    bracket_coords = []
    for bracket in brackets:
        bracket_coords.append([bracket[0].x, bracket[0].y])
        bracket_coords.append([bracket[1].x, bracket[1].y])
    bracket_coords = np.array(bracket_coords, dtype=np.float32)

    return bracket_coords


def _get_bracket_cost(bracket_coords_pred, bracket_coords_gt) -> float:
    assert len(bracket_coords_pred) == len(bracket_coords_gt)
    n_bracket = int(len(bracket_coords_pred) / 2)

    bracket_costs = np.ones((n_bracket, n_bracket), dtype=np.float32) * 1e3
    for i in range(n_bracket):
        midpoint_pred = (bracket_coords_pred[i*2] + bracket_coords_pred[i*2+1]) / 2
        for j in range(n_bracket):
            midpoint_gt = (bracket_coords_gt[j*2] + bracket_coords_gt[j*2+1]) / 2
            bracket_costs[i, j] = np.linalg.norm(midpoint_gt - midpoint_pred)

    row_ind, col_ind = linear_sum_assignment(bracket_costs)
    bracket_cost = bracket_costs[row_ind, col_ind].mean()

    return bracket_cost


def _sgroup_equal(sgroup_pred, sgroup_gt) -> bool:
    properties_pred = sgroup_pred.GetPropsAsDict()
    properties_gt = sgroup_gt.GetPropsAsDict()
    SCN_pred = properties_pred.get("CONNECT", "HT")
    SCN_gt = properties_gt.get("CONNECT", "HT")
    SMT_pred = properties_pred.get("LABEL", "")
    SMT_gt = properties_gt.get("LABEL", "")

    if not str(SCN_pred).lower() == str(SCN_gt).lower():
        return False
    if not str(SMT_pred).lower() == str(SMT_gt).lower():
        return False

    return True


def compare_molblocks(molblock_pred: str, molblock_gt: str) -> dict[str, Any]:
    # TODO: check how Hs are exactly handled
    mol_pred = Chem.MolFromMolBlock(molblock_pred, sanitize=False, removeHs=False, strictParsing=True)
    mol_gt = Chem.MolFromMolBlock(molblock_gt, sanitize=False, removeHs=False, strictParsing=True)

    stereo_bonds_pred = parse_molblock(molblock_pred)
    stereo_bonds_gt = parse_molblock(molblock_gt)

    if mol_pred is None:
        metrics = {
            "atom_precision": 0.0,
            "atom_recall": 0.0,
            "atom_f1": 0.0,
            "bond_precision": 0.0,
            "bond_recall": 0.0,
            "bond_f1": 0.0,
            "sgroup_precision": 0.0,
            "sgroup_recall": 0.0,
            "sgroup_f1": 0.0,
            "exact_match": 0.0
        }
        return metrics

    n_atom_pred = mol_pred.GetNumAtoms()
    n_atom_gt = mol_gt.GetNumAtoms()
    assert n_atom_pred == len(mol_pred.GetAtoms())
    assert n_atom_gt == len(mol_gt.GetAtoms())

    coords_pred, bbox_pred = _get_norm_coords(mol_pred)
    coords_gt, bbox_gt = _get_norm_coords(mol_gt)

    atom_costs = np.ones((n_atom_pred, n_atom_gt), dtype=np.float32) * 1e3
    for i, coord_pred in enumerate(coords_pred):
        for j, coord_gt in enumerate(coords_gt):
            atom_costs[i, j] = np.linalg.norm(coord_gt - coord_pred)

    row_ind, col_ind = linear_sum_assignment(atom_costs)
    # [print(f"{r}, {c}") for r, c in zip(row_ind, col_ind)]

    atom_precisions = np.zeros(n_atom_pred, dtype=np.float32)
    atom_recalls = np.zeros(n_atom_gt, dtype=np.float32)
    forward_map = {}
    reverse_map = {}
    for r, c in zip(row_ind, col_ind):
        forward_map[r] = c
        reverse_map[c] = r
        a_pred = mol_pred.GetAtomWithIdx(int(r))
        a_gt = mol_gt.GetAtomWithIdx(int(c))

        if _atom_equal(a_pred, a_gt):
            atom_precisions[r] = 1.0
            atom_recalls[c] = 1.0

    atom_precision = np.mean(atom_precisions) if atom_precisions.size else 0.0
    atom_recall = np.mean(atom_recalls) if atom_recalls.size else 0.0
    if atom_precision == 0.0 and atom_recall == 0.0:
        atom_f1 = 0.0
    else:
        atom_f1 = 2 * atom_precision * atom_recall / (atom_precision + atom_recall)

    # e.g., predicted bond (1 , 2) <=> gt bond (3, 4)
    bond_precisions = []
    bond_recalls = []
    for b_pred in mol_pred.GetBonds():
        begin_atom_i_pred = b_pred.GetBeginAtomIdx()
        end_atom_i_pred = b_pred.GetEndAtomIdx()
        try:
            begin_atom_i_gt = int(forward_map[begin_atom_i_pred])
            end_atom_i_gt = int(forward_map[end_atom_i_pred])
        except KeyError:
            bond_precisions.append(0.0)
            continue

        b_gt = mol_gt.GetBondBetweenAtoms(
            begin_atom_i_gt,
            end_atom_i_gt
        )
        stereo_bond_type_pred = stereo_bonds_pred.get((begin_atom_i_pred, end_atom_i_pred), 0)
        stereo_bond_type_gt = stereo_bonds_gt.get((begin_atom_i_gt, end_atom_i_gt), 0)
        b_type_pred = _get_bond_type(b_pred, stereo_bond_type_pred)
        b_type_gt = _get_bond_type(b_gt, stereo_bond_type_gt)

        if b_gt and b_type_pred == b_type_gt:
            bond_precisions.append(1.0)
        else:
            bond_precisions.append(0.0)

    for b_gt in mol_gt.GetBonds():
        begin_atom_i_gt = b_gt.GetBeginAtomIdx()
        end_atom_i_gt = b_gt.GetEndAtomIdx()
        try:
            begin_atom_i_pred = int(reverse_map[begin_atom_i_gt])
            end_atom_i_pred = int(reverse_map[end_atom_i_gt])
        except KeyError:
            bond_recalls.append(0.0)
            continue

        b_pred = mol_pred.GetBondBetweenAtoms(
            begin_atom_i_pred,
            end_atom_i_pred
        )
        stereo_bond_type_pred = stereo_bonds_pred.get((begin_atom_i_pred, end_atom_i_pred), 0)
        stereo_bond_type_gt = stereo_bonds_gt.get((begin_atom_i_gt, end_atom_i_gt), 0)
        b_type_pred = _get_bond_type(b_pred, stereo_bond_type_pred)
        b_type_gt = _get_bond_type(b_gt, stereo_bond_type_gt)

        if b_pred and b_type_pred == b_type_gt:
            bond_recalls.append(1.0)
        else:
            bond_recalls.append(0.0)

    bond_precision = np.mean(bond_precisions) if bond_precisions else 0.0
    bond_recall = np.mean(bond_recalls) if bond_recalls else 0.0
    if bond_precision == 0.0 and bond_recall == 0.0:
        bond_f1 = 0.0
    else:
        bond_f1 = 2 * bond_precision * bond_recall / (bond_precision + bond_recall)

    sgroups_pred = Chem.GetMolSubstanceGroups(mol_pred)
    sgroups_gt = Chem.GetMolSubstanceGroups(mol_gt)
    n_sgroup_pred = len(sgroups_pred)
    n_sgroup_gt = len(sgroups_gt)

    sgroup_costs = np.ones((n_sgroup_pred, n_sgroup_gt), dtype=np.float32) * 1e3
    for i, sgroup_pred in enumerate(sgroups_pred):
        brackets_pred = sgroup_pred.GetBrackets()
        bracket_coords_pred = _get_bracket_coords(brackets_pred)
        if bracket_coords_pred.size:
            bracket_coords_pred, _ = normalize_nodes(bracket_coords_pred, bbox=bbox_pred)
        else:
            continue

        for j, sgroup_gt in enumerate(sgroups_gt):
            brackets_gt = sgroup_gt.GetBrackets()
            bracket_coords_gt = _get_bracket_coords(brackets_gt)
            if bracket_coords_gt.size:
                bracket_coords_gt, _ = normalize_nodes(bracket_coords_gt, bbox=bbox_gt)
            else:
                continue

            if not len(brackets_pred) == len(brackets_gt):
                sgroup_costs[i, j] = 1e3
            else:
                sgroup_costs[i, j] = _get_bracket_cost(bracket_coords_pred, bracket_coords_gt)

    row_ind, col_ind = linear_sum_assignment(sgroup_costs)

    sgroup_precisions = np.zeros(n_sgroup_pred, dtype=np.float32)
    sgroup_recalls = np.zeros(n_sgroup_gt, dtype=np.float32)
    for r, c in zip(row_ind, col_ind):
        sgroup_pred = sgroups_pred[int(r)]
        sgroup_gt = sgroups_gt[int(c)]
        sgroup_cost = sgroup_costs[int(r), int(c)]

        if _sgroup_equal(sgroup_pred, sgroup_gt) and sgroup_cost < sgroup_cost_threshold:
            sgroup_precisions[int(r)] = 1.0
            sgroup_recalls[int(c)] = 1.0

    sgroup_precision = np.mean(sgroup_precisions) if sgroup_precisions.size else 0.0
    sgroup_recall = np.mean(sgroup_recalls) if sgroup_recalls.size else 0.0
    if sgroup_precision == 0.0 and sgroup_recall == 0.0:
        sgroup_f1 = 0.0
    else:
        sgroup_f1 = 2 * sgroup_precision * sgroup_recall / (sgroup_precision + sgroup_recall)

    exact_match = (atom_f1 == 1.0) and (bond_f1 == 1.0) and (sgroup_f1 == 1.0)

    metrics = {
        "atom_precision": atom_precision,
        "atom_recall": atom_recall,
        "atom_f1": atom_f1,
        "bond_precision": bond_precision,
        "bond_recall": bond_recall,
        "bond_f1": bond_f1,
        "sgroup_precision": sgroup_precision,
        "sgroup_recall": sgroup_recall,
        "sgroup_f1": sgroup_f1,
        "exact_match": exact_match
    }

    return metrics


def main(args):
    test_filelist = args.test_filelist
    pred_root_path = args.pred_root_path

    exact_matches = {}
    atom_precisions= {}
    atom_recalls = {}
    atom_f1s = {}
    bond_precisions= {}
    bond_recalls = {}
    bond_f1s = {}
    sgroup_precisions= {}
    sgroup_recalls = {}
    sgroup_f1s = {}

    with open(test_filelist, "r") as f:
        for line in f:
            molfile_gt = line.strip().replace(".png", ".corrected.mol")
            molfile_pred = line.strip().replace(".png", ".predicted.mol")
            molfile_pred = "/".join(molfile_pred.split("/")[2:])
            molfile_pred = os.path.join(pred_root_path, molfile_pred)

            with open(molfile_gt, "r") as f_gt:
                molblock_gt = f_gt.read()
            with open(molfile_pred, "r") as f_pred:
                molblock_pred = f_pred.read()
            metrics = compare_molblocks(molblock_pred, molblock_gt)
            mol_gt = Chem.MolFromMolBlock(molblock_gt, sanitize=False, removeHs=False, strictParsing=True)
            atom_count = mol_gt.GetNumAtoms()
            sgroups_gt = Chem.GetMolSubstanceGroups(mol_gt)
            bracket_count = 0
            for sgroup_gt in sgroups_gt:
                bracket_count += len(sgroup_gt.GetBrackets())

            count = atom_count // 10 * 10
            count = min(count, 50)
            # count = bracket_count
            if count in exact_matches:
                exact_matches[count].append(metrics["exact_match"])
                atom_precisions[count].append(metrics["atom_precision"])
                atom_recalls[count].append(metrics["atom_recall"])
                atom_f1s[count].append(metrics["atom_f1"])
                bond_precisions[count].append(metrics["bond_precision"])
                bond_recalls[count].append(metrics["bond_recall"])
                bond_f1s[count].append(metrics["bond_f1"])
                sgroup_precisions[count].append(metrics["sgroup_precision"])
                sgroup_recalls[count].append(metrics["sgroup_recall"])
                sgroup_f1s[count].append(metrics["sgroup_f1"])
            else:
                exact_matches[count] = [metrics["exact_match"]]
                atom_precisions[count] = [metrics["atom_precision"]]
                atom_recalls[count] = [metrics["atom_recall"]]
                atom_f1s[count] = [metrics["atom_f1"]]
                bond_precisions[count] = [metrics["bond_precision"]]
                bond_recalls[count] = [metrics["bond_recall"]]
                bond_f1s[count] = [metrics["bond_f1"]]
                sgroup_precisions[count] = [metrics["sgroup_precision"]]
                sgroup_recalls[count] = [metrics["sgroup_recall"]]
                sgroup_f1s[count] = [metrics["sgroup_f1"]]


            print(f"molfile_gt: {molfile_gt}, metrics: {metrics}")

    print(pred_root_path)
    for count in sorted(exact_matches.keys()):
        # print(f"count: {count} - {count+19}, "
        print(f"count: {count}, occurrences: {len(exact_matches[count])}, "
              f"Exact matches: {np.mean(exact_matches[count]): .2f}, "
              # f"AP: {np.mean(atom_precisions[count]): .4f}, "
              # f"AR: {np.mean(atom_recalls[count]): .4f}, "
              f"Atom F1: {np.mean(atom_f1s[count]): .4f}, "
              # f"BP: {np.mean(bond_precisions[count]): .4f}, "
              # f"BR: {np.mean(bond_recalls[count]): .4f}, "
              f"Bond F1: {np.mean(bond_f1s[count]): .4f}, "
              # f"SP: {np.mean(sgroup_precisions[count]): .4f}, "
              # f"SR: {np.mean(sgroup_recalls[count]): .4f}, "
              f"Sgroup F1: {np.mean(sgroup_f1s[count]): .4f}")


if __name__ == "__main__":
    args = get_args()
    main(args)
