import argparse
import csv
import numpy as np
import os
import sys
from collections import Counter
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from typing import Any

from utilities.binning import bucket_of_molblock
from utilities.canonical_bigsmiles_api import (
    FAILED_BIGSMILES,
    FIELDNAMES,
    STATUS_BIGSMILES_FAILED,
    STATUS_SUCCESS,
    ServerUnavailableError,
    add_server_args,
    api_from_args,
    check_servers,
    load_existing_rows,
    sanitize,
    write_rows,
)
from utilities.paths import (
    gt_molfile_for_image,
    gt_tsv_key,
    iter_filelist,
    pred_molfile_for_image,
)

sgroup_cost_threshold = 1.0

# Ground truth we have no canonical BigSMILES for at all.
GT_MISSING = "GT_MISSING"


def get_args():
    parser = argparse.ArgumentParser(
        description="Score predicted molblocks against the ground truth"
    )
    parser.add_argument("--test_filelist", type=str, default=None, required=True)
    parser.add_argument("--pred_root_path", type=str, default=None, required=True)
    parser.add_argument("--canonical_match", action="store_true",
                        help="also score canonical BigSMILES equality; needs "
                             "both services up")
    parser.add_argument("--gt_canonical_tsv", type=str,
                        default="data/PolymerLit/canonical_bigsmiles.tsv")
    parser.add_argument("--data_root", type=str, default="data/PolymerLit",
                        help="only used to key into --gt_canonical_tsv")
    parser.add_argument("--gt_status_valid", type=str, default="SUCCESS,NOOP",
                        help="ground-truth statuses whose canonical BigSMILES "
                             "is usable; anything else counts as a non-match")
    parser.add_argument("--pred_canonical_tsv", type=str, default=None,
                        help="cache of predicted canonicals; defaults to "
                             "<pred_root_path>/canonical_bigsmiles.pred.tsv")
    parser.add_argument("--no_pred_cache", action="store_true",
                        help="do not read or write the prediction cache")
    add_server_args(parser)

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


def load_gt_canonical(tsv_path: str) -> dict[str, tuple[str, str]]:
    """TSV key -> (canonical_bigsmiles, status).

    Older TSVs have no status column. Two of the three cases are still
    decidable from the strings, but "canonical equals its input" is not: it
    means either a genuine failure or an already-canonical molecule. Those are
    reported so the file can be regenerated with --reclassify rather than
    silently scored as if they were valid.
    """
    gt = {}
    ambiguous = 0

    with open(tsv_path, "r", newline="") as tsvfile:
        for row in csv.DictReader(tsvfile, delimiter="\t"):
            status = (row.get("status") or "").strip()
            if not status:
                if row["bigsmiles"] == FAILED_BIGSMILES:
                    status = STATUS_BIGSMILES_FAILED
                elif row["canonical_bigsmiles"] != row["bigsmiles"]:
                    status = STATUS_SUCCESS
                else:
                    ambiguous += 1
                    status = ""
            gt[row["path"]] = (row["canonical_bigsmiles"], status)

    print(f"Loaded {len(gt)} ground-truth canonical BigSMILES from {tsv_path}")
    if ambiguous:
        print(
            f"WARNING: {ambiguous} rows have no status and cannot be judged "
            f"(canonical equals its input). Regenerate with:\n"
            f"    python -m utilities.get_all_canonical_bigsmiles --reclassify"
        )

    return gt


def score_canonical(api, molblock_pred: str, canonical_gt: str,
                    gt_status: str, valid_statuses: set) -> tuple:
    """Compare a prediction's canonical BigSMILES with the ground truth.

    Returns (is_match, pred_bigsmiles, pred_canonical, pred_status).

    A ground truth we cannot canonicalize counts as a non-match rather than
    being dropped, so the metric stays conservative: every test sample is in
    the denominator. Such rows are not sent to the server at all, and are
    tallied separately so the ceiling they impose stays visible.
    """
    if gt_status not in valid_statuses:
        return False, "", "", gt_status or GT_MISSING

    pred_bigsmiles = api.molblock_to_bigsmiles(molblock_pred)
    if pred_bigsmiles == FAILED_BIGSMILES:
        return False, pred_bigsmiles, pred_bigsmiles, STATUS_BIGSMILES_FAILED

    pred_canonical, pred_status = api.canonicalize_with_status(pred_bigsmiles)
    is_match = sanitize(pred_canonical).strip() == sanitize(canonical_gt).strip()

    return is_match, pred_bigsmiles, pred_canonical, pred_status


def main(args):
    assert os.path.isdir("data"), "run from the repo root"

    pred_root_path = args.pred_root_path

    exact_matches = {}
    atom_precisions = {}
    atom_recalls = {}
    atom_f1s = {}
    bond_precisions = {}
    bond_recalls = {}
    bond_f1s = {}
    sgroup_precisions = {}
    sgroup_recalls = {}
    sgroup_f1s = {}
    canonical_matches = {}

    api = None
    gt_canonical = {}
    valid_statuses = set()
    cache_rows = {}
    cache_path = None
    # Ground truths whose own canonical BigSMILES could not be obtained; they
    # count as non-matches, and this is what makes the ceiling visible.
    uncanonicalizable = Counter()
    server_aborted = False

    if args.canonical_match:
        gt_canonical = load_gt_canonical(args.gt_canonical_tsv)
        valid_statuses = {
            s.strip() for s in args.gt_status_valid.split(",") if s.strip()
        }
        print(f"Treating these GT statuses as valid: {sorted(valid_statuses)}")
        api = api_from_args(args)
        check_servers(api)

        if not args.no_pred_cache:
            cache_path = args.pred_canonical_tsv or os.path.join(
                pred_root_path, "canonical_bigsmiles.pred.tsv"
            )
            cache_rows = load_existing_rows(cache_path)
            if cache_rows:
                print(f"Reusing {len(cache_rows)} cached predictions")

    for image_path in iter_filelist(args.test_filelist):
        molfile_gt = gt_molfile_for_image(image_path)
        molfile_pred = pred_molfile_for_image(image_path, pred_root_path)

        with open(molfile_gt, "r") as f_gt:
            molblock_gt = f_gt.read()
        if not os.path.exists(molfile_pred):
            raise FileNotFoundError(
                f"No prediction at {molfile_pred}\n"
                f"Run predict.py for this experiment first, e.g.\n"
                f"    sh scripts/submit_predict.sh"
            )
        with open(molfile_pred, "r") as f_pred:
            molblock_pred = f_pred.read()

        metrics = compare_molblocks(molblock_pred, molblock_gt)
        # Same bucketing the test split was stratified on, so the per-bucket
        # occurrence counts come out as the split intended.
        count = bucket_of_molblock(molblock_gt)
        assert count is not None, f"RDKit could not read {molfile_gt}"

        if count not in exact_matches:
            exact_matches[count] = []
            atom_precisions[count] = []
            atom_recalls[count] = []
            atom_f1s[count] = []
            bond_precisions[count] = []
            bond_recalls[count] = []
            bond_f1s[count] = []
            sgroup_precisions[count] = []
            sgroup_recalls[count] = []
            sgroup_f1s[count] = []
            canonical_matches[count] = []

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

        if args.canonical_match:
            gt_key = gt_tsv_key(image_path, args.data_root)
            canonical_gt, gt_status = gt_canonical.get(gt_key, ("", ""))

            if gt_status not in valid_statuses:
                # No usable ground truth, so this can never match. Counted as
                # a non-match rather than dropped, and never sent to the
                # server. Tallied so the ceiling it imposes stays visible.
                uncanonicalizable[gt_status or GT_MISSING] += 1
                canonical_matches[count].append(0.0)
            else:
                cached = cache_rows.get(image_path)
                if cached:
                    is_match = bool(cached["canonical_bigsmiles"]) and \
                        cached["canonical_bigsmiles"] == \
                        sanitize(canonical_gt).strip()
                else:
                    try:
                        is_match, pred_bigsmiles, pred_canonical, pred_status \
                            = score_canonical(
                                api, molblock_pred, canonical_gt, gt_status,
                                valid_statuses
                            )
                    except ServerUnavailableError as e:
                        print(f"Canonical matching aborted: {e}")
                        args.canonical_match = False
                        server_aborted = True
                        is_match = False
                        pred_canonical = None

                    if pred_canonical is not None and cache_path:
                        cache_rows[image_path] = {
                            "path": image_path,
                            "bigsmiles": sanitize(pred_bigsmiles),
                            "canonical_bigsmiles":
                                sanitize(pred_canonical).strip(),
                            "status": pred_status
                        }
                        write_rows(list(cache_rows.values()), cache_path)

                canonical_matches[count].append(float(is_match))

        print(f"molfile_gt: {molfile_gt}, metrics: {metrics}")

    print(pred_root_path)
    for count in sorted(exact_matches.keys()):
        line = (
            f"count: {count}, occurrences: {len(exact_matches[count])}, "
            f"Exact matches: {np.mean(exact_matches[count]): .2f}, "
            f"Atom F1: {np.mean(atom_f1s[count]): .4f}, "
            f"Bond F1: {np.mean(bond_f1s[count]): .4f}, "
            f"Sgroup F1: {np.mean(sgroup_f1s[count]): .4f}"
        )
        if canonical_matches.get(count):
            line += f", Canon match: {np.mean(canonical_matches[count]): .4f}"
        print(line)

    if any(canonical_matches.values()):
        overall = [m for ms in canonical_matches.values() for m in ms]
        total_uncanonicalizable = sum(uncanonicalizable.values())
        print(
            f"Canonical match (all {len(overall)} samples): "
            f"{np.mean(overall): .4f}"
        )
        print(
            f"  of which {total_uncanonicalizable} have no usable ground-truth "
            f"canonical and can never match, capping this at "
            f"{1 - total_uncanonicalizable / len(overall): .4f}"
        )
        for status, n in sorted(uncanonicalizable.items()):
            print(f"    {status}: {n}")

    if server_aborted:
        # The geometry report above is complete and correct; only the
        # canonical metric is partial, so say so loudly rather than exit 0.
        print(
            "Canonical matching did not finish: a service went away. The "
            "geometry metrics above are complete; rerun to finish the "
            "canonical metric (cached results are reused)."
        )
        sys.exit(1)


if __name__ == "__main__":
    args = get_args()
    main(args)
