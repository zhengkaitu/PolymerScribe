"""Collect the test samples that match exactly but not canonically.

`exact_match` is a geometry verdict: the predicted molblock has the same atoms,
bonds and S-groups as the ground truth. `canonical_match` is a string verdict on
the canonical BigSMILES derived from that molblock. They disagree more often
than they should, and the disagreements are only inspectable side by side, so
this copies each one into its own directory:

    <out_root>/<category>/<flattened image path>/
        image.png       the source image
        gt.mol          the manually corrected ground truth
        pred.mol        the prediction (geometrically identical, by definition)
        info.txt        both BigSMILES strings and both canonicalization statuses

Nothing is sent to either service: the ground truth comes from
`canonical_bigsmiles.tsv` and the prediction from the cache `evaluate.py` wrote
beside the predictions. Run `evaluate.py --canonical_match` first so that cache
exists.
"""
import argparse
import csv
import os
import shutil
import sys

from evaluate import compare_molblocks
from utilities.canonical_bigsmiles_api import (
    STATUS_BIGSMILES_FAILED,
    VALID_STATUSES,
    load_existing_rows,
    sanitize,
)
from utilities.paths import (
    gt_molfile_for_image,
    gt_tsv_key,
    iter_filelist,
    pred_molfile_for_image,
    stem_path,
)

# Why an exactly-matching prediction still failed to match canonically. The
# first two are the benchmark's own ceiling; the last two are the interesting
# ones, where both sides produced a canonical string and they still differ.
CAT_GT_UNUSABLE = "gt_unusable"
CAT_PRED_BIGSMILES_FAILED = "pred_bigsmiles_failed"
CAT_PRED_CANON_FAILED = "pred_canon_failed"
CAT_CANON_DIFFERS = "canon_differs"
CAT_UNCACHED = "uncached"

SUMMARY_FIELDNAMES = [
    "category",
    "image",
    "case_dir",
    "gt_status",
    "pred_status",
    "gt_bigsmiles",
    "pred_bigsmiles",
    "gt_canonical_bigsmiles",
    "pred_canonical_bigsmiles",
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Copy out the samples where exact_match is true but "
                    "canonical_match is false"
    )
    parser.add_argument("--test_filelist", type=str, default=None, required=True)
    parser.add_argument("--pred_root_path", type=str, default=None, required=True)
    parser.add_argument("--out_root", type=str, default=None,
                        help="defaults to <pred_root_path>/exact_not_canonical")
    parser.add_argument("--gt_canonical_tsv", type=str,
                        default="data/PolymerLit/canonical_bigsmiles.tsv")
    parser.add_argument("--data_root", type=str, default="data/PolymerLit",
                        help="only used to key into --gt_canonical_tsv")
    parser.add_argument("--pred_canonical_tsv", type=str, default=None,
                        help="defaults to "
                             "<pred_root_path>/canonical_bigsmiles.pred.tsv")
    parser.add_argument("--overwrite", action="store_true",
                        help="remove an existing --out_root first")

    return parser.parse_args()


def load_gt_rows(tsv_path: str) -> dict:
    """TSV key -> the whole row, so info.txt can show the pre-canonical string."""
    with open(tsv_path, "r", newline="") as tsvfile:
        return {row["path"]: row for row in csv.DictReader(tsvfile, delimiter="\t")}


def classify(gt_row: dict, pred_row: dict) -> tuple[str, bool]:
    """Return (category, is_canonical_match) for one sample.

    Mirrors what evaluate.py concluded, but from the cache rather than the
    server, and says *why* rather than just yes/no.
    """
    gt_status = (gt_row or {}).get("status", "")
    if gt_status not in VALID_STATUSES:
        return CAT_GT_UNUSABLE, False

    if pred_row is None:
        return CAT_UNCACHED, False

    gt_canonical = sanitize(gt_row["canonical_bigsmiles"]).strip()
    pred_canonical = pred_row["canonical_bigsmiles"].strip()
    if pred_canonical and pred_canonical == gt_canonical:
        return "", True

    if pred_row["status"] == STATUS_BIGSMILES_FAILED:
        return CAT_PRED_BIGSMILES_FAILED, False
    if pred_row["status"] not in VALID_STATUSES:
        return CAT_PRED_CANON_FAILED, False

    return CAT_CANON_DIFFERS, False


def case_dirname(image_path: str) -> str:
    """A flat, unique directory name for an image nested at any depth."""
    return stem_path(image_path).replace("/", "__")


def write_info(path: str, image_path: str, category: str,
               gt_row: dict, pred_row: dict) -> None:
    gt_row = gt_row or {}
    pred_row = pred_row or {}
    lines = [
        f"image                     : {image_path}",
        f"category                  : {category}",
        "",
        f"gt   status               : {gt_row.get('status', '')}",
        f"pred status               : {pred_row.get('status', '(not cached)')}",
        "",
        f"gt   bigsmiles            : {gt_row.get('bigsmiles', '')}",
        f"pred bigsmiles            : {pred_row.get('bigsmiles', '')}",
        "",
        f"gt   canonical_bigsmiles  : {gt_row.get('canonical_bigsmiles', '')}",
        f"pred canonical_bigsmiles  : {pred_row.get('canonical_bigsmiles', '')}",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(args):
    assert os.path.isdir("data"), "run from the repo root"

    out_root = args.out_root or os.path.join(
        args.pred_root_path, "exact_not_canonical"
    )
    if os.path.exists(out_root):
        if not args.overwrite:
            sys.exit(f"{out_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(out_root)

    gt_rows = load_gt_rows(args.gt_canonical_tsv)
    pred_tsv = args.pred_canonical_tsv or os.path.join(
        args.pred_root_path, "canonical_bigsmiles.pred.tsv"
    )
    pred_rows = load_existing_rows(pred_tsv)
    print(f"Loaded {len(gt_rows)} ground-truth rows and "
          f"{len(pred_rows)} cached predictions")

    summary = []
    n_total = 0
    n_exact = 0
    n_canonical = 0

    for image_path in iter_filelist(args.test_filelist):
        molfile_gt = gt_molfile_for_image(image_path)
        molfile_pred = pred_molfile_for_image(image_path, args.pred_root_path)
        if not os.path.exists(molfile_pred):
            raise FileNotFoundError(f"No prediction at {molfile_pred}")

        with open(molfile_gt, "r") as f:
            molblock_gt = f.read()
        with open(molfile_pred, "r") as f:
            molblock_pred = f.read()

        n_total += 1
        exact_match = compare_molblocks(molblock_pred, molblock_gt)["exact_match"]

        gt_row = gt_rows.get(gt_tsv_key(image_path, args.data_root))
        pred_row = pred_rows.get(image_path)
        category, canonical_match = classify(gt_row, pred_row)

        n_exact += int(bool(exact_match))
        n_canonical += int(canonical_match)

        if not exact_match or canonical_match:
            continue

        case_dir = os.path.join(out_root, category, case_dirname(image_path))
        os.makedirs(case_dir, exist_ok=True)
        shutil.copyfile(molfile_gt, os.path.join(case_dir, "gt.mol"))
        shutil.copyfile(molfile_pred, os.path.join(case_dir, "pred.mol"))
        if os.path.exists(image_path):
            shutil.copyfile(image_path, os.path.join(case_dir, "image.png"))
        write_info(os.path.join(case_dir, "info.txt"),
                   image_path, category, gt_row, pred_row)

        summary.append({
            "category": category,
            "image": image_path,
            "case_dir": os.path.relpath(case_dir, out_root),
            "gt_status": (gt_row or {}).get("status", ""),
            "pred_status": (pred_row or {}).get("status", ""),
            "gt_bigsmiles": (gt_row or {}).get("bigsmiles", ""),
            "pred_bigsmiles": (pred_row or {}).get("bigsmiles", ""),
            "gt_canonical_bigsmiles":
                (gt_row or {}).get("canonical_bigsmiles", ""),
            "pred_canonical_bigsmiles":
                (pred_row or {}).get("canonical_bigsmiles", ""),
        })

    os.makedirs(out_root, exist_ok=True)
    summary_path = os.path.join(out_root, "summary.tsv")
    summary.sort(key=lambda row: (row["category"], row["image"]))
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{n_total} test samples: {n_exact} exact, {n_canonical} canonical")
    print(f"{len(summary)} exact but not canonical, written to {out_root}")
    by_category = {}
    for row in summary:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
    for category, n in sorted(by_category.items()):
        print(f"  {category}: {n}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)
