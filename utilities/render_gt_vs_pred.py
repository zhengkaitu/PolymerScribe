"""Render ground truth against prediction, side by side, for visual triage.

`predict.py` already draws a source-image / prediction pair, which answers "did
the model read this image". It does not answer "where does the prediction
differ from the ground truth", because the ground truth is never drawn. That
question only has a visual answer -- the metrics say an S-group is wrong, not
which bracket moved -- so this renders all three panels:

    source image | ground truth | prediction

with the per-sample metrics in the title. RDKit draws SRU brackets and their
labels, so an S-group-only failure is visible as a bracket in the wrong place
rather than as a number.

Output is grouped so the interesting samples cluster together, and named with
the ground truth's heavy-atom count so a directory listing sorts by molecule
size:

    <out_root>/<group>/<natoms>__<flattened image path>.png

The default grouping is by failure mode, which separates the two cases that
look identical in the aggregate metrics: a prediction whose skeleton is perfect
and only the S-groups are wrong, versus one that misread atoms or bonds.

Canonical-match columns are filled in when the TSVs are there and quietly left
blank when they are not, so this runs without either server and without having
run `evaluate.py --canonical_match` first.

Run as a module from the repo root:

    python -m utilities.render_gt_vs_pred \
        --test_filelist experiments/full_rerun_fully_random_800/full_rerun_fully_random_800_test.filelist.txt \
        --pred_root_path predictions/image_comparison_full_rerun_fully_random_800
"""
import argparse
import csv
import io
import os
import shutil
import sys
import traceback as tb

import matplotlib
matplotlib.use("Agg")           # headless: this writes files, never a window
import matplotlib.pyplot as plt
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

from evaluate import compare_molblocks
from utilities.paths import (
    gt_molfile_for_image,
    gt_tsv_key,
    iter_filelist,
    pred_molfile_for_image,
    stem_path,
)

RDLogger.DisableLog("rdApp.*")

# Grouping by failure mode. A prediction that gets every atom and bond right
# and still misses exact_match failed on S-groups alone -- a different problem
# from a misread skeleton, and the aggregate metrics do not distinguish them.
# The S-group case splits again: reading the subscript text inside the bracket
# is OCR and fails on its own, so `sgroup_label` collects the samples whose
# only defect is that string, leaving `sgroup_geometry` for the ones where the
# brackets themselves are wrong.
GROUP_EXACT = "exact"
GROUP_SGROUP_LABEL = "sgroup_label"
GROUP_SGROUP_GEOMETRY = "sgroup_geometry"
GROUP_SKELETON = "skeleton_error"
GROUP_UNPARSEABLE = "unparseable"

VALID_STATUSES = ("SUCCESS", "NOOP")

# A landscape canvas, because these are polymer backbones: nearly all of them
# are far wider than they are tall, and a square canvas scales them down to fit
# a height they never use. Every panel is letterboxed to exactly this size so
# the three line up instead of floating at whatever offset their own aspect
# ratio implies.
PANEL_W = 720
PANEL_H = 480


def get_args():
    parser = argparse.ArgumentParser(
        description="Render GT vs predicted molfiles side by side for visual "
                    "inspection"
    )
    parser.add_argument("--test_filelist", type=str, required=True)
    parser.add_argument("--pred_root_path", type=str, required=True)
    parser.add_argument("--out_root", type=str, default=None,
                        help="defaults to <pred_root_path>/gt_vs_pred")
    parser.add_argument("--gt_canonical_tsv", type=str,
                        default="data/PolymerLit/canonical_bigsmiles.tsv",
                        help="optional; only used to annotate canonical match")
    parser.add_argument("--data_root", type=str, default="data/PolymerLit",
                        help="only used to key into --gt_canonical_tsv")
    parser.add_argument("--pred_canonical_tsv", type=str, default=None,
                        help="defaults to "
                             "<pred_root_path>/canonical_bigsmiles.pred.tsv")
    parser.add_argument("--group_by", choices=["failure", "bucket", "none"],
                        default="failure",
                        help="subdirectory layout under --out_root")
    parser.add_argument("--only", choices=["all", "mismatch"], default="all",
                        help="'mismatch' skips the samples that match exactly")
    parser.add_argument("--no_image", action="store_true",
                        help="omit the source-image panel")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--overwrite", action="store_true",
                        help="remove an existing --out_root first")

    return parser.parse_args()


def load_tsv(path: str) -> dict:
    """path -> row, or {} when the TSV is absent. Annotation is optional."""
    if not path or not os.path.exists(path):
        return {}

    with open(path, "r", newline="") as tsvfile:
        return {row["path"]: row for row in csv.DictReader(tsvfile, delimiter="\t")}


def letterbox(img: Image.Image, width: int, height: int) -> Image.Image:
    """Fit an image into a fixed canvas on white, preserving aspect ratio.

    Every panel ends up the same pixel size, so matplotlib lays the three out
    on a shared baseline. Without this a tall source image and a wide molecule
    render at different heights and the eye has to re-register between panels.
    """
    scale = min(width / img.width, height / img.height)
    resized = img.convert("RGB").resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2,
                           (height - resized.height) // 2))

    return canvas


def render_molblock(molblock: str, width: int = PANEL_W, height: int = PANEL_H):
    """A molblock as a PIL image, or None if RDKit cannot parse it.

    sanitize=False because these molblocks carry R-group and wildcard atoms
    that would not survive sanitization, and useMolBlockWedging so the drawn
    stereo matches what the file actually says rather than what RDKit would
    infer.
    """
    mol = Chem.MolFromMolBlock(molblock, sanitize=False)
    if mol is None:
        return None

    # Without this an aliased atom draws as its element symbol -- "C" or "R" --
    # and the abbreviation that distinguishes it is invisible. Same trick as
    # predict.py:draw_comparison.
    for atom in mol.GetAtoms():
        try:
            atom.SetProp("atomLabel", atom.GetProp("molFileAlias"))
        except KeyError:
            pass

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().useMolBlockWedging = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def heavy_atom_count(molblock: str) -> int:
    """Matches how evaluate.py buckets samples, so the two agree."""
    mol = Chem.MolFromMolBlock(molblock, sanitize=False)

    return mol.GetNumHeavyAtoms() if mol is not None else -1


def sgroup_summary(molblock: str) -> str:
    """"SRU:n, SRU:1-x" -- the S-groups as drawn, for the title."""
    mol = Chem.MolFromMolBlock(molblock, sanitize=False)
    if mol is None:
        return "(unparseable)"

    parts = []
    for sgroup in Chem.GetMolSubstanceGroups(mol):
        props = sgroup.GetPropsAsDict()
        label = props.get("LABEL", "")
        parts.append(f"{props.get('TYPE', '?')}:{label}" if label
                     else str(props.get("TYPE", "?")))

    return ", ".join(parts) if parts else "(none)"


def classify(metrics: dict) -> str:
    if metrics.get("exact_match"):
        return GROUP_EXACT
    if metrics["atom_f1"] == 0.0 and metrics["bond_f1"] == 0.0:
        return GROUP_UNPARSEABLE
    if metrics["atom_f1"] == 1.0 and metrics["bond_f1"] == 1.0:
        # exact_match_nolabel forgives the subscript text and nothing else, so
        # it is exactly the test for "only the label is wrong". Absent from a
        # prediction scored before that metric existed; those fall through to
        # the geometry bucket, as they did before the split.
        if metrics.get("exact_match_nolabel"):
            return GROUP_SGROUP_LABEL

        return GROUP_SGROUP_GEOMETRY

    return GROUP_SKELETON


def canonical_verdict(gt_row: dict, pred_row: dict) -> str:
    """A short human-readable canonical-match verdict for the title."""
    if gt_row is None and pred_row is None:
        return "n/a"

    gt_status = (gt_row or {}).get("status", "?")
    if gt_status not in VALID_STATUSES:
        return f"no (GT {gt_status}, can never match)"
    if pred_row is None:
        return "n/a (not cached)"

    gt_canonical = (gt_row.get("canonical_bigsmiles") or "").strip()
    pred_canonical = (pred_row.get("canonical_bigsmiles") or "").strip()
    if pred_canonical and pred_canonical == gt_canonical:
        return "yes"

    return f"no (pred {pred_row.get('status', '?')})"


def figure_title(image_path: str, metrics: dict, group: str, natoms: int,
                 canon: str, gt_sgroups: str, pred_sgroups: str) -> str:
    return "\n".join([
        image_path,
        f"{group}   |   heavy atoms {natoms}   |   exact {bool(metrics['exact_match'])}"
        f"   |   canonical {canon}",
        f"atom F1 {metrics['atom_f1']:.3f}   bond F1 {metrics['bond_f1']:.3f}"
        f"   sgroup F1 {metrics['sgroup_f1']:.3f}",
        f"GT sgroups   [{gt_sgroups}]",
        f"pred sgroups [{pred_sgroups}]",
    ])


def draw_case(image_path: str, molblock_gt: str, molblock_pred: str,
              metrics: dict, group: str, natoms: int, canon: str,
              out_path: str, show_image: bool, dpi: int) -> None:
    panels = []
    if show_image and os.path.exists(image_path):
        panels.append(("source image", Image.open(image_path)))
    panels.append(("ground truth", render_molblock(molblock_gt)))
    panels.append(("prediction", render_molblock(molblock_pred)))

    # Five title lines above three equal panels. The rect leaves room for the
    # title; tight_layout alone would let it overlap the top panel.
    fig = plt.figure(figsize=(5.0 * len(panels), 4.2))
    try:
        for i, (label, img) in enumerate(panels, start=1):
            axis = fig.add_subplot(1, len(panels), i)
            if img is None:
                axis.text(0.5, 0.5, "unparseable", ha="center", va="center",
                          family="monospace")
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
            else:
                axis.imshow(letterbox(img, PANEL_W, PANEL_H))
            axis.set_title(label, fontsize=10)
            axis.axis("off")

        fig.suptitle(
            figure_title(image_path, metrics, group, natoms, canon,
                         sgroup_summary(molblock_gt),
                         sgroup_summary(molblock_pred)),
            fontsize=8, family="monospace", y=0.995, va="top",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.83))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
    finally:
        # Without this the loop leaks a figure per sample.
        plt.close(fig)


def out_filename(image_path: str, natoms: int) -> str:
    """Heavy-atom count first, so a directory listing sorts by molecule size."""
    flat = stem_path(image_path).replace("/", "__")

    return f"{natoms:03d}__{flat}.png"


def main(args):
    assert os.path.isdir("data"), "run from the repo root"

    out_root = args.out_root or os.path.join(args.pred_root_path, "gt_vs_pred")
    if os.path.exists(out_root):
        if not args.overwrite:
            sys.exit(f"{out_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(out_root)

    gt_rows = load_tsv(args.gt_canonical_tsv)
    pred_rows = load_tsv(args.pred_canonical_tsv or os.path.join(
        args.pred_root_path, "canonical_bigsmiles.pred.tsv"
    ))
    if not gt_rows or not pred_rows:
        print("Canonical TSVs missing or empty; rendering without canonical "
              "annotation")

    counts = {}
    n_rendered = 0
    n_skipped = 0

    for image_path in iter_filelist(args.test_filelist):
        molfile_gt = gt_molfile_for_image(image_path)
        molfile_pred = pred_molfile_for_image(image_path, args.pred_root_path)
        if not os.path.exists(molfile_pred):
            raise FileNotFoundError(f"No prediction at {molfile_pred}")

        with open(molfile_gt, "r") as f:
            molblock_gt = f.read()
        with open(molfile_pred, "r") as f:
            molblock_pred = f.read()

        metrics = compare_molblocks(molblock_pred, molblock_gt)
        group = classify(metrics)
        counts[group] = counts.get(group, 0) + 1

        if args.only == "mismatch" and group == GROUP_EXACT:
            n_skipped += 1
            continue

        natoms = heavy_atom_count(molblock_gt)
        canon = canonical_verdict(
            gt_rows.get(gt_tsv_key(image_path, args.data_root)),
            pred_rows.get(image_path),
        )

        if args.group_by == "failure":
            subdir = group
        elif args.group_by == "bucket":
            subdir = f"atoms_{min(natoms // 10 * 10, 50):02d}"
        else:
            subdir = ""

        out_path = os.path.join(out_root, subdir,
                                out_filename(image_path, natoms))
        try:
            draw_case(image_path, molblock_gt, molblock_pred, metrics, group,
                      natoms, canon, out_path, not args.no_image, args.dpi)
        except Exception:
            print(f"Error rendering {image_path}")
            tb.print_exc()
            continue

        n_rendered += 1
        print(f"[{n_rendered}] {group:>14}  {out_path}")

    print(f"\nWrote {n_rendered} figures to {out_root}"
          + (f" ({n_skipped} exact matches skipped)" if n_skipped else ""))
    for group in sorted(counts):
        print(f"  {group:>14}: {counts[group]}")


if __name__ == "__main__":
    main(get_args())
