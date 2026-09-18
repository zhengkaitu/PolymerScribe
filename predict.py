import argparse
import cv2
import io
import matplotlib.pyplot as plt
import os
import torch
import traceback as tb
from molscribe import MolScribe
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from utilities.paths import (
    gt_molfile_for_image,
    iter_filelist,
    normalize_image_path,
    pred_figure_for_image,
    pred_molfile_for_image,
)

import warnings
warnings.filterwarnings('ignore')


# Fallback when no --filelist is given: the whole corpus, for generating
# comparison figures beyond the test split.
SOURCES = [
    "data/PolymerLit/PolymerLit-MT_processed",
    "data/PolymerLit/PolymerLit-Olsen_processed/bigsmiles_manuscript",
    "data/PolymerLit/PolymerLit-Olsen_processed/bigsmiles_si",
    "data/PolymerLit/PolymerLit-Olsen_processed/canonicalization_manuscript",
    "data/PolymerLit/PolymerLit-Olsen_processed/canonicalization_si",
    "data/PolymerLit/PolymerLit-Olsen_processed/non-covalent_manuscript",
    "data/PolymerLit/PolymerLit-Olsen_processed/non-covalent_si",
    "data/PolymerLit/PolymerLit-OA_processed/generic/acspolymersau",
    "data/PolymerLit/PolymerLit-OA_processed/generic/acsmacrolett",
    "data/PolymerLit/PolymerLit-OA_processed/generic/macromolecules",
    "data/PolymerLit/PolymerLit-OA_processed/ladder/acsmacrolett",
    "data/PolymerLit/PolymerLit-OA_processed/ladder/angewchemie",
    "data/PolymerLit/PolymerLit-OA_processed/ladder/chemengjournal",
    "data/PolymerLit/PolymerLit-OA_processed/ladder/chemicalscience",
    "data/PolymerLit/PolymerLit-OA_processed/ladder/digitaldiscovery",
    "data/PolymerLit/PolymerLit-OA_processed/ladder/faradaydiscussions",
    "data/PolymerLit/PolymerLit-OA_processed/ladder/macromolecules",
    "data/PolymerLit/PolymerLit-OA_processed/ladder/polymer"
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Run PolymerScribe over images and write molblocks"
    )
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--id', type=str, default="", required=True)
    parser.add_argument("--diff", action='store_true',
                        help="only process images without a .corrected.mol")
    parser.add_argument("--filelist", type=str, default=None,
                        help="predict only these images (e.g. a test "
                             "filelist); without it, the whole corpus is run")
    parser.add_argument("--no_figure", action="store_true",
                        help="skip the side-by-side comparison figure, which "
                             "dominates runtime")

    return parser.parse_args()


def iter_source_images(sources: list[str]):
    """Every *.png under the fallback source directories."""
    for source in sources:
        with os.scandir(source) as it:
            for entry in sorted(it, key=lambda x: x.name):
                if entry.is_file() and entry.name.endswith(".png"):
                    yield normalize_image_path(
                        os.path.join(source, entry.name)
                    )


def draw_comparison(image_path: str, molblock: str, figure_path: str) -> None:
    """Save an input-image / rendered-molblock side-by-side figure."""
    fig = plt.figure(figsize=(8, 4))
    try:
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.imread(image_path))

        plt.subplot(1, 2, 2)
        try:
            mol = Chem.MolFromMolBlock(molblock, sanitize=False)
            # draw the RGroups
            for a in mol.GetAtoms():
                try:
                    a.SetProp("atomLabel", a.GetProp("molFileAlias"))
                except KeyError:
                    pass

            drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
            opts = drawer.drawOptions()
            opts.useMolBlockWedging = True  # keep wedges as in molfile
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()

            png = drawer.GetDrawingText()
            plt.imshow(Image.open(io.BytesIO(png)))
        except Exception:
            print(f"Error rendering {image_path}")
            tb.print_exc()

        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        plt.savefig(figure_path)
    finally:
        # Without this the loop leaks a figure per image.
        plt.close(fig)


def process_image(model, image_path: str, pred_root: str, args) -> None:
    corrected_fn = gt_molfile_for_image(image_path)

    if args.diff and os.path.exists(corrected_fn):
        return

    if not args.model_path:
        assert os.path.exists(corrected_fn), corrected_fn

    print(f"Processing {image_path}")

    if model:
        output = model.predict_image_file(
            image_path,
            return_atoms_bonds=False,
            return_confidence=False
        )
        molblock = output["molfile"]

        # The prediction mirrors the image's own path, so evaluate.py can find
        # it without knowing anything about how the corpus is nested.
        pred_path = pred_molfile_for_image(image_path, pred_root)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        with open(pred_path, "w") as of:
            of.write(molblock)
    else:
        with open(corrected_fn, "r") as f:
            molblock = f.read()

    if not args.no_figure:
        draw_comparison(
            image_path, molblock, pred_figure_for_image(image_path, pred_root)
        )


def main(args):
    assert os.path.isdir("data"), "run from the repo root"

    device = torch.device('cuda')
    model = MolScribe(args.model_path, device) if args.model_path else None

    pred_root = f"predictions/image_comparison_{args.id}"
    os.makedirs(pred_root, exist_ok=True)

    if args.filelist:
        image_paths = list(iter_filelist(args.filelist))
        print(f"Predicting {len(image_paths)} images from {args.filelist}")
    else:
        image_paths = list(iter_source_images(SOURCES))
        print(f"Predicting {len(image_paths)} images from the whole corpus")

    for image_path in image_paths:
        process_image(model, image_path, pred_root, args)


if __name__ == "__main__":
    main(get_args())
