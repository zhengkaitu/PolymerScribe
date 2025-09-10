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
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

import warnings 
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--id', type=str, default="", required=True)
    parser.add_argument("--diff", action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda')
    model = MolScribe(args.model_path, device) if args.model_path else None

    sources = [
        ("./data", "mt_images_processed"),
        ("./data/olsen_images_processed", "bigsmiles_manuscript"),
        ("./data/olsen_images_processed", "bigsmiles_si"),
        ("./data/olsen_images_processed", "canonicalization_manuscript"),
        ("./data/olsen_images_processed", "canonicalization_si"),
        ("./data/olsen_images_processed", "non-covalent_manuscript"),
        ("./data/olsen_images_processed", "non-covalent_si"),
        ("./data/realistic_images_processed", "generic/acspolymersau"),
        ("./data/realistic_images_processed", "generic/acsmacrolett"),
        ("./data/realistic_images_processed", "generic/macromolecules"),
        ("./data/realistic_images_processed", "ladder/acsmacrolett"),
        ("./data/realistic_images_processed", "ladder/angewchemie"),
        ("./data/realistic_images_processed", "ladder/chemengjournal"),
        ("./data/realistic_images_processed", "ladder/chemicalscience"),
        ("./data/realistic_images_processed", "ladder/digitaldiscovery"),
        ("./data/realistic_images_processed", "ladder/faradaydiscussions"),
        ("./data/realistic_images_processed", "ladder/macromolecules"),
        ("./data/realistic_images_processed", "ladder/polymer")
    ]

    root_output_path = f"predictions/image_comparison_{args.id}"
    os.makedirs(root_output_path, exist_ok=True)

    for parent, child in sources:
        path = os.path.join(parent, child)
        images_output_path = os.path.join(root_output_path, child)
        os.makedirs(images_output_path, exist_ok=True)

        with os.scandir(path) as it:
            for entry in sorted(it, key=lambda x: x.name):
                corrected_fn = os.path.join(path, f"{entry.name[:-4]}.corrected.mol")

                if not entry.is_file():
                    continue
                if not entry.name.endswith(".png"):
                    continue
                if args.diff and os.path.exists(corrected_fn):
                    continue

                if not args.model_path:
                    assert os.path.exists(corrected_fn), corrected_fn

                image_path = os.path.join(path, entry.name)
                print(f"Processing {image_path}")

                if model:
                    output = model.predict_image_file(
                        image_path,
                        return_atoms_bonds=False,
                        return_confidence=False
                    )
                    molblock = output["molfile"]
                    with open(f"{images_output_path}/{entry.name[:-4]}.predicted.mol", "w") as of:
                        of.write(molblock)
                else:
                    with open(corrected_fn, "r") as f:
                        molblock = f.read()

                # print(molblock)
                # exit(0)
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.imread(image_path))

                plt.subplot(1, 2, 2)
                try:
                    mol = Chem.MolFromMolBlock(molblock, sanitize=False)
                    # print(f"mol: {mol}")
                    # draw the RGroups
                    for a in mol.GetAtoms():
                        try:
                            a.SetProp("atomLabel", a.GetProp("molFileAlias"))
                        except KeyError:
                            pass

                    # img = Draw.MolToImage(mol)
                    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
                    opts = drawer.drawOptions()
                    opts.useMolBlockWedging = True  # keep wedges as in molfile
                    drawer.DrawMolecule(mol)
                    drawer.FinishDrawing()

                    # Convert drawing to an IPython Image
                    png = drawer.GetDrawingText()
                    img = Image.open(io.BytesIO(png))

                    plt.imshow(img)
                except Exception as e:
                    print(f"Error processing {image_path}")
                    tb.print_exc()

                plt.savefig(f"{images_output_path}/{entry.name[:-4]}.predicted.png")
