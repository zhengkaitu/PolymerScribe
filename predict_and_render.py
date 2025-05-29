import argparse
import cv2
import matplotlib.pyplot as plt
import os
import torch
from molscribe import MolScribe
from rdkit import Chem
from rdkit.Chem import Draw

import warnings 
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    args = parser.parse_args()

    device = torch.device('cuda')
    model = MolScribe(args.model_path, device)

    images_root_path = "./data/doc_images"
    sources = [
        "bigsmiles_manuscript",
        "bigsmiles_si",
        "canonicalization_manuscript",
        "canonicalization_si",
        "non-covalent_manuscript",
        "non-covalent_si"
    ]
    images_root_output_path = "./data/doc_images/predictions"

    images_root_path = "./data/si_mol"
    sources = ["."]
    images_root_output_path = "./data/si_mol/predictions"

    # images_root_path = "./data/realistic_images"
    # sources = [
    #     "acs_applied_polymer_materials",
    #     "acs_macro_letters",
    #     "acs_polymers_au",
    #     "macromolecules",
    #     "polymer_chemistry"
    # ]
    # images_root_output_path = "./data/realistic_images/predictions"

    os.makedirs(images_root_output_path, exist_ok=True)
    for source in sources:
        path = os.path.join(images_root_path, source)
        with os.scandir(path) as it:
            for entry in sorted(it, key=lambda x: x.name):
                if entry.is_file():
                    # if entry.name.endswith(".png") or entry.name.endswith(".jpg"):
                    if not entry.name == "203.png":
                        continue
                    if entry.name.endswith(".png"):
                        image_path = os.path.join(path, entry.name)
                        print(f"Processing {image_path}")
                    else:
                        continue
                else:
                    continue

                output = model.predict_image_file(
                    image_path,
                    return_atoms_bonds=False,
                    return_confidence=False
                )
                molblock = output["molfile"]
                print(output)
                # exit(0)
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.imread(image_path))

                plt.subplot(1, 2, 2)
                try:
                    mol = Chem.MolFromMolBlock(molblock)
                    img = Draw.MolToImage(mol)
                    plt.imshow(img)
                except:
                    print(f"Error processing {image_path}")
                    pass

                plt.savefig(f"{images_root_output_path}/{entry.name}")
