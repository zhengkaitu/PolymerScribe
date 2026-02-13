import csv
import os
import requests
import traceback as tb
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm


class BigSMILESAPI:
    def __init__(self, url: str, port: int) -> None:
        self.url = url
        self.port = port
        self.session = requests.Session()

    def molblock_to_bigsmiles(self, molblock: str) -> str:
        uri = f"{self.url}:{self.port}/api/molblock-to-bigsmiles"
        data = {
            "molblock_string": molblock
        }

        resp = self.session.post(uri, json=data)

        result = resp.json()
        bigsmiles = result.get("data", "failed conversion")

        return bigsmiles

    def bigsmiles_to_molblock(self, bigsmiles: str) -> str:
        uri = f"{self.url}:{self.port}/api/bigsmiles-to-molblock"
        data = {
            "bigsmiles": bigsmiles
        }

        resp = self.session.post(uri, json=data)

        result = resp.json()
        molblock = result.get("data", "failed conversion")

        return molblock


def main():
    bigsmiles_api = BigSMILESAPI(url="http://0.0.0.0", port=3318)
    ofp = "bigsmiles_analysis"
    os.makedirs(ofp, exist_ok=True)

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

    rows = []

    for parent, child in sources:
        path = os.path.join(parent, child)
        os.makedirs(f"bigsmiles_analysis/{child}", exist_ok=True)

        with os.scandir(path) as it:
            for entry in tqdm(sorted(it, key=lambda x: x.name)):
                if not entry.is_file():
                    continue
                if not entry.name.endswith(".corrected.mol"):
                    continue

                corrected_fn = os.path.join(path, entry.name)
                with open(corrected_fn, "r") as f:
                    molblock = f.read()
                converted_bigsmiles = \
                    bigsmiles_api.molblock_to_bigsmiles(molblock)
                if converted_bigsmiles == "failed conversion":
                    converted_molblock = ""
                else:
                    converted_molblock = \
                        bigsmiles_api.bigsmiles_to_molblock(converted_bigsmiles)

                row = {
                    "corrected_fn": corrected_fn,
                    "converted_bigsmiles": converted_bigsmiles,
                    # "corrected_molblock": molblock,
                    # "converted_molblock": converted_molblock
                }
                rows.append(row)

                ofn = os.path.join(ofp, child, entry.name)
                with open(ofn, "w") as of:
                    of.write(molblock)

                converted_fn = entry.name.replace("corrected", "converted")
                ofn = os.path.join(ofp, child, converted_fn)
                with open(ofn, "w") as of:
                    of.write(converted_molblock)

                mol1 = Chem.MolFromMolBlock(molblock)
                mol2 = Chem.MolFromMolBlock(converted_molblock)

                legends = [entry.name.rstrip(".mol"), converted_fn.rstrip(".mol")]

                try:
                    img = Draw.MolsToGridImage(
                        [mol1, mol2],
                        molsPerRow=2,
                        subImgSize=(300, 300),
                        legends=legends,
                        returnPNG=False
                    )
                except Exception as e:
                    tb.print_exc()

                img_fn = entry.name.replace("corrected.mol", "png")
                ofn = os.path.join(ofp, child, img_fn)
                img.save(ofn)

    ofn = os.path.join(ofp, "converted_bigsmiles_gt.csv")
    with open(ofn, "w") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "corrected_fn",
                "converted_bigsmiles",
                # "corrected_molblock",
                # "converted_molblock"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
