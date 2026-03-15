import argparse
import copy
import logging
import os
import requests
import sys
import torch
import traceback
import uvicorn
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from molscribe import MolScribe
from rdkit import RDLogger

app = FastAPI()

base_response = {
    "status": "FAIL",
    "error": "",
    "result": {}
}


def parse_args():
    parser = argparse.ArgumentParser("rxnmapper_server")
    parser.add_argument("--server_ip", help="Server IP to use", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", help="Server port to use", type=int, default=3310)
    parser.add_argument("--model_path", help="Model path", type=str, default="output/r1a/best.pth")
    parser.add_argument("--log_file", help="Log file", type=str, default="polymerscribe_server")

    return parser.parse_args()


class PolymerScribeAPI:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = MolScribe(args.model_path, self.device)
        self.bigsmiles_url = "http://0.0.0.0:3318/api/molblock-to-bigsmiles"
        self.session = requests.Session()

    def predict_molblock(self, image_path: str) -> str:
        output = self.model.predict_image_file(
            image_path,
            return_atoms_bonds=False,
            return_confidence=False
        )
        molblock = output["molfile"]

        return molblock

    def molblock_to_bigsmiles(self, molblock: str) -> str:
        try:
            response = self.session.post(
                url=self.bigsmiles_url,
                json={"molblock_string": molblock},
                timeout=30
            )
            output = response.json()
            bigsmiles = output["data"]

            return bigsmiles

        except:
            return "Failed to convert molblock to bigsmiles."


@app.post("/predict-polymer-image/")
def PolymerScribeService(png_file: UploadFile, return_bigsmiles: bool = False):
    response = copy.deepcopy(base_response)

    try:
        fn = png_file.filename
        print(f"Uploaded filename: {fn}")
        contents = png_file.file.read()
        with open(fn, "wb") as f:
            f.write(contents)

        molblock = polymserscribe_api.predict_molblock(image_path=fn)

        if not molblock:
            response["status"] = "FAIL"
            response["error"] = "Molblock prediction failed."

            return response

        result = {"molblock": molblock}

        if return_bigsmiles:
            bigsmiles = polymserscribe_api.molblock_to_bigsmiles(molblock)
            result["bigsmiles"] = bigsmiles

        response["result"] = result
        response["status"] = "SUCCESS"

        return response

    except Exception:
        response["error"] = f"Unhandled error during PolymerScribe prediction, traceback: " \
                            f"{traceback.format_exc()}"
        traceback.print_exc()

        return response


if __name__ == "__main__":
    args = parse_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")

    os.makedirs(f"./logs", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"./logs/{args.log_file}.{dt}.log")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # set up apis
    polymserscribe_api = PolymerScribeAPI()

    # start running
    uvicorn.run(app, host=args.server_ip, port=args.server_port)
