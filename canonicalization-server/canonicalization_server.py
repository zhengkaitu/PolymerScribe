import argparse
import copy
import logging
import os
import shutil
import sys
import traceback
import uvicorn
from canon_tools import canonicalize_bigsmiles
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
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
    parser.add_argument("--server_port", help="Server port to use", type=int, default=3319)
    parser.add_argument("--model_path", help="Model path", type=str, default="output/r1a/best.pth")
    parser.add_argument("--log_file", help="Log file", type=str, default="canonicalization_server")

    return parser.parse_args()


class CanonicalizationInput(BaseModel):
    bigsmiles: str


class CanonicalizationAPI:
    @staticmethod
    def canonicalize(bigsmiles: str) -> str:
        output_folder = os.path.join("Output", bigsmiles)
        os.makedirs(output_folder, exist_ok=True)

        canonical = canonicalize_bigsmiles(
            bigsmiles=bigsmiles,
            output_folder=output_folder,
            plot=False
        )

        # remove output folder as part of the cleanup
        shutil.rmtree(output_folder)

        return canonical


@app.post("/canonicalize-bigsmiles/")
def CanonicalizationService(data: CanonicalizationInput):
    response = copy.deepcopy(base_response)
    bigsmiles = data.bigsmiles

    try:
        canonical = canonicalization_api.canonicalize(bigsmiles)
        result = {"canonical_bigsmiles": canonical}

        if canonical == bigsmiles:
            response["error"] = "Canonical BigSMILES is the same as the original, possible error!"

        response["result"] = result
        response["status"] = "SUCCESS"

        return response

    except Exception:
        response["error"] = f"Unhandled error during BigSMILES canonicalization, traceback: " \
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
    canonicalization_api = CanonicalizationAPI()

    # start running
    uvicorn.run(app, host=args.server_ip, port=args.server_port)
