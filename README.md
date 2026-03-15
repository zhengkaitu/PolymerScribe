# PolymerScribe

This is the repository for PolymerScribe, an image-to-graph model that translates a polymer structure to molfile, and subsequently to BigSMILES.

## Deployment (requiring Docker)

### 1. Build and start the containerized service for BigSMILES conversion.

This service provides endpoints for two-way translation between molblocks and BigSMILES. It is required if you want to obtain the BigSMILES (in addition to the molblock) of the image being recognized.

```shell
$ cd bigsmiles-server
$ make build-bigsmiles-image
$ make start-bigsmiles-service
$ cd ..
```

The service can be stopped when no longer needed via

```shell
$ cd bigsmiles-server
$ make stop-bigsmiles-service
$ cd ..
```

### 2. Build and start the containerized service for polymer structure recognition using PolymerScribe.

```shell
$ make build-polymerscribe-image
$ make start-polymerscribe-service
````

The service can be stopped when no longer needed via

```shell
$ make stop-polymerscribe-service
```

### 3. Query the recognition service
We provide sample query commands in `scripts/query.sh`, which can be executed from the command line

```shell
$ bash scripts/query.sh
```

The responses from the service will be printed to the terminal, e.g.,
```shell
{ 
  "status": "SUCCESS",
  "error": "",
  "result": {
    "molblock": str,
    "bigsmiles": str
  }
}
```

## Training and benchmarking (coming soon)

[//]: # (### 1. Create the Conda environment)

[//]: # ()
[//]: # (```shell)

[//]: # ($ conda create -y -n polymerscribe -c conda-forge python=3.9 ipykernel jupyterlab=4.0.13 packaging=21.3)

[//]: # ($ conda activate polymerscribe)

[//]: # ($ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116)

[//]: # ($ pip install -r requirements.txt)

[//]: # (```)

[//]: # ()
[//]: # (The following steps assume that the `polymerscribe` environment has been activated.)

[//]: # ()
[//]: # (### 2. Prepare the data)

[//]: # ()
[//]: # (```shell)

[//]: # ($ bash scripts/download_polymerlit_data.sh)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (### 3. Download pretrained MolScribe checkpoint)

[//]: # ()
[//]: # (```shell)

[//]: # ($ bash scripts/download_molscribe_checkpoint.sh)

[//]: # (```)

[//]: # ()
[//]: # (### 4. Train PolymerScribe)

[//]: # ()
[//]: # (```shell)

[//]: # ($ bash scripts/train_polymerlit.sh)

[//]: # (```)

[//]: # ()
[//]: # (### 5. Predict and evaluate with PolymerScribe)

[//]: # ()
[//]: # (```shell)

[//]: # ($ bash scripts/predict_polymerlit.sh)

[//]: # ($ bash scripts/evaluate_polymerlit.sh)

[//]: # (```)
