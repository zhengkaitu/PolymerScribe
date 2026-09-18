# PolymerScribe

This is the repository for PolymerScribe, an image-to-graph model that translates a polymer structure to molfile, and subsequently to BigSMILES.

## Deployment (requiring Docker)

- **OS**: Ubuntu (>= 20.04) recommended.
- **docker** (\>= version 20). See [Install docker](https://docs.docker.com/engine/install/).
- **git** and **make**.

### 1. Build and start the containerized service for BigSMILES conversion.

This service provides endpoints for two-way translation between molblocks and BigSMILES, based on [Deagen et al.](https://doi.org/10.1021/acs.macromol.3c01378) It is required if you want to obtain the BigSMILES (in addition to the molblock) of the image being recognized.

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

### 2. Build and start the containerized service for BigSMILES canonicalization.

This service provides endpoints for BigSMILES canonicalization, based on [Leão et al.](https://doi.org/10.1021/acs.jcim.5c02784) It is required if you want to obtain the canonical BigSMILES, especially if you want to evaluate performance with consideration of graph isomorphism.

```shell
$ cd canonicalization-server
$ make build-canonicalization-image
$ make start-canonicalization-service
$ cd ..
```

The service can be stopped when no longer needed via

```shell
$ cd canonicalization-server
$ make stop-canonicalization-service
$ cd ..
```

### 3. Build and start the containerized service for polymer structure recognition using PolymerScribe.

```shell
$ sh scripts/download_polymerscribe_model.sh
$ make build-polymerscribe-image
$ make start-polymerscribe-service
````

The service can be stopped when no longer needed via

```shell
$ make stop-polymerscribe-service
```

### 3. Query the recognition service

Once started, the API documentation would be available at `http://0.0.0.0:3310/docs`, where you can upload polymer structures (as .png image), try out the API, and get the molblock string (and optionally the BigSMILES representation) in return.

We also provide sample query commands in `scripts/query.sh`, which can be executed from the command line

```shell
$ sh scripts/query.sh
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

## Utilities

### Canonical BigSMILES for the PolymerLit ground truth

`utilities/get_all_canonical_bigsmiles.py` walks every `*.corrected.mol` under
`data/PolymerLit/` (at any depth), converts each molblock to BigSMILES via the
bigsmiles-server, canonicalizes it via the canonicalization-server, and writes
`data/PolymerLit/canonical_bigsmiles.tsv` with four tab-separated columns: the
path relative to `data/PolymerLit/`, the BigSMILES, the canonical BigSMILES,
and a status.

Both services must be running:

```shell
$ cd bigsmiles-server && make build-bigsmiles-image && make start-bigsmiles-service && cd ..
$ cd canonicalization-server && make build-canonicalization-image && make start-canonicalization-service && cd ..
$ python -m utilities.get_all_canonical_bigsmiles
```

Run it as a module from the repo root, as above, so that its `utilities.*`
imports resolve. The script refuses to start if either service is unreachable,
and prints the `make` target that starts it.

The status column records how the canonicalization went, and is what
downstream code uses to decide whether a ground truth is usable:

| status | meaning |
|---|---|
| `SUCCESS` | canonicalized to a different string |
| `NOOP` | already canonical; the canonical form equals the input |
| `NO_STOCHASTIC_OBJECT` | no stochastic object to enumerate, input returned unchanged |
| `FAIL` | canonicalization failed |
| `BIGSMILES_FAILED` | no BigSMILES could be obtained; both columns read `failed to obtain BigSMILES` |

The column matters because `SUCCESS` is the only case that is self-evident from
the strings alone: on any failure the input is handed back unchanged, which
looks exactly like a molecule that was already canonical. If you have an older
three-column TSV, fill the column in without recomputing the rows that are
already settled:

```shell
$ python -m utilities.get_all_canonical_bigsmiles --reclassify
```

Some PolymerLit structures contain nested stochastic objects that make the
canonicalization server's graph enumeration exhaust memory, so
`start-canonicalization-service` runs the container under a memory limit and a
restart policy (see `canonicalization-server/Makefile`, overridable via
`MEMORY_LIMIT` and `RESTART_POLICY`). The script cooperates with this: it
distinguishes a slow molecule (recorded as a failure), from an input that
crashes the service (retried once, then recorded as a failure), from a service
that is gone for good (the run aborts rather than emitting placeholder rows).

Progress is written to the TSV after every row. If a run aborts, restart the
service and continue where it left off:

```shell
$ cd canonicalization-server && make stop-canonicalization-service && make start-canonicalization-service && cd ..
$ python -m utilities.get_all_canonical_bigsmiles --resume
```

## Training and benchmarking

All commands are run from the repo root, in the order given. Each stage is a
serial shell loop over the experiments, with a single block of variables at the
top of the script to edit (`EXP_NO`, `SPLIT`, `COUNTS`).

### 1. Create the Conda environment

```shell
$ conda create -y -n polymerscribe -c conda-forge python=3.9 ipykernel jupyterlab=4.0.13 packaging=21.3
$ conda activate polymerscribe
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
```

The following steps assume that the `polymerscribe` environment has been
activated, and that `data/PolymerLit/` is populated.

### 2. Create the train/val/test filelists

```shell
$ python -m utilities.create_filelists_for_all_splits
```

This writes, for each training-set size in `--counts` (default
`0 200 400 600 800`):

```
experiments/<exp>_<split>_<count>/<exp>_<split>_<count>_{train,val,test}.filelist.txt
```

A filelist holds one path per line. A line ending in `/` is a directory whose
`*.png` files are all included; any other line is a single image. The synthetic
subsets (PolymerLit-MT and PolymerLit-Olsen) are emitted as directory lines and
are always entirely in train, so only the PolymerLit-OA images are split.

Val and test are drawn only from images whose ground-truth canonical BigSMILES
is valid (status `SUCCESS` or `NOOP`), so that every held-out sample can be
scored on canonical BigSMILES. Images without one still go to train, where they
are just as useful: every png/mol pair was manually inspected and corrected.
This needs `data/PolymerLit/canonical_bigsmiles.tsv` with its status column, so
run the utility above first.

**The `ladder` subset is exempt from that filter.** Essentially no ladder
structure survives canonicalization, so filtering would empty the ladder
holdout entirely; it is split 40/5/5 like before. Those samples are still
scored on the geometry metrics, but they can never match on canonical
BigSMILES — a limitation of the current canonicalization engine, not of the
model. See the note under step 6.

### 3. Preprocess

```shell
$ sh scripts/submit_preprocess.sh
```

Reads each experiment's filelists and writes the corresponding
`*_{train,val,test}.processed.csv` beside them.

### 4. Train

```shell
$ sh scripts/submit_train.sh
```

Trains one model per training-set size into `output/<exp>_<split>_<count>/`,
logging to `logs/<exp>/`. Hyperparameters (`BATCH_SIZE`, `LR`, `EPOCH`) are in
the edit block at the top.

Before the sweep, this also writes a **cold-start** checkpoint to
`output/<exp>_cold_start/` via `train.py --save_init_and_exit`: the pretrained
MolScribe weights mapped into the PolymerScribe architecture, saved before
train step 0. Its atom and bond heads are fully pretrained, but the widened
edge head and the bracket capability are randomly initialized, so it is the
honest "before any training" reference point.

Validation runs on the full val set. `train.py --val_limit N` truncates it to
the first N rows for a quick smoke test.

### 5. Predict

```shell
$ sh scripts/submit_predict.sh
```

Writes `<stem>.predicted.mol` under `predictions/image_comparison_<id>/`,
mirroring each image's own path so that evaluation can find it regardless of
how deeply the subset is nested.

By default only the test split is predicted, which is all evaluation reads. Set
`FULL_CORPUS=1` in the edit block to run the whole corpus instead and produce
the side-by-side comparison figures for every image, at roughly 17x the GPU
cost.

### 6. Evaluate

```shell
$ sh scripts/submit_evaluate.sh
```

Prints metrics bucketed by ground-truth heavy atom count (in tens, with
everything at 50 or above in one bucket) and writes them to
`logs/<exp>/<id>.evaluate.log`:

- `Atom F1`, `Bond F1`, `Sgroup F1` — Hungarian matching on normalized 2D
  coordinates, so they measure the recovered structure rather than string
  equality.
- `Exact matches` — all three F1s equal to 1.
- `Canon match` — the predicted molblock converted to BigSMILES, canonicalized,
  and compared with the ground-truth canonical BigSMILES. This is the
  graph-isomorphism-aware number, and it needs both services up. Drop
  `--canonical_match` (the `CANONICAL` variable) to skip it and get the
  geometry metrics alone.

`Canon match` is deliberately conservative: **every** test sample is in the
denominator, including those whose own ground truth could not be canonicalized,
which therefore can never match. The report prints how many those are and the
ceiling they impose, broken down by status. Because the generic pool is already
filtered to valid canonicals, those samples are exactly the `ladder` ones — 5 of
100 in the current split, so `Canon match` is capped at 0.95. Predicted
canonicals are cached to `<pred_root>/canonical_bigsmiles.pred.tsv`, so
re-running evaluation is free.
