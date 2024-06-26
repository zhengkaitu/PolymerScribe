IFS=","

export NUM_GPUS_PER_NODE=1
export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=1230

export BATCH_SIZE=256
export ACCUM_STEP=2
export EPOCH=30

export PRINT_FREQ=100

export TRAIN_FILE=polyBERT_len85_0_tokenized_train.csv
export VALID_FILE=polyBERT_len85_0_tokenized_valid.csv
export TEST_FILE=polyBERT_len85_0_tokenized_test.csv
export VOCAB_FILE=polymerscribe/vocab/vocab_polybert.json

mkdir -p logs_slurm/polybert
for args in \
  baseline_0,4e-4 \
  baseline_1,8e-4; \
  do set -- $args;
  export EXP_NO=$1;
  export LR=$2;
  export SAVE_PATH=output/polybert/$1;
  LLsub scripts/train_image2seq.sh -s 20 -g volta:1 -T 48:00:00 -o logs_slurm/polybert/train_"$1"
  done;
