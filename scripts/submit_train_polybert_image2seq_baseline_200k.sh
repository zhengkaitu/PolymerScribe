IFS=","

export NUM_GPUS_PER_NODE=2
export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=1230

export BATCH_SIZE=256
export ACCUM_STEP=1

mkdir -p logs_slurm/polybert
for args in \
  baseline_0,4e-4 \
  baseline_1,8e-4; \
  do set -- $args;
  export EXP_NO=$1;
  export LR=$2;
  export SAVE_PATH=output/polybert/$1;
  LLsub scripts/train_image2seq.sh -s 20 -g volta:2 -T 24:00:00 -o logs_slurm/polybert/train_"$1"
  done;
