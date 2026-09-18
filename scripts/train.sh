#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

ACCUM_STEP=1

LOAD_PATH=${LOAD_PATH:-ckpts/swin_base_char_aux_1m680k.pth}
# Cold-start runs set DO_VAL="" and EXTRA_TRAIN_ARGS=--save_init_and_exit, so
# the model is written out before training and no validation pass is run.
DO_VAL=${DO_VAL:---do_val}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
mkdir -p "$SAVE_PATH"

set -x

python train.py \
    --data_path data \
    --train_files "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --vocab_file molscribe/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --num_bond_type 9 \
    --load_path "$LOAD_PATH" \
    --encoder_lr "$LR" \
    --decoder_lr "$LR" \
    --save_path "$SAVE_PATH" --save_mode last \
    --label_smoothing 0.1 \
    --epochs "$EPOCH" \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup_ratio 0.02 \
    --print_freq 200 \
    --do_train \
    $DO_VAL \
    $EXTRA_TRAIN_ARGS \
    --fp16 --backend nccl 2>&1
