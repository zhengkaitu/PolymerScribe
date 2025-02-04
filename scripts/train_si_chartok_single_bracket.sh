#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

BATCH_SIZE=8
ACCUM_STEP=1

LOAD_PATH=ckpts/swin_base_char_aux_1m680k.pth
SAVE_PATH=output/si/swin_base_char_molscribe_finetuned
mkdir -p ${SAVE_PATH}

set -x

python train.py \
    --data_path data \
    --train_file si_mol/single_bracket_train.csv \
    --valid_file si_mol/single_bracket_val.csv \
    --vocab_file molscribe/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --load_path $LOAD_PATH \
    --encoder_lr 4e-4 \
    --decoder_lr 4e-4 \
    --save_path $SAVE_PATH --save_mode last --load_ckpt last \
    --label_smoothing 0.1 \
    --epochs 50 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.02 \
    --print_freq 200 \
    --do_train --do_valid \
    --fp16 --backend nccl 2>&1
