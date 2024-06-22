#!/bin/bash

module load anaconda/2023b
module load cuda/11.6
module load nccl/2.11.4-cuda11.6
source activate polymerscribe

mkdir -p "$SAVE_PATH"

torchrun \
    --nproc_per_node="$NUM_GPUS_PER_NODE" \
    --nnodes="$NUM_NODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py \
    --data_path data \
    --train_file "$TRAIN_FILE" \
    --aux_file "$AUX_FILE" \
    --coords_file aux_file \
    --valid_file "$VALID_FILE" \
    --test_file "$TEST_FILE" \
    --vocab_file "$VOCAB_FILE" \
    --formats chartok_coords,edges \
    --dynamic_indigo --augment --mol_augment \
    --include_condensed \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --encoder_lr "$LR" \
    --decoder_lr "$LR" \
    --save_path "$SAVE_PATH" --save_mode all \
    --label_smoothing 0.1 \
    --epochs "$EPOCH" \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps "$ACCUM_STEP" \
    --use_checkpoint \
    --warmup 0.02 \
    --print_freq "$PRINT_FREQ" \
    --do_train --do_valid --do_test \
    --fp16 --backend nccl \
    2>&1
