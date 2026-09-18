# ---------------- edit here ----------------
EXP_NO=full_rerun
SPLIT=fully_random
COUNTS="0 200 400 600 800"
export BATCH_SIZE=8
export LR=4e-4
export EPOCH=20
# -------------------------------------------

mkdir -p logs/$EXP_NO

# Cold start: the model before train step 0, i.e. the pretrained weights
# mapped into the PolymerScribe architecture with the widened heads still
# randomly initialized. It needs a processed train/val pair only to build the
# model, so it reuses the smallest experiment's.
COLD_ID=${EXP_NO}_cold_start
BASE_ID=${EXP_NO}_${SPLIT}_0
export SAVE_PATH=output/${COLD_ID}
export TRAIN_FILE=experiments/${BASE_ID}/${BASE_ID}_train.processed.csv
export VAL_FILE=experiments/${BASE_ID}/${BASE_ID}_val.processed.csv
DO_VAL="" EXTRA_TRAIN_ARGS="--save_init_and_exit" \
    sh scripts/train.sh > logs/$EXP_NO/"${COLD_ID}".log 2>&1

ARGS_LIST=""
for count in $COUNTS; do
    ARGS_LIST="$ARGS_LIST ${SPLIT}_${count}"
done

for args in $ARGS_LIST; do
    OLD_IFS=$IFS; IFS=","; set -- $args; IFS=$OLD_IFS
    export SUFFIX=$1
    export ID=${EXP_NO}_${SUFFIX}
    export SAVE_PATH=output/${ID}
    export TRAIN_FILE=experiments/${ID}/${ID}_train.processed.csv
    export VAL_FILE=experiments/${ID}/${ID}_val.processed.csv
    sh scripts/train.sh > logs/$EXP_NO/"${ID}".log 2>&1
done
