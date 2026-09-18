# ---------------- edit here ----------------
EXP_NO=full_rerun
SPLIT=fully_random
COUNTS="0 200 400 600 800"
# Extra ids that have a checkpoint but no training run of their own.
EXTRA_IDS="cold_start"
CKPT=swin_base_transformer_last.pth
# Set to 1 to predict the whole corpus instead of just the test split. That
# produces the side-by-side comparison figures for every image, at ~17x the
# GPU cost, and is not needed for evaluation.
FULL_CORPUS=0
# -------------------------------------------

mkdir -p logs/$EXP_NO

ARGS_LIST=""
for count in $COUNTS; do
    ARGS_LIST="$ARGS_LIST ${SPLIT}_${count}"
done
ARGS_LIST="$ARGS_LIST $EXTRA_IDS"

for args in $ARGS_LIST; do
    OLD_IFS=$IFS; IFS=","; set -- $args; IFS=$OLD_IFS
    export SUFFIX=$1
    export ID=${EXP_NO}_${SUFFIX}
    # cold_start is a checkpoint, not a data split, so it borrows the test
    # filelist of the base experiment. val/test are identical across counts.
    case "$SUFFIX" in
        cold_start) TEST_ID=${EXP_NO}_${SPLIT}_0 ;;
        *)          TEST_ID=$ID ;;
    esac

    if [ "$FULL_CORPUS" = "1" ]; then
        SCOPE=""
    else
        SCOPE="--filelist=experiments/${TEST_ID}/${TEST_ID}_test.filelist.txt --no_figure"
    fi

    python predict.py \
        --id="$ID" \
        --model_path=output/"$ID"/"$CKPT" \
        $SCOPE > logs/$EXP_NO/"${ID}".predict.log 2>&1
done
