# ---------------- edit here ----------------
EXP_NO=full_rerun
SPLIT=fully_random
COUNTS="0 200 400 600 800"
EXTRA_IDS="cold_start"
# Canonical BigSMILES matching needs both services up; drop this flag to get
# the geometry metrics alone.
CANONICAL="--canonical_match"
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
    export TEST_FL=experiments/"$TEST_ID"/"${TEST_ID}"_test.filelist.txt
    export PRED_ROOT_PATH=predictions/image_comparison_"$ID"
    python evaluate.py \
        --test_filelist="$TEST_FL" \
        --pred_root_path="$PRED_ROOT_PATH" \
        $CANONICAL 2>&1 | tee logs/$EXP_NO/"${ID}".evaluate.log
done
