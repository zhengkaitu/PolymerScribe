# ---------------- edit here ----------------
EXP_NO=full_rerun
SPLIT=fully_random
COUNTS="0 200 400 600 800"
# -------------------------------------------

ARGS_LIST=""
for count in $COUNTS; do
    ARGS_LIST="$ARGS_LIST ${SPLIT}_${count}"
done

for args in $ARGS_LIST; do
    OLD_IFS=$IFS; IFS=","; set -- $args; IFS=$OLD_IFS
    export SUFFIX=$1
    export EXP_NO
    python preprocess.py --expt_id="${EXP_NO}_${SUFFIX}"
done
