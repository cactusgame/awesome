#!/usr/bin/env bash

# usage: sh run_training.sh --algo_id prebuild_v1

set -x
ROOT_PATH=/root
ALGO_BASE_PATH=/root/src

TRAIN_OPTIONS=""

while [ "${1:-}" != "" ]; do
    case "$1" in
        "--algo_id")
            shift
            ALGO_ID=$1
            TRAIN_OPTIONS="${TRAIN_OPTIONS} --algo_id=${ALGO_ID} "
            ;;

    esac
    shift
done

echo "Training Model for"
echo "AlgoID: "${ALGO_ID}
echo ""
echo "TRAINING OPTIONS: "${TRAIN_OPTIONS}

# Export Model details as environment vars.
export ALGO_ID=${ALGO_ID}

mkdir -p /tmp/log

# Launch TrainingPipeline entry script.
export PYTHONPATH=${PYTHONPATH}:${ROOT_PATH}:.:${ALGO_BASE_PATH}
python -u ${ALGO_BASE_PATH}/algorithm/${ALGO_ID}/main.py ${TRAIN_OPTIONS}
# python -u src/algorithm/prebuild_v1/main.py ${TRAIN_OPTIONS}

if [ "$?" = "0" ]; then
    rm -f /tmp/log/train-${ALGO_ID}.log
else
   echo "training error"
fi

sleep 30
exit 0