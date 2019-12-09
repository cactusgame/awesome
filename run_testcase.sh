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
        "--train_steps")
            shift
            TRAIN_STEPS=$1
            TRAIN_OPTIONS="${TRAIN_OPTIONS} --train_steps=${TRAIN_STEPS} "
            ;;
        "--download_feature_db")
            shift
            DOWNLOAD_FEATURE_DB=$1
            TRAIN_OPTIONS="${TRAIN_OPTIONS} --download_feature_db=${DOWNLOAD_FEATURE_DB} "
            ;;
        "--do_preprocessing")
            shift
            DO_PREPROCESSING=$1
            TRAIN_OPTIONS="${TRAIN_OPTIONS} --do_preprocessing=${DO_PREPROCESSING} "
            ;;
        "--upload_model")
            shift
            UPLOAD_MODEL=$1
            TRAIN_OPTIONS="${TRAIN_OPTIONS} --upload_model=${UPLOAD_MODEL} "
            ;;
    esac
    shift
done

echo "Training Model for"
echo "ALGO_ID: "${ALGO_ID}
echo "TRAIN_STEPS: "${TRAIN_STEPS}
echo "DOWNLOAD_FEATURE_DB: "${DOWNLOAD_FEATURE_DB}
echo "DO_PREPROCESSING: "${DO_PREPROCESSING}
echo "UPLOAD_MODEL: "${UPLOAD_MODEL}
echo ""
echo "TRAINING OPTIONS: "${TRAIN_OPTIONS}

# Export Model details as environment vars.
export ALGO_ID=${ALGO_ID}
export TRAIN_STEPS=${TRAIN_STEPS}
export DOWNLOAD_FEATURE_DB=${DOWNLOAD_FEATURE_DB}
export DO_PREPROCESSING=${DO_PREPROCESSING}

mkdir -p /tmp/log

# Launch TrainingPipeline entry script.
export PYTHONPATH=${PYTHONPATH}:${ROOT_PATH}:.:${ALGO_BASE_PATH}
python -u ${ALGO_BASE_PATH}/research/bolling.py

echo "do testcase complete"

sleep 10
exit 0