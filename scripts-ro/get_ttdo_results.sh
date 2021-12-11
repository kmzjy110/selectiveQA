#!/bin/bash
set -eu -o pipefail

if [ $# -lt 3 ]
then
    echo "Usage: $0 model_dir target_prefix output_dir"
    exit 1
fi

MODEL_DIR=$1
INPUT_DATASET_PREFIX=$2
OUTPUT_DIR=$3

echo
echo "Evaluating model ${MODEL_DIR} on ${INPUT_DATASET_PREFIX}"
echo

echo "Test-time dropout (mean):"
python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=mean --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo

echo "Test-time dropout (-var):"
python3 run_experiment.py ${MODEL_DIR} qa ttdo --ttdo_type=neg_var --target_prefix=${INPUT_DATASET_PREFIX} --output_dir=${OUTPUT_DIR}
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo


