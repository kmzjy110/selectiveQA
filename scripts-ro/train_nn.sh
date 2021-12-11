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

echo "Train NN Calibrator:"
declare -a EXPOSE_PREFIXES=( "hotpotqa" )
for EXPOSE_PREFIX in "${EXPOSE_PREFIXES[@]}"; do
        echo "Exposed to: ${EXPOSE_PREFIX}, tested on ${INPUT_DATASET_PREFIX}"
        python3 run_experiment.py ${MODEL_DIR} qa nn_train --target_prefix=${INPUT_DATASET_PREFIX} --expose_prefix=${EXPOSE_PREFIX} --output_dir=${OUTPUT_DIR} --bert_model bert-base-uncased --do_lower_case
    done
echo
echo "Wrote results to ${OUTPUT_DIR}"
echo
