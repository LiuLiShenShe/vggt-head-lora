#!/usr/bin/env bash
set -euo pipefail

INPUT_IMAGE=${1:-}
OUTPUT_DIR=${2:-}

if [[ -z "${INPUT_IMAGE}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: UniDepthV2/run.sh </abs/path/to/input_image> </abs/path/to/output_dir>"
  exit 1
fi

python UniDepthV2/infer.py --input "${INPUT_IMAGE}" --output "${OUTPUT_DIR}" --model "lpiccinelli/unidepth-v2-vitb14"
