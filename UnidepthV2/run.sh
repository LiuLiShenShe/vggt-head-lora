#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: ./run.sh UniDepthV2 </abs/path/to/input_image> </abs/path/to/output_dir>"
  exit 1
fi

MODEL="$1"; shift
INPUT_IMAGE="$1"; shift
OUTPUT_DIR="$1"; shift

if [[ ! -f "${INPUT_IMAGE}" ]]; then
  echo "Input image not found: ${INPUT_IMAGE}"; exit 1
fi
mkdir -p "${OUTPUT_DIR}"

if [[ "${MODEL}" != "UniDepthV2" ]]; then
  echo "Only UniDepthV2 is supported in this repo." >&2; exit 2
fi

bash UniDepthV2/run.sh "${INPUT_IMAGE}" "${OUTPUT_DIR}"
