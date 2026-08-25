#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-unidepth-suite}
IMAGE_TAG=${IMAGE_TAG:-latest}

# If arguments are provided, execute them inside the container; otherwise, open a shell
CMD_ARGS=("bash")
if [[ $# -gt 0 ]]; then
  CMD_ARGS=("bash" "-lc" "$*")
fi

HOST_ROOT=$(cd "$(dirname "$0")/.." && pwd)

RUNTIME_ARGS=( )
if command -v nvidia-smi >/dev/null 2>&1; then
  RUNTIME_ARGS+=(--gpus all)
fi

docker run --rm -it \
  "${RUNTIME_ARGS[@]}" \
  -v "${HOST_ROOT}":/workspace \
  -w /workspace \
  ${IMAGE_NAME}:${IMAGE_TAG} \
  "${CMD_ARGS[@]}"
