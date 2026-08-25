#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-unidepth-suite}
IMAGE_TAG=${IMAGE_TAG:-latest}

cd "$(dirname "$0")/.."

docker build -f dockers/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo "Built image: ${IMAGE_NAME}:${IMAGE_TAG}"
