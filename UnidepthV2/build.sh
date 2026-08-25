#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-unidepthv2}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Build the Docker image
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo "Built image: ${IMAGE_NAME}:${IMAGE_TAG}"
