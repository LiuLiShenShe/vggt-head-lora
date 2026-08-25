#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-unidepthv2}
IMAGE_TAG=${IMAGE_TAG:-latest}
INPUT_IMAGE=${1:-}
USE_CPU=${USE_CPU:-0}
MODEL=${MODEL:-lpiccinelli/unidepth-v2-vitb14}

if [[ -z "${INPUT_IMAGE}" ]]; then
  echo "Usage: ./run_inference.sh /absolute/path/to/image.(jpg|png|jpeg)"
  echo "       USE_CPU=1 MODEL=lpiccinelli/unidepth-v2-vits14 ./run_inference.sh /abs/path/to/image.jpg"
  exit 1
fi

if [[ ! -f "${INPUT_IMAGE}" ]]; then
  echo "Input image not found: ${INPUT_IMAGE}"
  exit 1
fi

# Ensure output directory exists on host
HOST_ROOT=$(pwd)
HOST_OUTPUT_DIR="${HOST_ROOT}/output"
mkdir -p "${HOST_OUTPUT_DIR}"

# Use NVIDIA runtime for GPU if available and not forcing CPU
RUNTIME_ARGS=( )
if [[ "${USE_CPU}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  RUNTIME_ARGS+=(--gpus all)
fi

# Limit threads to reduce segfault risk
ENV_ARGS=( -e USE_CPU=${USE_CPU} -e XFORMERS_DISABLED=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e MODEL=${MODEL} )

# Run container, mount project and image path, write outputs to ./output
docker run --rm -it \
  "${RUNTIME_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  -v "${HOST_ROOT}":/workspace \
  -v "${INPUT_IMAGE}":/workspace/input_image:ro \
  -w /workspace \
  ${IMAGE_NAME}:${IMAGE_TAG} \
  bash -lc "python - <<'PY' \ 
import os, torch, numpy as np
from PIL import Image
from pathlib import Path
from unidepth.models import UniDepthV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

image_path = Path('/workspace/input_image')
out_dir = Path('/workspace/output')
model_name = os.getenv('MODEL', 'lpiccinelli/unidepth-v2-vitb14')

# Select device
force_cpu = os.getenv('USE_CPU', '0') == '1'
if force_cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = UniDepthV2.from_pretrained(model_name).to(device)
model.eval()

# Load and process image
rgb_np = np.array(Image.open(image_path).convert('RGB'))
rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).to(device)

with torch.inference_mode():
    pred = model.infer(rgb)

# Depth to HxW
D = pred['depth'].detach().float().cpu()
while D.ndim > 2:
    D = D.squeeze(0)
D = D.numpy()

# Normalize
finite = np.isfinite(D)
scale = float(np.percentile(D[finite], 99)) if finite.any() else 1.0
if not np.isfinite(scale) or scale <= 0:
    scale = 1.0
vis = np.clip(D / scale, 0, 1)

# Save 16-bit depth
from PIL import Image as PILImage
(PILImage.fromarray((vis * 65535.0).astype(np.uint16), mode='I;16')).save(out_dir / 'depth.png')

# Save colorized depth
cmap = plt.get_cmap('turbo')
colored = (cmap(vis)[:, :, :3] * 255).astype(np.uint8)
PILImage.fromarray(colored).save(out_dir / 'depth_colored.png')

print('Saved:', str(out_dir / 'depth.png'))
print('Saved:', str(out_dir / 'depth_colored.png'))
PY"
