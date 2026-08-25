"""Smoke test for VGGT-1B: auto-download weights + forward pass."""
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (room scene, 7 frames)
import glob
image_names = sorted(glob.glob("examples/room/images/*.*"))[:7]
print(f"Using {len(image_names)} images:")
for n in image_names:
    print("  ", n)
images = load_and_preprocess_images(image_names).to(device)
print("input tensor:", images.shape, images.dtype)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

print("\n=== Smoke test PASSED ===")
for k, v in predictions.items():
    if torch.is_tensor(v):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
