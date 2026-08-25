"""Smoke test for Depth-Anything-3 with DA3METRIC-LARGE (monocular metric depth)."""
import glob
import torch
from depth_anything_3.api import DepthAnything3

device = "cuda" if torch.cuda.is_available() else "cpu"

# DA3METRIC-LARGE: monocular metric depth model (0.35B)
model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(device)

# Use VGGT example room images as multi-view input
image_names = sorted(glob.glob("/fj/VGGT+head+lora实验/vggt/examples/room/images/*.*"))[:5]
print(f"Using {len(image_names)} images")

with torch.no_grad():
    prediction = model.inference(image_names)

print("\n=== Smoke test PASSED ===")
d = prediction.depth
print(f"depth: shape={d.shape}, dtype={d.dtype}, min={d.min():.4f}, max={d.max():.4f}")
if hasattr(prediction, "aux") and prediction.aux:
    for k, v in prediction.aux.items():
        if torch.is_tensor(v):
            print(f"  aux/{k}: {tuple(v.shape)}")
