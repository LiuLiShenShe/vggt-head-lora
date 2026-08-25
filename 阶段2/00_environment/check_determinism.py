"""阶段2.0 gate 验证:三个模型确定性(重复推理一致性)检查。

对同一输入各推理两次,比对输出是否一致(VGGT bf16 autocast 下允许微小浮点容差)。
结果写入 00_environment/determinism_check.json。
"""
import glob
import json
import os
import sys

IMAGES = sorted(glob.glob("/fj/VGGT+head+lora实验/vggt/examples/room/images/*.*"))[:3]
OUT = "/fj/VGGT+head+lora实验/阶段2/00_environment/determinism_check.json"
results = {}

def max_abs_diff(a, b):
    import numpy as np
    return float(np.abs(a - b).max())

# ---- 1. VGGT (vggt_lora env, bf16 autocast) ----
def check_vggt():
    sys.path.insert(0, "/fj/VGGT+head+lora实验/vggt")
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    images = load_and_preprocess_images(IMAGES).to("cuda")
    outs = []
    with torch.no_grad():
        for _ in range(2):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                p = model(images)
            outs.append({k: (v.float().cpu().numpy() if torch.is_tensor(v) else v)
                         for k, v in p.items()})
    diffs = {k: max_abs_diff(outs[0][k], outs[1][k])
             for k in outs[0] if isinstance(outs[0][k], __import__("numpy").ndarray)}
    return {"max_abs_diff_per_key": diffs,
            "deterministic": all(d < 1e-2 for d in diffs.values()),
            "precision_mode": "bf16 autocast"}

# ---- 2. DA3METRIC-LARGE (da3 env, fp32) ----
def check_da3():
    import torch
    from depth_anything_3.api import DepthAnything3
    import numpy as np
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda")
    outs = []
    with torch.no_grad():
        for _ in range(2):
            pred = model.inference(IMAGES)
            outs.append(pred.depth.astype(np.float64))
    d = max_abs_diff(outs[0], outs[1])
    return {"max_abs_diff_depth": d, "deterministic": d == 0.0, "precision_mode": "fp32"}

# ---- 3. UniDepthV2 (unidepth env, fp32) ----
def check_unidepth():
    import torch
    from unidepth.models import UniDepthV2
    import numpy as np
    from PIL import Image
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to("cuda").eval()
    rgb = np.array(Image.open(IMAGES[0]).convert("RGB"))
    x = torch.from_numpy(rgb).permute(2, 0, 1)
    outs = []
    with torch.no_grad():
        for _ in range(2):
            pred = model.infer(x)
            outs.append(pred["depth"].cpu().numpy().astype(np.float64))
    d = max_abs_diff(outs[0], outs[1])
    return {"max_abs_diff_depth": d, "deterministic": d == 0.0, "precision_mode": "fp32"}

CHECKS = {"vggt": check_vggt, "da3metric": check_da3, "unidepth_v2": check_unidepth}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = CHECKS if which == "all" else {which: CHECKS[which]}
    results["_input_images"] = [os.path.basename(p) for p in IMAGES]
    for name, fn in todo.items():
        print(f"== checking {name} ...", flush=True)
        try:
            r = fn()
        except Exception as e:
            r = {"error": str(e), "deterministic": False}
        results[name] = r
        print(json.dumps(r, indent=2), flush=True)
        # free GPU memory between models
        import torch
        torch.cuda.empty_cache()
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT}")
