#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3A smoke test — 1 frame per model, must pass before batch inference.

Usage:
  # DA3:
  /home/test/miniconda3/envs/da3/bin/python run_smoke_test.py --model da3
  # UniDepth:
  /home/test/miniconda3/envs/unidepth/bin/python run_smoke_test.py --model unidepth
  # VGGT (any env with numpy):
  python run_smoke_test.py --model vggt
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)
from configs import load_sequence_meta, get_depth_path, get_mask_path, VGGT_RERUN

SEQ_ID = "plantview__langdon_4__05-03-24"


def test_vggt():
    """Load existing VGGT depth — no inference needed."""
    path = os.path.join(VGGT_RERUN, SEQ_ID, "depth_vggt.npy")
    assert os.path.exists(path), f"VGGT depth not found: {path}"
    d = np.load(path)
    assert d.ndim == 3, f"Expected (S,H,W), got {d.shape}"
    S, H, W = d.shape
    return {
        "model": "vggt", "path": path,
        "shape": list(d.shape), "dtype": str(d.dtype),
        "min": float(d.min()), "max": float(d.max()),
        "mean": float(d.mean()),
        "note": "already computed, no inference",
    }


def test_da3():
    """Load DA3Metric and infer 1 frame."""
    from PIL import Image
    import torch
    from depth_anything_3.api import DepthAnything3

    meta = load_sequence_meta(SEQ_ID)
    rgb_path = meta["rgb_paths"][0]
    assert os.path.exists(rgb_path), f"RGB not found: {rgb_path}"

    t0 = time.time()
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda")
    t_load = time.time() - t0

    with torch.no_grad():
        t0 = time.time()
        prediction = model.inference([rgb_path])
        t_infer = time.time() - t0

    depth = prediction.depth
    if hasattr(depth, "cpu"):
        depth = depth.cpu().numpy()
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]

    # Load reference and mask for sanity check
    ref_path = get_depth_path(meta["depth_dir"], rgb_path)
    ref = np.asarray(Image.open(ref_path)).astype(np.float64) * 0.001
    mask_path = get_mask_path(meta["mask_dir"], rgb_path)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0

    return {
        "model": "da3", "repo": "depth-anything/DA3METRIC-LARGE",
        "shape": list(depth.shape), "dtype": str(depth.dtype),
        "min": float(depth.min()), "max": float(depth.max()),
        "mean": float(depth.mean()),
        "ref_shape": list(ref.shape), "mask_shape": list(mask.shape),
        "load_time_s": round(t_load, 2), "infer_time_s": round(t_infer, 2),
        "rgb_path": rgb_path,
    }


def test_unidepth():
    """Load UniDepthV2 and infer 1 frame."""
    from PIL import Image
    import torch
    from unidepth.models import UniDepthV2

    meta = load_sequence_meta(SEQ_ID)
    rgb_path = meta["rgb_paths"][0]
    assert os.path.exists(rgb_path), f"RGB not found: {rgb_path}"

    t0 = time.time()
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to("cuda").eval()
    t_load = time.time() - t0

    rgb = np.array(Image.open(rgb_path))
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        t0 = time.time()
        result = model.infer(rgb_tensor.cuda())
        t_infer = time.time() - t0

    depth = result["depth"].squeeze().cpu().numpy()
    intrinsics = result.get("intrinsics", None)
    if intrinsics is not None:
        intrinsics = intrinsics.squeeze().cpu().numpy()
    confidence = result.get("confidence", None)
    if confidence is not None:
        confidence = confidence.squeeze().cpu().numpy()

    return {
        "model": "unidepth", "repo": "lpiccinelli/unidepth-v2-vitl14",
        "shape": list(depth.shape), "dtype": str(depth.dtype),
        "min": float(depth.min()), "max": float(depth.max()),
        "mean": float(depth.mean()),
        "intrinsics_shape": list(intrinsics.shape) if intrinsics is not None else None,
        "confidence_shape": list(confidence.shape) if confidence is not None else None,
        "load_time_s": round(t_load, 2), "infer_time_s": round(t_infer, 2),
        "rgb_path": rgb_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vggt", "da3", "unidepth", "all"], default="all")
    args = parser.parse_args()

    results = {}
    tests = {
        "vggt": test_vggt,
        "da3": test_da3,
        "unidepth": test_unidepth,
    }

    models_to_test = list(tests.keys()) if args.model == "all" else [args.model]

    for m in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {m}...")
        print(f"{'='*60}")
        try:
            r = tests[m]()
            results[m] = {"status": "PASS", **r}
            print(f"  PASS — shape={r['shape']}, range=[{r['min']:.4f}, {r['max']:.4f}]")
            if "load_time_s" in r:
                print(f"  load={r['load_time_s']}s  infer={r['infer_time_s']}s")
        except Exception as e:
            results[m] = {"status": "FAIL", "error": str(e)}
            print(f"  FAIL — {e}")

    out_path = os.path.join(PHASE3_DIR, "smoke_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Exit non-zero if any failed
    for m, r in results.items():
        if r["status"] == "FAIL":
            sys.exit(1)


if __name__ == "__main__":
    main()
