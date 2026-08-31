#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DA3Metric batch inference — produces depth_da3.npy for each sequence.

Usage (must run in da3 env):
  /home/test/miniconda3/envs/da3/bin/python run_da3_inference.py [--seq SEQ_ID] [--all]
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)
from configs import load_sequence_meta, SEQUENCES

import torch
from depth_anything_3.api import DepthAnything3


def infer_sequence(model, seq_meta, out_dir):
    """Run DA3 inference on all frames of one sequence, save depth_da3.npy."""
    os.makedirs(out_dir, exist_ok=True)
    rgb_paths = seq_meta["rgb_paths"]
    S = len(rgb_paths)
    depths = []
    failed = []

    for i, rp in enumerate(rgb_paths):
        try:
            with torch.no_grad():
                pred = model.inference([rp])
            d = pred.depth
            if hasattr(d, "cpu"):
                d = d.cpu().numpy()
            d = np.asarray(d, dtype=np.float32)
            if d.ndim == 3 and d.shape[0] == 1:
                d = d[0]
            depths.append(d)
        except Exception as e:
            failed.append((i, str(e)))
            # Append zeros as placeholder
            if depths:
                depths.append(np.zeros_like(depths[0]))
            else:
                depths.append(np.zeros((1, 1), dtype=np.float32))

        if (i + 1) % 50 == 0 or i == S - 1:
            print(f"  [{i+1}/{S}] {'FAIL' if failed and failed[-1][0]==i else 'OK'}")

    depth_stack = np.stack(depths, axis=0)  # (S, H, W)
    out_path = os.path.join(out_dir, "depth_da3.npy")
    np.save(out_path, depth_stack)
    print(f"  Saved {out_path}: shape={depth_stack.shape}, "
          f"range=[{depth_stack.min():.4f}, {depth_stack.max():.4f}]")

    if failed:
        fail_path = os.path.join(out_dir, "failed_frames.json")
        with open(fail_path, "w") as f:
            json.dump(failed, f)
        print(f"  {len(failed)} frames failed, logged to {fail_path}")

    return depth_stack.shape, len(failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, help="Specific sequence ID")
    parser.add_argument("--all", action="store_true", help="Run all sequences")
    args = parser.parse_args()

    if args.all:
        seqs = SEQUENCES
    elif args.seq:
        seqs = [(args.seq, any(s[0] == args.seq for s in SEQUENCES))]
    else:
        seqs = SEQUENCES[:1]  # Default: first sequence only

    print("Loading DA3Metric model...")
    t0 = time.time()
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda")
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    results = {}
    for seq_id, pose_fail in seqs:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id} (pose_fail={pose_fail})")
        print(f"{'='*60}")
        meta = load_sequence_meta(seq_id)
        out_dir = os.path.join(PHASE3_DIR, "da3", seq_id)
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        shape, n_fail = infer_sequence(model, meta, out_dir)
        elapsed = time.time() - t0
        results[seq_id] = {"shape": shape, "n_failed": n_fail, "time_s": round(elapsed, 1)}
        print(f"  Done in {elapsed:.1f}s ({n_fail} failures)")

    manifest_path = os.path.join(PHASE3_DIR, "da3", "INFERENCE_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
