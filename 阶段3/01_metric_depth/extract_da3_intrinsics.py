#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DA3 inference with intrinsics extraction — re-run saving depth + predicted intrinsics.

DA3 is fast (~0.07s/frame), so we re-run ALL frames to get intrinsics.
The original run_da3_inference.py only saved depth, not intrinsics.

Usage (must run in da3 env):
  /home/test/miniconda3/envs/da3/bin/python extract_da3_intrinsics.py [--seq SEQ_ID] [--all]
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
AUDIT_DIR = os.path.join(PHASE3_DIR, "phase3a1_audit")
sys.path.insert(0, PHASE3_DIR)
from configs import load_sequence_meta, SEQUENCES

import torch
from depth_anything_3.api import DepthAnything3


def infer_sequence(model, seq_meta, out_dir):
    """Run DA3 on all frames, save depth_da3.npy + intrinsics_da3.npy."""
    os.makedirs(out_dir, exist_ok=True)
    rgb_paths = seq_meta["rgb_paths"]
    S = len(rgb_paths)
    depths = []
    intrinsics_list = []
    failed = []

    for i, rp in enumerate(rgb_paths):
        try:
            with torch.no_grad():
                pred = model.inference([rp])

            # Depth: (N, H, W) -> (H, W)
            d = pred.depth
            if hasattr(d, "cpu"):
                d = d.cpu().numpy()
            d = np.asarray(d, dtype=np.float32)
            if d.ndim == 3 and d.shape[0] == 1:
                d = d[0]
            depths.append(d)

            # Intrinsics: (N, 3, 3) -> (3, 3) — predicted by cam_dec
            if pred.intrinsics is not None:
                ix = pred.intrinsics
                if hasattr(ix, "cpu"):
                    ix = ix.cpu().numpy()
                ix = np.asarray(ix, dtype=np.float32)
                if ix.ndim == 3 and ix.shape[0] == 1:
                    ix = ix[0]
                intrinsics_list.append(ix)
            else:
                # Fallback: should not happen for DA3METRIC
                intrinsics_list.append(np.zeros((3, 3), dtype=np.float32))
                failed.append((i, "intrinsics is None"))

        except Exception as e:
            failed.append((i, str(e)))
            if depths:
                depths.append(np.zeros_like(depths[0]))
                intrinsics_list.append(np.zeros((3, 3), dtype=np.float32))
            else:
                depths.append(np.zeros((1, 1), dtype=np.float32))
                intrinsics_list.append(np.zeros((3, 3), dtype=np.float32))

        if (i + 1) % 50 == 0 or i == S - 1:
            print(f"  [{i+1}/{S}] {'FAIL' if failed and failed[-1][0]==i else 'OK'}")

    depth_stack = np.stack(depths, axis=0)  # (S, H, W)
    intr_stack = np.stack(intrinsics_list, axis=0)  # (S, 3, 3)

    depth_path = os.path.join(out_dir, "depth_da3.npy")
    intr_path = os.path.join(out_dir, "intrinsics_da3.npy")
    np.save(depth_path, depth_stack)
    np.save(intr_path, intr_stack)

    print(f"  depth: {depth_path} shape={depth_stack.shape} range=[{depth_stack.min():.4f}, {depth_stack.max():.4f}]")
    print(f"  intrinsics: {intr_path} shape={intr_stack.shape}")
    # Print first frame intrinsics for sanity
    if intr_stack.shape[0] > 0:
        K0 = intr_stack[0]
        print(f"  frame 0 K: fx={K0[0,0]:.2f} fy={K0[1,1]:.2f} cx={K0[0,2]:.2f} cy={K0[1,2]:.2f}")
        focal_mean = (K0[0,0] + K0[1,1]) / 2
        print(f"  frame 0 focal_mean/300 = {focal_mean/300:.4f}")

    if failed:
        fail_path = os.path.join(out_dir, "failed_frames.json")
        with open(fail_path, "w") as f:
            json.dump(failed, f)
        print(f"  {len(failed)} frames failed")

    return depth_stack.shape, intr_stack.shape, len(failed)


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
        seqs = SEQUENCES[:1]

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
        out_dir = os.path.join(AUDIT_DIR, "da3", seq_id)
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        depth_shape, intr_shape, n_fail = infer_sequence(model, meta, out_dir)
        elapsed = time.time() - t0
        results[seq_id] = {
            "depth_shape": list(depth_shape),
            "intrinsics_shape": list(intr_shape),
            "n_failed": n_fail,
            "time_s": round(elapsed, 1),
        }
        print(f"  Done in {elapsed:.1f}s ({n_fail} failures)")

    manifest_path = os.path.join(AUDIT_DIR, "da3", "INFERENCE_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
