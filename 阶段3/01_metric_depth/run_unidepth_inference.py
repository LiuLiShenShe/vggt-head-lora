#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UniDepthV2 batch inference — produces depth_unidepth.npy for each sequence.

Also saves intrinsics_unidepth.npy and confidence_unidepth.npy.

Usage (must run in unidepth env):
  /home/test/miniconda3/envs/unidepth/bin/python run_unidepth_inference.py [--seq SEQ_ID] [--all]
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)
from configs import load_sequence_meta, SEQUENCES

import torch
from PIL import Image
from unidepth.models import UniDepthV2


def infer_sequence(model, seq_meta, out_dir):
    """Run UniDepthV2 inference on all frames of one sequence."""
    os.makedirs(out_dir, exist_ok=True)
    rgb_paths = seq_meta["rgb_paths"]
    S = len(rgb_paths)
    depths, intrinsics_list, confidences = [], [], []
    failed = []

    for i, rp in enumerate(rgb_paths):
        try:
            rgb = np.array(Image.open(rp))
            rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            with torch.no_grad():
                result = model.infer(rgb_tensor.cuda())

            d = result["depth"].squeeze().cpu().numpy().astype(np.float32)
            depths.append(d)

            intr = result.get("intrinsics", None)
            if intr is not None:
                intrinsics_list.append(intr.squeeze().cpu().numpy().astype(np.float32))

            conf = result.get("confidence", None)
            if conf is not None:
                confidences.append(conf.squeeze().cpu().numpy().astype(np.float32))

        except Exception as e:
            failed.append((i, str(e)))
            if depths:
                depths.append(np.zeros_like(depths[0]))
                if intrinsics_list:
                    intrinsics_list.append(np.zeros_like(intrinsics_list[0]))
                if confidences:
                    confidences.append(np.zeros_like(confidences[0]))
            else:
                depths.append(np.zeros((1, 1), dtype=np.float32))

        if (i + 1) % 50 == 0 or i == S - 1:
            print(f"  [{i+1}/{S}] {'FAIL' if failed and failed[-1][0]==i else 'OK'}")

    depth_stack = np.stack(depths, axis=0)
    np.save(os.path.join(out_dir, "depth_unidepth.npy"), depth_stack)
    print(f"  depth: shape={depth_stack.shape}, range=[{depth_stack.min():.4f}, {depth_stack.max():.4f}]")

    if intrinsics_list:
        intr_stack = np.stack(intrinsics_list, axis=0)
        np.save(os.path.join(out_dir, "intrinsics_unidepth.npy"), intr_stack)
        print(f"  intrinsics: shape={intr_stack.shape}")

    if confidences:
        conf_stack = np.stack(confidences, axis=0)
        np.save(os.path.join(out_dir, "confidence_unidepth.npy"), conf_stack)
        print(f"  confidence: shape={conf_stack.shape}")

    if failed:
        with open(os.path.join(out_dir, "failed_frames.json"), "w") as f:
            json.dump(failed, f)
        print(f"  {len(failed)} frames failed")

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
        seqs = SEQUENCES[:1]

    print("Loading UniDepthV2 model...")
    t0 = time.time()
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to("cuda").eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    results = {}
    for seq_id, pose_fail in seqs:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id} (pose_fail={pose_fail})")
        print(f"{'='*60}")
        meta = load_sequence_meta(seq_id)
        out_dir = os.path.join(PHASE3_DIR, "unidepth_v2", seq_id)
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        shape, n_fail = infer_sequence(model, meta, out_dir)
        elapsed = time.time() - t0
        results[seq_id] = {"shape": shape, "n_failed": n_fail, "time_s": round(elapsed, 1)}
        print(f"  Done in {elapsed:.1f}s ({n_fail} failures)")

    manifest_path = os.path.join(PHASE3_DIR, "unidepth_v2", "INFERENCE_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
