#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B.1 Step 3: Starting-Frame Sensitivity Test.

For the 3 FAIL dates, tests whether VGGT pose failure is sensitive to:
1. Starting frame offset (circular shifts)
2. Reverse frame order

Runs independent VGGT forward passes with 16 views at each offset.

Output: STARTING_FRAME_SENSITIVITY.csv
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = "/fj/VGGT+head+lora实验"
VGGT_ROOT = os.path.join(ROOT, "vggt")
sys.path.insert(0, VGGT_ROOT)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences", "plant_view")
OUT_DIR = os.path.join(PHASE3B, "02_pose_inference", "viewcount_outputs")

# Reuse evaluation functions
sys.path.insert(0, os.path.join(PHASE3B, "03_pose_evaluation"))
from evaluate_multoplant import (
    global_rotation_procrustes, rot_angle_deg, w2c_centers, horn_sim3
)

FAIL_DATES = ["12-03-24", "15-04-24", "19-03-24"]
N_VIEWS = 16
OFFSET_RANGE = list(range(0, 320, 4))  # 80 offsets (0°, ~4.5°, ~9°, ...)


def load_sequence(date):
    """Load sequence JSON for a langdon_4 date."""
    jp = os.path.join(SEQ_BASE, f"langdon_4__{date}.json")
    with open(jp) as f:
        return json.load(f)


def load_reference_poses(seq):
    """Load reference extrinsics from sequence JSON."""
    ext_path = seq.get("extrinsics_path")
    if not ext_path or not os.path.exists(ext_path):
        return None
    with open(ext_path) as f:
        ext_data = json.load(f)
    ref_exts = ext_data.get("extrinsics", [])
    return np.array([np.array(e["w2c"])[:3, :4] for e in ref_exts])


def evaluate_pose(ref_w2c, vggt_ext):
    """Evaluate pose for a subset using global Procrustes alignment."""
    n = len(vggt_ext)
    if n < 3:
        return None

    R_ref_c2w = ref_w2c[:, :3, :3].transpose(0, 2, 1)
    R_vggt_c2w = vggt_ext[:, :3, :3].transpose(0, 2, 1)

    Rg = global_rotation_procrustes(R_vggt_c2w, R_ref_c2w)

    rot_errors = []
    for i in range(n):
        R_aligned = Rg @ R_vggt_c2w[i]
        err = rot_angle_deg(R_aligned.T @ R_ref_c2w[i])
        rot_errors.append(err)
    rot_errors = np.array(rot_errors)

    centers_ref = w2c_centers(ref_w2c)
    centers_vggt = w2c_centers(vggt_ext)
    s, R_sim3, t_sim3 = horn_sim3(centers_vggt, centers_ref)
    centers_aligned = s * (R_sim3 @ centers_vggt.T).T + t_sim3
    center_diff = centers_aligned - centers_ref
    ref_spread = np.linalg.norm(centers_ref - centers_ref.mean(axis=0), axis=1).mean()
    center_errors_norm = np.linalg.norm(center_diff, axis=1) / max(ref_spread, 1e-10)

    rot_median = float(np.median(rot_errors))
    rot_p90 = float(np.percentile(rot_errors, 90))
    gate_pass = rot_median <= 10.0 and rot_p90 <= 20.0

    return {
        "rot_median": rot_median,
        "rot_p90": rot_p90,
        "center_median_norm": float(np.median(center_errors_norm)),
        "pose_gate": "PASS" if gate_pass else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser(description="Starting-frame sensitivity test")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16

    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    print("Model loaded.")

    all_results = []

    for date in FAIL_DATES:
        seq_id = f"plantview__langdon_4__{date}"
        print(f"\n{'='*60}")
        print(f"Date: {date} (FAIL)")
        print(f"{'='*60}")

        seq = load_sequence(date)
        rgb_paths = seq["rgb_paths"]
        S = len(rgb_paths)
        print(f"  Total frames: {S}")

        ref_w2c = load_reference_poses(seq)
        if ref_w2c is None:
            print(f"  SKIP: no reference poses")
            continue

        # --- Offset sensitivity ---
        offsets = [i for i in range(0, S, max(1, S // 80))]
        for offset in offsets:
            out_path = os.path.join(OUT_DIR, f"{seq_id}_offset{offset:04d}_n{N_VIEWS}.npz")

            if os.path.exists(out_path) and not args.force:
                # Load existing result
                data = np.load(out_path)
                vggt_ext = data["ext_w2c_vggt"]
                frame_idx = data["frame_idx"]
                ref_sub = ref_w2c[frame_idx]
                result = evaluate_pose(ref_sub, vggt_ext)
                if result:
                    result.update({"date": date, "offset": offset, "order": "forward",
                                   "n_views": N_VIEWS, "seq_id": seq_id})
                    all_results.append(result)
                    print(f"  offset={offset:4d}: rot_med={result['rot_median']:.2f}° gate={result['pose_gate']} (cached)")
                continue

            # Select frames starting at offset
            idx = np.arange(offset, min(offset + N_VIEWS, S))
            rgb = [rgb_paths[i] for i in idx]

            images = load_and_preprocess_images(rgb, mode="crop").to(device)
            H, W = images.shape[-2:]
            torch.manual_seed(42)
            t0 = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                tokens, ps_idx = model.aggregator(images.unsqueeze(0))
                pose_enc = model.camera_head(tokens)[-1]
                ext_w2c, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
            dt = time.time() - t0

            vggt_ext_n = ext_w2c.squeeze(0).float().cpu().numpy()
            np.savez_compressed(out_path, ext_w2c_vggt=vggt_ext_n, frame_idx=idx)

            ref_sub = ref_w2c[idx]
            result = evaluate_pose(ref_sub, vggt_ext_n)
            if result:
                result.update({"date": date, "offset": offset, "order": "forward",
                               "n_views": N_VIEWS, "seq_id": seq_id})
                all_results.append(result)
                print(f"  offset={offset:4d}: rot_med={result['rot_median']:.2f}° gate={result['pose_gate']} ({dt:.1f}s)")

        # --- Reverse order ---
        out_path_rev = os.path.join(OUT_DIR, f"{seq_id}_reverse_n{N_VIEWS}.npz")
        if os.path.exists(out_path_rev) and not args.force:
            data = np.load(out_path_rev)
            vggt_ext = data["ext_w2c_vggt"]
            frame_idx = data["frame_idx"]
            ref_sub = ref_w2c[frame_idx]
            result = evaluate_pose(ref_sub, vggt_ext)
            if result:
                result.update({"date": date, "offset": -1, "order": "reverse",
                               "n_views": N_VIEWS, "seq_id": seq_id})
                all_results.append(result)
                print(f"  reverse:  rot_med={result['rot_median']:.2f}° gate={result['pose_gate']} (cached)")
        else:
            # Take last 16 frames in reverse order
            idx = np.arange(S - N_VIEWS, S)[::-1]
            rgb = [rgb_paths[i] for i in idx]

            images = load_and_preprocess_images(rgb, mode="crop").to(device)
            H, W = images.shape[-2:]
            torch.manual_seed(42)
            t0 = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                tokens, ps_idx = model.aggregator(images.unsqueeze(0))
                pose_enc = model.camera_head(tokens)[-1]
                ext_w2c, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
            dt = time.time() - t0

            vggt_ext_n = ext_w2c.squeeze(0).float().cpu().numpy()
            np.savez_compressed(out_path_rev, ext_w2c_vggt=vggt_ext_n, frame_idx=idx)

            ref_sub = ref_w2c[idx]
            result = evaluate_pose(ref_sub, vggt_ext_n)
            if result:
                result.update({"date": date, "offset": -1, "order": "reverse",
                               "n_views": N_VIEWS, "seq_id": seq_id})
                all_results.append(result)
                print(f"  reverse:  rot_med={result['rot_median']:.2f}° gate={result['pose_gate']} ({dt:.1f}s)")

    # Save CSV
    if all_results:
        csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "STARTING_FRAME_SENSITIVITY.csv")
        fields = ["seq_id", "date", "offset", "order", "n_views",
                  "rot_median", "rot_p90", "center_median_norm", "pose_gate"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_results:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"\nSaved: {csv_path} ({len(all_results)} rows)")

    # Summary: best/worst offset per date
    print(f"\n{'='*60}")
    print("SENSITIVITY SUMMARY")
    print(f"{'='*60}")
    for date in FAIL_DATES:
        date_results = [r for r in all_results if r["date"] == date]
        if not date_results:
            continue
        best = min(date_results, key=lambda r: r["rot_median"])
        worst = max(date_results, key=lambda r: r["rot_median"])
        rescued = [r for r in date_results if r["pose_gate"] == "PASS"]
        print(f"  {date}:")
        print(f"    Best:  offset={best['offset']:4d} rot_med={best['rot_median']:.2f}° gate={best['pose_gate']}")
        print(f"    Worst: offset={worst['offset']:4d} rot_med={worst['rot_median']:.2f}° gate={worst['pose_gate']}")
        print(f"    Rescued: {len(rescued)}/{len(date_results)} offsets")


if __name__ == "__main__":
    main()
