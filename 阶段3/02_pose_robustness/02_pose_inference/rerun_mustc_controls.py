#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B.1 Step 4: MuST-C Same-Domain Control Inference.

Runs VGGT inference on MuST-C sequences to test cross-plant generalization.
MuST-C has reference poses available, so we can evaluate pose gate.

Outputs: npz per (seq_id, n) + evaluation CSV.
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
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences", "mustc")
OUT_DIR = os.path.join(PHASE3B, "02_pose_inference", "mustc_outputs")
EVAL_DIR = os.path.join(PHASE3B, "03_pose_evaluation")

# Reuse evaluation functions
sys.path.insert(0, os.path.join(PHASE3B, "03_pose_evaluation"))
from evaluate_multoplant import (
    global_rotation_procrustes, rot_angle_deg, w2c_centers, horn_sim3
)

N_FRAMES = [8, 16, 20]  # 20 is full for MuST-C


def find_sequence_json(seq_id):
    """Find sequence JSON for a MuST-C sequence."""
    for jp in glob.glob(os.path.join(SEQ_BASE, "*.json")):
        with open(jp) as f:
            meta = json.load(f)
        if meta["sequence_id"] == seq_id:
            return meta
    raise FileNotFoundError(f"Sequence JSON not found for {seq_id}")


def load_reference_poses(seq):
    """Load reference extrinsics and intrinsics."""
    ext_path = seq.get("extrinsics_path")
    int_path = seq.get("intrinsics_path")
    if not ext_path or not os.path.exists(ext_path):
        return None, None

    with open(ext_path) as f:
        ext_data = json.load(f)
    ref_exts = ext_data.get("extrinsics", [])
    ref_w2c = np.array([np.array(e["w2c"])[:3, :4] for e in ref_exts])

    ref_intr = None
    if int_path and os.path.exists(int_path):
        with open(int_path) as f:
            int_data = json.load(f)
        int_list = int_data.get("intrinsics", int_data.get("cameras", []))
        if isinstance(int_list, list) and len(int_list) > 0:
            first = int_list[0]
            if "fx" in first or "fl_x" in first:
                ref_intr = np.zeros((len(int_list), 3, 3))
                for i, cam in enumerate(int_list):
                    ref_intr[i, 0, 0] = cam.get("fx", cam.get("fl_x", 1))
                    ref_intr[i, 1, 1] = cam.get("fy", cam.get("fl_y", 1))
                    ref_intr[i, 0, 2] = cam.get("cx", 0)
                    ref_intr[i, 1, 2] = cam.get("cy", 0)
                    ref_intr[i, 2, 2] = 1

    return ref_w2c, ref_intr


def evaluate_pose(ref_w2c, vggt_ext, ref_intr=None, vggt_intr=None):
    """Evaluate pose for a subset."""
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

    focal_error = -1
    if ref_intr is not None and vggt_intr is not None and len(ref_intr) >= n:
        ref_fx = ref_intr[:n, 0, 0]
        vggt_fx = vggt_intr[:n, 0, 0]
        focal_error = float(np.mean(np.abs(vggt_fx - ref_fx) / ref_fx))

    rot_median = float(np.median(rot_errors))
    rot_p90 = float(np.percentile(rot_errors, 90))
    gate_pass = rot_median <= 10.0 and rot_p90 <= 20.0

    return {
        "rot_median": rot_median,
        "rot_p90": rot_p90,
        "center_median_norm": float(np.median(center_errors_norm)),
        "focal_error": focal_error,
        "pose_gate": "PASS" if gate_pass else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser(description="MuST-C same-domain control inference")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Find all MuST-C sequences
    seq_files = sorted(glob.glob(os.path.join(SEQ_BASE, "*.json")))
    seq_ids = []
    for jp in seq_files:
        with open(jp) as f:
            meta = json.load(f)
        seq_ids.append(meta["sequence_id"])

    print(f"Found {len(seq_ids)} MuST-C sequences: {seq_ids}")

    device, dtype = "cuda", torch.bfloat16
    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    print("Model loaded.")

    all_results = []

    for seq_id in seq_ids:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")
        print(f"{'='*60}")

        seq = find_sequence_json(seq_id)
        rgb_paths = seq["rgb_paths"]
        S = len(rgb_paths)
        print(f"  Total frames: {S}")

        ref_w2c, ref_intr = load_reference_poses(seq)
        has_ref = ref_w2c is not None and len(ref_w2c) >= 3
        print(f"  Reference poses: {'YES' if has_ref else 'NO'}")

        for n in N_FRAMES:
            if n > S:
                print(f"  n={n}: SKIP (only {S} frames)")
                continue

            out_path = os.path.join(OUT_DIR, f"{seq_id}_n{n}.npz")

            if os.path.exists(out_path) and not args.force:
                data = np.load(out_path)
                vggt_ext = data["ext_w2c_vggt"]
                vggt_intr = data["intr_vggt"]
                frame_idx = data["frame_idx"]
                print(f"  n={n}: EXISTS, cached")
            else:
                idx = np.linspace(0, S - 1, n).astype(int)
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

                vggt_ext = ext_w2c.squeeze(0).float().cpu().numpy()
                vggt_intr = intr.squeeze(0).float().cpu().numpy()
                frame_idx = idx

                np.savez_compressed(out_path, ext_w2c_vggt=vggt_ext, intr_vggt=vggt_intr, frame_idx=frame_idx)
                print(f"  n={n}: {dt:.1f}s")

            if has_ref:
                ref_sub = ref_w2c[frame_idx]
                ref_intr_sub = ref_intr[frame_idx] if ref_intr is not None and len(ref_intr) > max(frame_idx) else None
                result = evaluate_pose(ref_sub, vggt_ext, ref_intr_sub, vggt_intr)
                if result:
                    result.update({"seq_id": seq_id, "view_count": n, "n_total": S})
                    all_results.append(result)
                    print(f"    rot_med={result['rot_median']:.2f}° gate={result['pose_gate']} focal_err={result['focal_error']:.4f}")
            else:
                # Report focal length and camera center spread as sanity
                centers = np.einsum("sij,sj->si", vggt_ext[:, :3, :3].transpose(0, 2, 1), -vggt_ext[:, :3, 3])
                center_spread = float(np.linalg.norm(centers - centers.mean(0), axis=1).mean())
                fx_mean = float(np.mean(vggt_intr[:, 0, 0]))
                print(f"    focal_fx={fx_mean:.2f} center_spread={center_spread:.4f} (no reference)")

    # Save evaluation CSV
    if all_results:
        csv_path = os.path.join(EVAL_DIR, "MUSTC_CONTROL_RESULTS.csv")
        fields = ["seq_id", "view_count", "n_total", "rot_median", "rot_p90",
                  "center_median_norm", "focal_error", "pose_gate"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_results:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"\nSaved: {csv_path} ({len(all_results)} rows)")

    # Summary
    print(f"\n{'='*60}")
    print("MuST-C SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['seq_id']} n={r['view_count']}: rot_med={r['rot_median']:.2f}° gate={r['pose_gate']}")


if __name__ == "__main__":
    main()
