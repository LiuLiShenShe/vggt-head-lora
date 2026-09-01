#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B.1 Step 1b: Evaluate true view-count VGGT re-run outputs.

Loads npz from rerun_viewcount.py, evaluates against reference poses.
Reuses evaluation functions from evaluate_multoplant.py.

Outputs: TRUE_VIEWCOUNT_RESULTS.csv
"""
import os, sys, json, csv
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
VC_DIR = os.path.join(PHASE3B, "02_pose_inference", "viewcount_outputs")
OUT_DIR = os.path.join(PHASE3B, "03_pose_evaluation")

# Reuse evaluation functions from evaluate_multoplant.py
sys.path.insert(0, os.path.join(PHASE3B, "03_pose_evaluation"))
from evaluate_multoplant import (
    global_rotation_procrustes, rot_angle_deg, w2c_centers, horn_sim3
)


def find_sequence_json(seq_id):
    """Find sequence JSON for a given sequence_id."""
    for subdir in ["plant_view", "wheat3dgs", "mustc"]:
        pattern = os.path.join(SEQ_BASE, subdir, "*.json")
        for jp in glob.glob(pattern):
            with open(jp) as f:
                meta = json.load(f)
            if meta["sequence_id"] == seq_id:
                return meta
    raise FileNotFoundError(f"Sequence JSON not found for {seq_id}")


def load_reference_poses(seq):
    """Load reference extrinsics and intrinsics from sequence JSON."""
    ext_path = seq.get("extrinsics_path")
    int_path = seq.get("intrinsics_path")

    if not ext_path or not os.path.exists(ext_path):
        return None, None

    with open(ext_path) as f:
        ext_data = json.load(f)

    ref_exts = ext_data.get("extrinsics", [])
    ref_w2c = np.array([np.array(e["w2c"])[:3, :4] for e in ref_exts])  # (S, 3, 4)

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
        elif isinstance(int_list, dict) and "fl_x" in int_list:
            _n = len(ref_exts)
            ref_intr = np.zeros((_n, 3, 3))
            ref_intr[:, 0, 0] = int_list.get("fl_x", 1)
            ref_intr[:, 1, 1] = int_list.get("fl_y", int_list.get("fl_x", 1))
            ref_intr[:, 0, 2] = int_list.get("cx", 0)
            ref_intr[:, 1, 2] = int_list.get("cy", 0)
            ref_intr[:, 2, 2] = 1

    return ref_w2c, ref_intr


def evaluate_true_viewcount(ref_w2c, vggt_ext_w2c, ref_intr, vggt_intr):
    """Evaluate pose for a true view-count prediction (independent VGGT forward).

    vggt_ext_w2c: (n, 3, 4) from npz
    ref_w2c: (S, 3, 4) from reference — must match by frame_idx.
    """
    n = len(vggt_ext_w2c)
    if n < 3:
        return None

    # Convert to c2w for rotation alignment
    R_ref_c2w = ref_w2c[:, :3, :3].transpose(0, 2, 1)
    R_vggt_c2w = vggt_ext_w2c[:, :3, :3].transpose(0, 2, 1)

    # Global rotation alignment
    Rg = global_rotation_procrustes(R_vggt_c2w, R_ref_c2w)

    # Per-frame rotation error
    rot_errors = []
    for i in range(n):
        R_aligned = Rg @ R_vggt_c2w[i]
        err = rot_angle_deg(R_aligned.T @ R_ref_c2w[i])
        rot_errors.append(err)
    rot_errors = np.array(rot_errors)

    # Camera centers
    centers_ref = w2c_centers(ref_w2c)
    centers_vggt = w2c_centers(vggt_ext_w2c)

    # Sim3 alignment
    s, R_sim3, t_sim3 = horn_sim3(centers_vggt, centers_ref)
    centers_aligned = s * (R_sim3 @ centers_vggt.T).T + t_sim3

    # Center errors
    center_diff = centers_aligned - centers_ref
    ref_spread = np.linalg.norm(centers_ref - centers_ref.mean(axis=0), axis=1).mean()
    center_errors_norm = np.linalg.norm(center_diff, axis=1) / max(ref_spread, 1e-10)

    # Trajectory cosine
    if n > 1:
        traj_vggt = np.diff(centers_aligned, axis=0)
        traj_ref = np.diff(centers_ref, axis=0)
        cosines = []
        for i in range(len(traj_vggt)):
            nv, nr = np.linalg.norm(traj_vggt[i]), np.linalg.norm(traj_ref[i])
            if nv > 1e-10 and nr > 1e-10:
                cosines.append(np.dot(traj_vggt[i], traj_ref[i]) / (nv * nr))
        traj_cosine = float(np.mean(cosines)) if cosines else 0.0
    else:
        traj_cosine = 0.0

    # Focal error
    focal_error = -1
    if ref_intr is not None and vggt_intr is not None and len(ref_intr) >= n:
        ref_fx = ref_intr[:n, 0, 0]
        vggt_fx = vggt_intr[:n, 0, 0]
        focal_error = float(np.mean(np.abs(vggt_fx - ref_fx) / ref_fx))

    # Pose gate
    rot_median = float(np.median(rot_errors))
    rot_p90 = float(np.percentile(rot_errors, 90))
    gate_pass = rot_median <= 10.0 and rot_p90 <= 20.0

    # Failure type
    if gate_pass:
        failure_type = "PASS"
    else:
        center_collapse = np.mean(center_errors_norm) < 0.01
        if center_collapse:
            failure_type = "CENTER_COLLAPSE"
        elif rot_median > 10.0 and np.mean(center_errors_norm) > 0.5:
            failure_type = "ROTATION_AND_TRANSLATION"
        elif rot_median > 10.0:
            failure_type = "ROTATION_COLLAPSE"
        elif np.mean(center_errors_norm) > 0.5:
            failure_type = "TRANSLATION_COLLAPSE"
        else:
            failure_type = "UNKNOWN"

    return {
        "n_frames": n,
        "rot_median": rot_median,
        "rot_p90": rot_p90,
        "rot_mean": float(np.mean(rot_errors)),
        "center_median_norm": float(np.median(center_errors_norm)),
        "center_p90_norm": float(np.percentile(center_errors_norm, 90)),
        "trajectory_cosine": traj_cosine,
        "focal_error": focal_error,
        "pose_gate": "PASS" if gate_pass else "FAIL",
        "failure_type": failure_type,
    }


import glob


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find all npz files from the viewcount re-run
    npz_files = sorted(glob.glob(os.path.join(VC_DIR, "*.npz")))
    if not npz_files:
        print(f"No npz files found in {VC_DIR}")
        print("Run rerun_viewcount.py first.")
        return

    print(f"Found {len(npz_files)} npz files to evaluate")

    # Collect unique sequence IDs
    seq_ids = sorted(set(
        os.path.basename(f).rsplit("_n", 1)[0]
        for f in npz_files
    ))
    print(f"Sequences: {seq_ids}")

    all_results = []

    for seq_id in seq_ids:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")
        print(f"{'='*60}")

        try:
            seq = find_sequence_json(seq_id)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        ref_w2c, ref_intr = load_reference_poses(seq)
        if ref_w2c is None:
            print(f"  SKIP: no reference poses")
            continue

        for n in [8, 16, 24]:
            npz_path = os.path.join(VC_DIR, f"{seq_id}_n{n}.npz")
            if not os.path.exists(npz_path):
                print(f"  n={n}: NOT FOUND")
                continue

            data = np.load(npz_path)
            vggt_ext = data["ext_w2c_vggt"]  # (n, 3, 4)
            vggt_intr = data["intr_vggt"]     # (n, 3, 3)
            frame_idx = data["frame_idx"]      # (n,)

            # Extract reference poses for the same frames
            ref_subset = ref_w2c[frame_idx]    # (n, 3, 4)
            ref_intr_subset = ref_intr[frame_idx] if ref_intr is not None and len(ref_intr) > max(frame_idx) else None

            result = evaluate_true_viewcount(ref_subset, vggt_ext, ref_intr_subset, vggt_intr)
            if result is None:
                print(f"  n={n}: SKIP (< 3 frames)")
                continue

            row = {
                "sequence_id": seq_id,
                "view_count": n,
                "frame_indices": ",".join(str(int(i)) for i in frame_idx),
                **result,
            }
            all_results.append(row)

            gate_str = result["pose_gate"]
            print(f"  n={n:2d}: rot_med={result['rot_median']:.2f}° rot_p90={result['rot_p90']:.2f}° "
                  f"center={result['center_median_norm']:.4f} gate={gate_str} "
                  f"type={result['failure_type']}")

    # Save CSV
    if all_results:
        csv_path = os.path.join(OUT_DIR, "TRUE_VIEWCOUNT_RESULTS.csv")
        fields = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {csv_path} ({len(all_results)} rows)")

    # Summary comparison: Phase 3B subsample vs true re-run
    print(f"\n{'='*60}")
    print("COMPARISON: Phase 3B subsample vs TRUE re-run")
    print(f"{'='*60}")
    old_csv = os.path.join(OUT_DIR, "MULTIPLANT_POSE_RESULTS.csv")
    if os.path.exists(old_csv):
        with open(old_csv) as f:
            old_rows = {r["sequence_id"]: r for r in csv.DictReader(f)
                        if r["sequence_id"].startswith("plantview__langdon_4")}
        for seq_id in seq_ids:
            if not seq_id.startswith("plantview__langdon_4"):
                continue
            new_rows = [r for r in all_results if r["sequence_id"] == seq_id]
            old_row = old_rows.get(seq_id)
            if old_row and new_rows:
                nr = next((r for r in new_rows if r["view_count"] == 8), None)
                if nr:
                    print(f"  {seq_id}:")
                    print(f"    Phase 3B subsample 8v: rot_med={float(old_row['rot_median']):.2f}° gate={old_row['pose_gate']}")
                    print(f"    TRUE re-run 8v:       rot_med={nr['rot_median']:.2f}° gate={nr['pose_gate']}")


if __name__ == "__main__":
    main()
