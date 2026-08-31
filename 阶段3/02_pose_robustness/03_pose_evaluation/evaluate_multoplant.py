#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B Step 4: Multi-plant pose evaluation.

Evaluates VGGT pose at 8/16/24/36(full) views per sequence.
Uses existing eval_pose_v2 logic (Procrustes alignment).
Outputs per-frame and per-sequence metrics.
"""
import os, sys, json, csv
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
VGGT_OUT = os.path.join(ROOT, "阶段2", "02_vggt")
sys.path.insert(0, os.path.join(ROOT, "阶段2", "02_vggt"))

VIEW_COUNTS = [8, 16, 24]


def horn_sim3(src, dst):
    """dst ≈ s R src + t. Returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((sc ** 2).sum() / len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def global_rotation_procrustes(R_vg_c2w, R_ref_c2w):
    """Find Rg minimizing Σ ||Rg @ Rv_i - Rr_i||_F."""
    H = np.einsum("sij,skj->ik", R_vg_c2w, R_ref_c2w)
    U, _, Vt = np.linalg.svd(H)
    Rg = Vt.T @ U.T
    if np.linalg.det(Rg) < 0:
        Vt[-1, :] *= -1
        Rg = Vt.T @ U.T
    return Rg


def rot_angle_deg(R):
    """Rotation angle in degrees from rotation matrix."""
    cos_val = (np.trace(R) - 1) / 2
    cos_val = np.clip(cos_val, -1, 1)
    return np.degrees(np.arccos(cos_val))


def w2c_centers(ext_w2c_3x4):
    """Extract camera centers from 3x4 w2c extrinsics (matching existing eval)."""
    R = ext_w2c_3x4[:, :3, :3]
    t = ext_w2c_3x4[:, :3, 3]
    return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)


def evaluate_pose_subset(ref_ext_w2c, vggt_ext_w2c, ref_intrinsics, vggt_intrinsics, indices):
    """Evaluate pose for a subset of frames.

    ref_ext_w2c: (S, 3, 4) w2c reference
    vggt_ext_w2c: (S, 4, 4) w2c VGGT (padded)
    """
    n = len(indices)
    if n < 3:
        return None

    ref_subset = ref_ext_w2c[indices]  # (n, 3, 4)
    vggt_subset_4x4 = vggt_ext_w2c[indices]  # (n, 4, 4)

    # Convert to c2w for rotation alignment (matching existing eval)
    R_ref_c2w = ref_subset[:, :3, :3].transpose(0, 2, 1)  # (n, 3, 3)
    R_vggt_c2w = vggt_subset_4x4[:, :3, :3].transpose(0, 2, 1)  # (n, 3, 3)

    # Global rotation alignment
    Rg = global_rotation_procrustes(R_vggt_c2w, R_ref_c2w)

    # Per-frame rotation error after alignment
    rot_errors = []
    for i in range(n):
        R_aligned = Rg @ R_vggt_c2w[i]
        err = rot_angle_deg(R_aligned.T @ R_ref_c2w[i])
        rot_errors.append(err)
    rot_errors = np.array(rot_errors)

    # Camera centers (from 3x4 w2c)
    ref_3x4 = ref_subset[:, :3, :4]
    vggt_3x4 = vggt_subset_4x4[:, :3, :4]
    centers_ref = w2c_centers(ref_3x4)
    centers_vggt = w2c_centers(vggt_3x4)

    # Sim3 alignment for scale
    s, R_sim3, t_sim3 = horn_sim3(centers_vggt, centers_ref)
    centers_aligned = s * (R_sim3 @ centers_vggt.T).T + t_sim3

    # Center errors (normalized by reference spread)
    center_diff = centers_aligned - centers_ref
    ref_spread = np.linalg.norm(centers_ref - centers_ref.mean(axis=0), axis=1).mean()
    center_errors_norm = np.linalg.norm(center_diff, axis=1) / max(ref_spread, 1e-10)

    # Trajectory cosine similarity
    if n > 1:
        traj_vggt = np.diff(centers_aligned, axis=0)
        traj_ref = np.diff(centers_ref, axis=0)
        cosines = []
        for i in range(len(traj_vggt)):
            nv = np.linalg.norm(traj_vggt[i])
            nr = np.linalg.norm(traj_ref[i])
            if nv > 1e-10 and nr > 1e-10:
                cosines.append(np.dot(traj_vggt[i], traj_ref[i]) / (nv * nr))
        traj_cosine = float(np.mean(cosines)) if cosines else 0.0
    else:
        traj_cosine = 0.0

    # Neighbor RPE (relative pose between consecutive frames)
    neighbor_rot_rpe = []
    neighbor_trans_rpe = []
    for i in range(n - 1):
        # Relative rotation (c2w convention)
        R_rel_ref = R_ref_c2w[i].T @ R_ref_c2w[i + 1]
        R_rel_vggt = (Rg @ R_vggt_c2w[i]).T @ (Rg @ R_vggt_c2w[i + 1])
        neighbor_rot_rpe.append(rot_angle_deg(R_rel_ref.T @ R_rel_vggt))

        # Relative translation
        t_rel_ref = R_ref_c2w[i].T @ (centers_ref[i + 1] - centers_ref[i])
        t_rel_vggt = (Rg @ R_vggt_c2w[i]).T @ (centers_aligned[i + 1] - centers_aligned[i])
        if np.linalg.norm(t_rel_ref) > 1e-10:
            neighbor_trans_rpe.append(np.linalg.norm(t_rel_vggt - t_rel_ref) / np.linalg.norm(t_rel_ref))

    # Focal length error
    if ref_intrinsics is not None and vggt_intrinsics is not None:
        ref_fx = ref_intrinsics[indices, 0, 0]
        vggt_fx = vggt_intrinsics[indices, 0, 0]
        focal_error = float(np.mean(np.abs(vggt_fx - ref_fx) / ref_fx))
    else:
        focal_error = -1

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
        "neighbor_rpe_rot": float(np.mean(neighbor_rot_rpe)) if neighbor_rot_rpe else -1,
        "neighbor_rpe_trans": float(np.mean(neighbor_trans_rpe)) if neighbor_trans_rpe else -1,
        "focal_error": focal_error,
        "pose_gate": "PASS" if gate_pass else "FAIL",
        "failure_type": failure_type,
    }


def load_sequence_data(seq_id):
    """Load reference and VGGT poses for a sequence."""
    # Find VGGT output dir
    vggt_dir = None
    for candidate in [
        os.path.join(VGGT_OUT, "wheat3dgs", seq_id),
        os.path.join(VGGT_OUT, "plant_view_3d", seq_id),
        os.path.join(VGGT_OUT, "v2_clean_rerun", seq_id),
        os.path.join(VGGT_OUT, "mustc", seq_id),
    ]:
        if os.path.exists(candidate):
            vggt_dir = candidate
            break

    if vggt_dir is None:
        return None

    # Load VGGT predictions (may be 3x4 or 4x4)
    vggt_w2c_raw = np.load(os.path.join(vggt_dir, "extrinsic_w2c.npy"))
    if vggt_w2c_raw.ndim == 3 and vggt_w2c_raw.shape[1] == 3 and vggt_w2c_raw.shape[2] == 4:
        # Pad 3x4 → 4x4
        S = vggt_w2c_raw.shape[0]
        vggt_w2c = np.zeros((S, 4, 4), dtype=vggt_w2c_raw.dtype)
        vggt_w2c[:, :3, :4] = vggt_w2c_raw
        vggt_w2c[:, 3, 3] = 1
    else:
        vggt_w2c = vggt_w2c_raw
    vggt_intr = np.load(os.path.join(vggt_dir, "intrinsic_vggt.npy"))

    # Load reference poses
    ref_dir = None
    json_name = seq_id
    # Strip dataset prefixes for JSON lookup
    for prefix in ["plantview__", "wheat3dgs__", "mustc__"]:
        if json_name.startswith(prefix):
            json_name = json_name[len(prefix):]
            break

    for json_dir in [
        os.path.join(SEQ_BASE, "plant_view"),
        os.path.join(SEQ_BASE, "wheat3dgs"),
        os.path.join(SEQ_BASE, "mustc"),
    ]:
        jp = os.path.join(json_dir, f"{json_name}.json")
        if os.path.exists(jp):
            with open(jp) as f:
                meta = json.load(f)
            ext_path = meta.get("extrinsics_path")
            int_path = meta.get("intrinsics_path")
            if ext_path and os.path.exists(ext_path):
                ref_dir = os.path.dirname(ext_path)
                ref_ext_data = json.load(open(ext_path))
                ref_int_data = json.load(open(int_path)) if int_path and os.path.exists(int_path) else None
                break

    if ref_dir is None:
        return None

    # Parse reference extrinsics (store as 3x4 like existing eval)
    ref_exts = ref_ext_data.get("extrinsics", [])
    ref_w2c = np.array([np.array(e["w2c"])[:3, :4] for e in ref_exts])

    # Parse reference intrinsics
    ref_intr = None
    if ref_int_data:
        int_list = ref_int_data.get("intrinsics", ref_int_data.get("cameras", []))
        if isinstance(int_list, list) and len(int_list) > 0:
            # Per-camera intrinsics list
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
            # Shared intrinsics dict (Wheat3DGS style)
            _n_intr = len(ref_exts)
            ref_intr = np.zeros((_n_intr, 3, 3))
            ref_intr[:, 0, 0] = int_list.get("fl_x", 1)
            ref_intr[:, 1, 1] = int_list.get("fl_y", int_list.get("fl_x", 1))
            ref_intr[:, 0, 2] = int_list.get("cx", 0)
            ref_intr[:, 1, 2] = int_list.get("cy", 0)
            ref_intr[:, 2, 2] = 1
        elif isinstance(ref_int_data, dict) and "fl_x" in ref_int_data:
            # Top-level shared intrinsics
            _n_intr = len(ref_exts)
            ref_intr = np.zeros((_n_intr, 3, 3))
            ref_intr[:, 0, 0] = ref_int_data.get("fl_x", 1)
            ref_intr[:, 1, 1] = ref_int_data.get("fl_y", ref_int_data.get("fl_x", 1))
            ref_intr[:, 0, 2] = ref_int_data.get("cx", 0)
            ref_intr[:, 1, 2] = ref_int_data.get("cy", 0)
            ref_intr[:, 2, 2] = 1

    # Ensure same length
    n = min(len(ref_w2c), len(vggt_w2c))

    return {
        "ref_w2c": ref_w2c[:n],
        "vggt_w2c": vggt_w2c[:n],
        "ref_intrinsics": ref_intr[:n] if ref_intr is not None and len(ref_intr) >= n else None,
        "vggt_intrinsics": vggt_intr[:n],
        "n_full": n,
    }


def main():
    os.makedirs(os.path.join(PHASE3B, "03_pose_evaluation"), exist_ok=True)

    # Load view sampling manifest
    manifest_path = os.path.join(PHASE3B, "00_protocol", "VIEW_SAMPLING_MANIFEST.json")
    with open(manifest_path) as f:
        view_manifest = json.load(f)

    # Load inventory
    inv_path = os.path.join(PHASE3B, "01_dataset_inventory", "MULTIPLANT_POSE_DATASET_INVENTORY.csv")
    with open(inv_path) as f:
        inventory = {r["sequence_id"]: r for r in csv.DictReader(f)}

    # Load canopy characterization
    canopy_path = os.path.join(PHASE3B, "04_scene_characterization", "CANOPY_CHARACTERIZATION.csv")
    canopy_data = {}
    if os.path.exists(canopy_path):
        with open(canopy_path) as f:
            canopy_data = {r["sequence_id"]: r for r in csv.DictReader(f)}

    all_results = []
    all_frame_results = []

    for seq_id in view_manifest:
        if seq_id not in inventory:
            print(f"  SKIP {seq_id}: not in inventory")
            continue

        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")
        print(f"{'='*60}")

        data = load_sequence_data(seq_id)
        if data is None:
            print(f"  SKIP: could not load data")
            continue

        vm = view_manifest[seq_id]
        canopy = canopy_data.get(seq_id, {})

        for vc in VIEW_COUNTS + ["full"]:
            vc_str = str(vc)
            if vc_str == "full":
                indices = list(range(data["n_full"]))
                actual_vc = data["n_full"]
            else:
                indices = vm.get(vc_str, [])
                actual_vc = len(indices)

            if actual_vc < 3:
                print(f"  {vc} views: SKIP (only {actual_vc} frames)")
                continue

            result = evaluate_pose_subset(
                data["ref_w2c"], data["vggt_w2c"],
                data["ref_intrinsics"], data["vggt_intrinsics"],
                indices
            )
            if result is None:
                continue

            row = {
                "sequence_id": seq_id,
                "plant_id": inventory[seq_id]["plant_id"],
                "dataset": inventory[seq_id]["dataset"],
                "date": inventory[seq_id]["date"],
                "canopy_fraction": canopy.get("canopy_fraction_mean", -1),
                "background_fraction": canopy.get("background_fraction_mean", -1),
                "density_class": canopy.get("density_class", "unknown"),
                "view_count": actual_vc,
                **result,
            }
            all_results.append(row)

            # Per-frame metrics for full evaluation
            if vc_str == "full":
                R_ref_c2w_full = data["ref_w2c"][indices, :3, :3].transpose(0, 2, 1)
                R_vggt_c2w_full = data["vggt_w2c"][indices, :3, :3].transpose(0, 2, 1)
                Rg_full = global_rotation_procrustes(R_vggt_c2w_full, R_ref_c2w_full)
                for i, idx in enumerate(indices):
                    R_aligned = Rg_full @ R_vggt_c2w_full[i]
                    err = rot_angle_deg(R_aligned.T @ R_ref_c2w_full[i])
                    all_frame_results.append({
                        "sequence_id": seq_id,
                        "frame_idx": idx,
                        "rot_error_deg": float(err),
                    })

            gate_str = result["pose_gate"]
            print(f"  {str(vc):5s} views: rot_med={result['rot_median']:.2f}° rot_p90={result['rot_p90']:.2f}° "
                  f"center={result['center_median_norm']:.4f} gate={gate_str} "
                  f"type={result['failure_type']}")

    # Save main results CSV
    if all_results:
        csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "MULTIPLANT_POSE_RESULTS.csv")
        fields = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {csv_path} ({len(all_results)} rows)")

    # Save per-frame results
    if all_frame_results:
        csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "PER_FRAME_ROT_ERRORS.csv")
        fields = list(all_frame_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_frame_results)
        print(f"Saved: {csv_path} ({len(all_frame_results)} rows)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for seq_id in view_manifest:
        seq_results = [r for r in all_results if r["sequence_id"] == seq_id]
        if not seq_results:
            continue
        r24 = [r for r in seq_results if r["view_count"] == 24]
        r16 = [r for r in seq_results if r["view_count"] == 16]
        r8 = [r for r in seq_results if r["view_count"] == 8]
        rf = [r for r in seq_results if r["view_count"] > 24]

        def _s(r):
            return f"rot={r['rot_median']:.1f}° gate={r['pose_gate']}" if r else "N/A"

        print(f"  {seq_id}:")
        print(f"    full: {_s(rf[0] if rf else None)}")
        print(f"    24v:  {_s(r24[0] if r24 else None)}")
        print(f"    16v:  {_s(r16[0] if r16 else None)}")
        print(f"    8v:   {_s(r8[0] if r8 else None)}")


if __name__ == "__main__":
    main()
