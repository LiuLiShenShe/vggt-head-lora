#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B.1 Step 2: Acquisition Anomaly Audit.

Compares 3 PASS vs 3 FAIL langdon_4 dates on image-level and
sequence-level metrics to identify acquisition anomalies.

Metrics: resolution, brightness, saturation, blur, file size,
trajectory geometry, angular baseline, intrinsic consistency,
COLMAP quality, capture order.

Outputs: ACQUISITION_COMPARISON.csv, ACQUISITION_SUMMARY.json
"""
import os, sys, json, csv, glob
import numpy as np
from PIL import Image

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
PHASE1 = os.path.join(ROOT, "阶段1-数据集", "3D Plant View", "langdon_4")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences", "plant_view")
OUT_DIR = os.path.join(PHASE3B, "05_failure_analysis")

PASS_DATES = ["05-03-24", "13-02-24", "20-02-24"]
FAIL_DATES = ["12-03-24", "15-04-24", "19-03-24"]
ALL_DATES = PASS_DATES + FAIL_DATES


def load_sequence_meta(date):
    """Load sequence JSON for a langdon_4 date."""
    jp = os.path.join(SEQ_BASE, f"langdon_4__{date}.json")
    with open(jp) as f:
        return json.load(f)


def compute_image_metrics(rgb_dir, sample_n=20):
    """Compute image-level metrics from a sample of RGB files."""
    files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if not files:
        return {}

    # Sample evenly
    if len(files) > sample_n:
        indices = np.linspace(0, len(files) - 1, sample_n).astype(int)
        files = [files[i] for i in indices]

    sizes = []
    brightnesses = []
    saturations = []
    blur_scores = []
    resolutions = []

    for fp in files:
        sz = os.path.getsize(fp)
        sizes.append(sz)

        img = Image.open(fp).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        resolutions.append((arr.shape[1], arr.shape[0]))  # (W, H)

        # Brightness: mean luminance
        brightness = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        brightnesses.append(float(np.mean(brightness)))

        # Saturation: max(chroma) - min(chroma) in HSV
        r, g, b = arr[:,:,0]/255, arr[:,:,1]/255, arr[:,:,2]/255
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        chroma = mx - mn
        saturations.append(float(np.mean(chroma)))

        # Blur: Laplacian variance (lower = blurrier)
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        lap = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        from scipy.signal import convolve2d
        laplacian = convolve2d(gray, lap, mode='valid')
        blur_scores.append(float(np.var(laplacian)))

    return {
        "frame_count": len(glob.glob(os.path.join(rgb_dir, "*.png"))),
        "resolution_w": resolutions[0][0] if resolutions else 0,
        "resolution_h": resolutions[0][1] if resolutions else 0,
        "file_size_mean": float(np.mean(sizes)),
        "file_size_std": float(np.std(sizes)),
        "brightness_mean": float(np.mean(brightnesses)),
        "brightness_std": float(np.std(brightnesses)),
        "saturation_mean": float(np.mean(saturations)),
        "saturation_std": float(np.std(saturations)),
        "blur_laplacian_var_mean": float(np.mean(blur_scores)),
        "blur_laplacian_var_std": float(np.std(blur_scores)),
        "first_frame_brightness": brightnesses[0] if brightnesses else -1,
        "first_frame_blur": blur_scores[0] if blur_scores else -1,
    }


def compute_trajectory_metrics(ext_w2c):
    """Compute trajectory geometry from w2c extrinsics (S, 3, 4)."""
    S = ext_w2c.shape[0]
    R = ext_w2c[:, :3, :3]
    t = ext_w2c[:, :3, 3]
    centers = np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)

    # Total path length
    diffs = np.diff(centers, axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Angular span (max angle between any two centers relative to centroid)
    centroid = centers.mean(axis=0)
    vecs = centers - centroid
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vecs_normed = vecs / norms
    # Gram matrix of dot products
    dots = vecs_normed @ vecs_normed.T
    angular_span = float(np.degrees(np.arccos(np.clip(np.min(dots), -1, 1))))

    # Adjacent-frame angular baseline
    adj_angles = []
    for i in range(S - 1):
        v1 = vecs[i] / max(np.linalg.norm(vecs[i]), 1e-10)
        v2 = vecs[i + 1] / max(np.linalg.norm(vecs[i + 1]), 1e-10)
        cos_a = np.clip(np.dot(v1, v2), -1, 1)
        adj_angles.append(float(np.degrees(np.arccos(cos_a))))

    # Trajectory direction consistency (clockwise vs counterclockwise)
    # Use cross product of consecutive displacement vectors
    if S > 2:
        cross_products = []
        for i in range(len(diffs) - 1):
            c = np.cross(diffs[i], diffs[i + 1])
            cross_products.append(c[2])  # z-component indicates CW/CCW
        direction_sign = np.sign(np.mean(cross_products))
        capture_order = "CLOCKWISE" if direction_sign < 0 else "COUNTERCLOCKWISE"
    else:
        capture_order = "UNKNOWN"

    return {
        "trajectory_path_length": path_length,
        "trajectory_angular_span_deg": angular_span,
        "adj_baseline_mean_deg": float(np.mean(adj_angles)) if adj_angles else -1,
        "adj_baseline_min_deg": float(np.min(adj_angles)) if adj_angles else -1,
        "adj_baseline_max_deg": float(np.max(adj_angles)) if adj_angles else -1,
        "adj_baseline_std_deg": float(np.std(adj_angles)) if adj_angles else -1,
        "capture_order": capture_order,
        "camera_center_x_range": float(centers[:, 0].max() - centers[:, 0].min()),
        "camera_center_y_range": float(centers[:, 1].max() - centers[:, 1].min()),
        "camera_center_z_range": float(centers[:, 2].max() - centers[:, 2].min()),
    }


def compute_intrinsic_consistency(seq_meta):
    """Check intrinsic consistency across frames."""
    int_path = seq_meta.get("intrinsics_path")
    if not int_path or not os.path.exists(int_path):
        return {}

    with open(int_path) as f:
        int_data = json.load(f)

    # Check if intrinsics are shared or per-frame
    int_list = int_data.get("intrinsics", int_data.get("cameras", []))
    if isinstance(int_list, list) and len(int_list) > 0:
        first = int_list[0]
        if "fx" in first or "fl_x" in first:
            fxs = [c.get("fx", c.get("fl_x", 0)) for c in int_list]
            fys = [c.get("fy", c.get("fl_y", 0)) for c in int_list]
            return {
                "intrinsics_shared": False,
                "fx_range": float(max(fxs) - min(fxs)),
                "fy_range": float(max(fys) - min(fys)),
                "fx_mean": float(np.mean(fxs)),
                "fy_mean": float(np.mean(fys)),
            }
    elif isinstance(int_list, dict) and "fl_x" in int_list:
        return {
            "intrinsics_shared": True,
            "fx_range": 0.0,
            "fy_range": 0.0,
            "fx_mean": float(int_list.get("fl_x", 0)),
            "fy_mean": float(int_list.get("fl_y", int_list.get("fl_x", 0))),
        }

    return {"intrinsics_shared": "unknown"}


def compute_transforms_dual_intrinsic(date):
    """Analyze the dual-robot intrinsic split from transforms.json."""
    transforms_dir = os.path.join(PHASE1, date, "transforms", "adjusted")
    tj = os.path.join(transforms_dir, "transforms.json")
    if not os.path.exists(tj):
        return {}

    with open(tj) as f:
        tdata = json.load(f)

    frames = tdata.get("frames", [])
    if not frames:
        return {}

    fxs = []
    for fr in frames:
        fi = fr.get("fl_x", fr.get("fx", 0))
        fxs.append(fi)

    if not fxs:
        return {}

    # Count distinct intrinsic clusters
    fx_arr = np.array(fxs)
    fx_unique = np.unique(np.round(fx_arr, 2))
    n_clusters = len(fx_unique)

    # Count frames per cluster
    cluster_counts = {}
    for fx in fx_unique:
        cluster_counts[str(fx)] = int(np.sum(np.abs(fx_arr - fx) < 1.0))

    return {
        "transforms_frame_count": len(frames),
        "transforms_n_intrinsic_clusters": n_clusters,
        "transforms_intrinsic_clusters": cluster_counts,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_metrics = []

    for date in ALL_DATES:
        print(f"\n{'='*60}")
        print(f"Date: {date} ({'PASS' if date in PASS_DATES else 'FAIL'})")
        print(f"{'='*60}")

        seq = load_sequence_meta(date)
        rgb_dir = os.path.join(PHASE1, date, "images", "rgb")

        # Image-level metrics
        img_metrics = compute_image_metrics(rgb_dir)
        print(f"  Frames: {img_metrics.get('frame_count', '?')}")
        print(f"  Resolution: {img_metrics.get('resolution_w', '?')}x{img_metrics.get('resolution_h', '?')}")
        print(f"  Brightness: {img_metrics.get('brightness_mean', -1):.2f}")
        print(f"  Blur (Laplacian var): {img_metrics.get('blur_laplacian_var_mean', -1):.2f}")

        # Trajectory metrics
        ext_path = seq.get("extrinsics_path")
        traj_metrics = {}
        if ext_path and os.path.exists(ext_path):
            with open(ext_path) as f:
                ext_data = json.load(f)
            ref_exts = ext_data.get("extrinsics", [])
            if ref_exts:
                ref_w2c = np.array([np.array(e["w2c"])[:3, :4] for e in ref_exts])
                traj_metrics = compute_trajectory_metrics(ref_w2c)
                print(f"  Path length: {traj_metrics.get('trajectory_path_length', -1):.4f}")
                print(f"  Capture order: {traj_metrics.get('capture_order', '?')}")
                print(f"  Adj baseline: {traj_metrics.get('adj_baseline_mean_deg', -1):.2f}°")

        # Intrinsic consistency
        intr_metrics = compute_intrinsic_consistency(seq)
        if intr_metrics:
            print(f"  Intrinsics shared: {intr_metrics.get('intrinsics_shared', '?')}")
            print(f"  fx range: {intr_metrics.get('fx_range', -1):.4f}")

        # Dual-robot intrinsic analysis
        dual_metrics = compute_transforms_dual_intrinsic(date)
        if dual_metrics:
            print(f"  Transforms intrinsic clusters: {dual_metrics.get('transforms_n_intrinsic_clusters', '?')}")

        row = {
            "date": date,
            "group": "PASS" if date in PASS_DATES else "FAIL",
            **img_metrics,
            **traj_metrics,
            **intr_metrics,
            **dual_metrics,
        }
        all_metrics.append(row)

    # Save per-date CSV
    csv_path = os.path.join(OUT_DIR, "ACQUISITION_COMPARISON.csv")
    if all_metrics:
        fields = list(all_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_metrics)
        print(f"\nSaved: {csv_path}")

    # Summary: PASS vs FAIL group stats
    pass_rows = [r for r in all_metrics if r["group"] == "PASS"]
    fail_rows = [r for r in all_metrics if r["group"] == "FAIL"]

    summary = {}
    for key in ["brightness_mean", "saturation_mean", "blur_laplacian_var_mean",
                 "trajectory_path_length", "adj_baseline_mean_deg",
                 "fx_range", "transforms_n_intrinsic_clusters"]:
        pass_vals = [r[key] for r in pass_rows if key in r and r[key] != -1]
        fail_vals = [r[key] for r in fail_rows if key in r and r[key] != -1]
        if pass_vals and fail_vals:
            summary[key] = {
                "pass_mean": float(np.mean(pass_vals)),
                "pass_std": float(np.std(pass_vals)),
                "fail_mean": float(np.mean(fail_vals)),
                "fail_std": float(np.std(fail_vals)),
                "diff": float(np.mean(fail_vals) - np.mean(pass_vals)),
            }

    summary_path = os.path.join(OUT_DIR, "ACQUISITION_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("PASS vs FAIL SUMMARY")
    print(f"{'='*60}")
    for key, stats in summary.items():
        direction = "HIGHER" if stats["diff"] > 0 else "LOWER"
        print(f"  {key}:")
        print(f"    PASS: {stats['pass_mean']:.4f} ± {stats['pass_std']:.4f}")
        print(f"    FAIL: {stats['fail_mean']:.4f} ± {stats['fail_std']:.4f}")
        print(f"    Δ: {stats['diff']:+.4f} ({direction} in FAIL)")


if __name__ == "__main__":
    main()
