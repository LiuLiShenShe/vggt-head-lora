#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic test: verify that orientation-only metric is independent of center scale.

Constructs a scene with correct rotations but scale-distorted centers.
Verifies that per-segment orientation error stays small even when centers are wrong.
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(
    "/fj/VGGT+head+lora实验", "阶段3", "02_pose_robustness", "03_pose_evaluation"))
from evaluate_multoplant import global_rotation_procrustes, rot_angle_deg, w2c_centers, horn_sim3


def make_circular_trajectory(n_frames, radius=5.0):
    """Generate circular trajectory with correct rotations."""
    angles = np.linspace(0, 2 * np.pi * n_frames / (n_frames + 1), n_frames)
    centers = np.zeros((n_frames, 3))
    centers[:, 0] = radius * np.cos(angles)
    centers[:, 1] = radius * np.sin(angles)

    # c2w rotations: cameras look at center
    R_c2w = np.zeros((n_frames, 3, 3))
    for i in range(n_frames):
        forward = -centers[i] / np.linalg.norm(centers[i])
        up = np.array([0, 0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_c2w[i] = np.column_stack([right, up, -forward])

    return centers, R_c2w


def make_w2c(R_c2w, centers):
    """Convert c2w rotations + centers to 3x4 w2c extrinsics."""
    R_w2c = R_c2w.transpose(0, 2, 1)
    t_w2c = -np.einsum("sij,sj->si", R_w2c, centers)
    ext = np.zeros((len(centers), 3, 4))
    ext[:, :3, :3] = R_w2c
    ext[:, :3, 3] = t_w2c
    return ext


def test_orientation_independent_of_center_scale():
    """Correct rotations + non-uniform scale distortion → orientation error ≈ 0."""
    n = 20
    centers, R_c2w = make_circular_trajectory(n, radius=5.0)
    ext_ref = make_w2c(R_c2w, centers)

    # Non-uniform scale distortion (simulates drift — horn_sim3 cannot fix this)
    scale_factors = 1.0 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, n))
    centers_distorted = centers * scale_factors[:, None]
    ext_pred = make_w2c(R_c2w, centers_distorted)

    # Orientation-only evaluation
    R_ref = ext_ref[:, :3, :3].transpose(0, 2, 1)
    R_vggt = ext_pred[:, :3, :3].transpose(0, 2, 1)
    Rg = global_rotation_procrustes(R_vggt, R_ref)

    rot_errors = np.array([rot_angle_deg((Rg @ R_vggt[i]).T @ R_ref[i]) for i in range(n)])
    rot_med = float(np.median(rot_errors))

    # Center evaluation — Sim(3) cannot correct non-uniform distortion
    c_ref = w2c_centers(ext_ref)
    c_pred = w2c_centers(ext_pred)
    s, R_s, t_s = horn_sim3(c_pred, c_ref)
    c_aligned = s * (R_s @ c_pred.T).T + t_s
    c_err = np.linalg.norm(c_aligned - c_ref, axis=1)
    ref_spread = np.linalg.norm(c_ref - c_ref.mean(0), axis=1).mean()
    c_norm = c_err / max(ref_spread, 1e-10)

    print(f"Orientation-only: rot_med={rot_med:.4f}° (should be ~0°)")
    print(f"Center-only:      cen_med={np.median(c_norm):.4f} (should be large)")
    print(f"Scale estimated:  s={s:.4f}")

    assert rot_med < 1.0, f"Orientation error should be ~0° with correct rotations, got {rot_med}°"
    assert np.median(c_norm) > 0.05, f"Center error should be large with non-uniform distortion, got {np.median(c_norm)}"
    print("\nPASS: test_orientation_independent_of_center_scale")


def test_center_only_detects_shape_distortion():
    """Scale-distorted circular trajectory → low trajectory cosine."""
    n = 30
    centers, R_c2w = make_circular_trajectory(n, radius=5.0)
    ext_ref = make_w2c(R_c2w, centers)

    # Non-uniform scale distortion (simulates drift)
    scale_factors = 1.0 + 0.3 * np.sin(np.linspace(0, 3 * np.pi, n))
    centers_distorted = centers * scale_factors[:, None]
    ext_pred = make_w2c(R_c2w, centers_distorted)

    c_ref = w2c_centers(ext_ref)
    c_pred = w2c_centers(ext_pred)
    s, R_s, t_s = horn_sim3(c_pred, c_ref)
    c_aligned = s * (R_s @ c_pred.T).T + t_s

    # Trajectory cosine
    dv = np.diff(c_aligned, axis=0)
    dr = np.diff(c_ref, axis=0)
    cosines = [np.dot(dv[i], dr[i]) / (np.linalg.norm(dv[i]) * np.linalg.norm(dr[i]))
               for i in range(len(dv))]
    tc = np.mean(cosines)

    print(f"Trajectory cosine: {tc:.4f} (should be <1.0 due to shape distortion)")
    assert tc < 0.99, f"Trajectory cosine should show distortion, got {tc}"
    print("PASS: test_center_only_detects_shape_distortion")


if __name__ == "__main__":
    test_orientation_independent_of_center_scale()
    test_center_only_detects_shape_distortion()
    print("\nAll orientation metric tests PASSED")
