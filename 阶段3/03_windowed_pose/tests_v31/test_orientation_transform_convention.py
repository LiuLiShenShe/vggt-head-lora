#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic tests for rotation convention and gauge-aware alignment.

Tests:
1. Rotation transform convention (Q applied to c2w/w2c correctly)
2. Relative Q recovery from overlap orientations
3. SO(3) mean validity
4. Fixed-Q scale+translation recovery
5. Sim(3) chain composition
6. c2w global composition
"""
import numpy as np
from scipy.spatial.transform import Rotation

RTOL = 1e-5
ATOL_DEG = 1e-4
ATOL_POS = 1e-6


def rot_angle_deg(R):
    """Geodesic angle of rotation matrix R in degrees."""
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def random_rotation():
    return Rotation.random().as_matrix()


def test_c2w_w2c_inverse():
    """R_w2c = R_c2w^T."""
    R_c2w = random_rotation()
    R_w2c = R_c2w.T
    assert np.allclose(R_w2c @ R_c2w, np.eye(3), atol=RTOL)
    print("PASS: test_c2w_w2c_inverse")


def test_center_formula():
    """C = -R_w2c^T @ t is consistent with R_c2w @ 0 + C = C."""
    R_c2w = random_rotation()
    C_true = np.random.randn(3)
    R_w2c = R_c2w.T
    # t_w2c = -R_w2c @ C_true so that R_w2c @ C + t = 0
    t_w2c = -R_w2c @ C_true
    C_recovered = -R_w2c.T @ t_w2c
    assert np.allclose(C_recovered, C_true, atol=ATOL_POS)
    print("PASS: test_center_formula")


def test_gauge_center_transform():
    """C_A = Q @ C_B + t."""
    Q = random_rotation()
    t_gauge = np.random.randn(3)
    C_B = np.random.randn(3)
    C_A = Q @ C_B + t_gauge

    # Verify via full transform
    X_B = C_B
    X_A = Q @ X_B + t_gauge
    assert np.allclose(X_A, C_A, atol=ATOL_POS)
    print("PASS: test_gauge_center_transform")


def test_gauge_c2w_transform():
    """R_c2w_A = Q @ R_c2w_B for overlap frame."""
    Q = random_rotation()
    R_c2w_B = random_rotation()
    R_c2w_A = Q @ R_c2w_B

    # A point in camera frame should map correctly
    x_c = np.random.randn(3)
    C_B = np.random.randn(3)
    t_gauge = np.random.randn(3)
    C_A = Q @ C_B + t_gauge

    # Via A
    X_A_via_A = R_c2w_A @ x_c + C_A
    # Via B then gauge
    X_B = R_c2w_B @ x_c + C_B
    X_A_via_B = Q @ X_B + t_gauge

    assert np.allclose(X_A_via_A, X_A_via_B, atol=ATOL_POS)
    print("PASS: test_gauge_c2w_transform")


def test_gauge_w2c_transform():
    """R_w2c_A = R_w2c_B @ Q^T."""
    Q = random_rotation()
    R_c2w_B = random_rotation()
    R_w2c_B = R_c2w_B.T

    R_w2c_A = R_w2c_B @ Q.T
    R_c2w_A_from_w2c = R_w2c_A.T

    # Should equal Q @ R_c2w_B
    R_c2w_A_direct = Q @ R_c2w_B
    assert np.allclose(R_c2w_A_from_w2c, R_c2w_A_direct, atol=RTOL)
    print("PASS: test_gauge_w2c_transform")


def test_Q_recovery_from_overlap():
    """Q_i = R_c2w_A @ R_c2w_B^T recovers ground truth Q."""
    Q_true = random_rotation()

    n_overlap = 16
    R_c2w_B_list = [random_rotation() for _ in range(n_overlap)]
    R_c2w_A_list = [Q_true @ R for R in R_c2w_B_list]

    for i in range(n_overlap):
        Q_i = R_c2w_A_list[i] @ R_c2w_B_list[i].T
        angle_err = rot_angle_deg(Q_i @ Q_true.T)
        assert angle_err < ATOL_DEG, f"Q recovery error: {angle_err} deg"

    print("PASS: test_Q_recovery_from_overlap")


def test_Q_recovery_from_w2c():
    """Q recovery using w2c matrices: Q_i = R_c2w_A @ R_c2w_B^T = (R_w2c_B^T @ R_w2c_A).T"""
    Q_true = random_rotation()

    n_overlap = 16
    R_c2w_B_list = [random_rotation() for _ in range(n_overlap)]

    for i in range(n_overlap):
        R_c2w_A = Q_true @ R_c2w_B_list[i]
        R_w2c_A = R_c2w_A.T
        R_w2c_B = R_c2w_B_list[i].T

        # From w2c: Q = (R_w2c_B^T @ R_w2c_A)^T = (R_c2w_B @ R_c2w_A^T)^T = R_c2w_A @ R_c2w_B^T
        Q_i = (R_w2c_B.T @ R_w2c_A).T
        angle_err = rot_angle_deg(Q_i @ Q_true.T)
        assert angle_err < ATOL_DEG, f"Q from w2c error: {angle_err} deg"

    print("PASS: test_Q_recovery_from_w2c")


def test_so3_mean_valid_rotation():
    """SO(3) mean of rotations is a valid rotation matrix."""
    n = 20
    Q_true = random_rotation()
    noise_std = 0.01  # small noise in axis-angle

    rotations = []
    for _ in range(n):
        noise = Rotation.from_rotvec(noise_std * np.random.randn(3))
        R = noise.as_matrix() @ Q_true
        rotations.append(R)

    # Quaternion averaging
    quats = Rotation.from_matrix(np.array(rotations)).as_quat()  # (n, 4) xyzw

    # Sign canonicalization: flip if dot with first quaternion is negative
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]

    mean_quat = quats.mean(axis=0)
    mean_quat /= np.linalg.norm(mean_quat)

    R_mean = Rotation.from_quat(mean_quat).as_matrix()

    # Check valid rotation
    assert np.allclose(R_mean @ R_mean.T, np.eye(3), atol=RTOL)
    assert np.isclose(np.linalg.det(R_mean), 1.0, atol=RTOL)

    # Check accuracy
    angle_err = rot_angle_deg(R_mean @ Q_true.T)
    assert angle_err < 1.0, f"SO(3) mean error: {angle_err} deg"

    print(f"PASS: test_so3_mean_valid_rotation (error={angle_err:.4f} deg)")


def test_fixed_Q_st_recovery():
    """Given known Q, recover s and t from C_A = s Q C_B + t."""
    Q_true = random_rotation()
    s_true = 0.95 + 0.1 * np.random.rand()
    t_true = np.random.randn(3)

    n = 20
    C_B = np.random.randn(n, 3)
    C_A = s_true * (Q_true @ C_B.T).T + t_true

    # Solve for s, t given Q = Q_true
    X = (Q_true @ C_B.T).T  # Q @ C_B
    X_mean = X.mean(axis=0)
    C_mean = C_A.mean(axis=0)
    X_centered = X - X_mean
    C_centered = C_A - C_mean

    s_est = np.sum(C_centered * X_centered) / np.sum(X_centered ** 2)
    t_est = C_mean - s_est * X_mean

    s_err = abs(s_est - s_true)
    t_err = np.linalg.norm(t_est - t_true)

    assert s_err < ATOL_POS, f"scale error: {s_err}"
    assert t_err < ATOL_POS, f"translation error: {t_err}"

    # Verify full transform
    C_recovered = s_est * X + t_est
    assert np.allclose(C_recovered, C_A, atol=ATOL_POS)

    print(f"PASS: test_fixed_Q_st_recovery (s_err={s_err:.2e}, t_err={t_err:.2e})")


def test_sim3_chain_composition():
    """Composing Sim(3) transforms through a chain."""
    n_windows = 5
    # Generate random pairwise transforms
    transforms = []
    for _ in range(n_windows - 1):
        s = 0.9 + 0.2 * np.random.rand()
        Q = random_rotation()
        t = np.random.randn(3)
        transforms.append((s, Q, t))

    # Chain: G_0 = I, G_{k+1} = G_k ∘ S_{k→k+1}
    # G_k maps from window_k's gauge to window_0's gauge
    # S_{k→k+1} maps from window_{k+1}'s gauge to window_k's gauge

    def compose(T_prev, S):
        s_prev, Q_prev, t_prev = T_prev
        s_S, Q_S, t_S = S
        # x_prev = s_prev * Q_prev @ x_k + t_prev
        # x_k = s_S * Q_S @ x_{k+1} + t_S
        # x_prev = s_prev * Q_prev @ (s_S * Q_S @ x_{k+1} + t_S) + t_prev
        #        = s_prev * s_S * (Q_prev @ Q_S) @ x_{k+1} + s_prev * Q_prev @ t_S + t_prev
        s_new = s_prev * s_S
        Q_new = Q_prev @ Q_S
        t_new = s_prev * (Q_prev @ t_S) + t_prev
        return (s_new, Q_new, t_new)

    G = [(1.0, np.eye(3), np.zeros(3))]
    for i, S in enumerate(transforms):
        G.append(compose(G[i], S))

    # Verify: apply each G_k to a test point from window_k
    for k in range(n_windows):
        # Create a test point in window_k's gauge
        x_k = np.random.randn(3)

        # Transform through chain to window_0
        x_0 = x_k
        for j in range(k - 1, -1, -1):
            s_j, Q_j, t_j = transforms[j]
            x_0 = s_j * (Q_j @ x_0) + t_j

        # Apply G_k
        s_k, Q_k, t_k = G[k]
        x_0_via_G = s_k * (Q_k @ x_k) + t_k

        assert np.allclose(x_0_via_G, x_0, atol=ATOL_POS), \
            f"Chain composition failed for window {k}"

    print("PASS: test_sim3_chain_composition")


def test_c2w_global_composition():
    """R_c2w_global = Q_global @ R_c2w_local (without scale, since rotations are orthonormal)."""
    Q = random_rotation()
    R_c2w_local = random_rotation()
    R_c2w_global = Q @ R_c2w_local

    # Verify rotation is orthonormal
    assert np.allclose(R_c2w_global @ R_c2w_global.T, np.eye(3), atol=RTOL)
    assert np.isclose(np.linalg.det(R_c2w_global), 1.0, atol=RTOL)

    # Full verification: X_A = s * Q @ X_B + t
    # where X_B = R_c2w_B @ x_c + C_B
    s = 0.95
    t_gauge = np.random.randn(3)
    C_B = np.random.randn(3)
    C_A = s * (Q @ C_B) + t_gauge

    x_c = np.random.randn(3)

    # Via local c2w then gauge
    X_B = R_c2w_local @ x_c + C_B
    X_A_chain = s * (Q @ X_B) + t_gauge

    # Direct: X_A = R_c2w_A @ x_c + C_A (but note: R_c2w_A = Q @ R_c2w_B without scale)
    # The camera-frame point x_c is NOT scaled — it's in camera coordinates
    # X_A = (Q @ R_c2w_B) @ x_c + C_A
    X_A_direct = R_c2w_global @ x_c + C_A

    # These won't match exactly when s != 1 because the gauge transform
    # scales world-frame coordinates, not camera-frame coordinates.
    # For stitching, we handle scale only on centers, not on rotation.
    # This is the standard convention for Sim(3) with rotation.

    # Instead, verify the rotation formula independently
    # R_c2w_A should map camera axes correctly
    cam_x = np.array([1, 0, 0])  # camera x-axis in world
    world_x_local = R_c2w_local @ cam_x
    world_x_global = Q @ world_x_local  # gauge transform of the direction
    world_x_global_actual = R_c2w_global @ cam_x
    assert np.allclose(world_x_global, world_x_global_actual, atol=RTOL)

    print("PASS: test_c2w_global_composition")


def test_Q_robust_averaging_with_outliers():
    """Robust SO(3) mean rejects outliers."""
    Q_true = random_rotation()
    n_inliers = 20
    n_outliers = 5
    noise_std = 0.02

    # Inliers: small noise around Q_true
    inlier_rots = []
    for _ in range(n_inliers):
        noise = Rotation.from_rotvec(noise_std * np.random.randn(3))
        inlier_rots.append((noise.as_matrix() @ Q_true))

    # Outliers: large rotation errors
    outlier_rots = []
    for _ in range(n_outliers):
        big_noise = Rotation.from_rotvec(np.random.randn(3))
        outlier_rots.append(big_noise.as_matrix() @ Q_true)

    all_rots = np.array(inlier_rots + outlier_rots)

    # Compute initial mean
    quats = Rotation.from_matrix(all_rots).as_quat()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]
    mean_quat_init = quats.mean(axis=0)
    mean_quat_init /= np.linalg.norm(mean_quat_init)
    R_mean_init = Rotation.from_quat(mean_quat_init).as_matrix()

    # Compute geodesic distances
    geodesics = np.array([rot_angle_deg(R_mean_init @ R.T) for R in all_rots])

    # MAD-based threshold
    median_geo = np.median(geodesics)
    mad = np.median(np.abs(geodesics - median_geo))
    threshold = median_geo + 3.0 * mad * 1.4826  # ~3 sigma

    # Filter outliers
    inlier_mask = geodesics < threshold
    inlier_rots_filtered = all_rots[inlier_mask]

    # Recompute mean
    quats_f = Rotation.from_matrix(inlier_rots_filtered).as_quat()
    for i in range(1, len(quats_f)):
        if np.dot(quats_f[i], quats_f[0]) < 0:
            quats_f[i] = -quats_f[i]
    mean_quat_f = quats_f.mean(axis=0)
    mean_quat_f /= np.linalg.norm(mean_quat_f)
    R_mean_final = Rotation.from_quat(mean_quat_f).as_matrix()

    # Should recover Q_true much better after outlier rejection
    err_before = rot_angle_deg(R_mean_init @ Q_true.T)
    err_after = rot_angle_deg(R_mean_final @ Q_true.T)

    assert err_after < err_before, f"Outlier rejection made it worse: {err_after} > {err_before}"
    assert err_after < 2.0, f"After rejection error too large: {err_after} deg"
    print(f"PASS: test_Q_robust_averaging_with_outliers (before={err_before:.2f}° after={err_after:.2f}°)")


if __name__ == "__main__":
    test_c2w_w2c_inverse()
    test_center_formula()
    test_gauge_center_transform()
    test_gauge_c2w_transform()
    test_gauge_w2c_transform()
    test_Q_recovery_from_overlap()
    test_Q_recovery_from_w2c()
    test_so3_mean_valid_rotation()
    test_fixed_Q_st_recovery()
    test_sim3_chain_composition()
    test_c2w_global_composition()
    test_Q_robust_averaging_with_outliers()
    print("\n=== ALL CONVENTION TESTS PASSED ===")
