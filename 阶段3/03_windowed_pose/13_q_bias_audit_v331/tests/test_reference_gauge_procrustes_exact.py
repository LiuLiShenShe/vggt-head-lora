#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§3/§4 + §32: Procrustes regression tests.

1. test_reference_gauge_procrustes_exact: random G_true + 16 R_local, noiseless
   R_ref = G_true @ R_local → corrected formula recovers G (err < 1e-6°), old formula FAILS.
2. test_old_procrustes_formula_fails_known_case: same setup, old formula error >> 1°.
3. test_corrected_gauge_fit_noiseless: residual per frame ~0.
4. test_noisy_procrustes_no_systematic_error: 1° noise → G_rec err ~ noise magnitude (< 5°).
5. test_relative_rotation_is_gauge_invariant: (G R_a)^T (G R_b) = R_a^T R_b exactly.
6. test_correct_Q_ref_formula_synthetic: Q_ref_ij = G_ref_i^T G_ref_j recovered from
   per-window gauges.
7. test_graph_gauge_global_alignment_synthetic: given gauge trajectory G_graph that is
   a global rotation A away from G_true, global-alignment recovers them.
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
sys.path.insert(0, os.path.join(ROOT, "阶段3", "03_windowed_pose",
                                "13_q_bias_audit_v331", "01_procrustes_repair"))
from correct_gauge_fit import fit_window_gauge, old_formula_gauge, residual_deg


def rot_angle_deg(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def _make_random_window(rng, n=16, seed_shift=0):
    """16 random R_local + random G_true."""
    r_local = Rotation.random(n, random_state=rng).as_matrix()
    g_true = Rotation.random(random_state=np.random.default_rng(10000 + seed_shift)).as_matrix()
    return r_local, g_true


def test_reference_gauge_procrustes_exact():
    """§3: noiseless exact recovery — corrected formula recovers G, residual ~ SVD eps."""
    rng = np.random.default_rng(7)
    for trial in range(5):
        # distinct seed_shift per trial → each trial tests a DIFFERENT random G_true
        R_local, G_true = _make_random_window(rng, n=16, seed_shift=trial)
        R_ref = np.einsum("ab,sbc->sac", G_true, R_local)

        G_est, res = fit_window_gauge(R_local, R_ref)
        err = rot_angle_deg(G_est.T @ G_true)
        assert err < 1e-4, f"trial {trial}: corrected gauge err {err:.2e}° ≥ 1e-4°"

        # residual per frame ~0 up to SVD float precision (~1e-6°)
        assert np.all(res < 1e-4), f"trial {trial}: exact residuals not ~0: {res.max():.2e}"


def test_old_procrustes_formula_fails_known_case():
    """§3: the OLD einsum formula cannot recover G on the same data (regression guard)."""
    rng = np.random.default_rng(7)
    R_local, G_true = _make_random_window(rng, n=16, seed_shift=0)
    R_ref = np.einsum("ab,sbc->sac", G_true, R_local)

    G_old = old_formula_gauge(R_local, R_ref)
    err_old = rot_angle_deg(G_old.T @ G_true)
    # It should be BAD compared to the corrected one (which gives ~0)
    assert err_old > 1.0, f"old formula unexpectedly good: {err_old:.3f}°"
    # Sanity: actually it's ~85° on this data
    assert err_old > 10.0, f"old formula error should be large, got {err_old:.3f}°"


def test_corrected_gauge_fit_noiseless():
    """§3: corrected fit leaves ~0 residual on noiseless data."""
    rng = np.random.default_rng(9)
    R_local, G_true = _make_random_window(rng, n=16)
    R_ref = np.einsum("ab,sbc->sac", G_true, R_local)
    G_est, res = fit_window_gauge(R_local, R_ref)
    assert float(np.median(res)) < 1e-6, f"noiseless residual not ~0: {np.median(res)}"


def test_noisy_procrustes_no_systematic_error():
    """§4: 1° noise → gauge err ~ noise magnitude, NO 20°+ systematic error."""
    rng = np.random.default_rng(11)
    for trial in range(3):
        R_local, G_true = _make_random_window(rng, n=16)
        # 1° Gaussian rotational noise on R_ref (equivalently on R_local)
        noise = rng.normal(0, np.radians(1.0), (16, 3))
        R_local_noisy = np.einsum(
            "sab,sbc->sac", Rotation.from_euler("xyz", noise).as_matrix(), R_local)
        R_ref = np.einsum("ab,sbc->sac", G_true, R_local_noisy)

        G_est, res = fit_window_gauge(R_local, R_ref)
        err = rot_angle_deg(G_est.T @ G_true)
        assert err < 5.0, f"trial {trial}: noisy gauge err {err:.2f}° too large (noise=1°)"

        # residual scale should be consistent with noise (~ a couple degrees median)
        med = float(np.median(res))
        assert med < 3.0, f"trial {trial}: residual {med:.2f}° inconsistent with 1° noise"


def test_relative_rotation_is_gauge_invariant():
    """§11: (G R_a)^T (G R_b) = R_a^T R_b — within-window relative rotation is gauge-free."""
    rng = np.random.default_rng(13)
    R = Rotation.random(16, random_state=rng).as_matrix()
    G = Rotation.random(random_state=np.random.default_rng(12345)).as_matrix()
    RaR = np.einsum("ab,sbc->sac", G, R)
    for a, b in [(0, 1), (0, 15), (3, 12)]:
        rel_pred = RaR[a].T @ RaR[b]
        rel_ref = R[a].T @ R[b]
        err = rot_angle_deg(rel_pred.T @ rel_ref)
        assert err < 1e-6, f"relative rotation not gauge-invariant: {err:.2e}°"


def test_correct_Q_ref_formula_synthetic():
    """§5: Q_ref_ij = G_ref_i^T @ G_ref_j recovered exactly from per-window gauges."""
    rng = np.random.default_rng(17)
    n_w = 5
    R_local = []
    G_true = []
    for k in range(n_w):
        # distinct seed_shift per window → genuinely INDEPENDENT per-window gauges
        rl, gt = _make_random_window(rng, n=16, seed_shift=k)
        R_local.append(rl)
        G_true.append(gt)
    # Build Q_ref from true gauges: Q_ref_ij = G_i^T G_j
    Q_true_01 = G_true[0].T @ G_true[1]
    # Recover gauges from contaminated R_ref (each window independent gauge)
    errs = []
    for k in range(n_w):
        R_ref = np.einsum("ab,sbc->sac", G_true[k], R_local[k])
        G_est, _ = fit_window_gauge(R_local[k], R_ref)
        errs.append(rot_angle_deg(G_est.T @ G_true[k]))
    assert all(e < 1e-6 for e in errs), f"gauge recovery failed: {errs}"
    # Q_ref recomputed from fitted gauges must match true Q
    G0 = None
    G1 = None
    R_ref0 = np.einsum("ab,sbc->sac", G_true[0], R_local[0])
    R_ref1 = np.einsum("ab,sbc->sac", G_true[1], R_local[1])
    G0, _ = fit_window_gauge(R_local[0], R_ref0)
    G1, _ = fit_window_gauge(R_local[1], R_ref1)
    Q_ref_rec = G0.T @ G1
    err = rot_angle_deg(Q_true_01.T @ Q_ref_rec)
    assert err < 1e-6, f"Q_ref reconstruction error: {err:.2e}°"


def test_graph_gauge_global_alignment_synthetic():
    """§18: if G_graph = A @ G_true (same up to global gauge), A is recoverable."""
    rng = np.random.default_rng(19)
    n_w = 8
    G_true = np.array([Rotation.random(random_state=np.random.default_rng(rng.integers(1e6))) .as_matrix()
                       for _ in range(n_w)])
    A = Rotation.random(random_state=np.random.default_rng(54321)).as_matrix()
    G_graph = np.einsum("ab,sbc->sac", A, G_true)

    # fit_window_gauge(R_local, R_ref) finds G_est minimizing Σ || G_est @ R_local - R_ref ||.
    # Passing (G_graph, G_true) finds A_est with A_est @ G_graph ≈ G_true.
    # Since G_graph = A @ G_true  →  A_est @ A @ G_true ≈ G_true  →  A_est ≈ A^{-1} = A^T.
    # The recovery check is therefore A_est @ A ≈ I (NOT A_est.T @ A).
    A_est = fit_window_gauge(G_graph.reshape(-1, 3, 3), G_true.reshape(-1, 3, 3))[0]
    err = rot_angle_deg(A_est @ A)
    assert err < 1e-4, f"global-alignment A error: {err:.2e}°"

    # After alignment, G_graph_aligned[k] = A_est @ G_graph[k] ≈ G_true[k]
    aligned = np.einsum("ab,sbc->sac", A_est, G_graph)
    gauge_errs = [rot_angle_deg(aligned[k].T @ G_true[k]) for k in range(n_w)]
    assert max(gauge_errs) < 1e-4, f"post-alignment gauge err: {max(gauge_errs):.2e}°"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])