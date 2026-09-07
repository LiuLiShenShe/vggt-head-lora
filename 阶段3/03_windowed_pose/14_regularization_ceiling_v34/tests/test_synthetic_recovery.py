#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 test suite — synthetic drift model recovery.

Verifies the synthetic testbed reproduces the real drift statistics it is fitted
to (coherence, magnitude distribution, over-rotation fraction), that the
injected drift has the observed magnitude, that all gauge matrices are valid
SO(3), that scenario A reproduces CASE-C consistency (triangle residuals ≈ 0),
that scenario B matches the real transitivity noise, that sampling is
deterministic under a fixed seed, and that the drift is predominantly
monotonic.

Run:
    python tests/test_synthetic_recovery.py          # pytest-style functions
    pytest tests/test_synthetic_recovery.py -v
"""
import os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE34 = os.path.join(ROOT, "阶段3", "03_windowed_pose", "14_regularization_ceiling_v34")
GEN_DIR = os.path.join(PHASE34, "01_synthetic_drift_model")
EVAL_DIR = os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation")
for d in (GEN_DIR, EVAL_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from synthetic_drift_generator import (
    LANGDON_SEQ, fit_error_statistics, generate_true_trajectory,
    sample_coherent_errors, build_synthetic_edges, injected_drift,
    realized_stats, rot_angle_deg,
)

COHERENCE_TARGET = 0.756
MAG_MEDIAN_DEG = 1.53
MAG_MEAN_DEG = 2.50


def make_testbed(seed=0):
    """Standard 77-window testbed: returns (mag_stats, G_true, errors, meta, edgesA, edgesB)."""
    mag_stats = fit_error_statistics(LANGDON_SEQ)
    G_true, _ = generate_true_trajectory(LANGDON_SEQ, n_windows=77, seed=seed)
    errors, meta = sample_coherent_errors(
        76, mag_stats, mag_stats["coherence_target"], seed=seed)
    edgesA = build_synthetic_edges(LANGDON_SEQ, G_true, errors, scenario="A",
                                   threshold=8, seed=seed)
    edgesB = build_synthetic_edges(LANGDON_SEQ, G_true, errors, scenario="B",
                                   threshold=8, seed=seed)
    return mag_stats, G_true, errors, meta, edgesA, edgesB


def _triangle_residual_median(edges):
    q = {}
    for e in range(len(edges["i"])):
        q[(int(edges["i"][e]), int(edges["j"][e]))] = edges["Q"][e]
    res = []
    for k in range(74):
        if (k, k + 1) in q and (k + 1, k + 2) in q and (k, k + 2) in q:
            E = q[(k, k + 2)].T @ (q[(k, k + 1)] @ q[(k + 1, k + 2)])
            res.append(rot_angle_deg(E))
    return float(np.median(res)) if res else None


def _assert_so3(G, msg=""):
    assert G.ndim == 3 and G.shape[1] == G.shape[2] == 3, f"{msg} shape"
    for k in range(len(G)):
        E = G[k].T @ G[k] - np.eye(3)
        assert np.abs(E).max() < 1e-8, f"{msg} not orthonormal @{k}"
        assert abs(np.linalg.det(G[k]) - 1.0) < 1e-8, f"{msg} det != +1 @{k}"


# --------------------------------------------------------------------------
def test_generator_matches_real_statistics():
    _, G_true, errors, _, edgesA, _ = make_testbed(seed=0)
    rs = realized_stats(edgesA["G_chain"], edgesA["G_true"])

    assert abs(rs["coherence"] - COHERENCE_TARGET) < 0.02, \
        f"coherence {rs['coherence']} too far from {COHERENCE_TARGET}"
    assert abs(rs["median_mag_deg"] - MAG_MEDIAN_DEG) < 0.15, \
        f"median mag {rs['median_mag_deg']} != 1.53±0.15"
    assert abs(rs["mean_mag_deg"] - MAG_MEAN_DEG) < 0.3, \
        f"mean mag {rs['mean_mag_deg']} != 2.50±0.3"
    assert rs["over_rotation_prob"] >= 0.93, \
        f"over-rotation fraction {rs['over_rotation_prob']} < 0.93"


def test_injected_drift_magnitude():
    _, _, _, _, edgesA, _ = make_testbed(seed=0)
    drift = injected_drift(edgesA["G_chain"], edgesA["G_true"])
    assert drift[-1] > 100.0, f"final injected drift {drift[-1]:.1f}° not > 100°"


def test_so3_validity():
    _, G_true, _, _, edgesA, _ = make_testbed(seed=0)
    _assert_so3(G_true, "G_true")
    _assert_so3(edgesA["G_chain"], "G_chain")
    # anchor preserved: chain starts at the true gauge
    d = rot_angle_deg(edgesA["G_chain"][0].T @ G_true[0])
    assert d < 1e-6, f"anchor not preserved ({d:.2e}°)"


def test_multi_hop_case_c_consistency():
    """Scenario A: multi-hop edges are chain products → triangle residuals ≈ 0."""
    _, _, _, _, edgesA, _ = make_testbed(seed=0)
    med = _triangle_residual_median(edgesA)
    assert med is not None and med < 0.5, f"scenA triangle residual {med}° not < 0.5°"


def test_multi_hop_noise_matches_real():
    """Scenario B: multi-hop edges carry the real transitivity noise (~0.37°)."""
    _, _, _, _, _, edgesB = make_testbed(seed=0)
    med = _triangle_residual_median(edgesB)
    assert med is not None and 0.2 < med < 0.9, \
        f"scenB triangle residual {med}° outside 0.2–0.9° (expect ≈0.37·√2)"


def test_deterministic_seed():
    _, G_true, _, _, edgesA, _ = make_testbed(seed=0)
    _, _, errors2, _, edgesA2, _ = make_testbed(seed=0)
    assert np.array_equal(edgesA["G_chain"], edgesA2["G_chain"]), \
        "same seed produced different G_chain"
    assert np.array_equal(edgesA["G_true"], edgesA2["G_true"]), \
        "same seed produced different G_true"


def test_drift_monotonic():
    """≥92 % of consecutive windows increase the accumulated drift."""
    _, _, _, _, edgesA, _ = make_testbed(seed=0)
    drift = injected_drift(edgesA["G_chain"], edgesA["G_true"])
    frac = float(np.mean(np.diff(drift) > -1e-9))
    assert frac >= 0.92, f"monotonic drift fraction {frac:.3f} < 0.92"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    npass = 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
            npass += 1
        except AssertionError as e:
            print(f"FAIL {f.__name__}: {e}")
    print(f"\n{npass}/{len(fns)} passed")
    sys.exit(0 if npass == len(fns) else 1)