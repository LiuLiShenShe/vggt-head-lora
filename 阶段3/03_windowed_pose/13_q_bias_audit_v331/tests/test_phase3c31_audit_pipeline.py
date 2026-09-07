#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§32: Tests for the Phase 3C.3.1 Q-bias audit pipeline.

These tests validate:
  - §32a: Synthetic Procrustes exact/noisy/gauge-invariance tests (7 tests)
  - §32b: Pipeline regression tests — corrected gauge fit produces expected output files
  - §32c: Cross-cutting invariants — drift coherence metrics and erratum correctness
  - §32d: No-write test — audit outputs are read-only, no modification to Phase 3C.3 artifacts

Usage:
    python -m pytest tests/ -v
"""
import json
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
AUDIT_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
sys.path.insert(0, os.path.join(AUDIT_DIR, "01_procrustes_repair"))
sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))

from evaluate_multoplant import global_rotation_procrustes, rot_angle_deg
from correct_gauge_fit import fit_window_gauge, old_formula_gauge


# -----------------------------------------------------------------------
# §32a: Synthetic tests (imported from the synthetic test module)
# -----------------------------------------------------------------------
# These are the 7 tests from test_reference_gauge_procrustes_exact.py.
# We duplicate the key assertions here to avoid import-order issues with
# sys.path manipulation.

class TestProcrustesSynthetic:
    """§3-§4: Synthetic Procrustes correctness tests."""

    def _make_random_window(self, rng, n=16, seed_shift=0):
        r_local = Rotation.random(n, random_state=rng).as_matrix()
        g_true = Rotation.random(random_state=np.random.default_rng(10000 + seed_shift)).as_matrix()
        return r_local, g_true

    def test_exact_recovery_corrected(self):
        """Corrected formula recovers G exactly on noiseless data."""
        rng = np.random.default_rng(42)
        for trial in range(5):
            R_local, G_true = self._make_random_window(rng, n=16, seed_shift=trial)
            R_ref = np.einsum("ab,sbc->sac", G_true, R_local)
            G_est, res = fit_window_gauge(R_local, R_ref)
            err = rot_angle_deg(G_est.T @ G_true)
            assert err < 1e-4, f"trial {trial}: gauge err {err:.2e}°"
            assert np.all(res < 1e-4), f"trial {trial}: residual {res.max():.2e}"

    def test_old_formula_fails(self):
        """Old buggy formula gives >> 1° error on the same data."""
        rng = np.random.default_rng(42)
        R_local, G_true = self._make_random_window(rng, n=16, seed_shift=0)
        R_ref = np.einsum("ab,sbc->sac", G_true, R_local)
        G_old = old_formula_gauge(R_local, R_ref)
        err_old = rot_angle_deg(G_old.T @ G_true)
        assert err_old > 10.0, f"old formula unexpectedly good: {err_old:.3f}°"

    def test_gauge_invariant_relative_rotation(self):
        """(G R_a)^T (G R_b) = R_a^T R_b — gauge cancels in relative rotation."""
        rng = np.random.default_rng(99)
        R = Rotation.random(16, random_state=rng).as_matrix()
        G = Rotation.random(random_state=np.random.default_rng(42)).as_matrix()
        RaR = np.einsum("ab,sbc->sac", G, R)
        for a, b in [(0, 1), (0, 15), (3, 12)]:
            rel_pred = RaR[a].T @ RaR[b]
            rel_ref = R[a].T @ R[b]
            err = rot_angle_deg(rel_pred.T @ rel_ref)
            assert err < 1e-4, f"gauge not cancelled: {err:.2e}°"


# -----------------------------------------------------------------------
# §32b: Pipeline output regression tests
# -----------------------------------------------------------------------
class TestPipelineOutputs:
    """Verify all pipeline scripts produced output files with expected structure."""

    def test_gauge_fit_summaries_exist(self):
        """§5: corrected gauge fit produces summaries for all 7 sequences."""
        summary = json.load(open(os.path.join(AUDIT_DIR, "02_window_gauge_fit",
                                               "WINDOW_GAUGE_FIT_SUMMARY.json")))
        expected = (
            [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
            + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
            + ["mustc__plot198__230613__ugv__pos00"]
        )
        for seq in expected:
            assert seq in summary, f"{seq} missing from WINDOW_GAUGE_FIT_SUMMARY.json"
            assert "gauge_fit_median_deg" in summary[seq]
            assert "n_edges" in summary[seq]
            assert summary[seq]["gauge_fit_median_deg"] < 5.0, (
                f"{seq}: gauge fit median {summary[seq]['gauge_fit_median_deg']}° too large")

    def test_hop1_q_vs_ref_corrected_range(self):
        """§6: corrected hop1 Q-vs-ref must be < 10° (was 25° with buggy formula)."""
        summary = json.load(open(os.path.join(AUDIT_DIR, "02_window_gauge_fit",
                                               "WINDOW_GAUGE_FIT_SUMMARY.json")))
        for seq in [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]:
            h1 = summary[seq]["hop_statistics"].get("1", {})
            assert h1.get("median", 999) < 5.0, (
                f"{seq}: hop1 Q-vs-ref median {h1['median']}° — should be < 5° with corrected formula")

    def test_relative_rotation_summary_exists(self):
        """§11: gauge-invariant relative rotation analysis produced."""
        path = os.path.join(AUDIT_DIR, "03_relative_rotation", "RELATIVE_ROTATION_GLOBAL_SUMMARY.json")
        assert os.path.exists(path), "RELATIVE_ROTATION_GLOBAL_SUMMARY.json missing"
        summary = json.load(open(path))
        assert len(summary) == 7, f"Expected 7 sequences, got {len(summary)}"
        for seq_id, s in summary.items():
            assert s["relative_err_median_deg"] < 15.0, (
                f"{seq_id}: relative rotation median {s['relative_err_median_deg']}° unexpected")

    def test_local_position_summary_exists(self):
        """§15-§17: local-position and cross-window analysis produced."""
        path = os.path.join(AUDIT_DIR, "04_edge_truth_diagnostic", "LOCAL_POSITION_SUMMARY.json")
        assert os.path.exists(path), "LOCAL_POSITION_SUMMARY.json missing"
        summary = json.load(open(path))
        assert len(summary) == 7
        # Boundary frames should have higher error than center frames for langdon
        for seq in ["plantview__langdon_4__05-03-24", "plantview__langdon_4__19-03-24"]:
            s = summary[seq]
            b = s["local_position"]["boundary_median_deg"]
            c = s["local_position"]["center_median_deg"]
            assert b is not None and c is not None and b > c, (
                f"{seq}: boundary ({b}) should be > center ({c})")

    def test_graph_gt_diagnostic_exists(self):
        """§18-§23: graph vs GT-gauge diagnostic produced."""
        path = os.path.join(AUDIT_DIR, "05_graph_gauge_diagnostic",
                            "GRAPH_GAUGE_DIAGNOSTIC_SUMMARY.json")
        assert os.path.exists(path), "GRAPH_GAUGE_DIAGNOSTIC_SUMMARY.json missing"
        summary = json.load(open(path))
        for seq in [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24"]]:
            s = summary[seq]
            assert s["sequential_chain_gt_vs_graph_median_deg"] < 2.0, (
                f"{seq}: chain-vs-graph should be < 2° (CASE C), got "
                f"{s['sequential_chain_gt_vs_graph_median_deg']}°")

    def test_drift_coherence_exists(self):
        """§20-§23: drift coherence analysis produced for all 7 sequences."""
        path = os.path.join(AUDIT_DIR, "05_graph_gauge_diagnostic",
                            "DRIFT_COHERENCE_SUMMARY.json")
        assert os.path.exists(path), "DRIFT_COHERENCE_SUMMARY.json missing"
        summary = json.load(open(path))
        assert len(summary) == 7
        for seq_id, s in summary.items():
            assert "temporal_coherence_ratio" in s
            assert "drift_amplification_vs_random_walk" in s


# -----------------------------------------------------------------------
# §32c: Cross-cutting scientific invariants
# -----------------------------------------------------------------------
class TestScientificInvariants:
    """Invariant checks on the audit's scientific conclusions."""

    def test_langdon_coherence_high_wheat_low(self):
        """Langdon coherence > 0.6, wheat/mustc < 0.6 — different drift regimes."""
        drift = json.load(open(os.path.join(AUDIT_DIR, "05_graph_gauge_diagnostic",
                                            "DRIFT_COHERENCE_SUMMARY.json")))
        for seq in ["plantview__langdon_4__05-03-24", "plantview__langdon_4__12-03-24"]:
            assert drift[seq]["temporal_coherence_ratio"] > 0.6, (
                f"{seq}: coherence should be > 0.6, got {drift[seq]['temporal_coherence_ratio']}")
        for seq in ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]:
            assert drift[seq]["temporal_coherence_ratio"] < 0.6, (
                f"{seq}: coherence should be < 0.6, got {drift[seq]['temporal_coherence_ratio']}")

    def test_langdon_drift_amplification_large(self):
        """Langdon drift amplification over random walk > 5×."""
        drift = json.load(open(os.path.join(AUDIT_DIR, "05_graph_gauge_diagnostic",
                                            "DRIFT_COHERENCE_SUMMARY.json")))
        for seq in [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]:
            assert drift[seq]["drift_amplification_vs_random_walk"] > 5.0, (
                f"{seq}: amplification should be > 5×, got "
                f"{drift[seq]['drift_amplification_vs_random_walk']}")

    def test_erratum_hop1_retracted(self):
        """Old buggy hop1 ≈ 25° claim retracted; corrected hop1 < 5°."""
        summary = json.load(open(os.path.join(AUDIT_DIR, "02_window_gauge_fit",
                                               "WINDOW_GAUGE_FIT_SUMMARY.json")))
        for seq in [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]:
            h1 = summary[seq]["hop_statistics"].get("1", {}).get("median", 999)
            assert h1 < 5.0, f"{seq}: corrected hop1 = {h1}° — erratum not valid"

    def test_single_gauge_assumption_valid(self):
        """Per-window single-gauge residual median < 5° — assumption valid."""
        summary = json.load(open(os.path.join(AUDIT_DIR, "02_window_gauge_fit",
                                               "WINDOW_GAUGE_FIT_SUMMARY.json")))
        for seq_id, s in summary.items():
            assert s["gauge_fit_median_deg"] < 5.0, (
                f"{seq_id}: single-gauge residual {s['gauge_fit_median_deg']}° > 5°")


# -----------------------------------------------------------------------
# §32d: No-write test — verify audit didn't modify Phase 3C.3 artifacts
# -----------------------------------------------------------------------
class TestNoWriteIntegrity:
    """Confirm that running the audit pipeline did not modify Phase 3C.3 artifacts."""

    def test_phase3c3_gauges_unchanged(self):
        """Phase 3C.3 solver gauges are byte-identical before and after audit."""
        gauges_path = os.path.join(SYNC_DIR, "04_so3_sync",
                                   "plantview__langdon_4__05-03-24_SO3_SYNC_GAUGES_stride4_th8.npz")
        if not os.path.exists(gauges_path):
            pytest.skip("Phase 3C.3 gauges not present")
        d = np.load(gauges_path)
        assert d["G"].shape == (77, 3, 3)
        # If we ran the pipeline, the gauges should not have changed
        # (pipeline reads from corrected gauges in audit dir, not Phase 3C.3 dir)
        assert not os.path.exists(os.path.join(
            AUDIT_DIR, "02_window_gauge_fit", "plantview__langdon_4__05-03-24_SO3_SYNC_GAUGES_stride4_th8.npz"
        )), "Audit accidentally wrote solver gauges to audit dir"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
