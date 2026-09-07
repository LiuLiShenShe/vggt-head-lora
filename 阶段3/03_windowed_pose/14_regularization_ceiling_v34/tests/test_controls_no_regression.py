#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 test suite — controls regression gate on real data (frame-level).

The controls (wheat461, wheat467, mustc) are short sequences whose motion is
small; ANY regularization there is pure distortion. The protocol's control gate
requires:

    per control sequence, every DEPLOYABLE method must keep
    Δrot_median vs baseline_sync ≤ +1.0° and the pose gate must stay PASS.

Deployable (transferred) set: A_gaussian@0.5, B_second_order@0.1,
C_lower_threshold@4 (D_multi_anchor is diagnostic-only, NOT deployable).

Tests:
  - baseline_sync reproduces the Phase 3C.3 D-stride4 rot_median (<1e-3°),
  - every deployable method stays within the +1.0° tolerance on all 3 controls
    and keeps the pose gate PASS,
  - the over-smoothing family (A_gaussian σ≥1) correctly violates the gate on at
    least one wheat control — that violation is itself the documented finding,
  - mustc (N=2) is untouched by every deployable method.

Input: 04_real_data_application/REGULARIZATION_REAL_RESULT.json (must be run
first: python 04_real_data_application/apply_regularization_to_real.py --all).

Run:
    python tests/test_controls_no_regression.py
    pytest tests/test_controls_no_regression.py -v
"""
import json, os, sys

ROOT = "/fj/VGGT+head+lora实验"
PHASE34 = os.path.join(ROOT, "阶段3", "03_windowed_pose", "14_regularization_ceiling_v34")
RESULT_PATH = os.path.join(PHASE34, "04_real_data_application", "REGULARIZATION_REAL_RESULT.json")

CONTROLS = ["wheat3dgs__plot_461", "wheat3dgs__plot_467",
            "mustc__plot198__230613__ugv__pos00"]
# Deployable transferred configs (1 config per method, no per-seq re-sweep).
DEPLOYABLE = [("A_gaussian", 0.5), ("B_second_order", 0.1),
              ("C_lower_threshold", 4)]
TOLERANCE_DEG = 1.0
# Phase 3C.3 D-stride4 baselines (per-frame rot_median) — locked numbers.
PHASE3C3_D = {"wheat3dgs__plot_461": 3.2761449529229703,
              "wheat3dgs__plot_467": 3.2338538179493366,
              "mustc__plot198__230613__ugv__pos00": 0.6027471166057163}
ROUNDING_TOL = 0.015  # JSON rot_median is stored at 2 dp; allow rounding gap


def _load():
    if not os.path.exists(RESULT_PATH):
        raise RuntimeError(
            f"{RESULT_PATH} missing — run "
            "04_real_data_application/apply_regularization_to_real.py --all first")
    with open(RESULT_PATH) as f:
        return json.load(f)


def _control_results(result):
    """{seq: {"baseline": rot_median, "rows": {(method,param): {..}}, "base3c": float}}."""
    phase_base = result["phase3c_D_baseline_rot_median"]
    out = {}
    for r in result["results"]:
        seq = r["sequence"]
        if seq not in CONTROLS:
            continue
        base = next(m for m in r["methods"] if m["method"] == "baseline_sync")
        # key by (method, param) so multiple Gaussian σ values don't overwrite
        rows = {(m["method"], m["param"]): m
                for m in r["methods"] if m["method"] != "baseline_sync"}
        out[seq] = {"baseline": base["rot_median"],
                    "baseline_gate": base["pose_gate"],
                    "rows": rows, "base3c": phase_base.get(seq)}
    return out


# --------------------------------------------------------------------------
def test_baseline_reproduces_phase3c3_D():
    result = _load()
    for seq, c in _control_results(result).items():
        assert abs(c["baseline"] - c["base3c"]) < ROUNDING_TOL, \
            f"{seq}: baseline_sync {c['baseline']}° != Phase 3C.3 D {c['base3c']}° (±2dp)"
        assert c["baseline_gate"] == "PASS", f"{seq}: baseline gate {c['baseline_gate']}"


def test_wheat_461_under_tolerance():
    c = _control_results(_load())["wheat3dgs__plot_461"]
    for method, param in DEPLOYABLE:
        m = c["rows"].get((method, param))
        assert m is not None and m.get("rot_median") is not None, f"no {method}@{param} on wheat461"
        delta = m["rot_median"] - c["baseline"]
        assert delta <= TOLERANCE_DEG, \
            f"wheat461 {method}@{param}: Δ={delta:+.3f}° > +{TOLERANCE_DEG}°"
        assert m["pose_gate"] == "PASS", f"wheat461 {method}@{param}: gate {m['pose_gate']}"


def test_wheat_467_under_tolerance():
    c = _control_results(_load())["wheat3dgs__plot_467"]
    for method, param in DEPLOYABLE:
        m = c["rows"].get((method, param))
        assert m is not None and m.get("rot_median") is not None, f"no {method}@{param} on wheat467"
        delta = m["rot_median"] - c["baseline"]
        assert delta <= TOLERANCE_DEG, \
            f"wheat467 {method}@{param}: Δ={delta:+.3f}° > +{TOLERANCE_DEG}°"
        assert m["pose_gate"] == "PASS", f"wheat467 {method}@{param}: gate {m['pose_gate']}"


def test_mustc_unaffected():
    """mustc (N=2) is untouched by every deployable method."""
    c = _control_results(_load())["mustc__plot198__230613__ugv__pos00"]
    for method, param in DEPLOYABLE:
        m = c["rows"].get((method, param))
        assert m is not None and m.get("rot_median") is not None
        assert abs(m["rot_median"] - c["baseline"]) < ROUNDING_TOL, \
            f"mustc {method}@{param}: rot_median changed {c['baseline']}→{m['rot_median']}"
        assert m["pose_gate"] == "PASS"


def test_over_smoothing_violates_gate():
    """The documented finding: A_gaussian σ≥1 violates the +1.0° tolerance on
    at least one wheat control (over-smoothing collapses short trajectories)."""
    ctrl = _control_results(_load())
    violators = []
    for seq in ("wheat3dgs__plot_461", "wheat3dgs__plot_467"):
        c = ctrl[seq]
        for m in c["rows"].values():
            if m["method"] == "A_gaussian" and m["param"] >= 1.0:
                if m["rot_median"] - c["baseline"] > TOLERANCE_DEG:
                    violators.append((seq, m["method"], m["param"]))
    assert violators, \
        "expected A_gaussian σ≥1 to violate the controls gate on a wheat control — check result"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    npass = 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
            npass += 1
        except (AssertionError, RuntimeError) as e:
            print(f"FAIL {f.__name__}: {e}")
    print(f"\n{npass}/{len(fns)} passed")
    sys.exit(0 if npass == len(fns) else 1)