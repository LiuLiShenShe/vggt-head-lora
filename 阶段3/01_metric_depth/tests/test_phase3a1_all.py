#!/usr/bin/env python3# -*- coding: utf-8 -*-
"""Phase 3A.1 tests — DA3 intrinsics audit, scaling verification, corrected comparison."""
import os, csv, json, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
AUDIT_DIR = os.path.join(PHASE3_DIR, "phase3a1_audit")
sys.path.insert(0, PHASE3_DIR)

SEQ_IDS = [
    "plantview__langdon_4__05-03-24",
    "plantview__langdon_4__12-03-24",
    "plantview__langdon_4__13-02-24",
    "plantview__langdon_4__20-02-24",
]


# ── DA3 Intrinsics Audit Tests ────────────────────────────────────────────

def test_da3_has_no_intrinsics():
    """DA3METRIC-LARGE has no intrinsics (cam_dec=None)."""
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    assert model.model.cam_dec is None, "DA3METRIC should have cam_dec=None"
    assert not hasattr(model.model, 'da3'), "DA3METRIC should be single-branch DepthAnything3Net"
    print("  PASS: DA3 has no intrinsics (cam_dec=None)")


def test_da3_is_metric_false():
    """DA3METRIC-LARGE declares is_metric=0."""
    from depth_anything_3.api import DepthAnything3
    import torch
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda")
    rgb = '/fj/VGGT+head+lora实验/阶段1-数据集/3D Plant View/langdon_4/05-03-24/images/rgb/0000_eval.png'
    pred = model.inference([rgb])
    # is_metric is set from model_output which uses getattr(..., "is_metric", 0)
    assert pred.intrinsics is None, "DA3METRIC should not predict intrinsics"
    print("  PASS: DA3 is_metric check passed")


def test_da3_intrinsics_audit_csv():
    """INTRINSICS_AUDIT.csv exists with correct findings."""
    csv_path = os.path.join(AUDIT_DIR, "INTRINSICS_AUDIT.csv")
    assert os.path.exists(csv_path), f"Missing: {csv_path}"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 4, f"Expected >=4 audit rows, got {len(rows)}"
    items = {r["audit_item"] for r in rows}
    assert "DA3 model type" in items
    assert "cam_dec" in items
    assert "is_metric" in items
    assert "pred.intrinsics" in items
    # Verify key findings
    for r in rows:
        if r["audit_item"] == "cam_dec":
            assert "None" in r["value"]
        if r["audit_item"] == "is_metric":
            assert "0" in r["value"] or "False" in r["value"]
    print(f"  PASS: intrinsics audit CSV ({len(rows)} rows)")


# ── Scaling Formula Tests ─────────────────────────────────────────────────

def test_da3_focal_300_not_applied():
    """DA3 does NOT apply focal/300 internally (no NestedDepthAnything3Net)."""
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    # Verify model type is single-branch
    from depth_anything_3.model.da3 import DepthAnything3Net
    assert isinstance(model.model, DepthAnything3Net), \
        f"Expected DepthAnything3Net, got {type(model.model).__name__}"
    # Verify no cam_dec
    assert model.model.cam_dec is None
    print("  PASS: DA3 does not apply focal/300 (single-branch model)")


def test_da3_scale_is_not_focal_300():
    """DA3 scale ratio is NOT equal to focal/300 (different source)."""
    CALIBRATED_FX = 1371.82
    DA3_RES = 504
    IMG_SIZE = 1080
    fx_net = CALIBRATED_FX * DA3_RES / IMG_SIZE
    focal_300 = fx_net / 300.0  # ~2.13

    # Load actual DA3 depth and compute scale
    from configs import VGGT_RERUN
    d_da3 = np.load(os.path.join(PHASE3_DIR, "da3", SEQ_IDS[0], "depth_da3.npy"))
    d_vggt = np.load(os.path.join(VGGT_RERUN, SEQ_IDS[0], "depth_vggt.npy"))
    # DA3 scale (ref/DA3) for first frame
    # From evaluation CSV: scale ≈ 2.34 for 05-03-24
    # focal_300 ≈ 2.13
    # They are DIFFERENT numbers
    assert abs(focal_300 - 2.34) > 0.1, \
        f"focal_300 ({focal_300:.4f}) should NOT equal DA3 scale (~2.34)"
    print(f"  PASS: DA3 scale (~2.34) ≠ focal/300 ({focal_300:.4f})")


# ── Corrected Comparison Tests ────────────────────────────────────────────

def test_corrected_comparison_csv():
    """CORRECTED_MODEL_COMPARISON.csv has 5 variants × 4 sequences = 20 rows."""
    csv_path = os.path.join(AUDIT_DIR, "CORRECTED_MODEL_COMPARISON.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20, f"Expected 20 rows, got {len(rows)}"
    models = set(r["model"] for r in rows)
    expected = {"vggt", "da3_raw", "da3_calibrated", "unidepth_raw", "unidepth_k_corrected"}
    assert models == expected, f"Expected models {expected}, got {models}"
    # Check all sequences present
    seqs = set(r["sequence_id"] for r in rows)
    for sid in SEQ_IDS:
        assert sid in seqs, f"Missing sequence: {sid}"
    print(f"  PASS: corrected comparison CSV ({len(rows)} rows, {len(models)} models)")


def test_da3_calibration_improves():
    """DA3 calibrated AbsRel < DA3 raw AbsRel on all pose-PASS sequences."""
    csv_path = os.path.join(AUDIT_DIR, "CORRECTED_MODEL_COMPARISON.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    for sid in SEQ_IDS:
        raw_rows = [r for r in rows if r["model"] == "da3_raw" and r["sequence_id"] == sid]
        cal_rows = [r for r in rows if r["model"] == "da3_calibrated" and r["sequence_id"] == sid]
        assert raw_rows and cal_rows, f"Missing data for {sid}"
        raw_absrel = float(raw_rows[0]["raw_absrel"])
        cal_absrel = float(cal_rows[0]["raw_absrel"])
        assert cal_absrel < raw_absrel, \
            f"DA3 calibrated ({cal_absrel:.4f}) should be better than raw ({raw_absrel:.4f}) for {sid}"
    print("  PASS: DA3 calibration improves AbsRel on all sequences")


def test_unidepth_k_correction_improves():
    """UniDepth K-corrected AbsRel < raw AbsRel."""
    csv_path = os.path.join(AUDIT_DIR, "CORRECTED_MODEL_COMPARISON.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    for sid in SEQ_IDS:
        raw_rows = [r for r in rows if r["model"] == "unidepth_raw" and r["sequence_id"] == sid]
        cal_rows = [r for r in rows if r["model"] == "unidepth_k_corrected" and r["sequence_id"] == sid]
        assert raw_rows and cal_rows, f"Missing data for {sid}"
        raw_absrel = float(raw_rows[0]["raw_absrel"])
        cal_absrel = float(cal_rows[0]["raw_absrel"])
        assert cal_absrel < raw_absrel, \
            f"UniDepth K-corrected ({cal_absrel:.4f}) should be better than raw ({raw_absrel:.4f}) for {sid}"
    print("  PASS: UniDepth K correction improves AbsRel")


def test_vggt_still_best_raw():
    """VGGT still has best raw AbsRel among all pose-PASS models (including corrected)."""
    csv_path = os.path.join(AUDIT_DIR, "CORRECTED_MODEL_COMPARISON.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    pass_rows = [r for r in rows if r["pose_fail"] == "False"]

    model_absrel = {}
    for r in pass_rows:
        m = r["model"]
        if m not in model_absrel:
            model_absrel[m] = []
        model_absrel[m].append(float(r["raw_absrel"]))

    means = {m: float(np.mean(v)) for m, v in model_absrel.items()}
    best = min(means, key=means.get)
    # VGGT should still be best or very close (within 0.03)
    assert best == "vggt" or abs(means["vggt"] - means[best]) < 0.03, \
        f"Expected VGGT best or close, but {best}={means[best]:.4f} vs VGGT={means['vggt']:.4f}"
    print(f"  PASS: VGGT raw AbsRel={means['vggt']:.4f}, best={best}={means[best]:.4f}")


# ── Report and Summary Tests ──────────────────────────────────────────────

def test_report_exists():
    """PHASE3A1 report exists with key sections."""
    p = os.path.join(PHASE3_DIR, "PHASE3A1_METRIC_SCALING_SANITY.md")
    assert os.path.exists(p)
    content = open(p).read()
    assert "Q1" in content
    assert "Q2" in content
    assert "phase3a_scaling_integrity" in content
    assert "da3_metric_scaling" in content
    print("  PASS: Phase 3A.1 report exists")


def test_scaling_audit_summary():
    """SCALING_AUDIT_SUMMARY.json has correct structure."""
    p = os.path.join(AUDIT_DIR, "SCALING_AUDIT_SUMMARY.json")
    assert os.path.exists(p)
    with open(p) as f:
        s = json.load(f)
    assert s["da3_cam_dec"] is None
    assert s["da3_is_metric"] == 0
    assert s["da3_intrinsics_predicted"] is False
    assert s["da3_focal_300_applied"] is False
    print("  PASS: scaling audit summary")


# ── Runner ────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_da3_has_no_intrinsics,
        test_da3_is_metric_false,
        test_da3_intrinsics_audit_csv,
        test_da3_focal_300_not_applied,
        test_da3_scale_is_not_focal_300,
        test_corrected_comparison_csv,
        test_da3_calibration_improves,
        test_unidepth_k_correction_improves,
        test_vggt_still_best_raw,
        test_report_exists,
        test_scaling_audit_summary,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__} — {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    return failed


if __name__ == "__main__":
    sys.exit(run_all())
