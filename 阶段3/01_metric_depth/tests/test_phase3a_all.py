#!/usr/bin/env python3
"""Phase 3A tests — verify inference outputs, evaluation CSVs, and anchor integrity."""
import os, csv, json, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)

SEQ_IDS = [
    "plantview__langdon_4__05-03-24",
    "plantview__langdon_4__12-03-24",
    "plantview__langdon_4__13-02-24",
    "plantview__langdon_4__20-02-24",
]


# ── Inference output tests ──────────────────────────────────────────────

def test_da3_inference_output():
    """DA3 depth files exist, correct shape, float32, positive values."""
    for sid in SEQ_IDS:
        p = os.path.join(PHASE3_DIR, "da3", sid, "depth_da3.npy")
        assert os.path.exists(p), f"Missing: {p}"
        d = np.load(p)
        assert d.ndim == 3, f"{sid}: expected (S,H,W), got {d.shape}"
        assert d.dtype == np.float32, f"{sid}: expected float32, got {d.dtype}"
        assert d.min() > 0, f"{sid}: min depth must be > 0, got {d.min()}"
        assert d.max() < 100, f"{sid}: max depth must be < 100m, got {d.max()}"
    print("  PASS: da3 inference output")


def test_unidepth_inference_output():
    """UniDepth depth + intrinsics files exist, correct shapes."""
    for sid in SEQ_IDS:
        dp = os.path.join(PHASE3_DIR, "unidepth_v2", sid, "depth_unidepth.npy")
        assert os.path.exists(dp), f"Missing depth: {dp}"
        d = np.load(dp)
        assert d.ndim == 3, f"{sid}: expected (S,H,W), got {d.shape}"
        assert d.dtype == np.float32
        assert d.min() > 0

        ip = os.path.join(PHASE3_DIR, "unidepth_v2", sid, "intrinsics_unidepth.npy")
        assert os.path.exists(ip), f"Missing intrinsics: {ip}"
        intr = np.load(ip)
        assert intr.ndim == 3, f"{sid}: expected (S,3,3), got {intr.shape}"
        assert intr.shape[1:] == (3, 3)
    print("  PASS: unidepth inference output")


# ── Evaluation CSV tests ────────────────────────────────────────────────

def test_unified_evaluator():
    """DEPTH_MODEL_COMPARISON_FRAME.csv has correct columns, all 3 models × 4 seqs."""
    frame_csv = os.path.join(PHASE3_DIR, "evaluation", "DEPTH_MODEL_COMPARISON_FRAME.csv")
    assert os.path.exists(frame_csv)
    with open(frame_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "Frame CSV is empty"
    # Check columns
    expected_cols = {"model", "sequence_id", "frame_idx", "n_valid_pixels",
                     "raw_absrel", "raw_rmse", "scale_ratio", "aligned_absrel"}
    assert expected_cols.issubset(set(rows[0].keys())), f"Missing columns: {expected_cols - set(rows[0].keys())}"
    # Check models
    models_found = set(r["model"] for r in rows)
    assert models_found == {"vggt", "da3", "unidepth"}, f"Expected 3 models, got {models_found}"
    # Check sequences
    seqs_found = set(r["sequence_id"] for r in rows)
    for sid in SEQ_IDS:
        assert sid in seqs_found, f"Missing sequence: {sid}"
    print(f"  PASS: unified evaluator ({len(rows)} rows)")


def test_model_comparison_csv():
    """CSV values are numeric, no NaN in headline metrics."""
    seq_csv = os.path.join(PHASE3_DIR, "evaluation", "DEPTH_MODEL_COMPARISON_SEQ.csv")
    assert os.path.exists(seq_csv)
    with open(seq_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12, f"Expected 12 rows (3 models × 4 seqs), got {len(rows)}"
    for r in rows:
        for col in ["raw_absrel_mean", "raw_rmse_mean", "scale_mean", "scale_cv"]:
            val = float(r[col])
            assert not np.isnan(val), f"NaN in {r['model']}/{r['sequence_id']}/{col}"
            assert val >= 0, f"Negative value in {col}: {val}"
    print("  PASS: model comparison CSV")


# ── Scale anchor tests ──────────────────────────────────────────────────

def test_scale_anchor_no_gt():
    """Anchor values are computed from DA3/UniDepth, NOT from GT depth."""
    anchor_csv = os.path.join(PHASE3_DIR, "anchor", "SCALE_ANCHOR_VALUES.csv")
    assert os.path.exists(anchor_csv)
    with open(anchor_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "Anchor CSV is empty"
    # Check that anchors are between reasonable bounds (0.1 to 5.0)
    for r in rows:
        val = float(r["anchor_value"])
        assert 0.1 < val < 5.0, f"Anchor out of range: {val}"
    # Verify proxy models are da3/unidepth only
    proxies = set(r["proxy_model"] for r in rows)
    assert proxies == {"da3", "unidepth"}, f"Unexpected proxies: {proxies}"
    print(f"  PASS: scale anchor ({len(rows)} values, proxies={proxies})")


def test_anchored_metrics_exist():
    """ANCHORED_VGGT_METRICS.csv exists and has comparison data."""
    comp_csv = os.path.join(PHASE3_DIR, "anchor", "ANCHORED_VGGT_METRICS.csv")
    assert os.path.exists(comp_csv)
    with open(comp_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8, f"Expected 8 rows (2 proxies × 4 seqs), got {len(rows)}"
    for r in rows:
        assert float(r["anchored_absrel"]) > 0
        assert float(r["raw_absrel"]) > 0
    print(f"  PASS: anchored metrics ({len(rows)} rows)")


# ── Smoke test verification ─────────────────────────────────────────────

def test_smoke_test_results():
    """Smoke test results JSON exists with all models."""
    p = os.path.join(PHASE3_DIR, "smoke_test_results.json")
    assert os.path.exists(p)
    data = json.load(open(p))
    for model in ["vggt", "da3", "unidepth"]:
        assert model in data, f"Missing {model} in smoke test"
        assert data[model]["status"] == "PASS", f"{model} smoke test FAILED"
    print("  PASS: smoke test results")


# ── Figures exist ───────────────────────────────────────────────────────

def test_figures_exist():
    """Key figures were generated."""
    fig_dir = os.path.join(PHASE3_DIR, "figures")
    assert os.path.exists(fig_dir)
    expected = [
        "model_comparison_bar.png",
        "scale_ratio_distribution.png",
        "anchor_comparison.png",
    ]
    for name in expected:
        p = os.path.join(fig_dir, name)
        assert os.path.exists(p), f"Missing figure: {p}"
    print("  PASS: figures exist")


# ── Report exists ───────────────────────────────────────────────────────

def test_report_exists():
    """PHASE3A report exists and contains key sections."""
    p = os.path.join(PHASE3_DIR, "PHASE3A_METRIC_DEPTH_BENCHMARK.md")
    assert os.path.exists(p)
    content = open(p).read()
    assert "Q-A" in content
    assert "Q-B" in content
    assert "Q-C" in content
    assert "Route 3" in content or "Route3" in content
    print("  PASS: report exists")


def run_all():
    tests = [
        test_da3_inference_output,
        test_unidepth_inference_output,
        test_unified_evaluator,
        test_model_comparison_csv,
        test_scale_anchor_no_gt,
        test_anchored_metrics_exist,
        test_smoke_test_results,
        test_figures_exist,
        test_report_exists,
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
