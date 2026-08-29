"""P2 (v3.2.1): real foreground-depth evaluator must produce per-frame + per-sequence metrics for
the 4 plant_view sequences (incl 12-03-24) with both RAW and SCALE-ALIGNED columns. This replaces
the vacuous test_depth_foreground_metrics.py which only checked FOREGROUND_METRICS_V31.csv (geometry).
"""
import os, csv, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)

EXPECTED = ["plantview__langdon_4__05-03-24",
            "plantview__langdon_4__12-03-24",
            "plantview__langdon_4__13-02-24",
            "plantview__langdon_4__20-02-24"]


def test_depth_foreground_metrics_real():
    p = os.path.join(ROOT, "DEPTH_FOREGROUND_METRICS.csv")
    assert os.path.exists(p), "先运行 depth_foreground_eval_v321.py"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "DEPTH_FOREGROUND_METRICS.csv is empty (vacuous)"
    # columns present
    cols = rows[0].keys()
    for c in ("raw_absrel", "raw_rmse", "aligned_absrel", "aligned_rmse"):
        assert c in cols, f"missing column {c}"
    # all 4 sequences present
    seqs = {r["sequence_id"] for r in rows}
    for s in EXPECTED:
        assert s in seqs, f"sequence {s} absent from depth foreground metrics"


def test_depth_foreground_summary_real():
    p = os.path.join(ROOT, "DEPTH_FOREGROUND_SUMMARY.csv")
    assert os.path.exists(p)
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 4, f"expected 4 summary rows, got {len(rows)}"
    for r in rows:
        assert float(r["raw_absrel_mean"]) > 0, "raw_absrel must be > 0"
        assert float(r["aligned_absrel_mean"]) > 0, "aligned_absrel must be > 0"


if __name__ == "__main__":
    test_depth_foreground_metrics_real()
    test_depth_foreground_summary_real()
    print("ALL depth-evaluator-real tests passed")
