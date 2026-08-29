"""P0-7: depth metrics must separate plant-foreground and background/full-scene."""
import os, csv, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_plant_foreground_depth_metrics_exist():
    """DEPTH_FOREGROUND_METRICS.csv (real foreground-depth evaluator, P2) must exist with depth columns.

    NOTE: FOREGROUND_METRICS_V31.csv is GEOMETRY (Chamfer/F@mm), NOT depth — it must not be used as
    depth proof. The real depth evidence is DEPTH_FOREGROUND_METRICS.csv.
    """
    p = os.path.join(ROOT, "DEPTH_FOREGROUND_METRICS.csv")
    assert os.path.exists(p), "先运行 depth_foreground_eval_v321.py (real depth evaluator)"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "DEPTH_FOREGROUND_METRICS.csv empty"
    assert "raw_absrel" in rows[0] and "aligned_absrel" in rows[0], "missing depth columns"


def test_depth_validity_audit_exists():
    p = os.path.join(ROOT, "DEPTH_VALIDITY_AUDIT.json")
    assert os.path.exists(p), "先运行 DEPTH_VALIDITY_AUDIT.py"
    d = __import__("json").load(open(p))
    assert "sequences" in d
    for seq, info in d["sequences"].items():
        assert "depth_range_m" in info, f"{seq} missing depth_range_m"
        assert "bin_proportions_mean" in info, f"{seq} missing bin_proportions"


if __name__ == "__main__":
    test_plant_foreground_depth_metrics_exist()
    test_depth_validity_audit_exists()
    print("ALL depth foreground metrics tests passed")
