"""P0-7: depth metrics must separate plant-foreground and background/full-scene."""
import os, csv, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_plant_foreground_depth_metrics_exist():
    """FOREGROUND_METRICS_V31.csv must exist with depth-aware results."""
    p = os.path.join(ROOT, "FOREGROUND_METRICS_V31.csv")
    assert os.path.exists(p)
    with open(p) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    fg_rows = [r for r in rows if r.get("result_set", "").startswith("plant_foreground")]
    assert len(fg_rows) >= 1, "no foreground depth rows"


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
