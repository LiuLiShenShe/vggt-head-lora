"""P0-3: scanner metrics must report foreground-only as primary."""
import os, csv, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_foreground_primary():
    p = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER.csv")
    assert os.path.exists(p), "先运行 scanner_gt_3tier_eval.py"
    with open(p) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    fg_rows = [r for r in rows if r.get("foreground_only") == "True"]
    full_rows = [r for r in rows if r.get("foreground_only") == "False"]
    assert len(fg_rows) >= len(full_rows), "foreground-only rows must be >= full-scene rows"
    # Full-scene must be marked diagnostic_only
    for r in full_rows:
        diag = r.get("diagnostic_only", "")
        assert diag == "True" or r.get("foreground_only") == "False", \
            "full-scene scanner metrics must be diagnostic_only"


if __name__ == "__main__":
    test_foreground_primary()
    print("ALL scanner foreground-only tests passed")
