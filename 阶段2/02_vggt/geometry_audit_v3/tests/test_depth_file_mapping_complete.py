"""P0-8: 12-03-24 depth file mapping must be audited."""
import os, csv, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_depth_file_mapping_csv_exists():
    p = os.path.join(ROOT, "DEPTH_FILE_MAPPING_AUDIT.csv")
    assert os.path.exists(p), "先运行 DEPTH_VALIDITY_AUDIT.py (generates DEPTH_FILE_MAPPING_AUDIT.csv)"
    with open(p) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 4, f"expected >=4 sequences, got {len(rows)}"
    # 12-03-24 must be present
    r12 = [r for r in rows if "12-03-24" in r.get("sequence_id", "")]
    assert len(r12) == 1, "12-03-24 missing from depth file mapping audit"
    # Must not say "because pose failed" for depth availability
    status = r12[0].get("mapping_status", "")
    assert "pose" not in status.lower() or "mapping" in status.lower(), \
        f"12-03-24 depth status incorrectly references pose: {status}"


if __name__ == "__main__":
    test_depth_file_mapping_csv_exists()
    print("ALL depth file mapping audit tests passed")
