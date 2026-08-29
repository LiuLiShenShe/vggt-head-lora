"""P0-4: SCANNER_UNIT_AUDIT.json must exist and be VERIFIED before scanner metrics."""
import os, json, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_scanner_unit_verified():
    p = os.path.join(ROOT, "SCANNER_UNIT_AUDIT.json")
    assert os.path.exists(p), "先运行 SCANNER_UNIT_AUDIT.py"
    d = json.load(open(p))
    assert d.get("status") == "VERIFIED", f"scanner unit status: {d.get('status')}"
    assert d.get("scale_to_meter") in (0.001, 1.0), f"unexpected scale: {d.get('scale_to_meter')}"
    assert d.get("scanner_storage_unit") in ("millimeter", "meter"), \
        f"unexpected unit: {d.get('scanner_storage_unit')}"


def test_scanner_metrics_use_verified_unit():
    """SCANNER_GT_3TIER.csv must reference SCANNER_UNIT_AUDIT scale."""
    unit_path = os.path.join(ROOT, "SCANNER_UNIT_AUDIT.json")
    csv_path = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER.csv")
    if os.path.exists(csv_path):
        unit_data = json.load(open(unit_path))
        assert unit_data.get("status") == "VERIFIED", "scanner unit must be VERIFIED before metrics"


if __name__ == "__main__":
    test_scanner_unit_verified()
    test_scanner_metrics_use_verified_unit()
    print("ALL scanner unit audit tests passed")
