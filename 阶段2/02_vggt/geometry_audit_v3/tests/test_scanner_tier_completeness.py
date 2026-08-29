"""P0-2/P0-6 (v3.2.1): authoritative scanner manifest must point to V321, list 3 tiers, and the
deprecated (invalid) v3.2 artifacts must exist in quarantine. Also alignment_provenance.json must
correctly flag B as scanner-geometry-independent and C as upper-bound.
"""
import os, json, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_manifest_points_to_v321():
    m = os.path.join(ROOT, "SCANNER_GT_AUTHORITATIVE_MANIFEST.json")
    assert os.path.exists(m)
    d = json.load(open(m))
    assert d["evaluation_version"] == "v3.2.1"
    assert d["expected_tiers"] == ["A_raw", "B_refcam", "C_oracle"]
    # authoritative files exist
    for key in ("authoritative_csv", "authoritative_json"):
        assert os.path.exists(os.path.join(ROOT, d[key])), f"{key} missing"
    # hashes match on-disk
    import hashlib
    for key, attr in (("authoritative_csv", "csv_sha256"), ("authoritative_json", "json_sha256")):
        p = os.path.join(ROOT, d[key])
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h == d[attr], f"sha256 mismatch for {key}"


def test_deprecated_in_quarantine():
    assert os.path.exists(os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER_INVALID_v32.json"))
    assert os.path.exists(os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER_INVALID_v32.csv"))


def test_alignment_provenance_flags():
    p = os.path.join(ROOT, "scanner_gt", "alignment_provenance.json")
    assert os.path.exists(p)
    prov = json.load(open(p))
    t = prov["tiers"]
    assert t["B_refcam"]["uses_scanner_geometry_for_transform"] is False
    assert t["B_refcam"]["uses_reference_camera_pose"] is True
    assert t["C_oracle"]["uses_scanner_geometry_for_transform"] is True
    assert t["C_oracle"]["upper_bound"] is True
    assert t["A_raw"]["uses_scanner_geometry_for_transform"] is False


if __name__ == "__main__":
    test_manifest_points_to_v321()
    test_deprecated_in_quarantine()
    test_alignment_provenance_flags()
    print("ALL scanner-tier-completeness tests passed")
