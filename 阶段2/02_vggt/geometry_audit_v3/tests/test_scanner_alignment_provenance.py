"""P0-1/P0-6 (v3.2.1): scanner alignment must NOT use GT point geometry for camera-based tier.

Points to authoritative alignment_provenance.json (which encodes per-tier transform provenance)
and scanner_gt/SCANNER_GT_3TIER_V321.json (which has the three-tier structure).
"""
import os, json, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_camera_tier_no_gt_geometry():
    """alignment_provenance.json: B_refcam must be scanner-geometry-free."""
    p = os.path.join(ROOT, "scanner_gt", "alignment_provenance.json")
    assert os.path.exists(p), "先运行 scanner_gt_3tier_eval_v321.py"
    d = json.load(open(p))
    b = d["tiers"]["B_refcam"]
    assert b["uses_scanner_geometry_for_transform"] is False, \
        f"B_refcam must NOT use scanner GT geometry for transform"
    assert b["uses_reference_camera_pose"] is True
    upper = b.get("upper_bound")
    assert upper in (False, None), \
        f"B_refcam must not be an upper_bound, got {upper}"


def test_oracle_tier_flagged():
    """alignment_provenance.json: C_oracle must use scanner GT and be upper_bound."""
    p = os.path.join(ROOT, "scanner_gt", "alignment_provenance.json")
    d = json.load(open(p))
    c = d["tiers"]["C_oracle"]
    assert c["uses_scanner_geometry_for_transform"] is True
    assert c["upper_bound"] is True


def test_deprecated_json_quarantined():
    """Old invalid json is in quarantine, not at original path."""
    old = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER.json")
    quarantine = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER_INVALID_v32.json")
    assert not os.path.exists(old), "SCANNER_GT_3TIER.json should have been quarantined"
    assert os.path.exists(quarantine), "quarantine file missing"


if __name__ == "__main__":
    test_camera_tier_no_gt_geometry()
    test_oracle_tier_flagged()
    test_deprecated_json_quarantined()
    print("ALL scanner alignment provenance tests passed")
