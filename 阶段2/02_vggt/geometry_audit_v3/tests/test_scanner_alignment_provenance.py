"""P0-1: scanner alignment must NOT use GT point geometry for camera-based tier."""
import os, json, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_camera_tier_no_gt_geometry():
    p = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER.json")
    assert os.path.exists(p), "先运行 scanner_gt_3tier_eval.py"
    d = json.load(open(p))
    for seq_key, tiers in d.items():
        for tier_name, tier_data in tiers.items():
            if tier_name == "camera_sim3":
                assert tier_data.get("uses_test_reference_geometry") is False, \
                    f"camera_sim3 must not use GT geometry: {tier_data.get('uses_test_reference_geometry')}"
                assert tier_data.get("uses_test_reference_pose") is True
                assert tier_data.get("evaluation_only") is True


def test_oracle_tier_flagged():
    p = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER.json")
    d = json.load(open(p))
    for seq_key, tiers in d.items():
        for tier_name, tier_data in tiers.items():
            if tier_name == "oracle_geometry":
                assert tier_data.get("uses_test_reference_geometry") is True
                assert tier_data.get("upper_bound") is True
                assert tier_data.get("evaluation_only") is True


if __name__ == "__main__":
    test_camera_tier_no_gt_geometry()
    test_oracle_tier_flagged()
    print("ALL scanner alignment provenance tests passed")
