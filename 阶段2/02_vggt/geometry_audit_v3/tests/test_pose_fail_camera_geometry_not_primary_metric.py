"""P2-1: pose FAIL must not generate 'primary geometry accuracy' from camera alignment."""
import os, json, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_pose_fail_not_primary():
    p = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER.json")
    assert os.path.exists(p), "先运行 scanner_gt_3tier_eval.py"
    d = json.load(open(p))
    for seq_key, tiers in d.items():
        if "fail" in seq_key.lower() or "12-03" in seq_key or "19-03" in seq_key:
            # For pose FAIL sequences, camera_sim3 must be flagged as diagnostic
            cam = tiers.get("camera_sim3", {})
            if cam:
                status = cam.get("camera_aligned_geometry_status", "")
                assert status in ("INVALID_POSE", "DIAGNOSTIC_ONLY", ""), \
                    f"pose FAIL seq camera_sim3 not flagged: {status}"


if __name__ == "__main__":
    test_pose_fail_not_primary()
    print("ALL pose fail camera geometry test passed")
