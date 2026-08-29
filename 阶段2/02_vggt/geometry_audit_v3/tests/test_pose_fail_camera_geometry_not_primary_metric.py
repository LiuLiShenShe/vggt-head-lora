"""P2-1: pose FAIL must not generate 'primary geometry accuracy' from camera alignment."""
import os, json, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_pose_fail_not_primary():
    """pose-FAIL sequences: B_refcam geometry must be marked INVALID_POSE or equivalent, not primary.

    Uses the V321 scanner csv to confirm 19-03-24 (pose-FAIL) is not used as headline evidence.
    """
    p = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER_V321.csv")
    assert os.path.exists(p), "先运行 scanner_gt_3tier_eval_v321.py"
    import csv
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "V321 csv is empty"
    # all rows must be for 19-03-24 (only scanner-GT plant locally)
    for r in rows:
        assert r["date"] == "19-03-24"
        # B_refcam on pose-FAIL must NOT be used as primary geometry accuracy
        if r["tier"] == "B_refcam" and r["foreground_only"] == "True":
            # F@50mm must be 0 (honest, not fabricated)
            f50 = float(r["fscore_50mm"])
            assert f50 == 0.0, f"B_refcam fg on pose-FAIL should report F@50mm=0 (got {f50})"


if __name__ == "__main__":
    test_pose_fail_not_primary()
    print("ALL pose fail camera geometry test passed")
