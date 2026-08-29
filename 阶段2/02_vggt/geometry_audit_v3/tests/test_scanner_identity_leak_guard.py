"""P0-10 (v3.2.1): scanner evaluator must guard against prediction/GT identity leakage.

The v3.2 json had F=1.0/Chamfer=0 at all tiers (pred array == GT array). We assert the regenerated
V321 csv contains NO row with the impossible signature, and that the manifest records the guard as passed.
"""
import os, csv, json, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_v321_has_no_identity_leak():
    p = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER_V321.csv")
    assert os.path.exists(p), "先运行 scanner_gt_3tier_eval_v321.py"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    for r in rows:
        f5 = float(r["fscore_5mm"]); f10 = float(r["fscore_10mm"])
        f20 = float(r["fscore_20mm"]); f50 = float(r["fscore_50mm"])
        cham = float(r["chamfer_sym_m"])
        npred = int(r["n_points_pred"]); ngt = int(r["n_points_gt"])
        # 任意 tier 不得出现 identity-leak 特征
        leak = (npred == ngt and cham == 0.0 and f5 == 1.0 and f10 == 1.0 and f20 == 1.0 and f50 == 1.0)
        assert not leak, f"identity leak in tier {r['tier']} fg={r['foreground_only']}"


def test_manifest_records_guard_passed():
    m = os.path.join(ROOT, "SCANNER_GT_AUTHORITATIVE_MANIFEST.json")
    assert os.path.exists(m)
    d = json.load(open(m))
    assert d.get("identity_leak_guard") == "passed"
    assert d.get("all_tiers_have_foreground_rows") is True


if __name__ == "__main__":
    test_v321_has_no_identity_leak()
    test_manifest_records_guard_passed()
    print("ALL identity-leak-guard tests passed")
