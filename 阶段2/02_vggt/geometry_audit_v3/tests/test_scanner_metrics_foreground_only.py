"""P0-3/P0-7 (v3.2.1): scanner metrics must report foreground-only as PRIMARY, with all three
tiers (A_raw/B_refcam/C_oracle) having non-empty foreground rows. The v3.2 csv had empty B/C
foreground rows — this test now fails on that defect and passes only on the regenerated V321 csv."""
import os, csv, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def _load():
    p = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_3TIER_V321.csv")
    assert os.path.exists(p), "先运行 scanner_gt_3tier_eval_v321.py"
    with open(p) as f:
        return list(csv.DictReader(f))


def test_three_tiers_present():
    rows = _load()
    tiers = {r["tier"] for r in rows}
    assert tiers == {"A_raw", "B_refcam", "C_oracle"}, f"expected 3 tiers, got {tiers}"


def test_all_tiers_have_foreground_rows():
    rows = _load()
    fg = [r for r in rows if r.get("foreground_only") == "True"]
    assert len(fg) == 3, f"expected 3 foreground rows (one per tier), got {len(fg)}"
    for r in fg:
        n_pred = int(r["n_points_pred"])
        assert n_pred > 0, f"tier {r['tier']} foreground row has n_points_pred={n_pred} (empty — leak)"


def test_foreground_is_primary_not_diagnostic():
    rows = _load()
    fg_rows = [r for r in rows if r.get("foreground_only") == "True"]
    full_rows = [r for r in rows if r.get("foreground_only") == "False"]
    assert len(fg_rows) >= len(full_rows)
    # 主结论必须 reporting 在 foreground 行; full-scene 不得作为 headline (upper_bound 或 evaluation_only)
    for r in fg_rows:
        assert r["tier"] in {"A_raw", "B_refcam", "C_oracle"}


if __name__ == "__main__":
    test_three_tiers_present()
    test_all_tiers_have_foreground_rows()
    test_foreground_is_primary_not_diagnostic()
    print("ALL scanner foreground-only v321 tests passed")
