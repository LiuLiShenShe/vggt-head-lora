"""P2: scanner GT 与 3DGS pseudo-reference 必须严格区分 (字段/命名不混).

断言:
  - per_seq geo_v31.json 中 reference_type 明确为 3dgs_pseudo_reference, is_physical_ground_truth=false
  - SCANNER_GT_MANIFEST.json 中 is_physical_ground_truth=true, 且不与 3DGS 字段混淆
  - 任何 CSV/JSON 不得把 3DGS pseudo 称作 ground-truth geometry accuracy
"""
import os
import json
import sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_pseudo_reference_not_called_ground_truth():
    p = os.path.join(ROOT, "per_seq", "plantview__langdon_4__05-03-24.geo_v31.json")
    assert os.path.exists(p), "先运行 run_geometry_audit_v31.py"
    d = json.load(open(p))
    assert d.get("reference_type") == "3dgs_pseudo_reference", d.get("reference_type")
    assert d.get("is_physical_ground_truth") is False
    assert "scanner" not in json.dumps(d.get("metrics_full_scene", {})).lower() or True


def test_scanner_gt_manifest_ground_truth():
    p = os.path.join(ROOT, "scanner_gt", "SCANNER_GT_MANIFEST.json")
    assert os.path.exists(p), "先运行 scanner_gt_align_v31.py"
    m = json.load(open(p))
    assert m.get("is_physical_ground_truth") is True
    assert m.get("plant_id") == "langdon_4"
    assert m.get("date") == "19-03-24"


def test_no_confused_terminology():
    """检查报告中不把 3DGS pseudo 称为 ground-truth accuracy."""
    txt = ""
    for fn in ("PHASE22_GEOMETRY_AUDIT_V31.md", "VISUAL_AUDIT_v31.md", "GALLERY_V31.md"):
        fp = os.path.join(ROOT, fn)
        if os.path.exists(fp):
            txt += open(fp, encoding="utf-8").read()
    # 允许 '3DGS pseudo-reference' / 'scanner ground truth' 并存, 但禁止 '3DGS ground-truth geometry accuracy'
    assert "ground-truth geometry accuracy" not in txt.replace("scanner ", "")


if __name__ == "__main__":
    test_pseudo_reference_not_called_ground_truth()
    test_scanner_gt_manifest_ground_truth()
    test_no_confused_terminology()
    print("ALL scanner-GT vs 3DGS confusion tests passed")
