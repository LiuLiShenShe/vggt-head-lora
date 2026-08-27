"""阶段 2.2 Geometry Audit v3 — 不引用 v1 字段 / 不出现 '几何 14/17 通过' 措辞."""
import csv
import json
import os

BASE = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"


def _read_table():
    rows = []
    with open(os.path.join(BASE, "GEOMETRY_AUDIT_TABLE.csv")) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def test_no_v1_fields():
    """CSV/JSON 不出现 v1 字段 (pose_eval_v1, gate_source 等)."""
    rows = _read_table()
    for row in rows:
        for k in row:
            assert "pose_eval_v1" not in k.lower(), f"列 {k} 含 pose_eval_v1"
            assert "deprecated" not in k.lower() or k == "truncated_inlier_diagnostic_only", f"列 {k} 含 v1"
    for sid in ["plantview__langdon_4__05-03-24", "plantview__langdon_4__12-03-24"]:
        j = json.load(open(f"{BASE}/per_seq/{sid}.geo_v3.json"))
        assert "pose_eval_v1" not in json.dumps(j)
        for k in j:
            assert "pose_eval_v1" not in k


def test_no_geometry_gate_pass_label():
    """CSV/JSON/MD 不出现 '几何 14/17 通过' / 'geometry gate 14/17 pass' 措辞."""
    forbidden = ["几何质量 14/17", "几何 14/17 通过", "geometry gate 14/17", "14/17 pass"]
    rows = _read_table()
    text = json.dumps(rows, ensure_ascii=False).lower()
    for phrase in forbidden:
        assert phrase.lower() not in text, f"CSV 含禁止措辞 '{phrase}'"
    for md in ["VISUAL_AUDIT.md", "PHASE22_GEOMETRY_AUDIT_V3.md", "GALLERY.md"]:
        p = os.path.join(BASE, md)
        if os.path.exists(p):
            t = open(p).read().lower()
            for phrase in forbidden:
                assert phrase.lower() not in t, f"{md} 含禁止措辞 '{phrase}'"


def test_truncated_deprecation_flag():
    """CSV/JSON 有 truncated_inlier_diagnostic_only=true 列."""
    rows = _read_table()
    assert any("truncated_inlier_diagnostic_only" in k for k in rows[0].keys())
    for row in rows:
        val = row.get("truncated_inlier_diagnostic_only", "")
        assert val in ("True", "true", "1")


def test_geometry_gate_not_yet():
    """CSV geometry_gate 均为 not_yet_established (不反向设计通过率)."""
    rows = _read_table()
    for row in rows:
        assert row["geometry_gate"] in ("not_yet_established", ""), row["geometry_gate"]


if __name__ == "__main__":
    test_no_v1_fields()
    test_no_geometry_gate_pass_label()
    test_truncated_deprecation_flag()
    test_geometry_gate_not_yet()
    print("ALL v1-field tests passed")
