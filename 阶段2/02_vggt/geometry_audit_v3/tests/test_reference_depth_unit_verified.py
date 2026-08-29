"""P0-5: 参考深度单位必须经审计为 VERIFIED.

断言 DEPTH_UNIT_AUDIT.json 存在且 status==VERIFIED, depth_scale_to_meter==0.001.
"""
import os
import json
import sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_depth_unit_verified():
    p = os.path.join(ROOT, "DEPTH_UNIT_AUDIT.json")
    assert os.path.exists(p), "DEPTH_UNIT_AUDIT.json 未生成"
    a = json.load(open(p))
    assert a.get("status") == "VERIFIED", f"status={a.get('status')} (必须 VERIFIED 才报 raw metric depth)"
    assert abs(a.get("depth_scale_to_meter", 0) - 0.001) < 1e-9, \
        f"depth_scale_to_meter={a.get('depth_scale_to_meter')} (必须 0.001: uint16 毫米)"
    # sanity check 必须匹配 (参考中位深度与相机距离同量级)
    sc = a.get("camera_distance_sanity_check", {})
    assert sc.get("match") is True, f"camera distance sanity check 失败: {sc}"


def test_depth_audit_applies_scale():
    """depth_audit_v3 必须把参考深度乘 0.001 而非当作米."""
    import depth_audit_v3 as da
    assert abs(da.DEPTH_SCALE_TO_METER - 0.001) < 1e-9


if __name__ == "__main__":
    test_depth_unit_verified()
    test_depth_audit_applies_scale()
    print("ALL depth unit audit tests passed")
