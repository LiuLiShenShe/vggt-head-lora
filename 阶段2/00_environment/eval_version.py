"""评价版本字段强制 helper(P0-7)。

原则:正式 Gate/汇总/报告/后续脚本只能读 v2。
get_active_pose_eval() 在 meta 缺 active_evaluation_version 或版本非 v2 时直接 raise,
绝不静默回落 pose_eval_v1。

自测: python eval_version.py  (构造含 v1 字段但无 active 版本的 meta,断言 raise)
"""
import json


def get_active_pose_eval(meta: dict) -> dict:
    """返回当前生效的位姿评价结果(仅 v2)。任何不合规状态都 raise。"""
    if "active_evaluation_version" not in meta:
        raise ValueError(
            f"meta 缺少 active_evaluation_version 字段;含旧字段 "
            f"{[k for k in ('pose_eval', 'pose_eval_v1') if k in meta]}。"
            "请先运行 enforce_eval_version.py 迁移。禁止静默读取旧评价。")
    if meta["active_evaluation_version"] != "v2":
        raise ValueError(f"active_evaluation_version={meta['active_evaluation_version']!r},仅支持 v2")
    pe = meta.get("gate_source") and meta.get(meta["gate_source"])
    if meta.get("gate_source") != "pose_eval_v2":
        raise ValueError(f"gate_source={meta.get('gate_source')!r},必须为 'pose_eval_v2'")
    if not isinstance(pe, dict):
        raise ValueError("pose_eval_v2 缺失或类型错误")
    return pe


def apply_version_fields(meta: dict, pose_eval_v1):
    """在 meta 上写入版本治理字段(pose_eval 已由调用方改名为 pose_eval_v1 或置 None)。"""
    meta["active_evaluation_version"] = "v2"
    meta["pose_eval_v1"] = pose_eval_v1
    meta["gate_source"] = "pose_eval_v2"
    meta["deprecated_fields"] = ["pose_eval_v1"]
    return meta


def _selftest():
    # 1) 旧格式(v1 字段、无 active 版本)必须 raise
    m = {"pose_eval": {"rotation_error_deg": {"median": 99}}}
    try:
        get_active_pose_eval(m)
        raise AssertionError("应当 raise")
    except ValueError:
        pass
    # 2) 正确迁移后可读
    m2 = apply_version_fields({"pose_eval_v2": {"ok": 1}}, {"old": True})
    assert get_active_pose_eval(m2) == {"ok": 1}
    # 3) gate_source 错误必须 raise
    m3 = apply_version_fields({"pose_eval_v2": {}}, None)
    m3["gate_source"] = "pose_eval_v1"
    try:
        get_active_pose_eval(m3)
        raise AssertionError("应当 raise")
    except ValueError:
        pass
    print("eval_version selftest OK")


if __name__ == "__main__":
    _selftest()
