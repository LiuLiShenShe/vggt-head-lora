"""P1: four_path_v4 不再用 truncated_nn 作主判据.

断言: verdict_v4.json 的路徑字段中不含 nn_med / truncated 作主判定; 主判定为完整双向指标.
"""
import os
import json
import sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_v4_verdict_no_truncated():
    p = os.path.join(ROOT, "four_path_v4", "verdict_v4.json")
    assert os.path.exists(p), f"four_path_v4/verdict_v4.json 未生成 (先运行 four_path_v4.py)"
    v = json.load(open(p))
    flat = json.dumps(v)
    # verdict 中每路应含 fscore_10mm 等完整指标, 不得出现 nn_med 作主字段
    assert "nn_med" not in flat, "v4 verdict 不应出现 truncated nn_med"
    assert "truncated_nn" not in flat, "v4 verdict 不应出现 truncated_nn"
    # 每路必须含完整双向指标
    found_f = False
    for key, nblock in v.items():
        for n, block in nblock.items():
            if not isinstance(block, dict):
                continue  # 跳过 n_frames / depth_scale_source 等标量字段
            for route, metrics in block.items():
                if not isinstance(metrics, dict):
                    continue
                assert "fscore_10mm" in metrics, f"{key}/{n}/{route} 缺 fscore_10mm"
                assert "chamfer_symmetric_m" in metrics, f"{key}/{n}/{route} 缺 chamfer"
                found_f = True
    assert found_f


def test_v3_marked_not_sufficient():
    """four_path_v3 存档保留但标记 NOT SUFFICIENT (不删不改)."""
    p = os.path.join(ROOT, "four_path_v3", "metric_definitions.json")
    if os.path.exists(p):
        d = json.load(open(p))
        assert "diagnostic_only" in json.dumps(d)


if __name__ == "__main__":
    test_v4_verdict_no_truncated()
    test_v3_marked_not_sufficient()
    print("ALL four_path no-truncated verdict tests passed")
