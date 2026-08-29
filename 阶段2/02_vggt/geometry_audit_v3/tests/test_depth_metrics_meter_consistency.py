"""P0-6: raw 与 scale-aligned 深度指标单位一致 (均为米).

断言: 修正单位后 raw AbsRel 为真实米制误差 (非伪 1.0);
raw 与 aligned 指标同属米制, 不混用单位.
"""
import os
import sys
import json
import numpy as np

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE = "/fj/VGGT+head+lora实验/阶段2"


def test_depth_metrics_meter_consistency():
    import depth_audit_v3 as da
    seq = os.path.join(BASE, "01_sequences/sequences/plant_view/langdon_4__05-03-24.json")
    dd = np.load(os.path.join(BASE, "02_vggt", "v2_clean_rerun/plant_view_3d/plantview__langdon_4__05-03-24/depth_vggt.npy"))
    rps = json.load(open(seq))["rgb_paths"]
    res = da.depth_audit_sequence(seq, dd, rps, max_frames=10)
    agg = res["aggregate"]
    raw = agg["mean_raw_absrel"]
    aln = agg["mean_aligned_absrel"]
    # 修正单位后: raw 应为真实米制相对误差 (< 0.6, 不再是伪 1.0)
    assert raw < 0.6, f"raw AbsRel={raw} (单位未修正? 期望 <0.6)"
    assert aln < 0.6, f"aligned AbsRel={aln} 异常"
    # 两者同量级米制 (差异来自尺度缩放, 不应差 1000x)
    assert abs(np.log(raw) - np.log(aln)) < 2.0, "raw 与 aligned 单位不一致"
    # scale 因子应接近 1 (VGGT 米制), 不是 ~1000 或 ~0.001
    assert 0.5 < agg["mean_median_scale"] < 3.0, f"median_scale={agg['mean_median_scale']} (VGGT 应为米制, scale≈1)"


if __name__ == "__main__":
    test_depth_metrics_meter_consistency()
    print("ALL depth metric meter-consistency tests passed")
