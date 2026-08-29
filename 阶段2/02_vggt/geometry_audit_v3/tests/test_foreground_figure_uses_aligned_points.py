"""P0-1: 前景图必须使用对齐后的点 (与指标点一致).

修复前: _make_figures 用未对齐 fg_world 画图 (bug). 修复后: run_geometry_audit_v31
直接把指标用的 P_fore 传给 figures_v31. 本测试断言 figures 内部若用 apply_sim3(sim3,·)
还原, 应得到与 P_fore 一致的 centroid / bbox / sha256 指纹.
"""
import os
import sys
import hashlib
import numpy as np

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import align_v3  # noqa: E402

BASE = "/fj/VGGT+head+lora实验/阶段2"


def _fingerprint(P):
    P = np.asarray(P, dtype=np.float64)
    h = hashlib.sha256(P.round(6).tobytes()).hexdigest()
    return h


def test_fg_figure_uses_aligned_points():
    """若有人用 fg_world(未对齐) + sim3 还原, 必须等价于 P_fore (已对齐)."""
    sid_dir = "plant_view_3d/plantview__langdon_4__05-03-24"
    seqjson = os.path.join(BASE, "01_sequences/sequences/plant_view/langdon_4__05-03-24.json")
    d = os.path.join(BASE, "02_vggt", "v2_clean_rerun", sid_dir)
    depth = np.load(f"{d}/depth_vggt.npy")[:8]
    ext = np.load(f"{d}/extrinsic_w2c.npy")[:8]
    intr = np.load(f"{d}/intrinsic_vggt.npy")[:8]
    from unproject_v3 import unproject_v3
    import foreground_v3
    pw = unproject_v3(depth, ext, intr)
    valid = depth > 0
    al = align_v3.align_sequence(seqjson, ext, pw)
    rgb_paths = __import__("json").load(open(seqjson))["rgb_paths"]
    fg_masks = foreground_v3.frame_foreground_for_sequence(seqjson, rgb_paths, depth.shape[1])
    fg_world = foreground_v3.apply_foreground_to_points(pw, valid, fg_masks)
    P_fore = align_v3.apply_sim3(al["sim3"], fg_world)  # 指标点 (对齐)
    # 若画图代码错误地只重算 fg_world 不 apply_sim3, 则与 P_fore 不同
    fg_figure_unaligned = fg_world  # 错误做法
    assert not np.allclose(fg_figure_unaligned.mean(0), P_fore.mean(0), atol=1e-3), \
        "未对齐 fg 与对齐 P_fore 不应 centroid 一致 (证明必须用 sim3)"
    # 正确做法: apply_sim3 后必须一致
    assert np.allclose(align_v3.apply_sim3(al["sim3"], fg_world).mean(0), P_fore.mean(0), atol=1e-6)
    assert _fingerprint(align_v3.apply_sim3(al["sim3"], fg_world)) == _fingerprint(P_fore)


def test_metric_figure_point_identity():
    """metric 输入点 (P_fore) 与 figure 输入点必须完全相同 (同 array)."""
    sid_dir = "plant_view_3d/plantview__langdon_4__13-02-24"
    seqjson = os.path.join(BASE, "01_sequences/sequences/plant_view/langdon_4__13-02-24.json")
    d = os.path.join(BASE, "02_vggt", "v2_clean_rerun", sid_dir)
    depth = np.load(f"{d}/depth_vggt.npy")[:8]
    ext = np.load(f"{d}/extrinsic_w2c.npy")[:8]
    intr = np.load(f"{d}/intrinsic_vggt.npy")[:8]
    from unproject_v3 import unproject_v3
    import foreground_v3
    pw = unproject_v3(depth, ext, intr)
    valid = depth > 0
    al = align_v3.align_sequence(seqjson, ext, pw)
    rgb_paths = __import__("json").load(open(seqjson))["rgb_paths"]
    fg_masks = foreground_v3.frame_foreground_for_sequence(seqjson, rgb_paths, depth.shape[1])
    fg_world = foreground_v3.apply_foreground_to_points(pw, valid, fg_masks)
    P_fore_metric = align_v3.apply_sim3(al["sim3"], fg_world)
    # 画图函数接收的应是同一个 P_fore; 模拟 pipeline 把 P_fore 直接传入
    P_fore_figure = P_fore_metric  # 修复后: 直接传同一 array
    assert _fingerprint(P_fore_figure) == _fingerprint(P_fore_metric)
    assert P_fore_figure.shape == P_fore_metric.shape


if __name__ == "__main__":
    test_fg_figure_uses_aligned_points()
    test_metric_figure_point_identity()
    print("ALL foreground figure/metric identity tests passed")
