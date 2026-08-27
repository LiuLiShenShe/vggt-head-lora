"""P0-1 / P0-2: unproject_v3 ≡ 官方 VGGT 反投影 (vggt.utils.geometry.unproject_depth_map_to_point_map).

本测试为阶段 2.2 几何审计的 P0 门禁: 若 unproject_v3 与官方不一致, 审计管线不可信, 必须阻断。

运行: pytest geometry_audit_v3/tests/test_unprojection_vs_official.py
环境: da3 (torch 2.3.1, 可 import vggt)
"""
import os
import sys
import numpy as np
import pytest

# 让官方 VGGT 可 import
VGGT_ROOT = "/fj/VGGT+head+lora实验/vggt"
ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
for p in (VGGT_ROOT, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from unproject_v3 import unproject_v3, unproject_np_legacy, camera_centers  # noqa: E402

ATOL = 1e-3   # float32 量级的对照容差
RTOL = 1e-3


def _rand_se3(rng, n=4):
    """生成 n 个随机但合法的 SE3 w2c (R 正交, t 小)."""
    ext = np.zeros((n, 3, 4))
    for i in range(n):
        A = rng.standard_normal((3, 3))
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        ext[i, :3, :3] = R
        ext[i, :3, 3] = rng.standard_normal(3) * 0.5
    return ext


def _intr(n=4, H=16, W=16):
    K = np.zeros((n, 3, 3))
    for i in range(n):
        f = 200.0 + 50 * i
        K[i] = np.diag([f, f, 1.0])
        K[i, 0, 2] = W / 2.0
        K[i, 1, 2] = H / 2.0
    return K


def test_identity_camera():
    """identity w2c => world == cam coords (z=depth)."""
    n, H, W = 1, 16, 16
    rng = np.random.default_rng(0)
    depth = (rng.random((n, H, W)).astype(np.float32) + 0.5)
    ext = np.tile(np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=np.float32), (n, 1, 1))
    K = _intr(n, H, W)
    mine = unproject_v3(depth, ext, K)
    off = unproject_depth_map_to_point_map(depth[..., None], ext, K)
    assert mine.shape == off.shape == (n, H, W, 3)
    assert np.allclose(mine, off, atol=ATOL, rtol=RTOL)


def test_random_se3_matches_official():
    """随机 SE3 下 unproject_v3 必须 ≡ 官方 (批量)."""
    n, H, W = 6, 24, 32
    rng = np.random.default_rng(1)
    depth = (rng.random((n, H, W)).astype(np.float32) * 3 + 0.2)
    ext = _rand_se3(rng, n).astype(np.float32)
    K = _intr(n, H, W)
    mine = unproject_v3(depth, ext, K)
    off = unproject_depth_map_to_point_map(depth[..., None], ext, K)
    max_diff = float(np.max(np.abs(mine.astype(np.float64) - off.astype(np.float64))))
    assert max_diff < ATOL, f"unproject_v3 与官方最大差 {max_diff} >= atol {ATOL}"


def test_real_clean_rerun_data():
    """真实 clean-rerun 数据 (05-03-24, 12-03-24) 上 unproject_v3 ≡ 官方."""
    base = "/fj/VGGT+head+lora实验/阶段2/02_vggt"
    for sid in ["plantview__langdon_4__05-03-24", "plantview__langdon_4__12-03-24"]:
        d = np.load(f"{base}/v2_clean_rerun/plant_view_3d/{sid}/depth_vggt.npy")
        e = np.load(f"{base}/v2_clean_rerun/plant_view_3d/{sid}/extrinsic_w2c.npy")
        i = np.load(f"{base}/v2_clean_rerun/plant_view_3d/{sid}/intrinsic_vggt.npy")
        # 注: 必须用全分辨率 (518x518) 比较 —— 降采样深度但保留全分辨率内参会破坏
        # 像素<->内参映射, 导致对照失败 (这是测试陷阱, 不是代码 bug). 仅取前 2 帧全分辨率.
        d2, e2, i2 = d[:2], e[:2], i[:2]
        mine = unproject_v3(d2, e2, i2)
        off = unproject_depth_map_to_point_map(d2[..., None], e2, i2)
        max_diff = float(np.max(np.abs(mine.astype(np.float64) - off.astype(np.float64))))
        assert max_diff < ATOL, f"{sid}: unproject_v3 vs 官方 max diff {max_diff} >= {ATOL}"
        # 同时验证 unproject_v3 与已落盘的主推理 point_map_unprojected 一致 (全分辨率)
        pm = np.load(f"{base}/v2_clean_rerun/plant_view_3d/{sid}/point_map_unprojected.npy")
        pm2 = pm[:2]
        assert np.allclose(mine, pm2, atol=ATOL, rtol=RTOL), \
            f"{sid}: 与 point_map_unprojected 不一致 (max diff {float(np.abs(mine-pm2.astype(np.float64)).max())})"


def test_legacy_is_different_from_correct():
    """遗留 unproject_np_legacy (带 bug) 在随机 SE3 下必须明显偏离正确结果
    —— 证明旧 four_path_v2 确实用了错误反投影."""
    n, H, W = 4, 20, 20
    rng = np.random.default_rng(3)
    depth = (rng.random((n, H, W)).astype(np.float32) * 3 + 0.2)
    ext = _rand_se3(rng, n).astype(np.float32)
    K = _intr(n, H, W)
    correct = unproject_v3(depth, ext, K).astype(np.float64)
    legacy = unproject_np_legacy(depth, ext, K).astype(np.float64)
    diff = np.max(np.abs(correct - legacy))
    # 随机 SE3 下平移项偏差 ~ camera distance 量级, 应远大于数值容差
    assert diff > 1e-2, f"遗留版偏差 {diff} 太小, 未能证明 bug 存在"


if __name__ == "__main__":
    test_identity_camera()
    test_random_se3_matches_official()
    test_real_clean_rerun_data()
    test_legacy_is_different_from_correct()
    print("ALL P0 unprojection tests passed")
