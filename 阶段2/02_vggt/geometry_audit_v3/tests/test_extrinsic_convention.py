"""P0-1: 外部参数约定 (OpenCV w2c) 的单元测试.

验证:
  1. 相机中心 C = -R_w2c.T @ t_w2c
  2. closed_form_inverse_se3 逆成立: 对 w2c 求逆得到 c2w, 再逆回去是自身
  3. 正确反投影的平移项应为 -R.T @ t (而非 +t)
"""
import sys
import numpy as np
import pytest

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from unproject_v3 import unproject_v3, unproject_np_legacy, camera_centers  # noqa: E402


def _rand_se3(rng, n=5):
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


def test_camera_centers_convention():
    """C = -R.T @ t (OpenCV w2c 世界系相机中心)."""
    rng = np.random.default_rng(0)
    ext = _rand_se3(rng, 6)
    R, t = ext[:, :3, :3], ext[:, :3, 3]
    C = -np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), t)
    assert np.allclose(C, camera_centers(ext), atol=1e-9)


def test_inverse_se3_roundtrip():
    """w2c -> c2w -> w2c 往返一致 (closed_form_inverse_se3 等价)."""
    rng = np.random.default_rng(1)
    ext = _rand_se3(rng, 4)
    R, t = ext[:, :3, :3], ext[:, :3, 3]
    # c2w: R_c2w=R.T, t_c2w=-R.T@t
    R_c2w = np.transpose(R, (0, 2, 1))
    t_c2w = -np.einsum("sij,sj->si", R_c2w, t)
    # 再逆回 w2c
    R_back = np.transpose(R_c2w, (0, 2, 1))
    t_back = -np.einsum("sij,sj->si", R_back, t_c2w)
    assert np.allclose(R_back, R, atol=1e-9)
    assert np.allclose(t_back, t, atol=1e-9)


def test_correct_translation_term():
    """正确反投影的 world 平移项必须是 -R.T @ t, 不是 +t.

    构造: 纯平移相机 (R=I, t≠0)。正确 world 应整体平移 -t; 遗留版会平移 +t。
    """
    n, H, W = 1, 8, 8
    rng = np.random.default_rng(2)
    depth = (rng.random((n, H, W)).astype(np.float32) + 0.5)
    ext = np.zeros((n, 3, 4), dtype=np.float32)
    ext[0, :3, :3] = np.eye(3)
    ext[0, :3, 3] = [0.3, -0.2, 0.1]
    K = np.array([[[400, 0, W / 2], [0, 400, H / 2], [0, 0, 1]]], dtype=np.float64)

    correct = unproject_v3(depth, ext, K).astype(np.float64)
    legacy = unproject_np_legacy(depth, ext, K).astype(np.float64)

    # 正确版: world = cam - t (因 R=I, t_c2w=-t) => 整体平移 -t
    # 累积每个像素的 (legacy - correct) 应恒等于 2t (因为 legacy 用 +t)
    diff = legacy - correct
    # 取非零深度像素的平均偏移
    m = depth[0] > 1e-6
    mean_diff = diff[0, m].mean(axis=0)
    # 期望 ~ 2*t = [0.6, -0.4, 0.2]
    assert np.allclose(mean_diff, 2 * ext[0, :3, 3], atol=1e-3), f"偏移 {mean_diff} != 2t {2*ext[0,:3,3]}"


if __name__ == "__main__":
    test_camera_centers_convention()
    test_inverse_se3_roundtrip()
    test_correct_translation_term()
    print("ALL extrinsic convention tests passed")
