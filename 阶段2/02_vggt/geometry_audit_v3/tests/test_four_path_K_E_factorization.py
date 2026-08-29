"""P1-3: Four-Path v4 必须能隔离 K 与 E (B-K 与 B-E 可区分).

用合成数据 + unproject_v3 验证: 给定同一 depth, 只换 K 与只换 E 产生的点云几何不同
(否则无法归因 intrinsics vs extrinsics).
"""
import sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _rand_se3(rng, n=4):
    from scipy.spatial.transform import Rotation
    ext = np.zeros((n, 3, 4))
    for i in range(n):
        R = Rotation.random(random_state=rng).as_matrix()
        t = rng.uniform(-0.5, 0.5, 3)
        ext[i, :3, :3] = R
        ext[i, :3, 3] = t
    return ext


def test_K_E_factorization_isolates():
    from unproject_v3 import unproject_v3
    rng = np.random.default_rng(0)
    S, H, W = 4, 32, 32
    depth = rng.uniform(1.0, 3.0, (S, H, W)).astype(np.float32)
    # 同一 K 两种 E
    K = np.tile(np.array([[500., 0, 16], [0, 500, 16], [0, 0, 1]]), (S, 1, 1))
    E1 = _rand_se3(rng, S)
    E2 = _rand_se3(rng, S)
    A = unproject_v3(depth, E1, K).reshape(-1, 3)
    B_E = unproject_v3(depth, E2, K).reshape(-1, 3)  # 只换 E
    # 只换 K
    K2 = np.tile(np.array([[800., 0, 10], [0, 800, 10], [0, 0, 1]]), (S, 1, 1))
    B_K = unproject_v3(depth, E1, K2).reshape(-1, 3)  # 只换 K
    # 只换 E 与只换 K 应产生不同点云
    assert not np.allclose(A, B_E, atol=0.01), "换 E 后点云应改变"
    assert not np.allclose(A, B_K, atol=0.01), "换 K 后点云应改变"
    assert not np.allclose(B_K, B_E, atol=0.01), "B-K 与 B-E 必须可区分 (否则无法隔离内外参)"


if __name__ == "__main__":
    test_K_E_factorization_isolates()
    print("ALL four_path K/E factorization tests passed")
