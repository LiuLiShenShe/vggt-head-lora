"""阶段 2.2 Geometry Audit v3 — 修正后的反投影 (P0-1)。

背景:
  遗留自定义 `four_path_eval.unproject_np()` 含 OpenCV w2c->c2w 符号错误:
      world = R.T @ cam + t        # 错误: 加了 t_w2c (世界->相机平移)
  正确应为 (见 vggt/utils/geometry.py: closed_form_inverse_se3 + depth_to_world_coords_points):
      R_c2w = R_w2c.T
      t_c2w = -R_w2c.T @ t_w2c
      world = cam @ R_c2w.T + t_c2w      ≡  R_w2c.T @ cam - R_w2c.T @ t_w2c

本模块提供 numpy 实现 `unproject_v3`, 数值上严格等价于官方
`vggt.utils.geometry.unproject_depth_map_to_point_map`, 用于阶段 2.2 几何审计。

约定 (与官方一致):
  - extrinsics 为 OpenCV 世界->相机 (w2c), 形状 (S,3,4):  x_cam = R_w2c @ X_world + t_w2c
  - depth 为 (S,H,W) 或 (S,H,W,1), 单位与参考系一致 (VGGT 输出为 metric-ish, 尺度由场景决定)
  - 输出 world 点云 (S,H,W,3)
"""
from __future__ import annotations

import numpy as np


def _cam_coords_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """depth(H,W) -> cam coords(H,W,3). 与官方 depth_to_cam_coords_points 一致 (零 skew)."""
    H, W = depth.shape
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x_cam = (u - cu) * depth / fu
    y_cam = (v - cv) * depth / fv
    z_cam = depth
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float64)


def unproject_v3(depth, extrinsic, intrinsic):
    """修正后的反投影 (≡ 官方 unproject_depth_map_to_point_map).

    Args:
        depth: (S,H,W) 或 (S,H,W,1) float
        extrinsic: (S,3,4) OpenCV w2c
        intrinsic: (S,3,3) 或 (3,3)
    Returns:
        (S,H,W,3) float32 world points
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 4:
        depth = depth[..., 0]
    S, H, W = depth.shape
    ext = np.asarray(extrinsic, dtype=np.float64).reshape(S, 3, 4)
    intr = np.asarray(intrinsic, dtype=np.float64)
    if intr.ndim == 2:
        intr = np.repeat(intr[None], S, axis=0)

    R_w2c = ext[:, :3, :3]            # (S,3,3)
    t_w2c = ext[:, :3, 3]             # (S,3)
    R_c2w = np.transpose(R_w2c, (0, 2, 1))          # R.T
    # 官方 closed_form_inverse_se3: t_c2w = -R.T @ t
    t_c2w = -np.einsum("sij,sj->si", R_c2w, t_w2c)  # (S,3)

    world = np.empty((S, H, W, 3), dtype=np.float64)
    for s in range(S):
        cam = _cam_coords_from_depth(depth[s], intr[s])          # (H,W,3)
        world[s] = cam @ R_c2w[s].T + t_c2w[s]                  # ≡ R.T @ cam - R.T @ t
    return world.astype(np.float32)


def unproject_np_legacy(depth, extrinsic, intrinsic):
    """遗留 (BUGGY) 反投影, 仅用于审计对照 (证明旧版错误). 不要在新管线使用."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 4:
        depth = depth[..., 0]
    S, H, W = depth.shape
    ext = np.asarray(extrinsic, dtype=np.float64).reshape(S, 3, 4)
    intr = np.asarray(intrinsic, dtype=np.float64)
    if intr.ndim == 2:
        intr = np.repeat(intr[None], S, axis=0)

    x, y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    ones = np.ones_like(x)
    pix = np.stack([x, y, ones], axis=-1)[..., None]             # (H,W,3,1)
    K_inv = np.linalg.inv(intr)                                  # (S,3,3)
    cam = np.einsum("sij,hwjn->shwin", K_inv, pix)[..., 0]       # (S,H,W,3)
    cam = cam * depth[..., None]                                 # (S,H,W,3)
    R = ext[:, :3, :3]
    t = ext[:, :3, 3]
    world = np.einsum("sij,shwj->shwi", R.transpose(0, 2, 1), cam) + t[:, None, None, :]  # BUG: +t
    return world.astype(np.float32)


def camera_centers(extrinsic):
    """世界系相机中心 C = -R_w2c.T @ t_w2c (官方 centers 约定)."""
    ext = np.asarray(extrinsic, dtype=np.float64)
    R = ext[:, :3, :3]
    t = ext[:, :3, 3]
    return np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), -t)
