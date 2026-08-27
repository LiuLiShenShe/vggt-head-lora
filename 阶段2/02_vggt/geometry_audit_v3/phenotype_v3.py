"""阶段 2.2 Geometry Audit v3 — 表型指标 (plant_view, P0-8).

从对齐后的植株前景点云计算:
  - 株高 Eh (z 跨度, 经 Sim3 固定)
  - 冠层宽 (xy 平面直径)
  - bbox (3D, 每轴 min/max)
  - 占用体积 (凸包体积 / 体素计数)
pred vs ref 各报。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def _vertical_extents(P):
    """返回 (Eh, canopy_diam, bbox). 假设 z 为竖直 (经 dataparser 恢复后 Z↑)."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) == 0:
        return {"Eh_m": None, "canopy_diam_m": None, "bbox_m": None}
    mins = P.min(0)
    maxs = P.max(0)
    Eh = float(maxs[2] - mins[2])
    xspan = float(maxs[0] - mins[0])
    yspan = float(maxs[1] - mins[1])
    canopy_diam = float(np.hypot(xspan, yspan))  # 平面包络直径近似
    bbox = {"x": [float(mins[0]), float(maxs[0])],
            "y": [float(mins[1]), float(maxs[1])],
            "z": [float(mins[2]), float(maxs[2])]}
    return {"Eh_m": Eh, "canopy_diam_m": canopy_diam, "bbox_m": bbox}


def occupied_volume(P, voxel_size=0.01):
    """体素计数体积 (m^3) + 凸包体积."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 4:
        return {"volume_voxel_m3": None, "volume_convexhull_m3": None}
    # 体素计数
    vox = np.floor(P / voxel_size).astype(np.int64)
    n_vox = len(np.unique(vox, axis=0))
    vol_vox = float(n_vox * voxel_size ** 3)
    # 凸包
    try:
        hull = ConvexHull(P)
        vol_hull = float(hull.volume)
    except Exception:
        vol_hull = None
    return {"volume_voxel_m3": vol_vox, "volume_convexhull_m3": vol_hull}


def phenotype_block(pred_points_aligned, ref_points, voxel_size=0.01):
    """pred_points_aligned: 已 Sim3 对齐的预测前景点 (N,3); ref_points: 参考前景点 (M,3)."""
    block = {}
    for name, P in (("pred", pred_points_aligned), ("ref", ref_points)):
        b = _vertical_extents(P)
        v = occupied_volume(P, voxel_size)
        block[f"{name}_Eh_m"] = b["Eh_m"]
        block[f"{name}_canopy_diam_m"] = b["canopy_diam_m"]
        block[f"{name}_bbox_m"] = b["bbox_m"]
        block[f"{name}_volume_voxel_m3"] = v["volume_voxel_m3"]
        block[f"{name}_volume_convexhull_m3"] = v["volume_convexhull_m3"]
    # 差异
    if block.get("pred_Eh_m") is not None and block.get("ref_Eh_m") is not None:
        block["Eh_abs_err_m"] = float(abs(block["pred_Eh_m"] - block["ref_Eh_m"]))
    return block
