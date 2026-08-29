"""阶段 2.2 Geometry Audit v3.1 — 表型指标 (plant_view, P0-8 修复 + P0-9 离群敏感).

从对齐后的植株前景点云计算:
  - RAW: min/max 跨度 (保留, 用于发现 outlier)
  - ROBUST: P1-P99 / P2.5-P97.5 跨度 (对 1-3% 离群点不敏感)
  - 明确命名 bbox_width_x / bbox_width_y / bbox_xy_diagonal (不再用 canopy_diam 误导)
  - PCA 主轴 / 次轴宽 (oriented bounding box major / minor width)
  - 占用体积 (凸包 / 体素)

pred vs ref 各报, scale 由 Sim3 的 s 固定.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def _vertical_extents_raw(P):
    """RAW 跨度 (min/max). 仅用于发现 outlier, 不作为最终精度判定."""
    P = np.asarray(P, dtype=np.float64)
    mins = P.min(0)
    maxs = P.max(0)
    return {
        "Eh_raw_m": float(maxs[2] - mins[2]),
        "bbox_width_x_raw_m": float(maxs[0] - mins[0]),
        "bbox_width_y_raw_m": float(maxs[1] - mins[1]),
        "bbox_xy_diagonal_raw_m": float(np.hypot(maxs[0] - mins[0], maxs[1] - mins[1])),
        "bbox_raw_m": {"x": [float(mins[0]), float(maxs[0])],
                       "y": [float(mins[1]), float(maxs[1])],
                       "z": [float(mins[2]), float(maxs[2])]},
    }


def _vertical_extents_robust(P, lo=0.01, hi=0.99):
    """ROBUST 跨度 (P_lo-P_hi 百分位). 对极端 outlier 不敏感."""
    P = np.asarray(P, dtype=np.float64)
    r = {}
    for ax, name in enumerate(["z", "x", "y"]):
        qs = np.quantile(P[:, ax], [lo, hi])
        r[f"{name}_robust_span_m"] = float(qs[1] - qs[0])
    # 明确命名
    r["height_robust_m"] = r["z_robust_span_m"]
    r["bbox_width_x_robust_m"] = r["x_robust_span_m"]
    r["bbox_width_y_robust_m"] = r["y_robust_span_m"]
    r["bbox_xy_diagonal_robust_m"] = float(np.hypot(r["x_robust_span_m"], r["y_robust_span_m"]))
    return r


def _pca_widths(P):
    """PCA 主轴/次轴宽 (oriented bounding box major/minor width)."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 3:
        return {"pca_major_width_m": None, "pca_minor_width_m": None}
    c = P - P.mean(0)
    cov = c.T @ c / len(c)
    w, V = np.linalg.eigh(cov)
    # 主轴 = 最大特征值方向; 投影跨度
    out = {}
    for k, ax in enumerate([2, 1]):  # major, minor (两最大特征值)
        e = V[:, ax]
        proj = c @ e
        out["pca_major_width_m" if k == 0 else "pca_minor_width_m"] = float(proj.max() - proj.min())
    return out


def occupied_volume(P, voxel_size=0.01):
    """体素计数体积 (m^3) + 凸包体积."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 4:
        return {"volume_voxel_m3": None, "volume_convexhull_m3": None}
    vox = np.floor(P / voxel_size).astype(np.int64)
    n_vox = len(np.unique(vox, axis=0))
    vol_vox = float(n_vox * voxel_size ** 3)
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
        if P is None or len(P) == 0:
            continue
        b = {}
        b.update(_vertical_extents_raw(P))
        b.update(_vertical_extents_robust(P))
        b.update(_pca_widths(P))
        b.update(occupied_volume(P, voxel_size))
        for k, v in b.items():
            block[f"{name}_{k}"] = v
    # 误差 (robust 主指标) — 仅对标量 _m 字段, 跳过 dict (bbox_raw_m)
    pk = [k for k in block if k.startswith("pred_") and k.endswith("_m")
          and not isinstance(block[k], dict)]
    for k in pk:
        refk = "ref_" + k[len("pred_"):]
        if refk in block and isinstance(block[refk], (int, float)) and block[k] is not None and block[refk] is not None:
            ek = k[len("pred_"):].replace("_m", "_error_m")
            block[f"pred_{ek}"] = float(abs(block[k] - block[refk]))
    return block


def outlier_sensitivity(P, ref=None):
    """P0-9: 同一前景点云在不同 outlier 处理下的 phenotype 对比.

    返回 dict: regime -> {height_robust, bbox_width_x_robust, bbox_xy_diagonal_robust, pca_major, pca_minor}
    regimes:
      all                     : 全部点
      remove_gt_50mm         : 删除 pred->ref NN 距离 > 50mm 的点 (需 ref)
      remove_gt_20mm         : 删除 > 20mm
      statistical_outlier     : 移除 NN 均值+2std 的离群
      robust_percentile_only  : 仅用 P1-P99 截尾 (等价于 _vertical_extents_robust)
    """
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    regimes = {}
    variants = {
        "all": P,
    }
    if ref is not None:
        ref = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
        tQ = cKDTree(ref)
        d, _ = tQ.query(P, k=1, workers=-1)
        variants["remove_gt_50mm"] = P[d <= 0.050]
        variants["remove_gt_20mm"] = P[d <= 0.020]
        # statistical outlier: 移除 mean+2std
        mu, sd = d.mean(), d.std()
        variants["statistical_outlier"] = P[d <= mu + 2 * sd]

    for name, sub in variants.items():
        if len(sub) < 10:
            regimes[name] = None
            continue
        r = _vertical_extents_robust(sub)
        r.update(_pca_widths(sub))
        regimes[name] = r
    # robust percentile only 单独用全部点截尾
    regimes["robust_percentile_only"] = _vertical_extents_robust(P)
    return regimes
