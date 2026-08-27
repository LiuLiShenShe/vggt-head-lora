"""阶段 2.2 Geometry Audit v3 — 双向几何指标 (P0-4).

所有指标基于两团点云 P(预测, 已对齐到参考系) 与 Q(参考), 双向计算, 默认不截断。

指标:
  - Chamfer: CD_p2g = mean_p min_q ||p-q||; CD_g2p = mean_q min_p ||p-q||; CD_sym = (|P|·CD_p2g + |Q|·CD_g2p)/(|P|+|Q|)
  - 双向 nn 距离分布: median / P90 / P95 (pred->gt, gt->pred 各自)
  - Precision/Recall/F-score @ τ: τ ∈ {1%,2%,5%}·D (D=参考云 bbox 对角线) + 物理单位 {0.01,0.02,0.05}
  - 覆盖/离群: N_total, N_within, N_outside, within_ratio, outside_ratio (@ τ=5%·D)
  - truncated_inlier_nn_median: 仅 diagnostic_only=True (弃用主判据)

依赖: scipy.spatial.cKDTree (da3 已装).
"""
from __future__ import annotations

import numpy as np


def _nearest(src, dst_tree):
    d, _ = dst_tree.query(src, k=1, workers=-1)
    return d.astype(np.float64)


def bbox_diag(Q: np.ndarray) -> float:
    """参考云 Q 的 bbox 对角线 (米). D 在参考云自身坐标系稳定."""
    q = np.asarray(Q, dtype=np.float64)
    return float(np.linalg.norm(q.max(0) - q.min(0)))


def chamfer(P, Q):
    """返回 (CD_p2g, CD_g2p, CD_sym)."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if len(P) == 0 or len(Q) == 0:
        return float("nan"), float("nan"), float("nan")
    tP = _tree(P)
    tQ = _tree(Q)
    d_p2g = _nearest(P, tQ)        # 每个 P 点到最近 Q
    d_g2p = _nearest(Q, tP)        # 每个 Q 点到最近 P
    cd_p2g = float(d_p2g.mean())
    cd_g2p = float(d_g2p.mean())
    cd_sym = (len(P) * cd_p2g + len(Q) * cd_g2p) / (len(P) + len(Q))
    return cd_p2g, cd_g2p, cd_sym


def nn_distributions(P, Q):
    """双向 nn 距离分布 (非截断). 返回 dict: median/P90/P95 for pred->gt and gt->pred."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    tP, tQ = _tree(P), _tree(Q)
    d_p2g = _nearest(P, tQ)
    d_g2p = _nearest(Q, tP)
    return {
        "median_nn_pred2gt": float(np.median(d_p2g)),
        "p90_nn_pred2gt": float(np.percentile(d_p2g, 90)),
        "p95_nn_pred2gt": float(np.percentile(d_p2g, 95)),
        "median_nn_gt2pred": float(np.median(d_g2p)),
        "p90_nn_gt2pred": float(np.percentile(d_g2p, 90)),
        "p95_nn_gt2pred": float(np.percentile(d_g2p, 95)),
    }


def precision_recall_fscore(P, Q, tau):
    """tau: float 阈值 (米). 返回 (P, R, F). F=0 if P+R==0."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    tP, tQ = _tree(P), _tree(Q)
    d_p2g = _nearest(P, tQ)
    d_g2p = _nearest(Q, tP)
    prec = float((d_p2g <= tau).mean())
    rec = float((d_g2p <= tau).mean())
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f


def coverage_outlier(P, Q, tau):
    """@给定 τ 的覆盖/离群统计."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    tP, tQ = _tree(P), _tree(Q)
    d_p2g = _nearest(P, tQ)
    d_g2p = _nearest(Q, tP)
    n_pred_within = int((d_p2g <= tau).sum())
    n_gt_within = int((d_g2p <= tau).sum())
    return {
        "N_pred": int(len(P)),
        "N_gt": int(len(Q)),
        "N_pred_within": n_pred_within,
        "N_gt_within": n_gt_within,
        "N_pred_outside": int(len(P) - n_pred_within),
        "N_gt_outside": int(len(Q) - n_gt_within),
        "within_ratio_pred": float(n_pred_within / max(1, len(P))),
        "within_ratio_gt": float(n_gt_within / max(1, len(Q))),
    }


def truncated_inlier_nn_median(P, Q, trunc, min_pts=100):
    """弃用主判据: 截断 NN 中位数. diagnostic_only=True.

    仅保留 ||p-q|| <= trunc 的最近邻; 有效点 < min_pts 返回 None.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if len(P) < 1 or len(Q) < 1:
        return None, None, len(P), len(P)
    tQ = _tree(Q)
    d, _ = tQ.query(P, k=1, distance_upper_bound=trunc, workers=-1)
    n_beyond = int((~np.isfinite(d)).sum())
    d = d[np.isfinite(d)]
    if len(d) < min_pts:
        return None, None, len(P), n_beyond
    return float(np.median(d)), float(np.percentile(d, 90)), len(P), n_beyond


def full_metric_block(P, Q, D, abs_taus=(0.01, 0.02, 0.05)):
    """主入口: 给定已对齐 P, Q 与 D(参考 bbox 对角线), 返回完整指标 dict.

    若 D 为 None (参考系不可审计/非 metric), 则跳过所有阈值指标 (F/P/R, coverage),
    仅保留 Chamfer 与双向分布 (这些不依赖 D 但也不宜用于跨 frame 比较, 调用方会清空).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    block = {}
    cd_p2g, cd_g2p, cd_sym = chamfer(P, Q)
    block["chamfer_symmetric_m"] = cd_sym
    block["chamfer_pred2gt_m"] = cd_p2g
    block["chamfer_gt2pred_m"] = cd_g2p
    block.update(nn_distributions(P, Q))

    if D is not None and D > 0:
        norm_taus = {f"{int(p*100)}pctD": p * D for p in (0.01, 0.02, 0.05)}
        for name, tau in norm_taus.items():
            p, r, f = precision_recall_fscore(P, Q, tau)
            block[f"precision_{name}"] = p
            block[f"recall_{name}"] = r
            block[f"fscore_{name}"] = f
        for tau in abs_taus:
            p, r, f = precision_recall_fscore(P, Q, tau)
            block[f"precision_{tau:0.3f}m"] = p
            block[f"recall_{tau:0.3f}m"] = r
            block[f"fscore_{tau:0.3f}m"] = f
        cov = coverage_outlier(P, Q, 0.05 * D)
        block.update({f"cov5pctD_{k}": v for k, v in cov.items()})
        block["D_bbox_diag_m"] = D
        # diagnostic-only truncated NN
        trunc = 0.05 * D
        tmed, tp90, n_tot, n_beyond = truncated_inlier_nn_median(P, Q, trunc)
        block["truncated_inlier_nn_median_m"] = tmed
        block["truncated_inlier_nn_p90_m"] = tp90
        block["truncated_n_beyond"] = n_beyond
    block["truncated_inlier_diagnostic_only"] = True
    return block


def _tree(P):
    from scipy.spatial import cKDTree
    return cKDTree(np.asarray(P, dtype=np.float64))
