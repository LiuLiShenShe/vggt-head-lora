"""P0-4: 双向几何指标单元测试."""
import sys
import numpy as np
import pytest

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from geometry_metrics_v3 import (  # noqa: E402
    chamfer, nn_distributions, precision_recall_fscore, coverage_outlier,
    bbox_diag, truncated_inlier_nn_median,
)

rng = np.random.default_rng(0)


def _grid(n=200, seed=0):
    r = np.random.default_rng(seed)
    return r.random((n, 3)).astype(np.float64) * 2.0 - 1.0


def test_chamfer_perfect_alignment():
    P = _grid(300, 1)
    Q = P.copy()
    cd_p2g, cd_g2p, cd_sym = chamfer(P, Q)
    assert cd_p2g < 1e-9 and cd_g2p < 1e-9 and cd_sym < 1e-9


def test_chamfer_monotonic_with_separation():
    """Chamfer 随两团云分离距离单调增大; 同分布(小噪声)时远小于大分离."""
    P = _grid(600, 2)
    cd_self, _, _ = chamfer(P, P + rng.standard_normal(P.shape) * 1e-4)
    cd_far, _, _ = chamfer(P, P + 5.0)
    assert cd_self < 0.05, cd_self          # 几乎重合 -> 近 0
    assert cd_far > 1.0, cd_far             # 大分离 -> 很大
    # 单调: 分离 1.0 < 分离 5.0
    cd_mid, _, _ = chamfer(P, P + 1.0)
    assert cd_mid < cd_far


def test_nn_distributions_monotonic_noise():
    P = _grid(400, 3)
    Q = P + rng.standard_normal(P.shape) * 0.03  # 小幅噪声, 距离为正
    d = nn_distributions(P, Q)
    # 中位数应远小于 P95
    assert d["median_nn_pred2gt"] < d["p95_nn_pred2gt"]
    assert d["p95_nn_pred2gt"] > 0
    assert d["median_nn_gt2pred"] < d["p95_nn_gt2pred"]


def test_precision_recall_threshold_monotonic():
    P = _grid(400, 4)
    Q = P + 0.03  # 整体平移 0.03
    # τ 越小, precision/recall 越低
    p1, r1, f1 = precision_recall_fscore(P, Q, 0.01)
    p2, r2, f2 = precision_recall_fscore(P, Q, 0.05)
    assert p1 <= p2 and r1 <= r2
    # τ 远大于平移时几乎全命中
    p3, r3, f3 = precision_recall_fscore(P, Q, 0.1)
    assert p3 > 0.99 and r3 > 0.99 and f3 > 0.99


def test_fscore_perfect_is_one():
    P = _grid(200, 5)
    Q = P.copy()
    p, r, f = precision_recall_fscore(P, Q, 0.02)
    assert f > 0.99


def test_coverage_outlier_counts():
    P = _grid(200, 6)
    Q = P + 0.04
    c = coverage_outlier(P, Q, 0.05 * bbox_diag(Q))
    assert c["N_pred"] == 200 and c["N_gt"] == 200
    # 平移 0.04 < 0.05*D (D~2) -> 几乎全 within
    assert c["within_ratio_pred"] > 0.9


def test_truncated_inlier_diagnostic():
    # 稠密云: NN 间距 << 平移量, 使截断内保留多数点
    P = np.random.default_rng(7).random((5000, 3)).astype(np.float64) * 2 - 1
    Q = P + 0.04
    med, p90, n_tot, n_beyond = truncated_inlier_nn_median(P, Q, trunc=0.05)
    # 平移 0.04 < trunc 0.05 -> 大多数点被保留, median 接近平移量
    assert med is not None and abs(med - 0.04) < 0.01, (med, n_beyond, n_tot)
    assert n_beyond < n_tot


def test_bbox_diag_basic():
    Q = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float64)
    assert abs(bbox_diag(Q) - np.sqrt(1 + 4 + 9)) < 1e-9


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
    print("ALL metric tests passed")
