"""P0-4: F-score 指标单元测试 (阈值扫描 + 退化情形)."""
import sys
import numpy as np
import pytest

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from geometry_metrics_v3 import precision_recall_fscore, chamfer  # noqa: E402

rng = np.random.default_rng(42)


def _cloud(n, seed):
    return np.random.default_rng(seed).random((n, 3)).astype(np.float64) * 2 - 1


def test_fscore_zero_when_disjoint():
    """完全分离的两团云: τ 远小于间隔时 P=R=0, F=0 (不报错)."""
    P = _cloud(100, 1)
    Q = P + 100.0  # 远离
    p, r, f = precision_recall_fscore(P, Q, 0.02)
    assert p == 0.0 and r == 0.0 and f == 0.0


def test_fscore_symmetric_in_tau_monotonic():
    """固定错位, F@τ 随 τ 增大单调不减."""
    P = _cloud(200, 2)
    Q = P + 0.05
    fs = [precision_recall_fscore(P, Q, t)[2] for t in (0.01, 0.02, 0.05, 0.1, 0.3)]
    assert all(fs[i] <= fs[i + 1] for i in range(len(fs) - 1)), fs


def test_fscore_perfect_alignment_unity():
    P = _cloud(150, 3)
    Q = P.copy()
    p, r, f = precision_recall_fscore(P, Q, 0.03)
    assert p > 0.99 and r > 0.99 and f > 0.99


def test_fscore_balanced_definition():
    """P=R 时 F=P=R."""
    P = _cloud(120, 4)
    Q = P + 0.02
    p, r, f = precision_recall_fscore(P, Q, 0.05)
    assert abs(f - (2 * p * r / (p + r))) < 1e-12


def test_chamfer_nan_empty():
    P = _cloud(50, 5)
    cd = chamfer(P, np.zeros((0, 3)))
    assert all(np.isnan(x) for x in cd)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
    print("ALL fscore tests passed")
