"""P0-8: phenotype robust 指标对离群点不敏感 (raw min/max 敏感).

断言: 注入极端 outlier 后, robust (P1-P99) 高度变化极小, 而 raw (min/max) 高度变化巨大.
"""
import sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _make_plant(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0, 1.0, n)
    x = rng.uniform(-0.4, 0.4, n)
    y = rng.uniform(-0.4, 0.4, n)
    return np.stack([x, y, z], 1)


def test_phenotype_robust_outlier_resistance():
    import phenotype_v3 as ph
    clean = _make_plant()
    # 注入 2% 极端离群点 (z 拉到 10m)
    dirty = np.vstack([clean, np.array([[0.0, 0.0, 10.0]] * 100)])
    rc = ph._vertical_extents_raw(clean)["Eh_raw_m"]
    rd = ph._vertical_extents_raw(dirty)["Eh_raw_m"]
    rob_c = ph._vertical_extents_robust(clean)["height_robust_m"]
    rob_d = ph._vertical_extents_robust(dirty)["height_robust_m"]
    # raw 被离群点严重污染
    assert rd > rc + 5.0, f"raw Eh 应被离群点污染 (Δ={(rd-rc):.2f})"
    # robust 几乎不变
    assert abs(rob_d - rob_c) < 0.05, f"robust Eh 应对离群不敏感 (Δ={(rob_d-rob_c):.4f})"


def test_pca_widths_defined():
    import phenotype_v3 as ph
    P = _make_plant(3000)
    r = ph._pca_widths(P)
    assert r["pca_major_width_m"] is not None and r["pca_minor_width_m"] is not None
    assert r["pca_major_width_m"] >= r["pca_minor_width_m"] - 1e-6


if __name__ == "__main__":
    test_phenotype_robust_outlier_resistance()
    test_pca_widths_defined()
    print("ALL phenotype robust tests passed")
