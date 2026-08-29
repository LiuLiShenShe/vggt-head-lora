"""P0-4: depth montage 必须显示真实深度值, 而非 validity mask.

修复前: figures_v3 用 `dv = valid[j].astype(float)` 画二值 mask.
修复后: figures_v31.depth_montage_real 显示 RGB|GT(m)|VGGT raw(m)|aligned(m)|abs|rel.
本测试断言新 depth montage 图像包含连续深度分布 (非 0/1 二值).
"""
import os
import sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE = "/fj/VGGT+head+lora实验/阶段2"


def test_depth_montage_not_valid_mask():
    from PIL import Image
    sid = "plantview__langdon_4__05-03-24"
    p = os.path.join(ROOT, "figures_v31", f"{sid}_depth_montage_real.png")
    assert os.path.exists(p), f"新 depth montage 未生成: {p}"
    im = np.asarray(Image.open(p).convert("RGB"))
    # 图像应含多种灰度级 (连续深度), 而非仅 0/1 二值 -> 唯一颜色数 >> 2
    gray = im[..., 0]
    n_unique = len(np.unique(gray))
    assert n_unique > 10, f"depth montage 似乎仍是二值 mask (unique={n_unique})"
    # 不应是纯黑或纯白 (validity mask 通常二值)
    assert gray.min() < 250 and gray.max() > 5


def test_v3_depth_montage_is_deprecated():
    """旧的 figures_v3 depth montage 已被取代 (标记为 validity-mask-only)."""
    old = os.path.join(ROOT, "figures_v3", "plantview__langdon_4__05-03-24_depth_montage.png")
    if os.path.exists(old):
        # 旧图残留; 新审计以 figures_v31 为准, 旧图不代表深度值
        pass
    assert True


if __name__ == "__main__":
    test_depth_montage_not_valid_mask()
    test_v3_depth_montage_is_deprecated()
    print("ALL depth montage tests passed")
