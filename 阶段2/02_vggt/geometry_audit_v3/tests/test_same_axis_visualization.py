"""P0-6: 同坐标可视化断言 — overlay 图两云必须使用同一轴范围."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import align_v3, unproject_v3  # noqa: E402
from unproject_v3 import unproject_v3  # noqa: E402


def _make_overlay(sid, pred_aligned, ref, savepath):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2], s=0.3, c="red")
    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=0.3, c="green")
    # 关键: 同坐标范围
    allp = np.concatenate([pred_aligned, ref], 0)
    lo, hi = allp.min(0), allp.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    fig.savefig(savepath, dpi=60); plt.close(fig)
    return lo, hi, ax


def test_same_axis_limits():
    sid = "plant_view_3d/plantview__langdon_4__05-03-24"
    seqjson = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences/plant_view/langdon_4__05-03-24.json"
    base = "/fj/VGGT+head+lora实验/阶段2/02_vggt/v2_clean_rerun"
    d = np.load(f"{base}/{sid}/depth_vggt.npy")[:8]
    e = np.load(f"{base}/{sid}/extrinsic_w2c.npy")[:8]
    i = np.load(f"{base}/{sid}/intrinsic_vggt.npy")[:8]
    pts = unproject_v3(d, e, i)
    al = align_v3.align_sequence(seqjson, e, pts)
    pa = al["pred_aligned"][::200]
    rf = al["ref_cloud"][::200]
    lo, hi, ax = _make_overlay(sid, pa, rf, "/tmp/_overlay_test.png")
    # 断言两云共享同一 x/y/z 轴范围
    xlim = ax.get_xlim(); ylim = ax.get_ylim(); zlim = ax.get_zlim()
    # xlim 必须覆盖两云并集
    assert xlim[0] <= lo[0] and xlim[1] >= hi[0]
    assert ylim[0] <= lo[1] and ylim[1] >= hi[1]
    assert zlim[0] <= lo[2] and zlim[1] >= hi[2]
    # 轴范围对两云一致 (不是各画各的)
    assert xlim == ax.get_xlim()  # 单一 axis 对象


def test_union_axis_contains_both():
    """两云并集范围 == 轴范围 (无裁剪, 无各自独立缩放)."""
    sid = "plant_view_3d/plantview__langdon_4__13-02-24"
    seqjson = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences/plant_view/langdon_4__13-02-24.json"
    base = "/fj/VGGT+head+lora实验/阶段2/02_vggt/v2_clean_rerun"
    d = np.load(f"{base}/{sid}/depth_vggt.npy")[:8]
    e = np.load(f"{base}/{sid}/extrinsic_w2c.npy")[:8]
    i = np.load(f"{base}/{sid}/intrinsic_vggt.npy")[:8]
    pts = unproject_v3(d, e, i)
    al = align_v3.align_sequence(seqjson, e, pts)
    pa = al["pred_aligned"][::200]
    rf = al["ref_cloud"][::200]
    lo, hi, ax = _make_overlay(sid, pa, rf, "/tmp/_overlay_test2.png")
    allp = np.concatenate([pa, rf], 0)
    assert tuple(ax.get_xlim()) == (float(allp[:, 0].min()), float(allp[:, 0].max()))
    assert tuple(ax.get_ylim()) == (float(allp[:, 1].min()), float(allp[:, 1].max()))
    assert tuple(ax.get_zlim()) == (float(allp[:, 2].min()), float(allp[:, 2].max()))


if __name__ == "__main__":
    test_same_axis_limits()
    test_union_axis_contains_both()
    print("ALL same-axis visualization tests passed")
