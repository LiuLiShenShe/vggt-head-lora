"""阶段 2.2 Geometry Audit v3.1 — 修正后可视化 (P0-1/2/3/4).

原则: **metric 输入点 ≡ figure 输入点**. 所有画图函数接收的是
主驱动中已经用于计算指标的同一些 array (P_fore, Q_fore, P_full, Q_full),
绝不在画图内部重新反投影/重新对齐 (修复 v3 _make_figures 漏 Sim3 的 bug).

所有 overlay / error 图:
  - same coordinate frame (两云均已对齐到参考系)
  - same Sim3 (传入即对齐后)
  - same axis limits (两云并集)
  - equal aspect ratio
  - same view/camera angle
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import cKDTree


def _union_limits(*clouds):
    allp = np.concatenate([c for c in clouds if c is not None and len(c) > 0], 0)
    lo, hi = allp.min(0), allp.max(0)
    return [(lo[i], hi[i]) for i in range(3)]


def _set_axes_lims(ax, lim):
    ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_zlim(lim[2])
    ax.set_box_aspect([lim[0][1]-lim[0][0], lim[1][1]-lim[1][0], lim[2][1]-lim[2][0]])


def overlay_fg_3d(figdir, sid, P_fore, Q_fore):
    """前景 3D overlay: 参考(绿) + VGGT(橙) 同坐标/同轴限."""
    lim = _union_limits(P_fore, Q_fore)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if Q_fore is not None and len(Q_fore) > 0:
        ax.scatter(Q_fore[:, 0], Q_fore[:, 1], Q_fore[:, 2], s=0.2, c="green", alpha=0.35, label="reference")
    ax.scatter(P_fore[:, 0], P_fore[:, 1], P_fore[:, 2], s=0.2, c="orange", alpha=0.45, label="VGGT pred")
    _set_axes_lims(ax, lim)
    ax.set_title(f"{sid}: foreground overlay 3D (aligned)")
    ax.legend()
    p = f"{figdir}/{sid}_overlay_fg_3d.png"
    fig.savefig(p, dpi=100); plt.close(fig)
    return p


def overlay_fg_ortho(figdir, sid, P_fore, Q_fore):
    """前景正交视图 XY / XZ / YZ, 同轴限, 用于判断叶片/冠层结构是否对齐."""
    lim = _union_limits(P_fore, Q_fore)
    paths = []
    for proj, name in [("xy", "overlay_fg_xy"), ("xz", "overlay_fg_xz"), ("yz", "overlay_fg_yz")]:
        fig, ax = plt.subplots(figsize=(6, 6))
        kw = dict(s=0.3, alpha=0.4)
        if Q_fore is not None and len(Q_fore) > 0:
            q = Q_fore[:, :2] if proj == "xy" else (Q_fore[:, [0, 2]] if proj == "xz" else Q_fore[:, [1, 2]])
            ax.scatter(q[:, 0], q[:, 1], c="green", label="reference", **kw)
        p = P_fore[:, :2] if proj == "xy" else (P_fore[:, [0, 2]] if proj == "xz" else P_fore[:, [1, 2]])
        ax.scatter(p[:, 0], p[:, 1], c="orange", label="VGGT pred", **kw)
        # 用 3D 并集范围投影到对应两轴
        if proj == "xy":
            ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
        elif proj == "xz":
            ax.set_xlim(lim[0]); ax.set_ylim(lim[2])
        else:
            ax.set_xlim(lim[1]); ax.set_ylim(lim[2])
        ax.set_aspect("equal")
        ax.set_title(f"{sid}: foreground {name}")
        ax.legend()
        pp = f"{figdir}/{sid}_{name}.png"
        fig.savefig(pp, dpi=100); plt.close(fig)
        paths.append(pp)
    return paths


def foreground_error_maps(figdir, sid, P_fore, Q_fore):
    """前景误差热图 (仅用 P_fore / Q_fore). 固定 mm 阈值上色.

    颜色固定阈值: 0-50 mm colorbar. pred2gt 用 P_fore 点 + 其到 ref 距离;
    gt2pred 用 Q_fore 点 + 其到 pred 距离 (避免点云长度不一致).
    """
    tQ = cKDTree(Q_fore)
    d_p2g, _ = tQ.query(P_fore, k=1, workers=-1)  # VGGT->ref (len P_fore)
    tP = cKDTree(P_fore)
    d_g2p, _ = tP.query(Q_fore, k=1, workers=-1)  # ref->VGGT (len Q_fore)
    lim = _union_limits(P_fore, Q_fore)
    paths = []
    # pred2gt: 在 VGGT 前景点上着色
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(P_fore[:, 0], P_fore[:, 1], P_fore[:, 2], s=0.3,
                    c=d_p2g * 1000.0, cmap="jet", vmin=0, vmax=50)
    _set_axes_lims(ax, lim)
    plt.colorbar(sc, ax=ax).set_label("nn dist (mm)")
    ax.set_title(f"{sid}: foreground error pred2gt (mm, 0-50mm colorbar)")
    p1 = f"{figdir}/{sid}_fg_error_pred2gt.png"
    fig.savefig(p1, dpi=100); plt.close(fig); paths.append(p1)
    # gt2pred: 在参考前景点上着色
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Q_fore[:, 0], Q_fore[:, 1], Q_fore[:, 2], s=0.3,
                    c=d_g2p * 1000.0, cmap="jet", vmin=0, vmax=50)
    _set_axes_lims(ax, lim)
    plt.colorbar(sc, ax=ax).set_label("nn dist (mm)")
    ax.set_title(f"{sid}: foreground error gt2pred (mm, 0-50mm colorbar)")
    p2 = f"{figdir}/{sid}_fg_error_gt2pred.png"
    fig.savefig(p2, dpi=100); plt.close(fig); paths.append(p2)
    return paths


def depth_montage_real(figdir, sid, rgb_paths, ref_depth_m, depth_vggt_raw, depth_vggt_aligned,
                       W_orig=1080, H_orig=1080, Hout=518, n_show=8):
    """P0-4: 真实深度 montage. 每代表帧一行:
      RGB | GT depth(m) | VGGT raw(m) | VGGT aligned(m) | Abs error(m) | Rel error
    所有 colorbar 带单位(m). 不再画 validity mask.
    """
    import os
    from PIL import Image
    idxs = list(range(0, len(rgb_paths), max(1, len(rgb_paths) // n_show)))[:n_show]
    fig, axes = plt.subplots(len(idxs), 6, figsize=(22, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]
    # VGGT grid -> 原图映射
    Hv, Wv = Hout, 518
    uu, vv = np.meshgrid(np.arange(Wv), np.arange(Hv))
    w_orig = (uu + 0.5) * (W_orig / 518.0) - 0.5
    h_orig = (vv + 0.5) * (H_orig / float(Hout)) - 0.5
    ox = np.clip(np.round(w_orig).astype(int), 0, W_orig - 1)
    oy = np.clip(np.round(h_orig).astype(int), 0, H_orig - 1)
    for ri, i in enumerate(idxs):
        # RGB
        axes[ri, 0].imshow(np.asarray(Image.open(rgb_paths[i]).convert("RGB")))
        axes[ri, 0].set_title(f"f{i} RGB"); axes[ri, 0].axis("off")
        # GT depth
        im = Image.open(ref_depth_path_for(rgb_paths[i], ref_depth_m))
        gt = np.asarray(im).astype(np.float64)
        gth, gtw = gt.shape[:2]
        if gth != H_orig or gtw != W_orig:
            gt = np.array(Image.fromarray(gt.astype(np.uint16)).resize((W_orig, H_orig), Image.Resampling.NEAREST), dtype=np.float64)
        gt_m = gt * 0.001  # 毫米 -> 米
        gtm = gt_m[oy, ox]
        _imshow_depth(axes[ri, 1], gtm, "GT depth (m)")
        # VGGT raw (已是 Hout×518 网格, 直接显示; gtm 同样取 Hv×Wv 便于对齐)
        dv = np.asarray(depth_vggt_raw[i], dtype=np.float64)  # (Hv, Wv)
        _imshow_depth(axes[ri, 2], dv, "VGGT raw (m)")
        # VGGT aligned (median scaling per frame)
        med_pred = np.median(dv[dv > 0]) if (dv > 0).any() else 1.0
        med_gt = np.median(gtm[gtm > 0]) if (gtm > 0).any() else 1.0
        sc = med_gt / med_pred if med_pred > 1e-6 else 1.0
        dva = dv * sc
        _imshow_depth(axes[ri, 3], dva, f"VGGT aligned (m) x{sc:.2f}")
        # abs error
        valid = (gtm > 0) & (gtm < 65) & (dv > 0)
        abserr = np.where(valid, np.abs(dva - gtm), np.nan)
        _imshow_depth(axes[ri, 4], abserr, "Abs error (m)", vmax=np.nanpercentile(abserr[valid], 95))
        # rel error
        relerr = np.where(valid, np.abs(dva - gtm) / np.maximum(gtm, 1e-3), np.nan)
        _imshow_depth(axes[ri, 5], relerr, "Rel error", vmax=np.nanpercentile(relerr[valid], 95))
    fig.suptitle(f"{sid}: real depth montage (m)")
    p = f"{figdir}/{sid}_depth_montage_real.png"
    fig.savefig(p, dpi=100); plt.close(fig)
    return p


def ref_depth_path_for(rgb_path, depth_dir):
    import os
    bn = os.path.splitext(os.path.basename(rgb_path))[0]
    return os.path.join(depth_dir, bn + ".png")


def _imshow_depth(ax, arr, title, vmax=None):
    arr = np.asarray(arr, dtype=np.float64)
    im = ax.imshow(arr, cmap="viridis", vmax=vmax)
    ax.set_title(title); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")
