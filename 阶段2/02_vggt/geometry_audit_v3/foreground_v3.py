"""阶段 2.2 Geometry Audit v3 — 前景掩膜提取 (P0-5, 优先级 2: 数据集分割).

策略:
  - plant_view: 用 sequence.extra.mask_dir 的逐帧二值 mask (1080²). mask 路径 = mask_dir/basename(rgb).png.
  - wheat3dgs: 用 extra.mask_dir 的 YOLO-SAM 实例 mask (多实例/视图); 同 cam 前缀的全部 mask 做 OR.
  - mustc: 无逐帧 mask -> 仅 FULL-SCENE (无前景集).

VGGT 518×Hout 网格 -> 原图 (W_orig,H_orig) 映射:
  new_height = Hout (VGGT 输出高)
  w_orig = (u + 0.5) * (W_orig/518) - 0.5
  v_orig = (v + 0.5) * (H_orig/new_height) - 0.5
  (crop 模式下 width 恒为较大维 -> new_height<=518 -> 无中心裁剪)
"""
from __future__ import annotations

import json
import os
import numpy as np
from PIL import Image


def _vggt_grid_to_orig(uv, W_orig, H_orig, Hout):
    """uv: (N,2) VGGT 像素坐标 -> 原图 (W,H) 坐标 (浮点)."""
    uv = np.asarray(uv, dtype=np.float64)
    w = (uv[:, 0] + 0.5) * (W_orig / 518.0) - 0.5
    h = (uv[:, 1] + 0.5) * (H_orig / float(Hout)) - 0.5
    return np.stack([w, h], axis=-1)


def _frame_foreground_mask(seq_json, frame_rgb_path, W_orig, H_orig, Hout):
    """返回 VGGT 网格 (Hout,518) 的 bool 前景掩膜 (前景=255 / 实例存在).

    Priority-2 数据集 mask 已对齐到原图 (mask 分辨率 == 原图分辨率).
    """
    d = json.load(open(seq_json))
    sid = d["dataset_id"]
    extra = d.get("extra", {})
    basename = os.path.basename(frame_rgb_path)

    if sid == "plant_view_3d":
        mask_dir = extra["mask_dir"]
        mp = os.path.join(mask_dir, basename + ".png")
        if not os.path.exists(mp):
            return None
        m = np.asarray(Image.open(mp).convert("L"))
        fg = (m > 0)
    elif sid == "wheat3dgs":
        mask_dir = extra["mask_dir"]
        # cam 前缀: 去掉扩展名, 实例 mask 形如 <cam>_NNN.png
        cam = os.path.splitext(basename)[0]
        fg = None
        if os.path.isdir(mask_dir):
            for fn in os.listdir(mask_dir):
                if fn.startswith(cam) and fn.endswith(".png"):
                    m = np.asarray(Image.open(os.path.join(mask_dir, fn)).convert("L"))
                    fgi = (m > 0)
                    fg = fgi if fg is None else (fg | fgi)
        if fg is None:
            return None
    else:  # mustc / 其他: 无 mask
        return None

    # 把 VGGT 网格每个像素映射到原图坐标, 采样 foreground
    Hv, Wv = Hout, 518
    uu, vv = np.meshgrid(np.arange(Wv), np.arange(Hv))
    grid = np.stack([uu.ravel(), vv.ravel()], axis=-1)
    orig = _vggt_grid_to_orig(grid, W_orig, H_orig, Hout)
    ox = np.round(orig[:, 0]).astype(int).clip(0, W_orig - 1)
    oy = np.round(orig[:, 1]).astype(int).clip(0, H_orig - 1)
    # mask 分辨率可能 != 原图 (plant 1080==原图; wheat 2998x4094==原图)
    mh, mw = fg.shape
    ox = ox.clip(0, mw - 1)
    oy = oy.clip(0, mh - 1)
    fg_flat = fg[oy, ox].reshape(Hv, Wv)
    return fg_flat


def frame_foreground_for_sequence(seq_json, rgb_paths, Hout):
    """预计算每帧的 VGGT-grid 前景掩膜列表 (或 None).

    Returns: list[np.bool (Hout,518)] or None (无 mask 数据集).
    """
    d = json.load(open(seq_json))
    # 原图尺寸: 读第一帧
    im = Image.open(rgb_paths[0])
    W_orig, H_orig = im.size
    masks = []
    has_any = False
    for p in rgb_paths:
        m = _frame_foreground_mask(seq_json, p, W_orig, H_orig, Hout)
        if m is not None:
            has_any = True
        masks.append(m)
    if not has_any:
        return None
    return masks


def apply_foreground_to_points(points_world, depth_valid, fg_masks):
    """根据每帧前景掩膜, 从 (S,H,W,3) 点云抽取前景点.

    points_world: (S,H,W,3), depth_valid: (S,H,W) bool, fg_masks: list[(H,W) bool] or None
    Returns: (N,3) 前景点 (仅 depth_valid 且 fg 为真的点), 或 None
    """
    if fg_masks is None:
        return None
    S, H, W = points_world.shape[:3]
    out = []
    for s in range(S):
        fg = fg_masks[s]
        if fg is None:
            continue
        m = depth_valid[s] & fg
        out.append(points_world[s][m])
    if not out:
        return None
    return np.concatenate(out, axis=0)
