"""阶段 2.2 Geometry Audit v3 — 单帧深度审计 (plant_view, P0-7).

参考深度: images/depth/*.png (720×720, 16-bit uint16, I;16).
VGGT 深度: depth_vggt.npy (S,Hout,518) float32.

映射:
  ref depth 720×720 → 原图 1080×1080: scale 1080/720, nearest-neighbor
  VGGT grid (v,u) → 原图 (W_orig, H_orig): 与 foreground_v3 相同映射
  最终在 VGGT 网格空间对齐

指标 (valid 像素 = ref_depth > 0 AND ref_depth < 65000):
  Raw: AbsRel / RMSE / logRMSE / δ1 / δ2 / δ3
  Aligned: median-scaling 后同样的 6 项
"""
from __future__ import annotations

import json
import os
import numpy as np
from PIL import Image

BASE = "/fj/VGGT+head+lora实验/阶段2"


def depth_audit_single_frame(depth_vggt_v, ref_depth_path, W_orig=1080, H_orig=1080, Hout=518):
    """单帧深度审计.

    Args:
        depth_vggt_v: (Hout,518) VGGT 深度
        ref_depth_path: str, 16-bit PNG (720x720 or same size as RGB)
    Returns:
        dict with raw + aligned metrics, or None if insufficient valid pixels
    """
    ref = np.asarray(Image.open(ref_depth_path)).astype(np.float64)
    ref_h, ref_w = ref.shape[:2]
    # ref depth 720→1080: 直接 nearest resize
    if ref_h != H_orig or ref_w != W_orig:
        ref = np.array(Image.fromarray(ref.astype(np.uint16)).resize(
            (W_orig, H_orig), Image.Resampling.NEAREST), dtype=np.float64)

    Hv, Wv = depth_vggt_v.shape
    # VGGT grid → 原图坐标
    uu, vv = np.meshgrid(np.arange(Wv), np.arange(Hv))
    w_orig = (uu + 0.5) * (W_orig / 518.0) - 0.5
    h_orig = (vv + 0.5) * (H_orig / float(Hout)) - 0.5
    ox = np.clip(np.round(w_orig).astype(int), 0, W_orig - 1)
    oy = np.clip(np.round(h_orig).astype(int), 0, H_orig - 1)
    ref_aligned = ref[oy, ox]  # (Hv,Wv)

    valid = (ref_aligned > 0) & (ref_aligned < 65000) & (depth_vggt_v > 0) & np.isfinite(depth_vggt_v)
    if valid.sum() < 100:
        return None

    d_pred = depth_vggt_v[valid].astype(np.float64)
    d_ref = ref_aligned[valid]

    result = {"N_valid": int(valid.sum()), "valid_ratio": float(valid.mean())}
    # raw
    result.update(_compute_depth_metrics(d_pred, d_ref, prefix="raw_"))
    # aligned (median scaling)
    med_pred = float(np.median(d_pred))
    med_ref = float(np.median(d_ref))
    if med_pred > 1e-6:
        scale = med_ref / med_pred
        d_pred_aligned = d_pred * scale
        result.update(_compute_depth_metrics(d_pred_aligned, d_ref, prefix="aligned_"))
        result["median_scale"] = float(scale)
    return result


def _compute_depth_metrics(d_pred, d_ref, prefix=""):
    ratio = d_pred / np.maximum(d_ref, 1e-10)
    ratio_inv = d_ref / np.maximum(d_pred, 1e-10)
    worse = np.maximum(ratio, ratio_inv)
    return {
        f"{prefix}absrel": float(np.mean(np.abs(d_pred - d_ref) / np.maximum(d_ref, 1e-10))),
        f"{prefix}rmse": float(np.sqrt(np.mean((d_pred - d_ref) ** 2))),
        f"{prefix}logrmse": float(np.sqrt(np.mean((np.log(np.maximum(d_pred, 1e-6)) - np.log(np.maximum(d_ref, 1e-6))) ** 2))),
        f"{prefix}delta1": float((worse < 1.1).mean()),
        f"{prefix}delta2": float((worse < 1.25).mean()),
        f"{prefix}delta3": float((worse < 1.5625).mean()),
    }


def depth_audit_sequence(seq_json, depth_vggt, rgb_paths, max_frames=40):
    """对一整个 plant_view 序列做帧级深度审计, 返回逐帧 + 汇总.

    depth_vggt: (S,Hout,518)
    rgb_paths: list[str]
    """
    d = json.load(open(seq_json))
    extra = d.get("extra", {})
    depth_dir = extra.get("depth_dir")
    if not depth_dir:
        return None  # 无参考深度 (wheat/mustc)

    S = depth_vggt.shape[0]
    step = max(1, S // max_frames)
    indices = list(range(0, S, step))[:max_frames]
    frame_results = []
    for i in indices:
        basename = os.path.splitext(os.path.basename(rgb_paths[i]))[0]
        ref_path = os.path.join(depth_dir, basename + ".png")
        if not os.path.exists(ref_path):
            continue
        r = depth_audit_single_frame(depth_vggt[i], ref_path)
        if r is not None:
            r["frame_idx"] = i
            frame_results.append(r)

    if not frame_results:
        return None

    # 汇总
    keys = [k for k in frame_results[0] if not k.startswith("frame")]
    agg = {}
    for k in keys:
        vals = [r[k] for r in frame_results if k in r]
        if isinstance(vals[0], (int, float, np.floating)):
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"] = float(np.std(vals))
            if k.endswith("absrel"):
                agg[f"median_{k}"] = float(np.median(vals))
    agg["n_frames_audited"] = len(frame_results)
    return {"frames": frame_results, "aggregate": agg}
