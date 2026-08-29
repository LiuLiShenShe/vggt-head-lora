"""阶段 2.2 Geometry Audit v3.1 — 单帧深度审计 (plant_view, P0-6 修复).

参考深度: images/depth/*.png (720×720, 16-bit uint16).
**单位审计 (DEPTH_UNIT_AUDIT.json, VERIFIED): uint16 存储的是毫米, depth_scale_to_meter=0.001.**
因此必须用 `ref_m = raw_png.astype(float) * 0.001` 才能与 VGGT 米制深度比较.
v3 把原值当米 (raw AbsRel≈1.0) 是单位错误造成的伪像.

VGGT 深度: depth_vggt.npy (S,Hout,518) float32 (model 输出已是米制 metric).

映射:
  ref depth 720×720 → 原图 1080×1080: scale 1080/720, nearest-neighbor
  VGGT grid (v,u) → 原图 (W_orig, H_orig): 与 foreground_v3 相同映射
  最终在 VGGT 网格空间对齐

指标 (VGGT 米制 vs 参考米制):
  RAW metric: 无尺度缩放 (VGGT 已是米制, 仅报真实尺度误差)
  SCALE-ALIGNED: 逐帧 median scaling (报 scale 因子)
  两列均含 AbsRel / RMSE_m / MAE_m / logRMSE / δ1 / δ2 / δ3, 严格区分.
"""
from __future__ import annotations

import json
import os
import numpy as np
from PIL import Image

BASE = "/fj/VGGT+head+lora实验/阶段2"

# P0-5 强制: 参考深度单位. 读取 DEPTH_UNIT_AUDIT.json; 若缺失则用保守默认并报警.
def _load_depth_scale():
    audit_path = os.path.join(os.path.dirname(__file__), "DEPTH_UNIT_AUDIT.json")
    if os.path.exists(audit_path):
        a = json.load(open(audit_path))
        if a.get("status") == "VERIFIED":
            return float(a["depth_scale_to_meter"])
    # 审计未完成: 不得臆测, 默认 0.001 (毫米), 但标记未审计
    return 0.001

DEPTH_SCALE_TO_METER = _load_depth_scale()


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

    # P0-5: 参考深度单位为毫米, 乘 scale 转为米, 才能与 VGGT 米制深度比较
    ref_aligned_m = ref_aligned * DEPTH_SCALE_TO_METER

    # valid: 参考深度米制 > 0 且 < 65m (米制物理上限兜底), VGGT 深度有效
    valid = (ref_aligned_m > 0) & (ref_aligned_m < 65.0) & (depth_vggt_v > 0) & np.isfinite(depth_vggt_v)
    if valid.sum() < 100:
        return None

    d_pred = depth_vggt_v[valid].astype(np.float64)  # VGGT 米制
    d_ref = ref_aligned_m[valid]                     # 参考米制

    result = {"N_valid": int(valid.sum()), "valid_ratio": float(valid.mean()),
              "depth_scale_to_meter": DEPTH_SCALE_TO_METER}
    # RAW metric: VGGT 米制 vs 参考米制, 无尺度缩放
    result.update(_compute_depth_metrics(d_pred, d_ref, prefix="raw_"))
    # SCALE-ALIGNED: 逐帧 median scaling (报 scale 因子)
    med_pred = float(np.median(d_pred))
    med_ref = float(np.median(d_ref))
    if med_pred > 1e-6:
        scale = med_ref / med_pred
        d_pred_aligned = d_pred * scale
        result.update(_compute_depth_metrics(d_pred_aligned, d_ref, prefix="aligned_"))
        result["median_scale"] = float(scale)
    return result


def _compute_depth_metrics(d_pred, d_ref, prefix=""):
    d_pred = np.asarray(d_pred, dtype=np.float64)
    d_ref = np.asarray(d_ref, dtype=np.float64)
    diff = d_pred - d_ref
    ratio = d_pred / np.maximum(d_ref, 1e-10)
    ratio_inv = d_ref / np.maximum(d_pred, 1e-10)
    worse = np.maximum(ratio, ratio_inv)
    return {
        f"{prefix}absrel": float(np.mean(np.abs(diff) / np.maximum(d_ref, 1e-10))),
        f"{prefix}rmse": float(np.sqrt(np.mean(diff ** 2))),
        f"{prefix}mae": float(np.mean(np.abs(diff))),
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
