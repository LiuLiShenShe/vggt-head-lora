#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 (v3.2.1): 真实 foreground-depth evaluator.

针对 plant_view_3d 序列 (05/13/20/12-03-24) 评估 VGGT 深度在 **plant foreground** 上的精度:
  - 有效前景像素 = plant_mask & ref_depth_valid & pred_depth_valid
  - RAW-SCALE (无尺度缩放, 报真实尺度误差) + SCALE-ALIGNED (median scaling, 仅相对形状诊断)
  - 每帧指标 -> DEPTH_FOREGROUND_METRICS.csv; 每序列汇总 -> DEPTH_FOREGROUND_SUMMARY.csv
  - 全场景诊断 -> DEPTH_FULLSCENE_DIAGNOSTIC.csv (绝不混入 headline)

含 12-03-24 (pose-FAIL 但参考深度可用 — 深度评估与 pose 无关).

复用 depth_audit_v3 的参考深度加载 (uint16 × scale_to_meter = 米) 与 foreground_v3 的掩膜.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from PIL import Image

BASE = "/fj/VGGT+head+lora实验/阶段2"
ROOT = os.path.join(BASE, "02_vggt", "geometry_audit_v3")

REPRESENTATIVES = [
    ("plant_view_3d/plantview__langdon_4__05-03-24", "plant_view/langdon_4__05-03-24.json"),
    ("plant_view_3d/plantview__langdon_4__12-03-24", "plant_view/langdon_4__12-03-24.json"),
    ("plant_view_3d/plantview__langdon_4__13-02-24", "plant_view/langdon_4__13-02-24.json"),
    ("plant_view_3d/plantview__langdon_4__20-02-24", "plant_view/langdon_4__20-02-24.json"),
]

sys.path.insert(0, ROOT)
import depth_audit_v3 as da3_mod                          # noqa: E402
import foreground_v3                                       # noqa: E402

DEPTH_SCALE_TO_METER = da3_mod.DEPTH_SCALE_TO_METER
OUT_METRICS = os.path.join(ROOT, "DEPTH_FOREGROUND_METRICS.csv")
OUT_SUMMARY = os.path.join(ROOT, "DEPTH_FOREGROUND_SUMMARY.csv")
OUT_DIAG = os.path.join(ROOT, "DEPTH_FULLSCENE_DIAGNOSTIC.csv")
W_ORIG, H_ORIG, HOUT = 1080, 1080, 518


def _load_ref_depth_aligned(ref_path):
    ref = np.asarray(Image.open(ref_path)).astype(np.float64)
    rh, rw = ref.shape[:2]
    if rh != H_ORIG or rw != W_ORIG:
        ref = np.array(Image.fromarray(ref.astype(np.uint16)).resize(
            (W_ORIG, H_ORIG), Image.Resampling.NEAREST), dtype=np.float64)
    uu, vv = np.meshgrid(np.arange(518), np.arange(HOUT))
    w_orig = (uu + 0.5) * (W_ORIG / 518.0) - 0.5
    h_orig = (vv + 0.5) * (H_ORIG / float(HOUT)) - 0.5
    ox = np.clip(np.round(w_orig).astype(int), 0, W_ORIG - 1)
    oy = np.clip(np.round(h_orig).astype(int), 0, H_ORIG - 1)
    return ref[oy, ox] * DEPTH_SCALE_TO_METER


def _depth_metrics(d_pred, d_ref):
    diff = d_pred - d_ref
    ratio = d_pred / np.maximum(d_ref, 1e-10)
    ratio_inv = d_ref / np.maximum(d_pred, 1e-10)
    worse = np.maximum(ratio, ratio_inv)
    med_pred = float(np.median(d_pred))
    med_ref = float(np.median(d_ref))
    scale = (med_ref / med_pred) if med_pred > 1e-6 else float("nan")
    out = {
        "raw_absrel": float(np.mean(np.abs(diff) / np.maximum(d_ref, 1e-10))),
        "raw_rmse": float(np.sqrt(np.mean(diff ** 2))),
        "raw_mae": float(np.mean(np.abs(diff))),
        "raw_logrmse": float(np.sqrt(np.mean((np.log(np.maximum(d_pred, 1e-6)) - np.log(np.maximum(d_ref, 1e-6))) ** 2))),
        "raw_delta1": float((worse < 1.1).mean()),
        "raw_delta2": float((worse < 1.25).mean()),
        "raw_delta3": float((worse < 1.5625).mean()),
        "median_scale": scale,
    }
    if med_pred > 1e-6:
        d_pred_a = d_pred * scale
        diff_a = d_pred_a - d_ref
        ratio_a = d_pred_a / np.maximum(d_ref, 1e-10)
        ratio_a_inv = d_ref / np.maximum(d_pred_a, 1e-10)
        worse_a = np.maximum(ratio_a, ratio_a_inv)
        out.update({
            "aligned_absrel": float(np.mean(np.abs(diff_a) / np.maximum(d_ref, 1e-10))),
            "aligned_rmse": float(np.sqrt(np.mean(diff_a ** 2))),
            "aligned_mae": float(np.mean(np.abs(diff_a))),
            "aligned_logrmse": float(np.sqrt(np.mean((np.log(np.maximum(d_pred_a, 1e-6)) - np.log(np.maximum(d_ref, 1e-6))) ** 2))),
            "aligned_delta1": float((worse_a < 1.1).mean()),
            "aligned_delta2": float((worse_a < 1.25).mean()),
            "aligned_delta3": float((worse_a < 1.5625).mean()),
        })
    return out


def main():
    frame_rows, diag_rows, summary_rows = [], [], []
    for sid_dir, seqjson_rel in REPRESENTATIVES:
        seqjson = os.path.join(BASE, "01_sequences", "sequences", seqjson_rel)
        sid = os.path.basename(sid_dir)
        d = json.load(open(seqjson))
        extra = d.get("extra", {})
        depth_dir = extra.get("depth_dir")
        if not depth_dir:
            print(f"SKIP {sid} (no ref depth_dir)")
            continue
        depth = np.load(os.path.join(BASE, "02_vggt", "v2_clean_rerun", sid_dir, "depth_vggt.npy"))
        if depth.ndim == 4:
            depth = depth[..., 0]
        S, Hout, W = depth.shape
        rgb_paths = d["rgb_paths"]
        fg_masks = foreground_v3.frame_foreground_for_sequence(seqjson, rgb_paths, Hout)
        seq_frame_metrics = []
        for i in range(S):
            bn = os.path.splitext(os.path.basename(rgb_paths[i]))[0]
            rp = os.path.join(depth_dir, bn + ".png")
            if not os.path.exists(rp):
                continue
            ref_m = _load_ref_depth_aligned(rp)              # (Hout,518)
            pred_v = depth[i].astype(np.float64)             # VGGT 米制
            # full-scene diagnostic
            valid_full = (ref_m > 0) & (ref_m < 65.0) & (pred_v > 0) & np.isfinite(pred_v)
            if valid_full.sum() >= 100:
                m_full = _depth_metrics(pred_v[valid_full], ref_m[valid_full])
                diag_rows.append({"sequence_id": sid, "frame_idx": i, **{k: round(v, 5) for k, v in m_full.items()}})
            # foreground-only (PRIMARY)
            if fg_masks is None or fg_masks[i] is None:
                continue
            fg = fg_masks[i]
            valid_fg = valid_full & fg
            if valid_fg.sum() >= 100:
                m_fg = _depth_metrics(pred_v[valid_fg], ref_m[valid_fg])
                n_fg = int(valid_fg.sum())
                frame_rows.append({
                    "sequence_id": sid, "frame_idx": i, "n_fg_valid_pixels": n_fg,
                    **{k: round(v, 5) for k, v in m_fg.items()},
                })
                seq_frame_metrics.append(m_fg)
        # summary
        if seq_frame_metrics:
            keys = seq_frame_metrics[0].keys()
            row = {"sequence_id": sid, "n_frames_with_fg_depth": len(seq_frame_metrics)}
            for k in keys:
                vals = np.array([m[k] for m in seq_frame_metrics if m[k] == m[k]])
                row[f"{k}_mean"] = round(float(np.mean(vals)), 5)
                row[f"{k}_median"] = round(float(np.median(vals)), 5)
                row[f"{k}_p90"] = round(float(np.percentile(vals, 90)), 5)
            summary_rows.append(row)
            print(f"[{sid}] fg-depth frames={len(seq_frame_metrics)} "
                  f"raw_AbsRel={row['raw_absrel_mean']:.3f} raw_RMSE={row['raw_rmse_mean']:.3f} "
                  f"aligned_AbsRel={row['aligned_absrel_mean']:.3f}")
        else:
            print(f"[{sid}] NO foreground depth pixels (mask overlap=0)")

    # write
    fc = ["sequence_id", "frame_idx", "n_fg_valid_pixels", "raw_absrel", "raw_mae", "raw_rmse",
          "raw_logrmse", "raw_delta1", "raw_delta2", "raw_delta3", "median_scale",
          "aligned_absrel", "aligned_mae", "aligned_rmse", "aligned_logrmse",
          "aligned_delta1", "aligned_delta2", "aligned_delta3"]
    _write(OUT_METRICS, frame_rows, fc)
    sc = ["sequence_id", "n_frames_with_fg_depth"]
    if summary_rows:
        for k in summary_rows[0]:
            if k not in sc:
                sc.append(k)
    _write(OUT_SUMMARY, summary_rows, sc)
    dc = ["sequence_id", "frame_idx", "raw_absrel", "raw_mae", "raw_rmse", "raw_logrmse",
          "raw_delta1", "raw_delta2", "raw_delta3", "median_scale",
          "aligned_absrel", "aligned_mae", "aligned_rmse", "aligned_logrmse",
          "aligned_delta1", "aligned_delta2", "aligned_delta3"]
    _write(OUT_DIAG, diag_rows, dc)
    print(f"\n-> {OUT_METRICS} ({len(frame_rows)} frames)")
    print(f"-> {OUT_SUMMARY} ({len(summary_rows)} seqs)")
    print(f"-> {OUT_DIAG} ({len(diag_rows)} frames, diagnostic_only)")


def _write(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (k not in r or r[k] is None or (isinstance(r[k], float) and r[k] != r[k])) else r[k]) for k in cols})


if __name__ == "__main__":
    main()
