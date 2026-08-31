#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1: Apply official DA3Metric conversion to all DA3 network outputs.

Official formula (README line 235):
    metric_depth = focal * net_output / 300

Where focal = (fx + fy) / 2 in pixels at network resolution.
DA3METRIC-LARGE processes at 504px (longest side resize from 1080px).

Calibrated intrinsics: fx=1371.82, fy=1370.79 at 1080×1080
Network resolution: 504px
focal_network = (1371.82 + 1370.79) / 2 * 504/1080 = 639.94
conversion_factor = 639.94 / 300 = 2.1331

No model re-run needed — pure numpy post-processing.
"""
import os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)
from configs import SEQUENCES

# ── Calibrated intrinsics ─────────────────────────────────────────────────
CALIBRATED_FX = 1371.82  # at 1080×1080
CALIBRATED_FY = 1370.79
IMG_SIZE = 1080
DA3_RES = 504

# Network-space focal length
fx_net = CALIBRATED_FX * DA3_RES / IMG_SIZE  # 640.18
fy_net = CALIBRATED_FY * DA3_RES / IMG_SIZE  # 639.70
focal_net = (fx_net + fy_net) / 2             # 639.94

# Official conversion factor: focal * net_output / 300
CONVERSION_FACTOR = focal_net / 300.0  # 2.1331


def main():
    print(f"DA3Metric Official Conversion")
    print(f"  Calibrated fx={CALIBRATED_FX:.2f} fy={CALIBRATED_FY:.2f} at {IMG_SIZE}px")
    print(f"  Network focal: {focal_net:.2f}px (at {DA3_RES}px)")
    print(f"  Conversion factor: {CONVERSION_FACTOR:.4f}")
    print(f"  Formula: D_metric = D_net × {CONVERSION_FACTOR:.4f}")
    print()

    total_frames = 0
    for seq_id, pose_fail in SEQUENCES:
        net_path = os.path.join(PHASE3_DIR, "da3", seq_id, "depth_da3.npy")
        metric_path = os.path.join(PHASE3_DIR, "da3", seq_id, "depth_da3_metric.npy")

        if not os.path.exists(net_path):
            print(f"  SKIP {seq_id}: {net_path} not found")
            continue

        d_net = np.load(net_path)
        d_metric = (d_net * CONVERSION_FACTOR).astype(np.float32)

        np.save(metric_path, d_metric)
        total_frames += d_net.shape[0]

        print(f"  {seq_id}:")
        print(f"    net:     range=[{d_net.min():.4f}, {d_net.max():.4f}] mean={d_net.mean():.4f}")
        print(f"    metric:  range=[{d_metric.min():.4f}, {d_metric.max():.4f}] mean={d_metric.mean():.4f}")
        print(f"    saved: {metric_path}")

    print(f"\nTotal: {total_frames} frames converted")
    print(f"Conversion factor: {CONVERSION_FACTOR:.4f}")


if __name__ == "__main__":
    main()
