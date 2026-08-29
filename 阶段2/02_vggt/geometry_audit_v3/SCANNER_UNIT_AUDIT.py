#!/usr/bin/env python3
"""
SCANNER_UNIT_AUDIT.py
---------------------
Verifies whether the scanner GT PLY point cloud is stored in meters or
millimeters by cross-referencing:
  1. PLY bounding box dimensions
  2. scan_metrics.json (ground-truth length/width/height in metres)
  3. Camera w2c extrinsic translation-z (from transforms.json)
  4. NeRFStudio config.yml (confirms data pipeline)

Outputs SCANNER_UNIT_AUDIT.json alongside this script.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

# ── paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = Path("/fj/VGGT+head+lora实验/阶段1-数据集/3D Plant View/langdon_4/19-03-24")

PLY_PATH = BASE / "ground_truth" / "scans" / "GTScanPC.ply"
SCAN_METRICS_PATH = BASE / "ground_truth" / "scan_metrics.json"
CONFIG_YML_PATHS = [
    BASE / "gaussian-splatting" / "undistorted" / "splatfacto" / "1" / "config.yml",
    BASE / "gaussian-splatting" / "undistorted_segmented" / "splatfacto" / "1" / "config.yml",
    BASE / "gaussian-splatting" / "undistorted_culled_bg" / "splatfacto" / "1" / "config.yml",
    BASE / "gaussian-splatting" / "colmap_splatfacto" / "splatfacto" / "1" / "config.yml",
]
TRANSFORMS_PATHS = [
    BASE / "transforms" / "original" / "transforms.json",
    BASE / "transforms" / "adjusted" / "transforms.json",
]
OUTPUT = SCRIPT_DIR / "SCANNER_UNIT_AUDIT.json"

# ── 1. Load PLY ──────────────────────────────────────────────────────────
print(f"[1/5] Loading PLY: {PLY_PATH}")
ply = PlyData.read(str(PLY_PATH))
vertex = ply["vertex"]
n_points = vertex.count
x = np.array(vertex["x"], dtype=np.float64)
y = np.array(vertex["y"], dtype=np.float64)
z = np.array(vertex["z"], dtype=np.float64)

ply_bbox = {
    "x": [float(x.min()), float(x.max())],
    "y": [float(y.min()), float(y.max())],
    "z": [float(z.min()), float(z.max())],
}
dx = float(x.max() - x.min())
dy = float(y.max() - y.min())
dz = float(z.max() - z.min())

# median distance from centroid (quick sanity check)
centroid = np.array([x.mean(), y.mean(), z.mean()])
dists = np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2 + (z - centroid[2]) ** 2)
median_dist = float(np.median(dists))

print(f"  n_points={n_points:,}  bbox=({dx:.2f}, {dy:.2f}, {dz:.2f})  median_dist={median_dist:.2f}")

# ── 2. Load scan_metrics.json ────────────────────────────────────────────
print(f"[2/5] Loading scan_metrics: {SCAN_METRICS_PATH}")
with open(SCAN_METRICS_PATH) as f:
    sm = json.load(f)
scan_metrics_m = [sm["length"], sm["width"], sm["height"]]
print(f"  length={sm['length']}  width={sm['width']}  height={sm['height']}")

# ── 3. Unit decision logic ──────────────────────────────────────────────
# PLY bbox (raw) vs scan_metrics (metres)
# ratio = max(PLY dim) / max(scan_metrics dim)
ply_max_dim = max(dx, dy, dz)
sm_max_dim = max(scan_metrics_m)
ratio = ply_max_dim / sm_max_dim

evidence = []

# Check if PLY is roughly in metres (ratio ~1) or millimetres (ratio ~1000)
if ratio > 100:
    # PLY is in millimetres
    scale_to_meter = 0.001
    unit = "millimeter"
    evidence.append(f"PLY bbox max dimension = {ply_max_dim:.1f}; scan_metrics max = {sm_max_dim:.4f} m")
    evidence.append(f"Ratio PLY/scan_metrics = {ratio:.1f} >> 10 => PLY is in millimetres")
    evidence.append(f"PLY bbox in metres (after /1000): ({dx/1000:.4f}, {dy/1000:.4f}, {dz/1000:.4f})")
    evidence.append(f"scan_metrics in metres: ({sm[sm['length'] and 'length']}, {sm['width']}, {sm['height']})")
elif ratio < 0.01:
    # PLY is in some other unit, unlikely
    scale_to_meter = 1.0
    unit = "meter"
    evidence.append(f"Ratio PLY/scan_metrics = {ratio:.4f} << 1; unusual, defaulting to metres")
else:
    # PLY is in metres
    scale_to_meter = 1.0
    unit = "meter"
    evidence.append(f"PLY bbox max dimension = {ply_max_dim:.4f}; scan_metrics max = {sm_max_dim:.4f} m")
    evidence.append(f"Ratio PLY/scan_metrics = {ratio:.2f} ~ 1 => PLY is in metres")

ply_bbox_dimensions_m = [dx * scale_to_meter, dy * scale_to_meter, dz * scale_to_meter]
bbox_vs_scan_metrics_ratio = float(np.mean([
    ply_bbox_dimensions_m[0] / sm["length"],
    ply_bbox_dimensions_m[1] / sm["width"],
    ply_bbox_dimensions_m[2] / sm["height"],
]))
evidence.append(f"PLY bbox dims (metres): [{ply_bbox_dimensions_m[0]:.4f}, {ply_bbox_dimensions_m[1]:.4f}, {ply_bbox_dimensions_m[2]:.4f}]")
evidence.append(f"scan_metrics (metres):  [{sm['length']:.4f}, {sm['width']:.4f}, {sm['height']:.4f}]")
evidence.append(f"Mean axis ratio (PLY_m / scan_metrics) = {bbox_vs_scan_metrics_ratio:.4f}")

# ── 4. Camera extrinsics z (from transforms.json) ──────────────────────
print("[3/5] Loading camera transforms for extrinsics z")
camera_tz_m = None
transforms_found = None
for tp in TRANSFORMS_PATHS:
    if tp.exists():
        with open(tp) as f:
            tdata = json.load(f)
        frames = tdata.get("frames", [])
        if frames and "transform_matrix" in frames[0]:
            transforms_found = tp
            # Extract z-translation from first 5 w2c matrices
            evidence.append(f"Camera transforms source: {tp}")
            z_vals = []
            for fi, frame in enumerate(frames[:5]):
                wm = frame["transform_matrix"]
                # w2c 4x4: translation is column 3, z = row 2, col 3
                tz = wm[2][3]
                z_vals.append(tz)
                evidence.append(f"  Frame {fi}: w2c translation z = {tz:.4f}")
            camera_tz_m = float(np.median(z_vals))
            evidence.append(f"Median camera_tz from first 5 frames: {camera_tz_m:.4f} m")
            break

if camera_tz_m is None:
    evidence.append("No transforms.json with w2c matrices found; camera_tz unavailable")

# Cross-check: camera z should be roughly same order as PLY z-range (in metres)
if camera_tz_m is not None and scale_to_meter == 0.001:
    cam_in_ply_units = camera_tz_m / scale_to_meter  # e.g. 1.5m -> 1500mm
    ply_z_min = ply_bbox["z"][0]
    ply_z_max = ply_bbox["z"][1]
    if cam_in_ply_units >= ply_z_min and cam_in_ply_units <= ply_z_max:
        evidence.append(f"Camera z ({camera_tz_m:.4f} m) falls within PLY z-range "
                        f"({ply_z_min:.1f}..{ply_z_max:.1f} mm) when converted => CONSISTENT")
    elif abs(cam_in_ply_units - ply_z_min) / (ply_z_max - ply_z_min) < 0.1 or \
         abs(cam_in_ply_units - ply_z_max) / (ply_z_max - ply_z_min) < 0.1:
        evidence.append(f"Camera z ({camera_tz_m:.4f} m) is near edge of PLY z-range => approximately consistent")
    else:
        evidence.append(f"Camera z ({camera_tz_m:.4f} m => {cam_in_ply_units:.1f} mm) "
                        f"vs PLY z [{ply_z_min:.1f}, {ply_z_max:.1f}] => NOTE: outside range "
                        "(cameras may be outside scan bounds)")

# ── 5. Load config.yml ───────────────────────────────────────────────────
print("[4/5] Checking NeRFStudio config.yml")
config_found = None
for cp in CONFIG_YML_PATHS:
    if cp.exists():
        config_found = str(cp)
        evidence.append(f"NeRFStudio config.yml found: {cp}")
        break
if config_found is None:
    evidence.append("No NeRFStudio config.yml found in expected locations")

# ── 6. Final status ──────────────────────────────────────────────────────
print("[5/5] Building output")

status = "VERIFIED" if 0.8 <= bbox_vs_scan_metrics_ratio <= 1.2 else "UNRESOLVED"
if status == "UNRESOLVED":
    evidence.append(f"WARNING: bbox_vs_scan_metrics_ratio = {bbox_vs_scan_metrics_ratio:.4f} "
                    "deviates from 1.0; unit assignment may be incorrect")

result = {
    "scanner_storage_unit": unit,
    "scale_to_meter": scale_to_meter,
    "evidence": evidence,
    "n_points": n_points,
    "ply_bbox": ply_bbox,
    "ply_bbox_dimensions_m": ply_bbox_dimensions_m,
    "scan_metrics_m": scan_metrics_m,
    "bbox_vs_scan_metrics_ratio": bbox_vs_scan_metrics_ratio,
    "camera_tz_m": camera_tz_m,
    "status": status,
}

with open(OUTPUT, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\nOutput written to: {OUTPUT}")
print(json.dumps(result, indent=2, ensure_ascii=False))
