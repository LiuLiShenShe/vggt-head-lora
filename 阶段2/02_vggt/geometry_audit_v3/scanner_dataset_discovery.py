#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P1: Scanner-GT dataset discovery — recursive scan of 3D Plant View dataset
for scanner ground-truth availability across multiple plants.

Generates SCANNER_DATASET_DISCOVERY.csv with per-plant summary.
"""
import os, sys, csv, json

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
sys.path.insert(0, ROOT)

DATASET = "/fj/VGGT+head+lora实验/阶段1-数据集/3D Plant View"

def check_dir(path):
    return os.path.isdir(path) if path else False

def count_files(path, ext=None):
    if not check_dir(path): return 0
    if ext:
        return len([f for f in os.listdir(path) if f.endswith(ext)])
    return len(os.listdir(path))

def main():
    rows = []
    # scan all plant dirs
    if not check_dir(DATASET):
        print(f"ERROR: dataset dir missing: {DATASET}")
        sys.exit(1)
    plant_ids = sorted([d for d in os.listdir(DATASET)
                        if os.path.isdir(os.path.join(DATASET, d))])
    print(f"Found {len(plant_ids)} plant dirs: {plant_ids[:10]}{'...' if len(plant_ids)>10 else ''}")

    for pid in plant_ids:
        pdir = os.path.join(DATASET, pid)
        # dates under each plant
        dates = sorted([d for d in os.listdir(pdir)
                        if os.path.isdir(os.path.join(pdir, d))])
        for date in dates:
            ddir = os.path.join(pdir, date)
            # RGB
            rgb_dir = os.path.join(ddir, "images", "rgb")
            rgb_available = check_dir(rgb_dir)
            n_rgb = count_files(rgb_dir, ".png") if rgb_available else 0
            # camera transforms
            cam_dir = os.path.join(ddir, "camera")
            camera_available = check_dir(cam_dir)
            # masks (plant_masks)
            mask_dir = os.path.join(ddir, "images", "plant_masks")
            mask_available = check_dir(mask_dir)
            n_mask = count_files(mask_dir, ".png") if mask_available else 0
            # depth
            depth_dir = os.path.join(ddir, "images", "depth")
            depth_available = check_dir(depth_dir)
            n_depth = count_files(depth_dir, ".png") if depth_available else 0
            # scanner GT
            gt_dir = os.path.join(ddir, "ground_truth")
            scanner_available = os.path.isfile(os.path.join(gt_dir, "scans", "GTScanPC.ply")) if check_dir(gt_dir) else False
            scan_metrics_available = os.path.isfile(os.path.join(gt_dir, "scans", "scan_metrics.json")) if check_dir(gt_dir) else False

            # local status
            if scanner_available:
                status = "scanner_gt_present"
            elif rgb_available and camera_available:
                status = "rgb_camera_only"
            else:
                status = "incomplete"

            rows.append({
                "plant_id": pid,
                "date": date,
                "rgb_available": rgb_available,
                "camera_available": camera_available,
                "mask_available": mask_available,
                "depth_available": depth_available,
                "scanner_available": scanner_available,
                "scan_metrics_available": scan_metrics_available,
                "n_rgb_frames": n_rgb,
                "n_mask_frames": n_mask,
                "n_depth_frames": n_depth,
                "local_status": status,
            })

    # Write CSV
    out_csv = os.path.join(ROOT, "SCANNER_DATASET_DISCOVERY.csv")
    fields = ["plant_id", "date", "rgb_available", "camera_available", "mask_available",
              "depth_available", "scanner_available", "scan_metrics_available",
              "n_rgb_frames", "n_mask_frames", "n_depth_frames", "local_status"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Summary
    n_plants = len(set(r["plant_id"] for r in rows))
    n_scanner = len([r for r in rows if r["scanner_available"]])
    n_scanner_plants = len(set(r["plant_id"] for r in rows if r["scanner_available"]))
    print(f"Discovery complete:")
    print(f"  Total rows (plant-date): {len(rows)}")
    print(f"  Unique plants: {n_plants}")
    print(f"  Rows with scanner GT: {n_scanner}")
    print(f"  Unique plants with scanner GT: {n_scanner_plants}")
    print(f"  CSV: {out_csv}")

    # Print scanner plants
    scanner_rows = [r for r in rows if r["scanner_available"]]
    for r in scanner_rows:
        print(f"  SCANNER: {r['plant_id']}/{r['date']} rgb={r['n_rgb_frames']} depth={r['n_depth_frames']} mask={r['n_mask_frames']}")

if __name__ == "__main__":
    main()
