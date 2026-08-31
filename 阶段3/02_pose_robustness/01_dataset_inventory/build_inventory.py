#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B Step 1: Build multi-plant dataset inventory.

Scans available data across datasets and produces:
  MULTIPLANT_POSE_DATASET_INVENTORY.csv
  DATASET_SUMMARY.json
"""
import os, sys, json, csv
import numpy as np
from PIL import Image

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
VGGT_OUT = os.path.join(ROOT, "阶段2", "02_vggt")
DATA_ROOT = os.path.join(ROOT, "阶段1-数据集")


def scan_wheat3dgs():
    """Scan Wheat3DGS plots."""
    rows = []
    dataset_dir = os.path.join(DATA_ROOT, "Wheat3DGS", "dataset")
    for plot_name in sorted(os.listdir(dataset_dir)):
        plot_dir = os.path.join(dataset_dir, plot_name)
        if not os.path.isdir(plot_dir):
            continue

        # Count images
        img_dir = os.path.join(plot_dir, "images")
        n_rgb = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0

        # Check COLMAP
        colmap_path = os.path.join(plot_dir, "sparse", "0", "images.txt")
        ref_available = os.path.exists(colmap_path)

        # Check masks
        mask_dir = os.path.join(plot_dir, "masks")
        mask_available = os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0

        # Check VGGT output
        vggt_dir = os.path.join(VGGT_OUT, "wheat3dgs", f"wheat3dgs__{plot_name}")
        vggt_available = os.path.exists(os.path.join(vggt_dir, "depth_vggt.npy"))

        # Parse plot number for plant_id
        plot_num = plot_name.replace("plot_", "")

        rows.append({
            "plant_id": f"wheat_{plot_num}",
            "date": "2023-06-30",
            "growth_week": "unknown",
            "dataset": "Wheat3DGS",
            "sequence_id": f"wheat3dgs__{plot_name}",
            "rgb_available": True,
            "n_rgb": n_rgb,
            "reference_camera_available": ref_available,
            "plant_mask_available": mask_available,
            "depth_available": vggt_available,
            "scanner_gt_available": False,
            "canopy_density_label": "TBD",
            "background_fraction": -1,
            "occlusion_label": "unknown",
            "selected_for_pose_benchmark": True,
            "reason": "independent plant, COLMAP poses, multi-view",
        })
    return rows


def scan_langdon4():
    """Scan langdon_4 multi-date sequences."""
    rows = []
    plant_dir = os.path.join(SEQ_BASE, "plant_view")
    for seq_name in sorted(os.listdir(plant_dir)):
        if not seq_name.startswith("langdon_4__"):
            continue
        if not os.path.isdir(os.path.join(plant_dir, seq_name)):
            continue

        # Load sequence JSON
        json_path = os.path.join(plant_dir, f"{seq_name}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            meta = json.load(f)

        n_rgb = len(meta.get("rgb_paths", []))
        date_str = seq_name.replace("langdon_4__", "")

        # Check VGGT output
        vggt_dir = os.path.join(VGGT_OUT, "v2_clean_rerun", f"plantview__{seq_name}")
        vggt_available = os.path.exists(os.path.join(vggt_dir, "depth_vggt.npy"))

        # Check masks
        mask_dir = meta.get("extra", {}).get("mask_dir", "")
        mask_available = os.path.exists(mask_dir) if mask_dir else False

        rows.append({
            "plant_id": "langdon_4",
            "date": date_str,
            "growth_week": "unknown",
            "dataset": "3DPlantView",
            "sequence_id": f"plantview__langdon_4__{date_str}",
            "rgb_available": True,
            "n_rgb": n_rgb,
            "reference_camera_available": True,
            "plant_mask_available": mask_available,
            "depth_available": vggt_available,
            "scanner_gt_available": False,
            "canopy_density_label": "TBD",
            "background_fraction": -1,
            "occlusion_label": "unknown",
            "selected_for_pose_benchmark": True,
            "reason": "longitudinal multi-date, known failures",
        })
    return rows


def scan_mustc():
    """Scan MuST-C sequences."""
    rows = []
    vggt_mustc = os.path.join(VGGT_OUT, "mustc")
    if not os.path.exists(vggt_mustc):
        return rows

    for seq_name in sorted(os.listdir(vggt_mustc)):
        seq_dir = os.path.join(vggt_mustc, seq_name)
        if not os.path.isdir(seq_dir):
            continue

        vggt_available = os.path.exists(os.path.join(seq_dir, "depth_vggt.npy"))

        # Parse sequence name: mustc__plot198__230613__ugv__pos00
        parts = seq_name.split("__")
        plot_id = parts[1] if len(parts) > 1 else "unknown"
        date_part = parts[2] if len(parts) > 2 else "unknown"

        rows.append({
            "plant_id": f"mustc_{plot_id}",
            "date": date_part,
            "growth_week": "unknown",
            "dataset": "MuST-C",
            "sequence_id": seq_name,
            "rgb_available": True,
            "n_rgb": 36,
            "reference_camera_available": True,
            "plant_mask_available": False,
            "depth_available": vggt_available,
            "scanner_gt_available": False,
            "canopy_density_label": "TBD",
            "background_fraction": -1,
            "occlusion_label": "unknown",
            "selected_for_pose_benchmark": True,
            "reason": "independent plant, different crop type",
        })
    return rows


def main():
    os.makedirs(PHASE3B, exist_ok=True)

    all_rows = []
    all_rows.extend(scan_wheat3dgs())
    all_rows.extend(scan_langdon4())
    all_rows.extend(scan_mustc())

    print(f"Total sequences found: {len(all_rows)}")

    # Count unique plants
    plants = set(r["plant_id"] for r in all_rows)
    print(f"Unique plant_ids: {len(plants)} — {sorted(plants)}")

    # Count per dataset
    by_dataset = {}
    for r in all_rows:
        ds = r["dataset"]
        by_dataset[ds] = by_dataset.get(ds, 0) + 1
    for ds, count in sorted(by_dataset.items()):
        print(f"  {ds}: {count} sequences")

    # Write CSV
    csv_path = os.path.join(PHASE3B, "01_dataset_inventory", "MULTIPLANT_POSE_DATASET_INVENTORY.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {csv_path}")

    # Summary JSON
    summary = {
        "total_sequences": len(all_rows),
        "n_unique_plants": len(plants),
        "plant_ids": sorted(plants),
        "by_dataset": by_dataset,
        "sequences": [{
            "sequence_id": r["sequence_id"],
            "plant_id": r["plant_id"],
            "date": r["date"],
            "n_rgb": r["n_rgb"],
            "dataset": r["dataset"],
        } for r in all_rows],
    }
    summary_path = os.path.join(PHASE3B, "01_dataset_inventory", "DATASET_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
