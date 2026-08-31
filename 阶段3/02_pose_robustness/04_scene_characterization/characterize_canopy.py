#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B Step 2: Canopy density characterization.

Computes per-sequence:
  - canopy_fraction (plant_mask_pixels / image_pixels)
  - background_fraction
  - edge_density (Sobel-based)

Then assigns LOW/MEDIUM/HIGH tertile labels.
"""
import os, sys, json, csv
import numpy as np
from PIL import Image

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
DATA_ROOT = os.path.join(ROOT, "阶段1-数据集")


def compute_canopy_metrics(mask_dir, rgb_paths, sample_n=8):
    """Compute canopy metrics for a sequence using sampled views.

    For instance masks (multiple small masks), union all mask files per view.
    For single plant masks, use directly.
    """
    if not mask_dir or not os.path.exists(mask_dir):
        return None

    # Detect mask type: list all mask files
    all_mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif'))]
    if not all_mask_files:
        return None

    # Sample frames
    n = len(rgb_paths)
    indices = np.linspace(0, n - 1, min(sample_n, n), dtype=int)

    canopy_fractions = []
    bg_fractions = []
    edge_densities = []

    for idx in indices:
        rp = rgb_paths[idx]
        basename = os.path.basename(rp)
        stem = os.path.splitext(basename)[0]

        # Find masks for this view: could be single or instance-level
        # Instance masks: stem_000.png, stem_001.png, ...
        # Single mask: stem.png
        view_masks = [f for f in all_mask_files if f.startswith(stem)]

        if not view_masks:
            continue

        # Load first mask to get image size
        first_mask = np.array(Image.open(os.path.join(mask_dir, view_masks[0])).convert("L"))
        h, w = first_mask.shape
        total = h * w

        # Union all instance masks for this view
        union_mask = np.zeros((h, w), dtype=bool)
        for mf in view_masks:
            m = np.array(Image.open(os.path.join(mask_dir, mf)).convert("L"))
            union_mask |= (m > 127)

        plant_pixels = union_mask.sum()
        canopy_frac = plant_pixels / total
        bg_frac = 1.0 - canopy_frac

        # Edge density
        from PIL import ImageFilter
        union_pil = Image.fromarray(union_mask.astype(np.uint8) * 255)
        edges = union_pil.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges)
        edge_dens = (edge_arr > 0).sum() / total

        canopy_fractions.append(canopy_frac)
        bg_fractions.append(bg_frac)
        edge_densities.append(edge_dens)

    if not canopy_fractions:
        return None

    return {
        "canopy_fraction_mean": float(np.mean(canopy_fractions)),
        "canopy_fraction_median": float(np.median(canopy_fractions)),
        "canopy_fraction_p90": float(np.percentile(canopy_fractions, 90)),
        "background_fraction_mean": float(np.mean(bg_fractions)),
        "background_fraction_median": float(np.median(bg_fractions)),
        "edge_density_mean": float(np.mean(edge_densities)),
        "n_sampled_views": len(canopy_fractions),
    }


def compute_canopy_from_rgb(rgb_paths, sample_n=8):
    """Estimate canopy from RGB using simple green-channel threshold (fallback)."""
    n = len(rgb_paths)
    indices = np.linspace(0, n - 1, min(sample_n, n), dtype=int)

    canopy_fractions = []
    for idx in indices:
        rp = rgb_paths[idx]
        if not os.path.exists(rp):
            continue
        rgb = np.array(Image.open(rp))
        # Simple green excess: G - (R+B)/2
        if rgb.ndim == 3 and rgb.shape[2] >= 3:
            g = rgb[:, :, 1].astype(float)
            r = rgb[:, :, 0].astype(float)
            b = rgb[:, :, 2].astype(float)
            green_excess = g - (r + b) / 2
            canopy = (green_excess > 20).sum() / green_excess.size
            canopy_fractions.append(canopy)

    if not canopy_fractions:
        return None
    return {
        "canopy_fraction_mean": float(np.mean(canopy_fractions)),
        "canopy_fraction_median": float(np.median(canopy_fractions)),
        "background_fraction_mean": float(1.0 - np.mean(canopy_fractions)),
    }


def load_sequence_paths(seq_id):
    """Load RGB paths for a sequence."""
    # Try plant_view first
    json_name = seq_id
    if json_name.startswith("plantview__"):
        json_name = json_name[len("plantview__"):]
    json_path = os.path.join(SEQ_BASE, "plant_view", f"{json_name}.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        return d.get("rgb_paths", []), d.get("extra", {}).get("mask_dir", "")

    # Try wheat3dgs
    json_path = os.path.join(SEQ_BASE, "wheat3dgs", f"{json_name.replace('wheat3dgs__', '')}.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        return d.get("rgb_paths", []), d.get("extra", {}).get("mask_dir", "")

    return [], ""


def main():
    inventory_path = os.path.join(PHASE3B, "01_dataset_inventory", "MULTIPLANT_POSE_DATASET_INVENTORY.csv")
    with open(inventory_path) as f:
        reader = csv.DictReader(f)
        inventory = list(reader)

    canopy_results = []

    for row in inventory:
        seq_id = row["sequence_id"]
        rgb_paths, mask_dir = load_sequence_paths(seq_id)

        if not rgb_paths:
            print(f"  SKIP {seq_id}: no RGB paths found")
            continue

        print(f"  {seq_id}: {len(rgb_paths)} frames, mask_dir={mask_dir}")

        # Try mask-based characterization
        metrics = None
        if mask_dir and os.path.exists(mask_dir):
            metrics = compute_canopy_metrics(mask_dir, rgb_paths)
            if metrics:
                print(f"    mask-based: canopy={metrics['canopy_fraction_mean']:.3f} "
                      f"bg={metrics['background_fraction_mean']:.3f}")

        # Fallback to RGB-based
        if metrics is None:
            metrics = compute_canopy_from_rgb(rgb_paths)
            if metrics:
                print(f"    rgb-based: canopy={metrics['canopy_fraction_mean']:.3f}")

        if metrics:
            canopy_results.append({
                "sequence_id": seq_id,
                "plant_id": row["plant_id"],
                "dataset": row["dataset"],
                **metrics,
            })
            # Update inventory
            row["canopy_density_label"] = "TBD"  # Will assign after tertiles
            row["background_fraction"] = str(metrics.get("background_fraction_mean", -1))

    # Assign tertile labels
    if canopy_results:
        canopy_vals = [r["canopy_fraction_mean"] for r in canopy_results]
        sorted_vals = sorted(canopy_vals)
        n = len(sorted_vals)
        t1 = sorted_vals[n // 3]
        t2 = sorted_vals[2 * n // 3]

        for r in canopy_results:
            cf = r["canopy_fraction_mean"]
            if cf <= t1:
                r["density_class"] = "LOW"
            elif cf <= t2:
                r["density_class"] = "MEDIUM"
            else:
                r["density_class"] = "HIGH"
            print(f"    {r['sequence_id']}: canopy={cf:.3f} → {r['density_class']}")

        # Save canopy characterization
        csv_path = os.path.join(PHASE3B, "04_scene_characterization", "CANOPY_CHARACTERIZATION.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        all_fields = ["sequence_id", "plant_id", "dataset", "canopy_fraction_mean",
                      "canopy_fraction_median", "canopy_fraction_p90",
                      "background_fraction_mean", "background_fraction_median",
                      "edge_density_mean", "n_sampled_views", "density_class"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(canopy_results)
        print(f"\nSaved: {csv_path}")

        # Update inventory with density labels
        density_map = {r["sequence_id"]: r["density_class"] for r in canopy_results}
        for row in inventory:
            if row["sequence_id"] in density_map:
                row["canopy_density_label"] = density_map[row["sequence_id"]]

        # Rewrite inventory
        inv_fields = ["plant_id", "date", "growth_week", "dataset", "sequence_id",
                      "rgb_available", "n_rgb", "reference_camera_available",
                      "plant_mask_available", "depth_available", "scanner_gt_available",
                      "canopy_density_label", "background_fraction", "occlusion_label",
                      "selected_for_pose_benchmark", "reason"]
        with open(inventory_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=inv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(inventory)
        print(f"Updated inventory: {inventory_path}")


if __name__ == "__main__":
    main()
