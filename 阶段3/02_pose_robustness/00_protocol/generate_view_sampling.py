#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B Step 3: Generate view sampling manifest.

For each sequence, creates nested view sets: 24 ⊃ 16 ⊃ 8.
Uniform sampling from full sequence.
"""
import os, sys, json
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")

VIEW_COUNTS = [8, 16, 24]


def load_sequence_rgb(seq_id):
    """Load RGB paths for a sequence."""
    json_name = seq_id
    if json_name.startswith("plantview__"):
        json_name = json_name[len("plantview__"):]

    # Try plant_view
    json_path = os.path.join(SEQ_BASE, "plant_view", f"{json_name}.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        return d.get("rgb_paths", [])

    # Try wheat3dgs
    wheat_name = json_name.replace("wheat3dgs__", "")
    json_path = os.path.join(SEQ_BASE, "wheat3dgs", f"{wheat_name}.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        return d.get("rgb_paths", [])

    return []


def generate_nested_views(n_full, max_views=24):
    """Generate nested uniform view sets: 24 ⊃ 16 ⊃ 8."""
    result = {}
    # Start from max_views as母集合
    n_sample = min(n_full, max_views)
    all_indices = np.linspace(0, n_full - 1, n_sample, dtype=int).tolist()
    result[str(max_views)] = all_indices

    # Subsample for 16 and 8
    for vc in [16, 8]:
        if vc <= n_sample:
            # Uniform sub-selection from the max_views set
            step = len(all_indices) / vc
            sub_indices = [all_indices[int(i * step)] for i in range(vc)]
            result[str(vc)] = sub_indices
        else:
            result[str(vc)] = all_indices[:vc]

    return result


def main():
    inventory_path = os.path.join(PHASE3B, "01_dataset_inventory", "MULTIPLANT_POSE_DATASET_INVENTORY.csv")
    import csv
    with open(inventory_path) as f:
        reader = csv.DictReader(f)
        inventory = list(reader)

    manifest = {}

    for row in inventory:
        seq_id = row["sequence_id"]
        rgb_paths = load_sequence_rgb(seq_id)
        n_full = len(rgb_paths)

        if n_full == 0:
            print(f"  SKIP {seq_id}: no RGB paths")
            continue

        view_sets = generate_nested_views(n_full)
        filenames = [os.path.basename(p) for p in rgb_paths]

        manifest[seq_id] = {
            "full_n": n_full,
            "frame_order_source": "COLMAP images.txt order" if "wheat" in seq_id else "sequence JSON order",
            "filenames": filenames,
        }
        for vc_str, indices in view_sets.items():
            manifest[seq_id][vc_str] = indices
            manifest[seq_id][f"{vc_str}_filenames"] = [filenames[i] for i in indices if i < len(filenames)]

        print(f"  {seq_id}: {n_full} full, 24={len(view_sets.get('24', []))}, "
              f"16={len(view_sets.get('16', []))}, 8={len(view_sets.get('8', []))}")

    # Save manifest
    os.makedirs(os.path.join(PHASE3B, "00_protocol"), exist_ok=True)
    manifest_path = os.path.join(PHASE3B, "00_protocol", "VIEW_SAMPLING_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {manifest_path}")


if __name__ == "__main__":
    main()
