#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B.1 Step 1: True view-count VGGT re-run.

Runs INDEPENDENT VGGT forward passes on 8/16/24-image subsets
(of Phase 3B subsampled from full-view predictions — that was invalid).

Pattern: four_path_infer.py (lines 38–78).
Output: npz per (seq_id, n) with ext_w2c, intrinsics, frame_idx.

Usage:
    conda activate vggt_lora
    python 02_pose_inference/rerun_viewcount.py [--force]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = "/fj/VGGT+head+lora实验"
VGGT_ROOT = os.path.join(ROOT, "vggt")
sys.path.insert(0, VGGT_ROOT)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
OUT_DIR = os.path.join(PHASE3B, "02_pose_inference", "viewcount_outputs")

# All langdon_4 dates
LANGDON4_DATES = ["05-03-24", "12-03-24", "13-02-24", "15-04-24", "19-03-24", "20-02-24"]

# Wheat3DGS controls (plots 461 and 467 — one low-canopy, one high-canopy)
WHEAT3DGS_CONTROLS = ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]

N_FRAMES = [8, 16, 24]


def find_sequence_json(seq_id):
    """Find the sequence JSON file for a given sequence_id."""
    for subdir in ["plant_view", "wheat3dgs", "mustc"]:
        pattern = os.path.join(SEQ_BASE, subdir, "*.json")
        for jp in glob.glob(pattern):
            with open(jp) as f:
                meta = json.load(f)
            if meta["sequence_id"] == seq_id:
                return meta
    raise FileNotFoundError(f"Sequence JSON not found for {seq_id}")


def run_single_inference(model, rgb_paths, device, dtype, frame_idx):
    """Run VGGT forward on a subset of images. Returns (ext_w2c, intr)."""
    images = load_and_preprocess_images(rgb_paths, mode="crop").to(device)
    H, W = images.shape[-2:]
    torch.manual_seed(42)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        tokens, ps_idx = model.aggregator(images.unsqueeze(0))
        pose_enc = model.camera_head(tokens)[-1]
        ext_w2c, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
    return (
        ext_w2c.squeeze(0).float().cpu().numpy(),  # (n, 3, 4)
        intr.squeeze(0).float().cpu().numpy(),       # (n, 3, 3)
        frame_idx,
    )


def main():
    ap = argparse.ArgumentParser(description="Phase 3B.1: True view-count VGGT re-run")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16

    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    print("Model loaded.")

    all_seqs = []

    # Langdon_4 sequences
    for date in LANGDON4_DATES:
        sid = f"plantview__langdon_4__{date}"
        all_seqs.append((sid, date))

    # Wheat3DGS controls
    for sid in WHEAT3DGS_CONTROLS:
        # Extract plot name for label
        plot = sid.split("__")[-1]
        all_seqs.append((sid, plot))

    total_runs = len(all_seqs) * len(N_FRAMES)
    completed = 0
    skipped = 0

    for seq_id, label in all_seqs:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id} ({label})")
        print(f"{'='*60}")

        try:
            seq = find_sequence_json(seq_id)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        rgb_paths = seq["rgb_paths"]
        print(f"  Total frames: {len(rgb_paths)}")

        for n in N_FRAMES:
            out_path = os.path.join(OUT_DIR, f"{seq_id}_n{n}.npz")

            if os.path.exists(out_path) and not args.force:
                print(f"  n={n}: EXISTS, skipping (use --force to overwrite)")
                skipped += 1
                completed += 1
                continue

            # Uniform frame selection (same as four_path_infer.py)
            idx = np.linspace(0, len(rgb_paths) - 1, n).astype(int)
            rgb = [rgb_paths[i] for i in idx]

            t0 = time.time()
            ext_w2c, intr, frame_idx = run_single_inference(model, rgb, device, dtype, idx)
            dt = time.time() - t0

            np.savez_compressed(
                out_path,
                ext_w2c_vggt=ext_w2c,
                intr_vggt=intr,
                frame_idx=frame_idx,
            )
            print(f"  n={n}: {dt:.1f}s -> {out_path}")
            completed += 1

    print(f"\n{'='*60}")
    print(f"DONE: {completed}/{total_runs} completed, {skipped} skipped")
    print(f"Output: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
