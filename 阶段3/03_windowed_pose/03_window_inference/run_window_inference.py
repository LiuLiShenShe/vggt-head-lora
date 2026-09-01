#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C P1: Windowed VGGT Inference.

Runs independent VGGT forward on overlapping windows covering full trajectory.

Window layout (window_size=W, overlap=O):
  W0: frames [0, W)
  W1: frames [W-O, 2W-O)
  W2: frames [2(W-O), 3W-O)
  ...

Usage:
    conda activate vggt_lora
    python 03_window_inference/run_window_inference.py [--force] [--window-size 16] [--overlap 8]
"""
import argparse, csv, glob, json, os, sys, time, hashlib
import numpy as np
import torch

ROOT = "/fj/VGGT+head+lora实验"
VGGT_ROOT = os.path.join(ROOT, "vggt")
sys.path.insert(0, VGGT_ROOT)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
OUT_BASE = os.path.join(PHASE3C, "03_window_inference", "window_outputs")

# Sequences
FAIL_DATES = ["12-03-24", "15-04-24", "19-03-24"]
PASS_DATES = ["05-03-24"]
WHEAT3DGS = ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
MUSTC = ["mustc__plot198__230613__ugv__pos00"]

ALL_SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in FAIL_DATES + PASS_DATES]
    + WHEAT3DGS + MUSTC
)


def find_sequence_json(seq_id):
    for subdir in ["plant_view", "wheat3dgs", "mustc"]:
        for jp in glob.glob(os.path.join(SEQ_BASE, subdir, "*.json")):
            with open(jp) as f:
                meta = json.load(f)
            if meta["sequence_id"] == seq_id:
                return meta
    raise FileNotFoundError(f"Sequence JSON not found for {seq_id}")


def compute_windows(S, window_size, overlap):
    """Compute window frame index arrays covering [0, S)."""
    step = window_size - overlap
    windows = []
    start = 0
    while start < S:
        end = min(start + window_size, S)
        windows.append(list(range(start, end)))
        if end >= S:
            break
        start += step
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--seq", nargs="*", help="Specific sequences only")
    args = ap.parse_args()

    os.makedirs(OUT_BASE, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16

    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    print("Model loaded.")

    sequences = args.seq if args.seq else ALL_SEQUENCES
    manifest_all = {}

    for seq_id in sequences:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")
        print(f"{'='*60}")

        try:
            seq = find_sequence_json(seq_id)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        rgb_paths = seq["rgb_paths"]
        S = len(rgb_paths)
        windows = compute_windows(S, args.window_size, args.overlap)
        print(f"  Total frames: {S}, windows: {len(windows)} "
              f"(size={args.window_size}, overlap={args.overlap})")

        seq_dir = os.path.join(OUT_BASE, seq_id)
        os.makedirs(seq_dir, exist_ok=True)

        window_manifest = {
            "sequence_id": seq_id,
            "total_frames": S,
            "window_size": args.window_size,
            "overlap": args.overlap,
            "n_windows": len(windows),
            "windows": [],
        }

        total_time = 0
        peak_vram = 0

        for wi, frame_indices in enumerate(windows):
            out_path = os.path.join(seq_dir, f"window_{wi:03d}.npz")
            overlap_prev = max(0, frame_indices[0] - (windows[wi-1][-1] if wi > 0 else -1) + 1)

            entry = {
                "window_id": wi,
                "original_frame_indices": frame_indices,
                "n_frames": len(frame_indices),
                "overlap_with_previous": overlap_prev,
            }

            if os.path.exists(out_path) and not args.force:
                data = np.load(out_path)
                entry["output_hash"] = hashlib.md5(open(out_path, "rb").read()).hexdigest()[:12]
                entry["runtime_s"] = float(data.get("runtime_s", 0))
                window_manifest["windows"].append(entry)
                print(f"  W{wi:3d}: frames {frame_indices[0]:4d}-{frame_indices[-1]:4d} "
                      f"(cached) overlap_prev={overlap_prev}")
                continue

            rgb = [rgb_paths[i] for i in frame_indices]
            images = load_and_preprocess_images(rgb, mode="crop").to(device)
            H, W = images.shape[-2:]
            torch.manual_seed(42)

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                tokens, ps_idx = model.aggregator(images.unsqueeze(0))
                pose_enc = model.camera_head(tokens)[-1]
                ext_w2c, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
                depth, dconf = model.depth_head(tokens, images.unsqueeze(0), ps_idx)
                pts3d, _ = model.point_head(tokens, images.unsqueeze(0), ps_idx)
            dt = time.time() - t0
            vram = torch.cuda.max_memory_allocated() / 1e9
            total_time += dt
            peak_vram = max(peak_vram, vram)

            ext_w2c_np = ext_w2c.squeeze(0).float().cpu().numpy()
            intr_np = intr.squeeze(0).float().cpu().numpy()
            depth_np = depth.squeeze(0).squeeze(-1).float().cpu().numpy()
            pts3d_np = pts3d.squeeze(0).squeeze(-2).float().cpu().numpy()

            np.savez_compressed(out_path,
                ext_w2c_vggt=ext_w2c_np,
                intr_vggt=intr_np,
                depth_vggt=depth_np,
                point_map=pts3d_np,
                frame_idx=np.array(frame_indices),
                runtime_s=dt,
                peak_vram_gb=vram,
            )

            entry["runtime_s"] = dt
            entry["peak_vram_gb"] = vram
            entry["output_hash"] = hashlib.md5(open(out_path, "rb").read()).hexdigest()[:12]
            window_manifest["windows"].append(entry)

            print(f"  W{wi:3d}: frames {frame_indices[0]:4d}-{frame_indices[-1]:4d} "
                  f"({dt:.1f}s, {vram:.1f}GB) overlap_prev={overlap_prev}")

        window_manifest["total_runtime_s"] = total_time
        window_manifest["peak_vram_gb"] = peak_vram

        # Save per-sequence manifest
        manifest_path = os.path.join(seq_dir, "WINDOW_RUN_MANIFEST.json")
        with open(manifest_path, "w") as f:
            json.dump(window_manifest, f, indent=2)

        manifest_all[seq_id] = window_manifest
        print(f"  Total: {total_time:.1f}s, peak VRAM: {peak_vram:.1f}GB")

    # Save global manifest
    global_manifest_path = os.path.join(OUT_BASE, "ALL_WINDOW_MANIFESTS.json")
    with open(global_manifest_path, "w") as f:
        json.dump(manifest_all, f, indent=2, default=str)
    print(f"\nSaved global manifest: {global_manifest_path}")


if __name__ == "__main__":
    main()
