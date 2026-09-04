#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3 Step 1: Stride-4 VGGT inference.

Runs real VGGT forward on stride-4 windows (window_size=16, step=4, overlap=12).

Key optimization (scientifically sound): stride-4 window k covers [4k, 4k+16).
Even-indexed window 2m covers [8m, 8m+16) == stride-8 window m — the SAME frame
indices and therefore the SAME real VGGT output already computed in Phase 3C.
These are reused verbatim (byte-identical npz), NOT re-run and NOT interpolated.
This also makes Methods A (stride-8) and C (stride-4) share identical window data
for overlapping windows, eliminating run-to-run nondeterminism as a confound.

Odd-indexed windows (start 8m+4) are genuinely new — real VGGT forward.

Everything is provenance-tracked in STRIDE4_WINDOW_MANIFEST.json (source per window).

Usage (GPU):
    conda activate vggt_lora
    python 01_stride4_inference/run_stride4_inference.py [--seq ...] [--force]

Usage (dry-run, no GPU — reports reuse vs new counts):
    python 01_stride4_inference/run_stride4_inference.py --dry-run
"""
import argparse, hashlib, json, os, time
import numpy as np
import torch

import sys
ROOT = "/fj/VGGT+head+lora实验"
VGGT_ROOT = os.path.join(ROOT, "vggt")
sys.path.insert(0, VGGT_ROOT)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
STRIDE8_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
OUT_BASE = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")

WINDOW_SIZE = 16
STRIDE = 4  # window step
OVERLAP = WINDOW_SIZE - STRIDE  # = 12, passed to compute_windows(S, size, overlap)

sys.path.insert(0, os.path.join(PHASE3C, "03_window_inference"))
from run_window_inference import find_sequence_json, compute_windows

FAIL_DATES = ["12-03-24", "15-04-24", "19-03-24"]
PASS_DATES = ["05-03-24"]
WHEAT3DGS = ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
MUSTC = ["mustc__plot198__230613__ugv__pos00"]
ALL_SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in FAIL_DATES + PASS_DATES]
    + WHEAT3DGS + MUSTC
)


def file_hash(path, n=12):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:n]


def find_stride8_match(stride8_windows, frame_indices):
    """Return stride-8 window id whose frame_idx == frame_indices (tuple compare)."""
    key = tuple(int(f) for f in frame_indices)
    for w8 in stride8_windows:
        if tuple(int(f) for f in w8["original_frame_indices"]) == key:
            return w8
    return None


def load_stride8_windows(seq_id):
    """Load stride-8 window manifest and return its windows list."""
    m_path = os.path.join(STRIDE8_DIR, seq_id, "WINDOW_RUN_MANIFEST.json")
    if not os.path.exists(m_path):
        # Fallback: infer from npz files
        return None
    with open(m_path) as f:
        return json.load(f)["windows"]


def dry_run():
    """No-GPU analysis: report window counts and reuse/new split."""
    print(f"{'Sequence':<45s} {'frames':>6s} {'windows':>7s} {'reuse':>5s} {'new':>5s}")
    totals = {"windows": 0, "reuse": 0, "new": 0}
    for seq_id in ALL_SEQUENCES:
        seq = find_sequence_json(seq_id)
        S = len(seq["rgb_paths"])
        windows = compute_windows(S, WINDOW_SIZE, OVERLAP)
        w8 = load_stride8_windows(seq_id) or []
        n_reuse = 0
        for w in windows:
            if find_stride8_match(w8, w):
                n_reuse += 1
        n_new = len(windows) - n_reuse
        totals["windows"] += len(windows)
        totals["reuse"] += n_reuse
        totals["new"] += n_new
        print(f"{seq_id:<45s} {S:6d} {len(windows):7d} {n_reuse:5d} {n_new:5d}")
    print(f"\nTOTAL windows={totals['windows']} reuse={totals['reuse']} new_inference={totals['new']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seq", nargs="*", help="Specific sequences only")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    os.makedirs(OUT_BASE, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16

    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    print("Model loaded.")

    sequences = args.seq if args.seq else ALL_SEQUENCES

    for seq_id in sequences:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")
        seq = find_sequence_json(seq_id)
        rgb_paths = seq["rgb_paths"]
        S = len(rgb_paths)

        windows = compute_windows(S, WINDOW_SIZE, OVERLAP)
        w8 = load_stride8_windows(seq_id) or []
        print(f"  Total frames: {S}, stride-4 windows: {len(windows)}")

        seq_dir = os.path.join(OUT_BASE, seq_id)
        os.makedirs(seq_dir, exist_ok=True)

        manifest = {
            "sequence_id": seq_id,
            "total_frames": S,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "overlap": WINDOW_SIZE - STRIDE,
            "n_windows": len(windows),
            "windows": [],
        }
        total_time = 0.0
        peak_vram = 0.0
        n_new = 0
        n_reuse = 0

        for wi, frame_indices in enumerate(windows):
            out_path = os.path.join(seq_dir, f"window_{wi:03d}.npz")
            entry = {
                "window_id": wi,
                "start_frame": int(frame_indices[0]),
                "original_frame_indices": list(frame_indices),
                "n_frames": len(frame_indices),
                "stride4_artifact": os.path.relpath(out_path, PHASE3C),
            }

            # 1) Reuse exact stride-8 match if available
            w8_match = find_stride8_match(w8, frame_indices)
            if w8_match is not None and not args.force:
                src = os.path.join(STRIDE8_DIR, seq_id, f"window_{w8_match['window_id']:03d}.npz")
                if os.path.exists(src):
                    os.makedirs(seq_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(src, out_path)
                    entry["source"] = "reuse_stride8"
                    entry["stride8_window_id"] = w8_match["window_id"]
                    entry["output_hash"] = file_hash(out_path)
                    entry["runtime_s"] = w8_match.get("runtime_s", 0)
                    n_reuse += 1
                    manifest["windows"].append(entry)
                    print(f"  W{wi:3d}: frames {frame_indices[0]:4d}-{frame_indices[-1]:4d} "
                          f"(reuse stride8 W{w8_match['window_id']})")
                    continue

            # 2) Reuse existing stride-4 artifact on disk
            if os.path.exists(out_path) and not args.force:
                entry["source"] = "new_inference_cached"
                entry["output_hash"] = file_hash(out_path)
                entry["runtime_s"] = float(np.load(out_path).get("runtime_s", 0))
                n_new += 1
                manifest["windows"].append(entry)
                print(f"  W{wi:3d}: frames {frame_indices[0]:4d}-{frame_indices[-1]:4d} (cached)")
                continue

            # 3) Real VGGT forward
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

            np.savez_compressed(
                out_path,
                ext_w2c_vggt=ext_w2c.squeeze(0).float().cpu().numpy(),
                intr_vggt=intr.squeeze(0).float().cpu().numpy(),
                depth_vggt=depth.squeeze(0).squeeze(-1).float().cpu().numpy(),
                point_map=pts3d.squeeze(0).squeeze(-2).float().cpu().numpy(),
                frame_idx=np.array(frame_indices),
                runtime_s=dt,
                peak_vram_gb=vram,
                stride4_window=wi,
            )
            entry["source"] = "new_inference"
            entry["runtime_s"] = dt
            entry["peak_vram_gb"] = vram
            entry["output_hash"] = file_hash(out_path)
            n_new += 1
            manifest["windows"].append(entry)
            print(f"  W{wi:3d}: frames {frame_indices[0]:4d}-{frame_indices[-1]:4d} "
                  f"({dt:.1f}s, {vram:.1f}GB) [NEW]")

        manifest["runtime_s"] = total_time
        manifest["peak_vram_gb"] = peak_vram
        manifest["n_reused_from_stride8"] = n_reuse
        manifest["n_new_inference"] = n_new

        with open(os.path.join(seq_dir, "WINDOW_RUN_MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  → {n_reuse} reused, {n_new} new; total {total_time:.1f}s")

    # Global stride-4 manifest at sync dir level
    all_manifest = {}
    for seq_id in sequences:
        p = os.path.join(OUT_BASE, seq_id, "WINDOW_RUN_MANIFEST.json")
        if os.path.exists(p):
            with open(p) as f:
                all_manifest[seq_id] = json.load(f)
    with open(os.path.join(SYNC_DIR, "01_stride4_inference", "STRIDE4_WINDOW_MANIFEST.json"), "w") as f:
        json.dump(all_manifest, f, indent=2, default=str)
    print(f"\nSaved manifest: {os.path.join(SYNC_DIR, '01_stride4_inference', 'STRIDE4_WINDOW_MANIFEST.json')}")


if __name__ == "__main__":
    main()