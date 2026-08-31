#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: UniDepthV2 calibrated-K inference pilot (20 frames × 4 sequences).

Uses the official UniDepthV2 API with calibrated intrinsics:
    camera = Pinhole(K=K_calibrated)
    result = model.infer(rgb_tensor, camera)

This feeds rays_gt to the decoder (decoder.py:400), overriding predicted rays.

Usage (must run in unidepth env):
  /home/test/miniconda3/envs/unidepth/bin/python run_unidepth_calibrated_k.py
"""
import os, sys, json, time
import numpy as np
import torch
from PIL import Image

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
AUDIT_DIR = os.path.join(PHASE3_DIR, "phase3a2_closure")
sys.path.insert(0, PHASE3_DIR)
from configs import load_sequence_meta, SEQUENCES

# ── Calibrated intrinsics ─────────────────────────────────────────────────
CALIBRATED_FX = 1371.82
CALIBRATED_FY = 1370.79
CALIBRATED_CX = 540.0
CALIBRATED_CY = 540.0

# Sample 20 frames per sequence: every 16th frame
SAMPLE_INTERVAL = 16
MAX_FRAMES = 20


def main():
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    # Build K tensor
    K_calibrated = torch.tensor([
        [CALIBRATED_FX, 0, CALIBRATED_CX],
        [0, CALIBRATED_FY, CALIBRATED_CY],
        [0, 0, 1],
    ], dtype=torch.float32)
    print(f"Calibrated K:\n{K_calibrated}")
    print(f"  fx={CALIBRATED_FX:.2f} fy={CALIBRATED_FY:.2f} cx={CALIBRATED_CX:.2f} cy={CALIBRATED_CY:.2f}")
    print()

    # Load model
    print("Loading UniDepthV2 model...")
    t0 = time.time()
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to("cuda").eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    all_results = {}

    for seq_id, pose_fail in SEQUENCES:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id} (pose_fail={pose_fail})")
        print(f"{'='*60}")

        meta = load_sequence_meta(seq_id)
        rgb_paths = meta["rgb_paths"]
        n_frames = len(rgb_paths)

        # Sample frames
        sample_indices = list(range(0, min(n_frames, MAX_FRAMES * SAMPLE_INTERVAL), SAMPLE_INTERVAL))[:MAX_FRAMES]
        print(f"  Sampling {len(sample_indices)} frames from {n_frames} total")

        depths_calK = []
        depths_auto = []
        intrinsics_calK = []
        failed = []

        for i, idx in enumerate(sample_indices):
            rp = rgb_paths[idx]
            try:
                rgb = np.array(Image.open(rp))
                rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                # Autonomous inference (no K)
                with torch.no_grad():
                    result_auto = model.infer(rgb_tensor.cuda())
                d_auto = result_auto["depth"].squeeze().cpu().numpy().astype(np.float32)
                depths_auto.append(d_auto)

                # Calibrated-K inference
                camera_calK = Pinhole(K=K_calibrated)
                with torch.no_grad():
                    result_calK = model.infer(rgb_tensor.cuda(), camera_calK)
                d_calK = result_calK["depth"].squeeze().cpu().numpy().astype(np.float32)
                depths_calK.append(d_calK)

                # Save intrinsics from calK run
                if "intrinsics" in result_calK:
                    ix = result_calK["intrinsics"].squeeze().cpu().numpy().astype(np.float32)
                    intrinsics_calK.append(ix)

            except Exception as e:
                failed.append((idx, str(e)))
                if depths_auto:
                    depths_auto.append(np.zeros_like(depths_auto[0]))
                    depths_calK.append(np.zeros_like(depths_calK[0]))

            if (i + 1) % 5 == 0 or i == len(sample_indices) - 1:
                status = "FAIL" if failed and failed[-1][0] == idx else "OK"
                print(f"  [{i+1}/{len(sample_indices)}] frame {idx} {status}")

        if depths_auto:
            auto_stack = np.stack(depths_auto, axis=0)
            calK_stack = np.stack(depths_calK, axis=0)

            # Save
            out_dir = os.path.join(PHASE3_DIR, "unidepth_v2", seq_id)
            auto_path = os.path.join(out_dir, "depth_unidepth_calK_pilot.npy")
            np.save(auto_path, calK_stack)
            print(f"\n  Saved calibrated-K depth: {auto_path} shape={calK_stack.shape}")

            if intrinsics_calK:
                ix_stack = np.stack(intrinsics_calK, axis=0)
                ix_path = os.path.join(out_dir, "intrinsics_unidepth_calK_pilot.npy")
                np.save(ix_path, ix_stack)
                print(f"  Saved calK intrinsics: {ix_path} shape={ix_stack.shape}")

            # Quick comparison
            auto_medians = [np.median(d[d > 0]) for d in auto_stack if d.max() > 0]
            calK_medians = [np.median(d[d > 0]) for d in calK_stack if d.max() > 0]
            print(f"  Autonomous depth median: {np.mean(auto_medians):.4f}")
            print(f"  Calibrated-K depth median: {np.mean(calK_medians):.4f}")
            print(f"  Ratio (calK/auto): {np.mean(calK_medians)/np.mean(auto_medians):.4f}")

            all_results[seq_id] = {
                "n_frames": len(sample_indices),
                "n_failed": len(failed),
                "auto_depth_median": float(np.mean(auto_medians)),
                "calK_depth_median": float(np.mean(calK_medians)),
                "ratio": float(np.mean(calK_medians) / np.mean(auto_medians)),
            }

    # Save manifest
    manifest_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_PILOT_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
