#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3 Step 5: Global camera generation.

For each window k, each local camera:
    R_c2w_global = G_k @ R_c2w_local
No sequential Q multiplication. Overlap-frame ownership: central-window preference.

Methods (4):
  A. STRIDE8_SEQUENTIAL_CHAIN  — stride-8 windows, G from sequential Q chain
  B. STRIDE8_SO3_GRAPH         — stride-8 windows, G from SO(3) solver (negative control, cr=0)
  C. STRIDE4_SEQUENTIAL_CHAIN  — stride-4 windows, G from sequential Q chain
  D. STRIDE4_SO3_GRAPH         — stride-4 windows, G from SO(3) solver (headline)

Translation/scale: frozen center-chain baseline is NOT touched here — this script
produces orientation-only global cameras (R_c2w per frame + original frame index).
Center trajectory is secondary (Phase 3C.2 retained).

Outputs (npz with R_c2w_global (n,3,3), original_frame_index (n,)):
  05_global_stitching/{seq}_METHOD{A,B,C,D}_GLOBAL_CAMERAS.npz
  STRIDE4_SO3GRAPH_GLOBAL_CAMERAS.npz (alias)  — headline

Usage:
    python 05_global_stitching/build_global_cameras.py [--seq ...]
"""
import argparse, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
SOLVER_DIR = os.path.join(SYNC_DIR, "04_so3_sync")
OUT_DIR = os.path.join(SYNC_DIR, "05_global_stitching")
STRIDE4_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")
STRIDE8_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PHASE3C, "05_global_stitching_v31"))
from run_gauge_stitching import c2w_rotations

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)


def load_windows(base, seq_id):
    seq_dir = os.path.join(base, seq_id)
    files = sorted(glob.glob(os.path.join(seq_dir, "window_*.npz")))
    windows = []
    for wf in files:
        d = np.load(wf)
        windows.append({"ext_w2c": d["ext_w2c_vggt"], "frame_idx": d["frame_idx"]})
    return windows


def central_owner(n_windows, wid):
    """Central-window preference: for stride-4, window k central index = (k+...)."""
    # Central window index = the window whose interval midpoint is closest to frame's midpoint.
    return wid


def build_global_rotations(windows, G, method_name, seq_id):
    """Apply G_k @ R_c2w_local to each frame, with central-window ownership.

    Ownership: frame f assigned to window k whose center frame is closest to f.
    """
    # Precompute window intervals
    intervals = []
    for w in windows:
        idx = np.asarray(w["frame_idx"], dtype=int)
        intervals.append((int(idx.min()), int(idx.max())))

    # Ownership: for each frame, choose the window whose midpoint is closest to the frame.
    owners = {}
    for k, w in enumerate(windows):
        mid = (intervals[k][0] + intervals[k][1]) / 2.0
        R_local = c2w_rotations(w["ext_w2c"])
        for i, f in enumerate(np.asarray(w["frame_idx"], dtype=int)):
            if f not in owners:
                owners[f] = (k, abs(f - mid))
            else:
                if abs(f - mid) < owners[f][1]:
                    owners[f] = (k, abs(f - mid))

    final_R = {}
    for k, w in enumerate(windows):
        R_local = c2w_rotations(w["ext_w2c"])
        for i, f in enumerate(np.asarray(w["frame_idx"], dtype=int)):
            if owners.get(f, (None,))[0] == k:
                final_R[f] = G[k] @ R_local[i]

    sorted_f = sorted(final_R.keys())
    R_global = np.array([final_R[f] for f in sorted_f])
    idx = np.array(sorted_f)
    return R_global, idx


def sequential_chain_G(windows):
    """G_{k+1} = G_k @ Q_{k,k+1} using adjacent overlap Q (robust mean)."""
    n = len(windows)
    G = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    from run_gauge_stitching import so3_robust_mean
    for k in range(n - 1):
        frames_a = set(np.asarray(windows[k]["frame_idx"], dtype=int))
        frames_b = set(np.asarray(windows[k + 1]["frame_idx"], dtype=int))
        overlap = sorted(frames_a & frames_b)
        if len(overlap) < 2:
            G[k + 1] = G[k]
            continue
        Ra = c2w_rotations(windows[k]["ext_w2c"])
        Rb = c2w_rotations(windows[k + 1]["ext_w2c"])
        f2a = {int(f): i for i, f in enumerate(np.asarray(windows[k]["frame_idx"], dtype=int))}
        f2b = {int(f): i for i, f in enumerate(np.asarray(windows[k + 1]["frame_idx"], dtype=int))}
        Q_list = [Ra[f2a[f]] @ Rb[f2b[f]].T for f in overlap]
        Q_star, _, _ = so3_robust_mean(np.array(Q_list))
        G[k + 1] = G[k] @ Q_star
    return G


def load_solver_G(seq_id, dataset, threshold):
    tag = f"{dataset}_th{threshold}"
    p = os.path.join(SOLVER_DIR, f"{seq_id}_SO3_SYNC_GAUGES_{tag}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["G"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()
    sequences = args.seq if args.seq else SEQUENCES

    for seq_id in sequences:
        print(f"\n=== {seq_id} ===")
        w8 = load_windows(STRIDE8_DIR, seq_id)
        w4 = load_windows(STRIDE4_DIR, seq_id)
        if not w8 or not w4:
            print("  windows missing — skip")
            continue

        # A: stride-8 chain
        G_a = sequential_chain_G(w8)
        R_a, idx_a = build_global_rotations(w8, G_a, "A", seq_id)
        np.savez(os.path.join(OUT_DIR, f"{seq_id}_METHOD_A_STRIDE8_CHAIN_GLOBAL_CAMERAS.npz"),
                 R_c2w_global=R_a, original_frame_index=idx_a, method="A_STRIDE8_SEQUENTIAL_CHAIN")
        print(f"  A stride8 chain: frames={len(idx_a)}")

        # B: stride-8 SO3 graph (negative control)
        G_b = load_solver_G(seq_id, "stride8", 8)
        if G_b is not None:
            R_b, idx_b = build_global_rotations(w8, G_b, "B", seq_id)
            np.savez(os.path.join(OUT_DIR, f"{seq_id}_METHOD_B_STRIDE8_SO3GRAPH_GLOBAL_CAMERAS.npz"),
                     R_c2w_global=R_b, original_frame_index=idx_b, method="B_STRIDE8_SO3_GRAPH")
            print(f"  B stride8 so3graph: frames={len(idx_b)}")
        else:
            print("  B stride8 so3graph: gauges not found — run solver --dataset stride8 first")

        # C: stride-4 chain
        G_c = sequential_chain_G(w4)
        R_c, idx_c = build_global_rotations(w4, G_c, "C", seq_id)
        np.savez(os.path.join(OUT_DIR, f"{seq_id}_METHOD_C_STRIDE4_CHAIN_GLOBAL_CAMERAS.npz"),
                 R_c2w_global=R_c, original_frame_index=idx_c, method="C_STRIDE4_SEQUENTIAL_CHAIN")
        print(f"  C stride4 chain: frames={len(idx_c)}")

        # D: stride-4 SO3 graph (headline)
        G_d = load_solver_G(seq_id, "stride4", 8)
        if G_d is not None:
            R_d, idx_d = build_global_rotations(w4, G_d, "D", seq_id)
            np.savez(os.path.join(OUT_DIR, f"{seq_id}_METHOD_D_STRIDE4_SO3GRAPH_GLOBAL_CAMERAS.npz"),
                     R_c2w_global=R_d, original_frame_index=idx_d, method="D_STRIDE4_SO3_GRAPH")
            np.savez(os.path.join(OUT_DIR, f"{seq_id}_STRIDE4_SO3GRAPH_GLOBAL_CAMERAS.npz"),
                     R_c2w_global=R_d, original_frame_index=idx_d, method="D_STRIDE4_SO3_GRAPH")
            print(f"  D stride4 so3graph: frames={len(idx_d)}")
        else:
            print("  D stride4 so3graph: gauges not found")


if __name__ == "__main__":
    main()