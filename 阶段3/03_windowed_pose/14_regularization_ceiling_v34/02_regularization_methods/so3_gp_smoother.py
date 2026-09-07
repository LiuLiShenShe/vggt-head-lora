#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Step 2b: SO(3) Gaussian-Process smoother.

Primary (GP on increments): per channel c ∈ {x,y,z}, independent GP over window
index t with SE kernel k(t,t') = σ_f² exp(−(t−t')²/2ℓ²); posterior mean
δ̃ = K(K + σ_n² I)⁻¹ δ_obs. Re-integrate via exact Exp composition.

Secondary (local-trend GP — the only genuinely bias-aware blind variant): fit a
constant-rate trend θ_k = k·Δ̄ from the FIRST n_trust windows only (where
accumulated drift is still small, a local "trusted" region — pure data, no GT),
then regress perturbations around that trend with the GP. Quantifies the best a
local drift-free anchoring can do without GT.

Gauge trajectory lives on SO(3); because increments are small (< 180°), the
Euclidean GP on increments is a valid linearization of the manifold structure.

Outputs: REGULARIZED_GAUGES_<seq>_gp_l{ℓ}.npz and _gptr_t{ℓ}.npz

Usage:
    python 02_regularization_methods/so3_gp_smoother.py [--seq ...]
"""
import argparse, json, os
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SOLVER_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33", "04_so3_sync")
OUT_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "02_regularization_methods")
os.makedirs(OUT_DIR, exist_ok=True)

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)

LENGTH_SCALES = [2, 5, 10, 20, 40]
N_TRUST = 5                # local trusted windows for the trend fit (drift ~2-4°)
SIGMA_N_DEG = 1.53         # per-step measurement noise (real langdon median)
FLOAT_EPS = 1e-10


# --------------------------------------------------------------------------
# SO(3) helpers
# --------------------------------------------------------------------------
def exp(rv):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rv).as_matrix()


def log(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def increments(G):
    return np.array([log(G[k].T @ G[k + 1]) for k in range(len(G) - 1)])


def integrate(G0, deltas):
    n = len(deltas) + 1
    G = np.zeros((n, 3, 3))
    G[0] = G0
    for k in range(n - 1):
        G[k + 1] = G[k] @ exp(deltas[k])
    return G


# --------------------------------------------------------------------------
# Kernel / GP
# --------------------------------------------------------------------------
def se_kernel(t, s, length_scale, sigma_f):
    d2 = (t - s) ** 2
    return sigma_f ** 2 * np.exp(-d2 / (2.0 * length_scale ** 2))


def gp_smooth_increments(delta, length_scale, sigma_f, sigma_n):
    """δ̃ = K(K + σ_n²I)⁻¹ δ, independent per channel (shared kernel)."""
    n = len(delta)
    t = np.arange(n, dtype=float)
    K = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            K[a, b] = se_kernel(t[a], t[b], length_scale, sigma_f)
    A = K + sigma_n ** 2 * np.eye(n)
    # small ridge for conditioning (plan failure-gate F2)
    cond = np.linalg.cond(A)
    if cond > 1e10:
        A = A + 1e-6 * np.trace(A) / n * np.eye(n)
    Kinv = np.linalg.solve(A, np.eye(n))
    return K @ Kinv @ delta  # (n,3)


def fit_local_trend(delta, n_trust=N_TRUST):
    """Mean rotvec of the first n_trust increments (local trusted drift-free trend)."""
    return delta[:n_trust].mean(axis=0)


def run_so3_gp(G_in, length_scale, mode="increments", sigma_n_deg=SIGMA_N_DEG):
    deltas = increments(G_in)
    sigma_f = float(np.std(deltas))
    sigma_n = np.radians(sigma_n_deg)
    if mode == "increments":
        delta_s = gp_smooth_increments(deltas, length_scale, sigma_f, sigma_n)
        return integrate(G_in[0], delta_s)
    elif mode == "local_trend":
        trend = fit_local_trend(deltas)
        # perturbations around the constant-rate trend k·trend
        pert = deltas - trend
        pert_s = gp_smooth_increments(pert, length_scale, sigma_f, sigma_n)
        return integrate(G_in[0], trend + pert_s)
    raise ValueError(f"unknown mode {mode}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def load_solver_gauges(seq_id):
    p = os.path.join(SOLVER_DIR, f"{seq_id}_SO3_SYNC_GAUGES_stride4_th8.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["G"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)

    runs = []
    for seq_id in sequences:
        G_in = load_solver_gauges(seq_id)
        if G_in is None:
            print(f"  SKIP {seq_id}: no solver gauges")
            continue
        n = len(G_in)
        print(f"=== {seq_id} (N={n}) ===")

        for length_scale in LENGTH_SCALES:
            G_out = run_so3_gp(G_in, length_scale, mode="increments")
            tag = f"gp_l{length_scale}"
            npz = os.path.join(OUT_DIR, f"REGULARIZED_GAUGES_{seq_id}_{tag}.npz")
            np.savez(npz, G=G_out, method="so3_gp", mode="increments",
                     length_scale=length_scale, anchor_window=0,
                     input_gauge="so3_sync_stride4_th8", n_windows=n)
            runs.append({"sequence": seq_id, "method": "so3_gp", "mode": "increments",
                         "length_scale": length_scale, "output": os.path.basename(npz)})

            G_out_t = run_so3_gp(G_in, length_scale, mode="local_trend")
            tag = f"gptr_t{length_scale}"
            npz = os.path.join(OUT_DIR, f"REGULARIZED_GAUGES_{seq_id}_{tag}.npz")
            np.savez(npz, G=G_out_t, method="so3_gp", mode="local_trend",
                     length_scale=length_scale, n_trust=N_TRUST, anchor_window=0,
                     input_gauge="so3_sync_stride4_th8", n_windows=n)
            runs.append({"sequence": seq_id, "method": "so3_gp", "mode": "local_trend",
                         "length_scale": length_scale, "output": os.path.basename(npz)})

    summary_path = os.path.join(OUT_DIR, "METHOD_SUMMARY.json")
    prev = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            prev = json.load(f)
    prev["so3_gp"] = {
        "length_scales": LENGTH_SCALES,
        "kernel": "SE, k(t,t') = σf² exp(−(t−t')²/2ℓ²)",
        "sigma_n_deg": SIGMA_N_DEG,
        "modes": ["increments", "local_trend (trend from first %d windows)" % N_TRUST],
        "gt_used": False,
        "runs": runs,
    }
    with open(summary_path, "w") as f:
        json.dump(prev, f, indent=2)
    print(f"\nUpdated METHOD_SUMMARY.json (so3_gp: {len(runs)} runs)")


if __name__ == "__main__":
    main()