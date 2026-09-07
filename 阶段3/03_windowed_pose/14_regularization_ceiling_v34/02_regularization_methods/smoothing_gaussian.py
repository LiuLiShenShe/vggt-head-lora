#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Step 2a: Gaussian smoothing (increment domain) + 2nd-order probe.

Primary: smooth per-step increments δ_k = Log(G_k^T G_{k+1}) with a Gaussian
kernel over window index, re-integrate G_out[k+1] = G_out[k] @ Exp(δ̃_k).

Second-order probe: min_δ̃ Σ‖δ̃−δ‖² + λ Σ‖δ̃_{k+1}−2δ̃_k+δ̃_{k-1}‖² (closed-form
tridiagonal ridge). Constant-rate coherent bias has zero 2nd difference, so it
should SURVIVE this penalty — empiricizes the plan's key mathematical insight.

The Gaussian low-pass preserves DC (kernel has unit mass at k=0), so a
constant-rate drift component is NOT suppressed by smoothing — expected U-shaped
rot_median vs σ curve (drift stays high, then true motion is destroyed).

Outputs: REGULARIZED_GAUGES_<seq>_gauss_s{σ}.npz and _2nd_l{λ}.npz

Usage:
    python 02_regularization_methods/smoothing_gaussian.py [--seq ...]
"""
import argparse, json, os
import numpy as np
from scipy.linalg import solve_banded
from scipy.spatial.transform import Rotation

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

GAUSS_SIGMAS = [0.5, 1, 2, 3, 5, 8, 12, 16]
SECOND_ORDER_LAMS = [0.1, 1, 10, 100]
KERNEL_TRUNC = 4  # kernel support: ±4σ


# --------------------------------------------------------------------------
# SO(3) helpers
# --------------------------------------------------------------------------
def exp(rv):
    return Rotation.from_rotvec(rv).as_matrix()


def log(R):
    return Rotation.from_matrix(R).as_rotvec()


def increments(G):
    """Per-step rotvec increments δ_k = Log(G_k^T G_{k+1})  (N-1, 3)."""
    n = len(G)
    return np.array([log(G[k].T @ G[k + 1]) for k in range(n - 1)])


def integrate(G0, deltas):
    """G[0] = G0; G[k+1] = G[k] @ Exp(delta[k]) — exact SO(3) composition."""
    n = len(deltas) + 1
    G = np.zeros((n, 3, 3))
    G[0] = G0
    for k in range(n - 1):
        G[k + 1] = G[k] @ exp(deltas[k])
    return G


# --------------------------------------------------------------------------
# Method 1a: Gaussian smoothing of increments
# --------------------------------------------------------------------------
def gaussian_smooth_increments(delta, sigma, kernel_trunc=KERNEL_TRUNC):
    """Kernel w_i = exp(−i²/2σ²), normalized; boundaries truncated+renormalized."""
    n = len(delta)
    out = np.zeros_like(delta)
    half = int(np.ceil(kernel_trunc * sigma))
    for k in range(n):
        i0 = max(0, k - half)
        i1 = min(n, k + half + 1)
        idx = np.arange(i0, i1)
        w = np.exp(-((idx - k) ** 2) / (2.0 * sigma ** 2))
        w = w / w.sum()
        out[k] = np.einsum("i,ij->j", w, delta[idx])
    return out


def run_gaussian_smoothing(G_in, sigma):
    deltas = increments(G_in)
    delta_s = gaussian_smooth_increments(deltas, sigma)
    return integrate(G_in[0], delta_s)


# --------------------------------------------------------------------------
# Method 1b: second-order (acceleration) penalty — closed form
# --------------------------------------------------------------------------
def second_order_penalized_increments(delta, lam):
    """min_δ̃ Σ‖δ̃−δ‖² + λ Σ‖δ̃_{k+1}−2δ̃_k+δ̃_{k-1}‖²  — per-channel tridiagonal.

    (I + λ DᵀD) δ̃ = δ  per channel, D = 2nd-difference matrix (banded).
    Returns smoothed increments (N-1, 3). For N-1 < 3 (short control
    sequences) the 2nd-difference operator is empty → no-op.
    """
    n = len(delta)
    if n < 3:
        return delta.copy()
    # system A = I + λ DᵀD is pentadiagonal; solve per channel via banded
    # representation of A: A[k,k]=1+6λ (interior), A[k,k±1]=-4λ, A[k,k±2]=λ;
    # boundaries adjust for the (N-3) length of the 2nd-difference operator.
    diag = np.full(n, 1.0 + 6.0 * lam)
    diag[0] = 1.0 + lam
    diag[1] = 1.0 + 5.0 * lam
    diag[-2] = 1.0 + 5.0 * lam
    diag[-1] = 1.0 + lam
    off1 = np.full(n - 1, -4.0 * lam)
    off1[0] = -2.0 * lam
    off1[-1] = -2.0 * lam
    off2 = np.full(n - 2, lam)
    # banded storage: ab[(m + i - k, k)] = A[i,k], m = 2 (superdiagonals)
    ab = np.zeros((5, n))
    ab[2, :] = diag
    ab[1, 1:] = off1
    ab[3, :-1] = off1
    ab[0, 2:] = off2
    ab[4, :-2] = off2
    out = np.zeros_like(delta)
    for c in range(3):
        out[:, c] = solve_banded((2, 2), ab, delta[:, c])
    return out


def run_second_order_smoothing(G_in, lam):
    deltas = increments(G_in)
    delta_s = second_order_penalized_increments(deltas, lam)
    return integrate(G_in[0], delta_s)


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

    methods = []

    for seq_id in sequences:
        G_in = load_solver_gauges(seq_id)
        if G_in is None:
            print(f"  SKIP {seq_id}: no solver gauges")
            continue
        n = len(G_in)
        print(f"=== {seq_id} (N={n}) ===")

        for sigma in GAUSS_SIGMAS:
            G_out = run_gaussian_smoothing(G_in, sigma)
            npz = os.path.join(OUT_DIR, f"REGULARIZED_GAUGES_{seq_id}_gauss_s{sigma}.npz")
            np.savez(npz, G=G_out, method="gaussian", sigma=sigma, anchor_window=0,
                     input_gauge="so3_sync_stride4_th8", n_windows=n)
            methods.append({"sequence": seq_id, "method": "gaussian",
                            "sigma": sigma, "output": os.path.basename(npz)})

        for lam in SECOND_ORDER_LAMS:
            G_out = run_second_order_smoothing(G_in, lam)
            npz = os.path.join(OUT_DIR, f"REGULARIZED_GAUGES_{seq_id}_2nd_l{lam}.npz")
            np.savez(npz, G=G_out, method="second_order", lambda_=lam, anchor_window=0,
                     input_gauge="so3_sync_stride4_th8", n_windows=n)
            methods.append({"sequence": seq_id, "method": "second_order",
                            "lambda": lam, "output": os.path.basename(npz)})

    with open(os.path.join(OUT_DIR, "METHOD_SUMMARY.json"), "w") as f:
        json.dump({
            "gaussian_sigmas": GAUSS_SIGMAS,
            "second_order_lambdas": SECOND_ORDER_LAMS,
            "kernel_trunc_sigma": KERNEL_TRUNC,
            "domain": "increment (rotvec) + exact Exp integration",
            "gt_used": False,
            "runs": methods,
        }, f, indent=2)
    print(f"\nSaved METHOD_SUMMARY.json ({len(methods)} runs)")


if __name__ == "__main__":
    main()