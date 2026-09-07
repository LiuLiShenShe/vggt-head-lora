#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Step 1: Synthetic drift generator.

Builds a controlled testbed with the SAME drift statistics as langdon:
  77 windows, ~24°/step smooth true rotation, per-step Q errors with
  LogNormal magnitudes (median 1.53°, mean 2.50°) and tuneable temporal
  coherence (target C = 0.76) injected onto the measured increments.

Scenarios:
  A "CASE-C consistent"      — multi-hop edges are exactly the chain product
                               (no new corrective information; matches real
                               transitivity residual ≈ 0.37° ≈ 0).
  B "mildly inconsistent"    — multi-hop edges get independent Exp(η),
                               η ~ N(0, (0.37°)² I₃) noise (matches the real
                               hop-2 direct-vs-composed residual 0.37°; the
                               only structure averaging can exploit).
  C "axis-varying bias"      — bias axis u0 rotates 2°/window (robustness).

Edge topology + weights are copied VERBATIM from ROTATION_GRAPH_EDGES.csv so
graph degree sequence, weights and hop distribution match reality.

Known ground truth: G_true and injected rv are saved; drift is exactly
Log(G_chain[k]^T G_true[k]) per window.

Outputs (in 01_synthetic_drift_model/):
  SYNTHETIC_DRIFT_SUMMARY.json  — model params + realized-vs-target stats

Usage:
    python 01_synthetic_drift_model/synthetic_drift_generator.py [--seq ...]
        [--scenario A|B|C] [--threshold 8] [--seed 0]
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
EDGES_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
SOLVER_DIR = os.path.join(SYNC_DIR, "04_so3_sync")
GT_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331", "02_window_gauge_fit")
OUT_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "01_synthetic_drift_model")
os.makedirs(OUT_DIR, exist_ok=True)

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)
LANGDON_SEQ = SEQUENCES[0]  # calibration reference

DEFAULT_MAG_DEG = 24.58        # real GT per-step median
DEFAULT_TRANSITIVITY_DEG = 0.37  # real hop-2 direct-vs-composed median
AXIS_VARY_DEG_PER_WINDOW = 2.0   # scenario C
OVER_ROTATION_PROB = 0.96        # fraction of steps over-rotating along u0


# --------------------------------------------------------------------------
# SO(3) helpers
# --------------------------------------------------------------------------
def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def log(R):
    """Exact Log map: SO(3) -> rotvec (radians)."""
    return Rotation.from_matrix(R).as_rotvec()


def exp(rv):
    """Exact Exp map: rotvec (radians) -> SO(3)."""
    return Rotation.from_rotvec(rv).as_matrix()


def exponentiate_path(anchors, deltas):
    """Cumulative composition G[k+1] = G[k] @ Exp(delta[k]).

    anchors: (N,3,3) base gauges; deltas: (N-1,3) rotvec increments.
    Returns full path (N,3,3) with G[0] = anchors[0].
    """
    N = len(anchors)
    G = anchors.copy()
    for k in range(N - 1):
        G[k + 1] = G[k] @ exp(deltas[k])
    return G


# --------------------------------------------------------------------------
# Real-data statistics fitting
# --------------------------------------------------------------------------
def _load_solver_gauges(seq_id):
    p = os.path.join(SOLVER_DIR, f"{seq_id}_SO3_SYNC_GAUGES_stride4_th8.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["G"]


def _load_gt_gauges(seq_id):
    p = os.path.join(GT_DIR, f"CORRECTED_WINDOW_GAUGES_{seq_id}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["G"]


def _per_step_errors(G, G_true):
    """Per-step error rotvecs rv_k = Log(Q_meas^T Q_gt) between consecutive windows.

    Q_meas = G[k]^T G[k+1] (measured), Q_gt = G_true[k]^T G_true[k+1].
    Mirrors drift_analysis.py's step-error definition.
    """
    n = min(len(G), len(G_true)) - 1
    rv = np.zeros((n, 3))
    for k in range(n):
        Q_meas = G[k].T @ G[k + 1]
        Q_gt = G_true[k].T @ G_true[k + 1]
        rv[k] = log(Q_meas @ Q_gt.T)
    return rv


def fit_error_statistics(seq_id=None):
    """Extract {logmean, logstd, u0, coherence_target, over_rotation_prob} from real data.

    Uses the graph-sync gauges vs corrected GT gauges per-step errors
    (graph sync ≈ chain, CASE C). Aggregates over langdon sequences when
    seq_id is None for a stable calibration reference.
    """
    seqs = [seq_id] if seq_id else [f"plantview__langdon_4__{d}" for d in
                                    ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    all_rv = []
    for s in seqs:
        G = _load_solver_gauges(s)
        Gt = _load_gt_gauges(s)
        if G is None or Gt is None:
            continue
        n = min(len(G), len(Gt))
        all_rv.append(_per_step_errors(G[:n], Gt[:n]))
    if not all_rv:
        raise FileNotFoundError("No solver/GT gauges available for error-stat fit")
    rv = np.concatenate(all_rv)
    mags = np.linalg.norm(rv, axis=1)                       # radians
    mean_dir = rv.sum(axis=0)
    if np.linalg.norm(mean_dir) > 1e-12:
        u0 = mean_dir / np.linalg.norm(mean_dir)
    else:
        u0 = np.array([0.0, 0.0, 1.0])
    coherence = float(np.linalg.norm(mean_dir) / np.sum(mags))
    # LogNormal parameters: logmean pinned to the target MEDIAN (exp(logmean) =
    # median), logstd from the real log-magnitudes so the mean/skew match.
    logmean = float(np.log(np.median(mags)))
    logstd = float(np.std(np.log(mags), ddof=1)) if len(mags) > 1 else 0.3
    proj = rv @ u0
    over_rot = float(np.mean(proj > 0))
    return {
        "logmean": logmean,
        "logstd": logstd,
        "median_deg": float(np.degrees(np.median(mags))),
        "mean_deg": float(np.degrees(np.mean(mags))),
        "u0": u0.tolist(),
        "coherence_target": round(coherence, 3),
        "over_rotation_prob": round(over_rot, 3),
        "n_samples": int(len(mags)),
    }


# --------------------------------------------------------------------------
# vMF direction sampling
# --------------------------------------------------------------------------
def _vMF_dir(mu, kappa, rng):
    """Sample one unit vector from von Mises-Fisher(mu, kappa) on S²."""
    mu = np.asarray(mu, float)
    mu = mu / np.linalg.norm(mu)
    n1 = np.array([1.0, 0.0, 0.0])
    if abs(n1 @ mu) > 0.9:
        n1 = np.array([0.0, 1.0, 0.0])
    n1 = n1 - (n1 @ mu) * mu
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(mu, n1)
    # sample z ~ exp(kappa z) on [-1,1] via inverse CDF (stability: expm1/log1p)
    u = rng.random()
    z = -1.0 + np.log1p(u * np.expm1(2.0 * kappa)) / kappa
    phi = 2.0 * np.pi * rng.random()
    return z * mu + np.sqrt(max(0.0, 1.0 - z * z)) * (np.cos(phi) * n1 + np.sin(phi) * n2)


def _vMF_path(mus, kappa, rng):
    """Sample unit vectors with per-step concentration axis mus (N,3)."""
    return np.array([_vMF_dir(mus[k], kappa, rng) for k in range(len(mus))])


def _realized_coherence(kappa, mus, mags, p_flip, rng_seed=12345):
    """Monte-Carlo coherence for a given kappa (calibration helper, own rng)."""
    rng = np.random.default_rng(rng_seed)
    dirs = _vMF_path(mus, kappa, rng)
    rvs = dirs * mags[:, None]
    flips = rng.random(len(mags)) < p_flip
    rvs[flips] *= -1.0
    return float(np.linalg.norm(rvs.sum(axis=0)) / np.sum(mags))


def _calibrate_kappa(mus, mags, target, p_flip, rng_seed=12345):
    """Bisect kappa so realized coherence ≈ target (monotone increasing)."""
    lo, hi = 0.05, 40.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        c = _realized_coherence(mid, mus, mags, p_flip, rng_seed)
        if c < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Public generators
# --------------------------------------------------------------------------
def generate_true_trajectory(seq_id=None, mag_deg=DEFAULT_MAG_DEG, seed=0,
                             n_windows=77, a0=None):
    """Smooth ~mag_deg/step SO(3) true trajectory with slow axis wobble.

    Magnitudes: LogNormal(μ=ln mag_deg, σ≈0.12) clipped to [15°, 28°],
    smoothed by moving-average(3). Axis a(t) wobbles sinusoidally around a0.
    a0 defaults to the real GT dominant (magnitude-weighted mean) axis.
    """
    rng = np.random.default_rng(seed)
    if a0 is None:
        a0 = _dominant_axis(seq_id)
    if np.linalg.norm(a0) < 1e-9:
        a0 = np.array([0.0, 1.0, 0.0])
    a0 = a0 / np.linalg.norm(a0)
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ a0) > 0.9:
        e1 = np.array([0.0, 0.0, 1.0])
    e1 = e1 - (e1 @ a0) * a0
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a0, e1)

    # smooth magnitudes
    logmean = np.log(np.radians(mag_deg))
    mags = rng.lognormal(mean=logmean, sigma=0.12, size=n_windows)
    mags = np.clip(mags, np.radians(15.0), np.radians(28.0))
    mags = np.convolve(mags, np.ones(3) / 3, mode="same")   # smooth
    mags = np.clip(mags, np.radians(15.0), np.radians(28.0))

    t = np.arange(n_windows - 1)
    axis = (a0[None, :] + 0.02 * np.sin(0.40 * t)[:, None] * e1[None, :]
            + 0.015 * np.sin(0.13 * t)[:, None] * e2[None, :])
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    deltas = mags[:-1][:, None] * axis
    G_true = exponentiate_path(np.stack([np.eye(3)] * n_windows), deltas)
    return G_true, {"a0": a0.tolist(), "axis_wobble_deg": 2.0,
                    "mag_deg": mag_deg, "n_windows": n_windows}


def _dominant_axis(seq_id):
    """Magnitude-weighted mean per-step GT rotation axis of a sequence."""
    if not seq_id:
        seq_id = LANGDON_SEQ
    Gt = _load_gt_gauges(seq_id)
    if Gt is None:
        return np.array([0.0, 1.0, 0.0])
    rvs = np.array([log(Gt[k].T @ Gt[k + 1]) for k in range(len(Gt) - 1)])
    mags = np.linalg.norm(rvs, axis=1)
    axis = (rvs * mags[:, None]).sum(axis=0)
    n = np.linalg.norm(axis)
    return axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def sample_coherent_errors(n_steps, mag_stats, coherence_target, seed=0,
                           axis_vary_deg=0.0, p_flip=1.0 - OVER_ROTATION_PROB):
    """Sample temporally-coherent per-step error rotvecs.

    rv[k] = m_k · u_k,  m_k ~ LogNormal(logmean, logstd), u_k ~ vMF(u0_k, κ).
    κ is calibrated so the *realized* coherence (full rv over the step count)
    matches coherence_target. axis_vary_deg rotates u0_k by that much per
    window (scenario C). p_flip flips the direction of a fraction of steps
    (reproduces real 96 % over-rotation).
    """
    mags = np.random.default_rng(seed).lognormal(
        mean=mag_stats["logmean"], sigma=mag_stats["logstd"], size=n_steps)
    # rescale so the SAMPLE median matches the target median exactly
    # (uniform scale leaves coherence invariant); removes n=76 sample fluctuation
    target_med = float(np.exp(mag_stats["logmean"]))
    sample_med = float(np.median(mags))
    if sample_med > 1e-12:
        mags = mags * (target_med / sample_med)
    u0 = np.asarray(mag_stats["u0"], float)
    u0 = u0 / np.linalg.norm(u0)
    if axis_vary_deg > 0:
        # slowly rotating mean axis (scenario C): rotate u0 around a fixed
        # axis orthogonal to u0 by axis_vary_deg per window.
        pivot = np.array([0.0, 0.0, 1.0])
        if abs(pivot @ u0) > 0.9:
            pivot = np.array([1.0, 0.0, 0.0])
        pivot = np.cross(u0, pivot)
        pivot = pivot / np.linalg.norm(pivot)
        rot_step = Rotation.from_rotvec(np.deg2rad(axis_vary_deg) * pivot)
        mus = [u0]
        for _ in range(1, n_steps):
            mus.append(rot_step.as_matrix() @ mus[-1])
        mus = np.array(mus)
    else:
        mus = np.stack([u0] * n_steps)

    kappa = _calibrate_kappa(mus, mags, coherence_target, p_flip)
    rng = np.random.default_rng(seed + 1)
    dirs = _vMF_path(mus, kappa, rng)
    rv = dirs * mags[:, None]
    flips = rng.random(n_steps) < p_flip
    rv[flips] *= -1.0
    realized = float(np.linalg.norm(rv.sum(axis=0)) / np.sum(mags))
    meta = {
        "kappa": round(float(kappa), 4),
        "realized_coherence": round(realized, 4),
        "axis_vary_deg": axis_vary_deg,
    }
    return rv, meta


def build_synthetic_edges(seq_id, G_true, errors, scenario="A",
                          threshold=8, transitivity_noise_deg=DEFAULT_TRANSITIVITY_DEG,
                          seed=0):
    """Build synthetic edge dict copied from the real topology.

    Scenario A: multi-hop Q = G_chain[i]^T G_chain[j]  (CASE-C consistent).
    Scenario B: ... + independent Exp(η) noise on multi-hop edges.
    hop-1 edges always carry the injected drift via measured increments.
    Returns dict {i, j, Q, weight, hop, G_chain} with i/j (E,), Q (E,3,3).
    """
    csv_path = os.path.join(EDGES_DIR, "ROTATION_GRAPH_EDGES.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"edge csv missing: {csv_path}")
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f)
                if r["sequence"] == seq_id and int(r["n_overlap"]) >= threshold]
    if not rows:
        raise ValueError(f"no edges for {seq_id} at th>={threshold}")

    i = np.array([int(r["window_i"]) for r in rows])
    j = np.array([int(r["window_j"]) for r in rows])
    w = np.array([float(r["weight"]) for r in rows])
    hop = np.array([int(r["window_distance"]) for r in rows])

    n = len(G_true)
    # measured hop-1 edge: error left-multiplied on the true relative rotation.
    # Q_meas[k] = Exp(rv_k) @ Q_gt[k]  ⇒  recovered per-step error is EXACTLY rv_k
    # (Log(Q_meas @ Q_gt^T) = rv_k), so injected statistics are under precise control.
    Q_gt_h1 = [G_true[k].T @ G_true[k + 1] for k in range(n - 1)]
    G_chain = np.stack([G_true[0]] * n)
    for k in range(n - 1):
        Q_meas = exp(errors[k]) @ Q_gt_h1[k]
        G_chain[k + 1] = G_chain[k] @ Q_meas

    rng = np.random.default_rng(seed + 2)
    Q = np.zeros((len(rows), 3, 3))
    for e in range(len(rows)):
        a, b = int(i[e]), int(j[e])
        Q[e] = G_chain[a].T @ G_chain[b]      # multi-hop: chain product (scenario A = consistent)
        if scenario == "B" and hop[e] > 1:
            Q[e] = exp(rng.normal(0.0, np.radians(transitivity_noise_deg), 3)) @ Q[e]

    drift = np.array([rot_angle_deg(G_chain[k].T @ G_true[k]) for k in range(n)])
    return {
        "i": i, "j": j, "Q": Q, "weight": w, "hop": hop,
        "G_chain": G_chain, "G_true": G_true, "drift_injected": drift,
    }


def injected_drift(G_chain, G_true):
    """Per-window injected drift in degrees."""
    n = min(len(G_chain), len(G_true))
    return np.array([rot_angle_deg(G_chain[k].T @ G_true[k]) for k in range(n)])


def realized_stats(G_chain, G_true):
    """Realized coherence / magnitudes / over-rotation from a drifted chain."""
    n = min(len(G_chain), len(G_true))
    rv = _per_step_errors(G_chain[:n], G_true[:n])
    mags = np.linalg.norm(rv, axis=1)
    mean_dir = rv.sum(axis=0)
    nn = np.linalg.norm(mean_dir)
    coherence = float(nn / np.sum(mags)) if np.sum(mags) > 0 else 0.0
    u0 = mean_dir / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
    over_rot = float(np.mean(rv @ u0 > 0))
    return {
        "coherence": round(coherence, 3),
        "median_mag_deg": round(float(np.degrees(np.median(mags))), 3),
        "mean_mag_deg": round(float(np.degrees(np.mean(mags))), 3),
        "over_rotation_prob": round(over_rot, 3),
        "final_drift_deg": round(float(np.degrees(np.linalg.norm(mean_dir))), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq")
    ap.add_argument("--scenario", choices=["A", "B", "C"], default="A")
    ap.add_argument("--threshold", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-windows", type=int, default=77)
    args = ap.parse_args()

    seq_id = args.seq if args.seq else LANGDON_SEQ
    mag_stats = fit_error_statistics(seq_id)
    G_true, traj_meta = generate_true_trajectory(seq_id, n_windows=args.n_windows,
                                                 seed=args.seed)
    axis_vary = AXIS_VARY_DEG_PER_WINDOW if args.scenario == "C" else 0.0
    errors, err_meta = sample_coherent_errors(
        args.n_windows - 1, mag_stats, mag_stats["coherence_target"],
        seed=args.seed, axis_vary_deg=axis_vary)
    edges = build_synthetic_edges(seq_id, G_true, errors, scenario=args.scenario,
                                  threshold=args.threshold, seed=args.seed)
    realized = realized_stats(edges["G_chain"], G_true)

    n_win = len(G_true)
    hop_counts = {}
    for h in sorted(set(edges["hop"].tolist())):
        hop_counts[int(h)] = int(np.sum(edges["hop"] == h))

    summary = {
        "sequence": seq_id,
        "scenario": args.scenario,
        "n_windows": n_win,
        "n_steps": n_win - 1,
        "n_edges": int(len(edges["i"])),
        "edge_hops": hop_counts,
        "threshold": args.threshold,
        "error_magnitudes_deg": {k: mag_stats[k] for k in
                                 ["median_deg", "mean_deg"]},
        "target_coherence": mag_stats["coherence_target"],
        "axis_vary_deg": axis_vary,
        "injected": {
            "param": err_meta,
            "final_drift_deg": round(float(edges["drift_injected"][-1]), 2),
            "w10_drift_deg": round(float(edges["drift_injected"][9]), 2),
            "w30_drift_deg": round(float(edges["drift_injected"][29]), 2),
        },
        "realized": realized,
        "trajectory": traj_meta,
        "validation": {
            "coherence_abs_err": round(abs(realized["coherence"] - mag_stats["coherence_target"]), 3),
            "median_mag_abs_err_deg": round(abs(realized["median_mag_deg"] - mag_stats["median_deg"]), 3),
            "note": ("coherence_abs_err only meaningful for scenarios A/B; scenario C "
                     "intentionally breaks coherence (rotating bias axis)") if args.scenario == "C" else "",
        },
    }

    # persist synthetic assets for the evaluation harness
    npz_path = os.path.join(OUT_DIR,
                            f"SYNTHETIC_{seq_id}_scen{args.scenario}_th{args.threshold}_seed{args.seed}.npz")
    np.savez(npz_path,
             G_true=edges["G_true"], G_chain=edges["G_chain"],
             rv=errors, edges_i=edges["i"], edges_j=edges["j"], Q=edges["Q"],
             weight=edges["weight"], hop=edges["hop"], drift_injected=edges["drift_injected"],
             scenario=args.scenario, threshold=args.threshold,
             target_coherence=mag_stats["coherence_target"],
             kappa=err_meta["kappa"], seed=args.seed, n_windows=n_win)

    json_path = os.path.join(OUT_DIR, "SYNTHETIC_DRIFT_SUMMARY.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"=== {seq_id} scenario {args.scenario} (th>={args.threshold}, seed={args.seed}) ===")
    print(f"  windows={n_win} steps={n_win-1} edges={len(edges['i'])} hops={hop_counts}")
    print(f"  target C={mag_stats['coherence_target']} realized C={realized['coherence']} "
          f"(|Δ|={summary['validation']['coherence_abs_err']})")
    print(f"  target med {mag_stats['median_deg']}° realized {realized['median_mag_deg']}° "
          f"(|Δ|={summary['validation']['median_mag_abs_err_deg']}°)")
    print(f"  injected drift: final={summary['injected']['final_drift_deg']}° "
          f"(w10={summary['injected']['w10_drift_deg']}°, w30={summary['injected']['w30_drift_deg']}°)")
    print(f"  kappa={err_meta['kappa']:.3f} over_rot={realized['over_rotation_prob']:.2f}")
    print(f"Saved: {json_path}")
    print(f"Saved: {npz_path}")


if __name__ == "__main__":
    main()