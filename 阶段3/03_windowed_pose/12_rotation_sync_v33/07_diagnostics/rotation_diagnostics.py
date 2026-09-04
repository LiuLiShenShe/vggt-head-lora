#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3 Step 7: GT-free diagnostics for the rotation graph.

Diagnostic questions (GT-free — no COLMAP reference used anywhere here):
  1. Are the pairwise Q measurements internally cycle-consistent?
       per triangle (i,j,k): cycle_error = angle(Q_ij Q_jk Q_ki)
     If cycle residual is small → measurements self-consistent, NOT random noise.
  2. Did the SO(3) sync converge to a consistent solution?
       postfit edge residual E_ij = Q_ij^T G_i^T G_j  (geodesic angle)
     Small postfit → solver found an internally-consistent gauge.
  3. Is there systematic Q bias by window distance?
       edge error vs reference (using global cameras + COLMAP for COMPARISON ONLY),
       bucketed by j - i ∈ {1, 2, 3}
     If hop-2/hop-3 Q's are biased while hop-1 are clean → context-dependent Q bias.

Interpretation key:
  - cycle_residual small  AND postfit small  AND edge-vs-ref error large
      => consistent-but-systematically-biased Q: graph sync CANNOT remove this.
      (Q_chain_accumulation root cause REJECTED; context-dependent Q bias CONFIRMED)
  - cycle_residual large  => noisy/inconsistent edges: more redundancy helps.

Outputs (in 07_diagnostics/):
  ROTATION_CYCLE_RESIDUALS.csv
  ROTATION_GRAPH_POSTFIT_RESIDUALS.csv
  EDGE_ERROR_BY_WINDOW_DISTANCE.csv
  DIAGNOSTICS_SUMMARY.json

Usage:
    python 07_diagnostics/rotation_diagnostics.py [--seq ...]
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
EDGES_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
SOLVER_DIR = os.path.join(SYNC_DIR, "04_so3_sync")
OUT_DIR = os.path.join(SYNC_DIR, "07_diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def load_edges(seq_id, dataset, threshold):
    path = os.path.join(EDGES_DIR,
                        "ROTATION_GRAPH_EDGES_STRIDE8.csv" if dataset == "stride8"
                        else "ROTATION_GRAPH_EDGES.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r["sequence"] == seq_id and int(r["n_overlap"]) >= threshold]
    if not rows:
        return None
    i = np.array([int(r["window_i"]) for r in rows])
    j = np.array([int(r["window_j"]) for r in rows])
    Q = np.array([[[float(r[f"q{rr}{cc}"]) for cc in range(3)] for rr in range(3)] for r in rows])
    w = np.array([float(r["weight"]) for r in rows])
    disp_med = np.array([float(r["Q_disp_med_deg"]) for r in rows])
    n_overlap = np.array([int(r["n_overlap"]) for r in rows])
    hop = np.array([int(r["window_distance"]) for r in rows])
    return i, j, Q, w, disp_med, n_overlap, hop


def load_G(seq_id, dataset, threshold):
    path = os.path.join(SOLVER_DIR, f"{seq_id}_SO3_SYNC_GAUGES_{dataset}_th{threshold}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["G"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    ap.add_argument("--threshold", type=int, default=8)
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)
    th = args.threshold

    cycle_rows, postfit_rows, edge_rows = [], [], []
    diag_summary = {}

    for seq_id in sequences:
        print(f"\n=== {seq_id} (min_overlap={th}) ===")

        # ---- Dataset 1: stride-4 redundant graph (headline) ----
        res = load_edges(seq_id, "stride4", th)
        if res is None:
            print("  stride4 edges missing — skip")
            continue
        i4, j4, Q4, w4, disp4, n_over4, hop4 = res

        # ---- Cycle residuals: triangles from stride-4 graph ----
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(int(i4.max()) + 1))
        for a, b in zip(i4, j4):
            g.add_edge(int(a), int(b))
        q_lookup = {}
        for a, b, q in zip(i4, j4, Q4):
            q_lookup[(int(a), int(b))] = q

        n_cycle = 0
        cyc_errs = []
        for a, b in g.edges():
            for c in g.neighbors(b):
                if c == a or not g.has_edge(a, c):
                    continue
                if a >= c:
                    continue  # canonical (a < b < c) via c > b
                if c < b:
                    continue
                # triangle (a, b, c) with a < b < c
                Qab = q_lookup.get((a, b), q_lookup.get((b, a), np.eye(3)).T)
                Qbc = q_lookup.get((b, c), q_lookup.get((c, b), np.eye(3)).T)
                Qca = q_lookup.get((c, a), q_lookup.get((a, c), np.eye(3)).T)
                E = Qab @ Qbc @ Qca
                cyc_errs.append(rot_angle_deg(E))
                n_cycle += 1
        # fallback canonical ordering with in-place orientation normalization
        # (first pass above already covers triangles a<b<c)

        if n_cycle == 0:
            # Fallback: brute-force triangles via node lists (robust)
            nodes = sorted(g.nodes())
            cyc_errs = []
            n_cycle = 0
            for ai, a in enumerate(nodes):
                for b in nodes[ai + 1:]:
                    if not g.has_edge(a, b):
                        continue
                    for c in nodes[ai + 2:]:
                        if not (g.has_edge(a, c) and g.has_edge(b, c)):
                            continue
                        Qab = q_lookup.get((a, b), q_lookup.get((b, a), np.eye(3)).T)
                        Qbc = q_lookup.get((b, c), q_lookup.get((c, b), np.eye(3)).T)
                        Qca = q_lookup.get((c, a), q_lookup.get((a, c), np.eye(3)).T)
                        cyc_errs.append(rot_angle_deg(Qab @ Qbc @ Qca))
                        n_cycle += 1

        cycle_stat = {
            "sequence": seq_id, "dataset": "stride4", "threshold": th,
            "n_triangles": n_cycle,
            "cycle_err_median_deg": float(np.median(cyc_errs)) if cyc_errs else -1,
            "cycle_err_p90_deg": float(np.percentile(cyc_errs, 90)) if cyc_errs else -1,
            "cycle_err_max_deg": float(np.max(cyc_errs)) if cyc_errs else -1,
        }
        cycle_rows.append(cycle_stat)

        # ---- Postfit residuals from solver gauges ----
        G4 = load_G(seq_id, "stride4", th)
        if G4 is not None:
            per_edge = []
            for e in range(len(i4)):
                a, b = i4[e], j4[e]
                E = Q4[e].T @ (G4[a].T @ G4[b])
                per_edge.append(rot_angle_deg(E))
            per_edge = np.array(per_edge)
            postfit_stat = {
                "sequence": seq_id, "dataset": "stride4", "threshold": th,
                "n_edges": len(per_edge),
                "postfit_mean_deg": float(np.mean(per_edge)),
                "postfit_median_deg": float(np.median(per_edge)),
                "postfit_p90_deg": float(np.percentile(per_edge, 90)),
                "postfit_max_deg": float(np.max(per_edge)),
            }
            postfit_rows.append(postfit_stat)
            print(f"  cycle: n={n_cycle} med={cycle_stat['cycle_err_median_deg']:.3f}° "
                  f"p90={cycle_stat['cycle_err_p90_deg']:.3f}°")
            print(f"  postfit: med={postfit_stat['postfit_median_deg']:.3f}° "
                  f"p90={postfit_stat['postfit_p90_deg']:.3f}°")

        # ---- Edge error by window distance (requires COLMAP ref — comparison ONLY) ----
        try:
            import glob
            seq_json = None
            for subdir in ["plant_view", "wheat3dgs", "mustc"]:
                for jp in glob.glob(os.path.join(ROOT, "阶段2", "01_sequences", "sequences",
                                                 subdir, "*.json")):
                    with open(jp) as f:
                        meta = json.load(f)
                    if meta.get("sequence_id") == seq_id:
                        seq_json = meta
                        break
                if seq_json:
                    break
            if seq_json is None:
                print("  (no seq json — skip edge-vs-ref)")
            else:
                # We compare each edge's Q to the reference-implied Q
                ext_path = seq_json.get("extrinsics_path")
                ref_w2c = None
                if ext_path and os.path.exists(ext_path):
                    with open(ext_path) as f:
                        ref_w2c = np.array([np.array(e["w2c"])[:3, :4]
                                            for e in json.load(f)["extrinsics"]])
                if ref_w2c is None:
                    print("  (no COLMAP ref — skip edge-vs-ref)")
                else:
                    # Load window frame indices + local rotations
                    w4dir = os.path.join(PHASE3C, "03_window_inference",
                                         "window_outputs_stride4", seq_id)
                    wins = []
                    for wf in sorted(glob.glob(os.path.join(w4dir, "window_*.npz"))):
                        d = np.load(wf)
                        wins.append({"ext": d["ext_w2c_vggt"], "idx": d["frame_idx"]})
                    n_w = len(wins)
                    R_ref_c2w = ref_w2c[:, :3, :3].transpose(0, 2, 1)  # (N,3,3) c2w

                    c2w = lambda w: np.asarray(w["ext"])[:, :3, :3].transpose(0, 2, 1)

                    # Fit per-window reference gauge G_k: min_G sum_f ||G @ R_local,f - R_ref,f||_F
                    # (Procrustes per window vs COLMAP). G_k maps local -> global reference.
                    G_ref = []
                    for k, w in enumerate(wins):
                        idx_k = np.asarray(w["idx"], dtype=int)
                        R_local = c2w(w)
                        R_refk = R_ref_c2w[idx_k]
                        H = np.einsum("sij,sik->jk", R_refk, R_local)
                        U, _, Vt = np.linalg.svd(H)
                        Gk = Vt.T @ U.T
                        if np.linalg.det(Gk) < 0:
                            Vt[-1, :] *= -1
                            Gk = Vt.T @ U.T
                        G_ref.append(Gk)
                    G_ref = np.array(G_ref)

                    hop_err = {1: [], 2: [], 3: []}
                    for e in range(len(i4)):
                        a, b = int(i4[e]), int(j4[e])
                        if a >= n_w or b >= n_w:
                            continue
                        # Reference-implied Q: Q_ref_ij = G_i^T G_j (true relative gauge)
                        Q_ref = G_ref[a].T @ G_ref[b]
                        # VGGT measured edge Q
                        Q_v = Q4[e]
                        err = rot_angle_deg(Q_ref.T @ Q_v)
                        hop = int(hop4[e])
                        if hop in hop_err:
                            hop_err[hop].append(err)

                    for hop_k in sorted(hop_err):
                        if hop_err[hop_k]:
                            vals = hop_err[hop_k]
                            edge_rows.append({
                                "sequence": seq_id, "dataset": "stride4",
                                "window_distance": hop_k,
                                "n_edges": len(vals),
                                "q_vs_ref_median_deg": float(np.median(vals)),
                                "q_vs_ref_p90_deg": float(np.percentile(vals, 90)),
                                "q_vs_ref_mean_deg": float(np.mean(vals)),
                            })
                            print(f"  hop{hop_k}: n={len(vals)} "
                                  f"Q-vs-COLMAP med={np.median(vals):.2f}° "
                                  f"p90={np.percentile(vals, 90):.2f}°")
        except Exception as ex:
            print(f"  edge-vs-ref skipped: {ex}")

        diag_summary[seq_id] = {
            "cycle_err_median_deg": cycle_stat["cycle_err_median_deg"],
            "cycle_err_p90_deg": cycle_stat["cycle_err_p90_deg"],
            "postfit_median_deg": postfit_stat["postfit_median_deg"] if G4 is not None else None,
            "n_triangles": n_cycle,
        }

    # ---- Save CSVs ----
    if cycle_rows:
        with open(os.path.join(OUT_DIR, "ROTATION_CYCLE_RESIDUALS.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cycle_rows[0].keys()))
            w.writeheader(); w.writerows(cycle_rows)
        print(f"\nSaved ROTATION_CYCLE_RESIDUALS.csv ({len(cycle_rows)} rows)")
    if postfit_rows:
        with open(os.path.join(OUT_DIR, "ROTATION_GRAPH_POSTFIT_RESIDUALS.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(postfit_rows[0].keys()))
            w.writeheader(); w.writerows(postfit_rows)
        print(f"Saved ROTATION_GRAPH_POSTFIT_RESIDUALS.csv ({len(postfit_rows)} rows)")
    if edge_rows:
        with open(os.path.join(OUT_DIR, "EDGE_ERROR_BY_WINDOW_DISTANCE.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(edge_rows[0].keys()))
            w.writeheader(); w.writerows(edge_rows)
        print(f"Saved EDGE_ERROR_BY_WINDOW_DISTANCE.csv ({len(edge_rows)} rows)")
    with open(os.path.join(OUT_DIR, "DIAGNOSTICS_SUMMARY.json"), "w") as f:
        json.dump(diag_summary, f, indent=2)
    print("Saved DIAGNOSTICS_SUMMARY.json")


if __name__ == "__main__":
    main()