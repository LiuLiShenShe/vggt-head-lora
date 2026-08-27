"""阶段 2.2 Geometry Audit v3 — 四路判别修正重算 (P0-3).

复用 four_path_data/*.npz (无需 GPU), 用 unproject_v3 替代旧 unproject_np.
输出 four_path_v3/{verdict.json, *_grid.png, metric_definitions.json}.
不覆盖 10_failures/four_path_discrimination/ 或 v2_clean_rerun_eval/four_path_discrimination/.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import sys
ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
sys.path.insert(0, ROOT)
from unproject_v3 import unproject_v3, camera_centers  # noqa: E402

BASE = "/fj/VGGT+head+lora实验/阶段2"
DEFAULT_DATA_DIR = os.path.join(BASE, "02_vggt/v2_clean_rerun_eval/four_path_data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "four_path_v3")
SEQS = {
    "success_05-03-24": "plantview__langdon_4__05-03-24",
    "fail_12-03-24": "plantview__langdon_4__12-03-24",
}
N_FRAMES = [8, 16, 24, 36]


def horn_sim3_params(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((sc ** 2).sum() / len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(params, P):
    s, R, t = params
    return s * (R @ P.T).T + t


def ref_intrinsics_crop(seq):
    from PIL import Image
    d = json.load(open(seq["intrinsics_path"]))
    K = np.array([[d["fl_x"], 0, d["cx"]], [0, d["fl_y"], d["cy"]], [0, 0, 1]], dtype=np.float64)
    w0, h0 = d["width"], d["height"]
    scale = 518.0 / w0
    crop_top = max(0.0, (h0 * scale - 518) / 2)
    K2 = K.copy()
    K2[0] *= scale
    K2[1] *= scale
    K2[1, 2] -= crop_top
    return K2


def truncated_nn(P_src, tree, trunc):
    d, _ = tree.query(P_src, k=1, distance_upper_bound=trunc)
    n_trunc = int((~np.isfinite(d)).sum())
    d = d[np.isfinite(d)]
    if len(d) < 100:
        return None, None, len(P_src), n_trunc
    return float(np.median(d)), float(np.percentile(d, 90)), len(P_src), n_trunc


def main():
    ap = argparse.ArgumentParser(description="四路判别 v3 · 修正反投影 · CPU")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    verdict = {}
    for key, sid in SEQS.items():
        seq = next(json.load(open(p)) for p in glob.glob(f"{BASE}/01_sequences/sequences/plant_view/*.json")
                   if json.load(open(p))["sequence_id"] == sid)
        ref_w2c_all = np.stack([np.array(e["w2c"], dtype=np.float64)[:3, :4]
                                for e in json.load(open(seq["extrinsics_path"]))["extrinsics"]])
        K_ref_crop = ref_intrinsics_crop(seq)
        import open3d as o3d
        ref_cloud = np.asarray(o3d.io.read_point_cloud(seq["reference_pointcloud"]).points)
        dp_path = os.path.join(os.path.dirname(seq["reference_pointcloud"]),
                               "dataparser_transforms.json")
        dp = json.load(open(dp_path))
        T_dp = np.array(dp["transform"], dtype=np.float64)
        scale_dp = float(dp.get("scale", 1.0))
        t_dp = T_dp[:3, 3]
        ref_cloud = (ref_cloud - t_dp) / scale_dp
        tree = cKDTree(ref_cloud)
        diag = float(np.linalg.norm(ref_cloud.max(0) - ref_cloud.min(0)))
        trunc = diag * 0.05

        rows = {}
        for n in N_FRAMES:
            z = np.load(os.path.join(args.data_dir, f"{key}_n{n}.npz"))
            dep, ext, intr, pmap = z["depth"], z["ext_w2c_vggt"], z["intr_vggt"], z["point_map_head"]
            idx = z["frame_idx"]
            C_ref = camera_centers(ref_w2c_all[idx])
            C_vg = camera_centers(ext)

            A = unproject_v3(dep, ext, intr).reshape(-1, 3)
            B = unproject_v3(dep, ref_w2c_all[idx], np.repeat(K_ref_crop[None], n, 0)).reshape(-1, 3)
            C = pmap.reshape(-1, 3)

            sA, RA, tA = horn_sim3_params(C_vg, C_ref)
            A_al = apply_sim3((sA, RA, tA), A)
            C_al = apply_sim3((sA, RA, tA), C)
            B_al = B

            out = {"n_frames": int(n)}
            for name, P in (("A_vggt_cam", A_al), ("B_ref_cam", B_al), ("C_point_head", C_al)):
                P_s = P[::max(1, len(P) // 80000)]
                m, p90, n_pts, n_tr = truncated_nn(P_s, tree, trunc)
                out[name] = {"nn_med": None if m is None else round(m, 5),
                             "nn_p90": None if p90 is None else round(p90, 5),
                             "n_points_sampled": n_pts,
                             "n_beyond_trunc": n_tr}
            out["trunc"] = round(trunc, 5)
            out["ref_diag"] = round(diag, 4)
            rows[n] = (A_al, B_al, C_al, out)

        fig = plt.figure(figsize=(16, 16))
        for r, n in enumerate(N_FRAMES):
            A_al, B_al, C_al, _ = rows[n]
            panels = [(A_al, f"A: VGGT depth+cam n={n} (v3)"),
                      (B_al, f"B: VGGT depth+ref cam n={n} (v3)"),
                      (C_al, f"C: point_head n={n} (v3)"),
                      (ref_cloud[::20], "D: reference GS")]
            for c, (P, ttl) in enumerate(panels):
                ax = fig.add_subplot(4, 4, r * 4 + c + 1, projection="3d")
                if P is not None and len(P):
                    s2 = P[::max(1, len(P) // 60000)]
                    ax.scatter(s2[:, 0], s2[:, 1], s2[:, 2], s=0.2, c=s2[:, 2], cmap="viridis")
                ax.set_title(ttl, fontsize=7)
        fig.savefig(os.path.join(args.out_dir, f"{key}_grid.png"), dpi=110)
        plt.close(fig)

        verdict[key] = {str(n): rows[n][3] for n in N_FRAMES}
        print(f"{key} (trunc={trunc:.4f}, ref diag={diag:.3f}):")
        for n in N_FRAMES:
            o = rows[n][3]
            fmt = lambda v: "None" if v is None else f"{v:.5f}"
            print(f"  n={n}: A {fmt(o['A_vggt_cam']['nn_med'])} | "
                  f"B {fmt(o['B_ref_cam']['nn_med'])} | C {fmt(o['C_point_head']['nn_med'])}")

    with open(os.path.join(args.out_dir, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    defs = {
        "nn_med": {"name": "truncated NN median", "diagnostic_only": True},
        "nn_p90": {"name": "truncated NN P90", "diagnostic_only": True},
        "n_beyond_trunc": {"name": "points beyond truncation"},
        "_note": "Corrected unprojection (v3): world = R_w2c.T @ cam - R_w2c.T @ t_w2c; replaces buggy v2 (+t_w2c)",
    }
    with open(os.path.join(args.out_dir, "metric_definitions.json"), "w") as f:
        json.dump(defs, f, indent=2)
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
