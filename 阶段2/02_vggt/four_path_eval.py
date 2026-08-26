"""四路判别实验 · 步骤2:组装路径 + 定量 + 出图(da3 环境)。

读取 four_path_data/*.npz,构造:
  A: unproject(VGGT depth, VGGT ext/intr) → 相机中心 Sim3 对齐到参考
  B: unproject(VGGT depth, 参考 w2c + crop 后参考内参) → 已在参考系
  C: point_head 直接 point map → 同 A 对齐
  D: 参考 GS splat.ply
定量:各路径到参考云截断 NN 距离(截断 3×参考云中位最近邻间距)。
出图 4×4 网格 + verdict.json → 10_failures/four_path_discrimination/。

用法: python four_path_eval.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/fj/VGGT+head+lora实验/阶段2"
DATA_DIR = os.path.join(BASE, "10_failures", "four_path_data")
OUT_DIR = os.path.join(BASE, "10_failures", "four_path_discrimination")
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


def centers(e):
    return np.einsum("sij,sj->si", e[:, :3, :3].transpose(0, 2, 1), -e[:, :3, 3])


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
    d = d[np.isfinite(d)]
    if len(d) < 100:
        return None, None
    return float(np.median(d)), float(np.percentile(d, 90))


def main():
    assert not os.path.exists(OUT_DIR), f"{OUT_DIR} 已存在"
    os.makedirs(OUT_DIR)
    import open3d as o3d
    from scipy.spatial import cKDTree

    verdict = {}
    for key, sid in SEQS.items():
        seq = next(json.load(open(p)) for p in glob.glob(f"{BASE}/01_sequences/sequences/plant_view/*.json")
                   if json.load(open(p))["sequence_id"] == sid)
        ref_w2c_all = np.stack([np.array(e["w2c"], dtype=np.float64)[:3, :4]
                                for e in json.load(open(seq["extrinsics_path"]))["extrinsics"]])
        K_ref_crop = ref_intrinsics_crop(seq)
        ref_cloud = np.asarray(o3d.io.read_point_cloud(seq["reference_pointcloud"]).points)
        # GS splat.ply 在 nerfstudio 训练坐标系(dataparser 平移了 Z),
        # 与 transforms.json 相机系差一个固定平移;从 dataparser_transforms.json 读取并还原:
        dp_path = os.path.join(os.path.dirname(seq["reference_pointcloud"]),
                               "dataparser_transforms.json")
        dp = json.load(open(dp_path))
        T_dp = np.array(dp["transform"], dtype=np.float64)   # 3x4: P_gs = T_dp @ P_orig
        scale_dp = float(dp.get("scale", 1.0))
        R_dp, t_dp = T_dp[:3, :3], T_dp[:3, 3]
        # 还原到 transforms.json 相机系:P_orig = (P_gs - t_dp) / scale_dp(此处 R=I, scale=1)
        ref_cloud = (ref_cloud - t_dp) / scale_dp
        print(f"  dataparser shift applied: t={t_dp.round(4)}, scale={scale_dp}")
        tree = cKDTree(ref_cloud)
        diag = float(np.linalg.norm(ref_cloud.max(0) - ref_cloud.min(0)))
        trunc = diag * 0.05   # 场景对角线的 5% 作截断(GS 云极密,中位间距 ~0.5mm 不适合作截断基准)

        rows = {}
        for n in N_FRAMES:
            z = np.load(os.path.join(DATA_DIR, f"{key}_n{n}.npz"))
            dep, ext, intr, pmap = z["depth"], z["ext_w2c_vggt"], z["intr_vggt"], z["point_map_head"]
            idx = z["frame_idx"]
            C_ref = centers(ref_w2c_all[idx])
            C_vg = centers(ext)

            A = unproject_np(dep[..., None], ext, intr).reshape(-1, 3)
            B = unproject_np(dep[..., None], ref_w2c_all[idx], np.repeat(K_ref_crop[None], n, 0)).reshape(-1, 3)
            C = pmap.reshape(-1, 3)

            sA, RA, tA = horn_sim3_params(C_vg, C_ref)
            A_al = apply_sim3((sA, RA, tA), A)
            C_al = apply_sim3((sA, RA, tA), C)
            B_al = B

            out = {"n_frames": int(n)}
            for name, P in (("A_vggt_cam", A_al), ("B_ref_cam", B_al), ("C_point_head", C_al)):
                P_s = P[::max(1, len(P) // 80000)]
                m, p90 = truncated_nn(P_s, tree, trunc)
                out[name] = {"nn_med": None if m is None else round(m, 5),
                             "nn_p90": None if p90 is None else round(p90, 5)}
            rows[n] = (A_al, B_al, C_al, out)

        fig = plt.figure(figsize=(16, 16))
        for r, n in enumerate(N_FRAMES):
            A_al, B_al, C_al, _ = rows[n]
            panels = [(A_al, f"A: VGGT depth+cam n={n}"),
                      (B_al, f"B: VGGT depth+ref cam n={n}"),
                      (C_al, f"C: point_head n={n}"),
                      (ref_cloud[::20], "D: reference GS")]
            for c, (P, ttl) in enumerate(panels):
                ax = fig.add_subplot(4, 4, r * 4 + c + 1, projection="3d")
                if P is not None and len(P):
                    s2 = P[::max(1, len(P) // 60000)]
                    ax.scatter(s2[:, 0], s2[:, 1], s2[:, 2], s=0.2, c=s2[:, 2], cmap="viridis")
                ax.set_title(ttl, fontsize=7)
        fig.savefig(os.path.join(OUT_DIR, f"{key}_grid.png"), dpi=110)
        plt.close(fig)

        verdict[key] = {str(n): rows[n][3] for n in N_FRAMES}
        print(f"{key} (trunc={trunc:.4f}, ref diag={diag:.3f}):")
        for n in N_FRAMES:
            o = rows[n][3]
            fmt = lambda v: "None" if v is None else f"{v:.5f}"
            print(f"  n={n}: A {fmt(o['A_vggt_cam']['nn_med'])} | "
                  f"B {fmt(o['B_ref_cam']['nn_med'])} | C {fmt(o['C_point_head']['nn_med'])}")

    with open(os.path.join(OUT_DIR, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(f"-> {OUT_DIR}")


def unproject_np(depth, extrinsic, intrinsic):
    """官方 geometry.py 同逻辑(numpy 版,供 da3 环境使用)。"""
    H, W = depth.shape[1:3]
    x, y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    ones = np.ones_like(x)
    pix = np.stack([x, y, ones], axis=-1)[..., None]          # (H,W,3,1)
    K_inv = np.linalg.inv(intrinsic)                          # (S,3,3)
    cam = np.einsum("sij,hwjn->shwin", K_inv, pix)[..., 0]    # (S,H,W,3)
    cam = cam * depth                                         # 乘深度(depth 已是 (S,H,W,1))
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    world = np.einsum("sij,shwj->shwi", R.transpose(0, 2, 1), cam) + t[:, None, None, :]
    return world


if __name__ == "__main__":
    main()
