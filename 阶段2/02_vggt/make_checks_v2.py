"""阶段2.2 检查图生成 v2(da3 环境)——审计修正版。

v1 问题(审计确认):点云对齐用无对应关系的随机配对做 Horn(尺度趋零,塌陷假象);
相机图红/蓝两系未对齐不可比。v2 修复:
- 相机中心逐帧对应 Sim3(Umeyama)→ 同一变换作用于相机与完整点云;
- mustc 参考云为 UTM 系(与 plot-local 相机不同系)→ FPFH-RANSAC + ICP 全局配准;
- 相机朝向用变换后的旋转矩阵绘制,红蓝同系比较;
- 输出写 checks_v2/(旧 checks/ 保留作审计痕迹)。

用法: python make_checks_v2.py <seq_dir> [<seq_dir2> ...]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/fj/VGGT+head+lora实验/阶段2/00_environment")

SEQ_BASE = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences"
MAX_PTS = 500_000


# ---------- 几何工具(带合成自检) ----------

def horn_sim3_params(src, dst):
    """Umeyama-Horn: 求 s,R,t 使 dst ≈ s R src + t(src/dst 逐点对应)。
    返回 (s, R, t)。列向量约定。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)          # (3,3)
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
    return s * (R @ np.asarray(P).T).T + t


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def _selftest():
    rng = np.random.RandomState(7)
    A = rng.randn(200, 3) * 3
    q = np.linalg.qr(rng.randn(3, 3))[0]
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    s_true, t_true = 2.7, np.array([5.0, -3.0, 1.5])
    B = s_true * (q @ A.T).T + t_true
    s, R, t = horn_sim3_params(A, B)
    err = np.abs(apply_sim3((s, R, t), A) - B).max()
    assert err < 1e-6 and abs(s - s_true) < 1e-9 and rot_angle_deg(R.T @ q) < 1e-6, err


_selftest()


# ---------- 可视化工具 ----------

def depth_to_color(d):
    import cv2
    d = d.copy()
    vmin, vmax = np.percentile(d[d > 0], [2, 98]) if (d > 0).any() else (0, 1)
    dn = np.clip((d - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]


def conf_to_color(c):
    import cv2
    lo, hi = np.percentile(c, [5, 95])
    cn = np.clip((c - lo) / max(hi - lo, 1e-9), 0, 1)
    return cv2.applyColorMap((cn * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1]


def grid(imgs, cols=6, size=200):
    import cv2
    rows = []
    for r in range(0, len(imgs), cols):
        row = imgs[r:r + cols]
        row = [cv2.resize(im, (size, int(size * im.shape[0] / im.shape[1]))) for im in row]
        h = min(im.shape[0] for im in row)
        canvas = np.full((h, sum(im.shape[1] for im in row), 3), 255, np.uint8)
        x = 0
        for im in row:
            canvas[:im.shape[0], x:x + im.shape[1]] = im
            x += im.shape[1]
        rows.append(canvas)
    W = max(r.shape[1] for r in rows)
    H = sum(r.shape[0] for r in rows)
    out = np.full((H, W, 3), 255, np.uint8)
    y = 0
    for r in rows:
        out[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0]
    return out


def save_ply(pts, colors_rgb01, path):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.clip(colors_rgb01, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(path, pc)


def depth_like_colors(P):
    d = P[:, 2] - P[:, 2].min()
    d = d / max(d.max(), 1e-9)
    return plt.cm.viridis(d)


def load_ref_cloud(ref_cloud_path):
    if ref_cloud_path.endswith(".txt"):
        return np.loadtxt(ref_cloud_path, usecols=(1, 2, 3))
    if ref_cloud_path.endswith(".las"):
        import laspy
        las = laspy.read(ref_cloud_path)
        return np.stack([las.x, las.y, las.z], axis=1)
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(ref_cloud_path).points)


def fpfh_ransac_icp(src_pts, dst_pts, voxel=None, seed=0):
    """open3d 全局配准:FPFH 特征 + RANSAC + ICP 精配。
    返回 4x4 变换(src→dst)与 fitness;失败返回 (None, 0)。"""
    import open3d as o3d

    def make_pc(pts, vox):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        if vox:
            pc = pc.voxel_down_sample(vox)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2 if vox else 1.0, max_nn=30))
        return pc

    # 尺度自适应 voxel:取 src 对角线的 2%
    diag = float(np.linalg.norm(src_pts.max(0) - src_pts.min(0)))
    vox = voxel or diag * 0.02
    src = make_pc(src_pts, vox)
    dst = make_pc(dst_pts, vox)
    src_f = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 5, max_nn=100))
    dst_f = o3d.pipelines.registration.compute_fpfh_feature(
        dst, o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 5, max_nn=100))
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, dst, src_f, dst_f, True, vox * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vox * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    icp = o3d.pipelines.registration.registration_icp(
        src, dst, vox * 2, res.transformation,
        o3d.pipeline if False else o3d.pipelines.registration.TransformationEstimationPointToPoint())
    T = np.asarray(icp.transformation)
    return T, float(icp.fitness)


# ---------- 主流程 ----------

def make_checks(seq_dir: str):
    meta = json.load(open(os.path.join(seq_dir, "prediction_meta.json")))
    sid = meta["sequence_id"]
    ds = meta["dataset_id"]
    ck = os.path.join(seq_dir, "checks_v2")
    os.makedirs(ck, exist_ok=True)

    depth = np.load(os.path.join(seq_dir, "depth_vggt.npy"))
    dconf = np.load(os.path.join(seq_dir, "depth_conf_vggt.npy"))
    pdir = np.load(os.path.join(seq_dir, "point_map_direct.npy"))
    pconf_d = np.load(os.path.join(seq_dir, "point_conf_direct.npy"))
    punj = np.load(os.path.join(seq_dir, "point_map_unprojected.npy"))
    ext_w2c = np.load(os.path.join(seq_dir, "extrinsic_w2c.npy")).astype(np.float64)

    ds_dir = os.path.join(SEQ_BASE,
                          {"plant_view_3d": "plant_view", "wheat3dgs": "wheat3dgs",
                           "mustc": "mustc", "terra_ref": "terraref"}[ds])
    seq = next(p for p in glob.glob(os.path.join(ds_dir, "*.json"))
               if json.load(open(p))["sequence_id"] == sid)
    seq = json.load(open(seq))
    S = len(depth)

    # VGGT 相机中心与 c2w 旋转
    R_vg = ext_w2c[:, :3, :3]
    t_vg = ext_w2c[:, :3, 3]
    C_vg = np.einsum("sij,sj->si", R_vg.transpose(0, 2, 1), -t_vg)
    R_vg_c2w = R_vg.transpose(0, 2, 1)

    # 参考相机
    ref_ext_path = seq.get("extrinsics_path")
    C_ref = R_ref_c2w = None
    if ref_ext_path and os.path.exists(ref_ext_path):
        ref = json.load(open(ref_ext_path))["extrinsics"]
        ref_w2c = np.stack([np.array(e["w2c"], dtype=np.float64)[:3, :4] for e in ref])
        R_ref_c2w = ref_w2c[:, :3, :3].transpose(0, 2, 1)
        C_ref = np.einsum("sij,sj->si", ref_w2c[:, :3, :3].transpose(0, 2, 1), -ref_w2c[:, :3, 3])

    # ---- 1. 输入缩略图 ----
    from PIL import Image
    n_show = min(S, 12)
    idx = np.linspace(0, S - 1, n_show).astype(int)
    thumbs = [np.array(Image.open(seq["rgb_paths"][i]).convert("RGB").resize((256, 256))) for i in idx]
    Image.fromarray(grid(thumbs)).save(os.path.join(ck, "input_thumbs.png"))

    # ---- 2/3. depth 彩色 & conf ----
    Image.fromarray(grid([depth_to_color(depth[i]) for i in idx])).save(os.path.join(ck, "depth_colored.png"))
    Image.fromarray(grid([conf_to_color(dconf[i]) for i in idx])).save(os.path.join(ck, "depth_conf.png"))

    # ---- 4. 相机图:Sim3(相机中心逐帧对应)对齐后同系比较 ----
    align_method = None
    sim3 = None
    if C_ref is not None:
        sim3 = horn_sim3_params(C_vg, C_ref)
        align_method = "camera-center Sim3 (per-frame correspondence)"
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    step = max(1, S // 40)
    scale = float(np.percentile(np.linalg.norm(C_vg - C_vg.mean(0), axis=1), 90)) / 8
    if C_ref is not None:
        # 画参考(蓝)
        for i in range(0, S, step):
            z = R_ref_c2w[i] @ np.array([0, 0, 1.0]) * scale
            ax.plot(*zip(C_ref[i], C_ref[i] + z), color="tab:blue", lw=0.8)
        ax.scatter(*C_ref[::step].T, s=4, c="tab:blue", label="ref cams")
        # 画 VGGT(对齐后,红):中心与朝向都经 Sim3 变换
        C_al = apply_sim3(sim3, C_vg)
        for i in range(0, S, step):
            z = (sim3[1] @ R_vg_c2w[i]) @ np.array([0, 0, 1.0]) * scale
            ax.plot(*zip(C_al[i], C_al[i] + z), color="tab:red", lw=0.8)
        ax.scatter(*C_al[::step].T, s=4, c="tab:red", label="VGGT cams (Sim3 aligned)")
    else:
        for i in range(0, S, step):
            z = R_vg_c2w[i] @ np.array([0, 0, 1.0]) * scale
            ax.plot(*zip(C_vg[i], C_vg[i] + z), color="tab:red", lw=0.8)
        ax.scatter(*C_vg[::step].T, s=4, c="tab:red", label="VGGT cams")
    ax.legend(); ax.set_title(f"{sid} cameras\n({align_method or 'no ref'})")
    fig.savefig(os.path.join(ck, "camera_frustums.png"), dpi=110); plt.close(fig)

    if C_ref is not None:
        fig2 = plt.figure(figsize=(10, 10)); a2 = fig2.add_subplot(projection="3d")
        a2.scatter(*C_ref.T, s=3, c="tab:blue", label="reference")
        a2.scatter(*C_al.T, s=3, c="tab:red", label="VGGT (Sim3 aligned)")
        a2.legend(); a2.set_title(f"ref vs VGGT camera centers ({align_method})")
        fig2.savefig(os.path.join(ck, "overlay_ref_vs_vggt_cams.png"), dpi=110)
        plt.close(fig2)

    # ---- 5/6. 两路点云 PLY(原始 VGGT 系,conf 过滤 + 下采样) ----
    def cloud_from(pmap, conf, path_ply):
        pts = pmap.reshape(-1, 3)
        cf = conf.reshape(-1)
        thr = np.percentile(cf, 30)
        m = cf >= thr
        pts, cf = pts[m], cf[m]
        if len(pts) > MAX_PTS:
            sel = np.random.RandomState(0).choice(len(pts), MAX_PTS, replace=False)
            pts, cf = pts[sel], cf[sel]
        dep = np.linalg.norm(pts - pts.mean(0), axis=1)
        dep_n = (dep - dep.min()) / max(dep.ptp(), 1e-9)
        save_ply(pts, plt.cm.turbo(dep_n)[:, :3], path_ply)
        return pts

    pd_pts = cloud_from(pdir, pconf_d, os.path.join(ck, "pointcloud_direct.ply"))
    pu_pts = cloud_from(punj, dconf, os.path.join(ck, "pointcloud_unprojected.ply"))

    # ---- 7. 参考点云对齐图(v2:相机中心 Sim3 变换整云 / mustc 用 FPFH-RANSAC+ICP) ----
    ref_cloud_path = seq.get("reference_pointcloud")
    if ref_cloud_path and os.path.exists(ref_cloud_path) and len(pu_pts) > 1000:
        try:
            ref_pts = load_ref_cloud(ref_cloud_path)
        except Exception as e:
            print(f"  ref cloud load fail: {e}")
            ref_pts = None
        if ref_pts is not None and len(ref_pts) > MAX_PTS:
            ref_pts = ref_pts[np.random.RandomState(0).choice(len(ref_pts), MAX_PTS, replace=False)]

        if ref_pts is not None and len(ref_pts) > 1000:
            vg_show = pu_pts[np.random.RandomState(0).choice(len(pu_pts), min(150_000, len(pu_pts)), replace=False)]
            if ds == "mustc":
                # LAS 为 UTM 系,与 plot-local 相机不同系:全局配准
                T, fit = fpfh_ransac_icp(vg_show[::4], ref_pts[::max(1, len(ref_pts)//150_000)])
                method = f"FPFH-RANSAC+ICP (fitness={fit:.2f})" if fit > 0.3 else "FPFH-RANSAC+ICP FAILED (no reliable alignment)"
                P_al = (T[:3, :3] @ vg_show.T).T + T[:3, 3] if fit > 0.3 else None
            else:
                # 参考云与参考相机同系:相机中心 Sim3 直接变换整云
                P_al = apply_sim3(sim3, vg_show)
                method = "camera-center Sim3 applied to full cloud"
            fig3 = plt.figure(figsize=(12, 6))
            a = fig3.add_subplot(1, 2, 1, projection="3d")
            a.scatter(ref_pts[::13, 0], ref_pts[::13, 1], ref_pts[::13, 2], s=0.3, c=depth_like_colors(ref_pts[::13]))
            a.set_title("reference cloud")
            a = fig3.add_subplot(1, 2, 2, projection="3d")
            if P_al is not None:
                a.scatter(P_al[::13, 0], P_al[::13, 1], P_al[::13, 2], s=0.3, c=depth_like_colors(P_al[::13]))
            a.set_title(f"VGGT unproj → ref frame\n[{method}]")
            fig3.savefig(os.path.join(ck, "align_refcloud_vs_vggt.png"), dpi=110)
            plt.close(fig3)
            print(f"  align: {method}")

    print(f"checks_v2 done -> {ck}")


if __name__ == "__main__":
    for sd in sys.argv[1:]:
        print(f"== checks_v2 {sd}")
        make_checks(sd)
