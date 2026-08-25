"""阶段2.2:检查图生成(da3 环境,需 open3d/matplotlib)。

用法: python make_checks.py <seq_dir> [<seq_dir2> ...]
seq_dir 即 run_vggt_inference.py 的输出目录(含 prediction_meta.json)。
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
from provenance import FROZEN  # noqa

SEQ_BASE = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences"
MAX_PTS = 500_000


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
        w = max(sum(im.shape[1] for im in row), cols * size)
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


def horn_sim3(src, dst):
    """Umeyama-Horn: 求 s,R,t 使 dst ≈ s R src + t。返回变换后的 src。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return (s * (R @ src.T).T + t)


def make_checks(seq_dir: str):
    meta = json.load(open(os.path.join(seq_dir, "prediction_meta.json")))
    sid = meta["sequence_id"]
    ds = meta["dataset_id"]
    ck = os.path.join(seq_dir, "checks")

    depth = np.load(os.path.join(seq_dir, "depth_vggt.npy"))
    dconf = np.load(os.path.join(seq_dir, "depth_conf_vggt.npy"))
    pdir = np.load(os.path.join(seq_dir, "point_map_direct.npy"))
    pconf_d = np.load(os.path.join(seq_dir, "point_conf_direct.npy"))
    punj = np.load(os.path.join(seq_dir, "point_map_unprojected.npy"))

    # sequence.json 文件名与输出目录名可能不同:直接扫描该数据集目录下所有 json 匹配 sequence_id
    ds_dir = os.path.join(SEQ_BASE,
                          {"plant_view_3d": "plant_view", "wheat3dgs": "wheat3dgs",
                           "mustc": "mustc", "terra_ref": "terraref"}[ds])
    seq_json = next(p for p in glob.glob(os.path.join(ds_dir, "*.json"))
                    if json.load(open(p))["sequence_id"] == sid)
    seq = json.load(open(seq_json))
    S = len(depth)

    from PIL import Image
    n_show = min(S, 12)
    idx = np.linspace(0, S - 1, n_show).astype(int)

    # 1. 输入缩略图
    thumbs = [np.array(Image.open(seq["rgb_paths"][i]).convert("RGB").resize((256, 256)))
              for i in idx]
    Image.fromarray(grid(thumbs)).save(os.path.join(ck, "input_thumbs.png"))

    # 2/3. depth 彩色 & conf
    Image.fromarray(grid([depth_to_color(depth[i]) for i in idx])).save(
        os.path.join(ck, "depth_colored.png"))
    Image.fromarray(grid([conf_to_color(dconf[i]) for i in idx])).save(
        os.path.join(ck, "depth_conf.png"))

    # 4. 相机视锥图(+参考相机叠加)
    ext_c2w = np.load(os.path.join(seq_dir, "extrinsic_c2w.npy"))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    C = ext_c2w[:, :3, 3]
    scale = float(np.percentile(np.linalg.norm(C - C.mean(0), axis=1), 90)) / 8
    for T in ext_c2w[::max(1, S // 40)]:
        z = T[:3, :3] @ np.array([0, 0, 1]) * scale
        ax.plot(*zip(T[:3, 3], T[:3, 3] + z), color="tab:red", lw=0.8)
    ax.scatter(*C[::max(1, S // 40)].T, s=4, c="red", label="VGGT cams")

    ref_ext_path = seq.get("extrinsics_path")
    if ref_ext_path and os.path.exists(ref_ext_path):
        ref = json.load(open(ref_ext_path))["extrinsics"]
        Rw = np.stack([np.array(e["w2c"])[:3, :3].T for e in ref])
        tw = np.stack([np.array(e["w2c"])[:3, 3] for e in ref])
        ref_C = -(Rw @ tw[..., None])[..., 0]          # ref cam centers (ref world)
        ref_z = np.einsum("nij,ni->ni", Rw, np.tile(np.array([0.0, 0.0, 1.0]), (len(Rw), 1))) * scale
        for i in range(0, len(ref_C), max(1, len(ref_C) // 40)):
            ax.plot(*zip(ref_C[i], ref_C[i] + ref_z[i]), color="tab:blue", lw=0.8)
        # Sim(3): 把 VGGT 相机中心对齐到参考相机中心再画
        aligned = horn_sim3(C, ref_C)
        ax.scatter(*ref_C[::max(1, len(ref_C) // 40)].T, s=4, c="blue", label="ref cams")
        ax.scatter(*aligned[::max(1, S // 40)].T, s=4, c="green", label="VGGT→ref align")
    ax.legend(); ax.set_title(f"{sid} cameras")
    fig.savefig(os.path.join(ck, "camera_frustums.png"), dpi=110); plt.close(fig)
    # 同图另存一份命名(规范要求 overlay_ref_vs_vggt_cams.png)
    if ref_ext_path:
        fig2 = plt.figure(figsize=(10, 10)); a2 = fig2.add_subplot(projection="3d")
        a2.scatter(*C.T, s=3, c="red", label="VGGT")
        if ref_ext_path and os.path.exists(ref_ext_path):
            a2.scatter(*ref_C.T, s=3, c="blue", label="reference")
        a2.legend(); a2.set_title("ref vs VGGT camera centers")
        fig2.savefig(os.path.join(ck, "overlay_ref_vs_vggt_cams.png"), dpi=110)
        plt.close(fig2)

    # 5/6. 两路点云 PLY(conf 过滤 + 下采样)+ 7. 对齐渲染图
    def cloud_from(pmap, conf, path_ply, name):
        pts = pmap.reshape(-1, 3)
        cf = conf.reshape(-1)
        thr = np.percentile(cf, 30)
        m = cf >= thr
        pts, cf = pts[m], cf[m]
        if len(pts) > MAX_PTS:
            sel = np.random.RandomState(0).choice(len(pts), MAX_PTS, replace=False)
            pts, cf = pts[sel], cf[sel]
        # 颜色按深度着色
        dep = np.linalg.norm(pts - pts.mean(0), axis=1)
        dep_n = ((dep - dep.min()) / max(dep.ptp(), 1e-9))
        colors = plt.cm.turbo(dep_n)[:, :3]
        save_ply(pts, colors, path_ply)
        return pts

    pd_pts = cloud_from(pdir, pconf_d, os.path.join(ck, "pointcloud_direct.ply"), "direct")
    pu_pts = cloud_from(punj, dconf, os.path.join(ck, "pointcloud_unprojected.ply"), "unproj")

    # 参考点云对齐图
    ref_cloud_path = seq.get("reference_pointcloud")
    if ref_cloud_path and os.path.exists(ref_cloud_path):
        import open3d as o3d
        try:
            if ref_cloud_path.endswith(".txt"):
                ref_pts = np.loadtxt(ref_cloud_path, usecols=(1, 2, 3))
            elif ref_cloud_path.endswith(".las"):
                import laspy
                las = laspy.read(ref_cloud_path)
                ref_pts = np.stack([las.x, las.y, las.z], axis=1)
            else:
                g = o3d.io.read_point_cloud(ref_cloud_path)
                ref_pts = np.asarray(g.points)
        except Exception as e:
            print(f"  ref cloud load fail: {e}")
            ref_pts = None
        if ref_pts is not None and len(ref_pts) > MAX_PTS:
            ref_pts = ref_pts[np.random.RandomState(0).choice(len(ref_pts), MAX_PTS, replace=False)]
        if ref_pts is not None and len(pu_pts) > 1000 and len(ref_pts) >= len(pu_pts[::7]):
            try:
                n_sub = len(pu_pts[::7])
                ref_sub = ref_pts[np.random.RandomState(0).choice(len(ref_pts), n_sub, replace=False)]
                vg_al = horn_sim3(pu_pts[::7], ref_sub)
                fig3 = plt.figure(figsize=(12, 6))
                for k, (P_, ttl) in enumerate([(ref_pts[::13], "reference cloud"),
                                               (vg_al[::13], "VGGT unproj → ref (Sim3)")]):
                    a = fig3.add_subplot(1, 2, k + 1, projection="3d")
                    a.scatter(P_[:, 0], P_[:, 1], P_[:, 2], s=0.3, c=depth_like_colors(P_))
                    a.set_title(ttl)
                fig3.savefig(os.path.join(ck, "align_refcloud_vs_vggt.png"), dpi=110)
                plt.close(fig3)
            except Exception as e:
                print(f"  align plot fail: {e}")

    print(f"checks done -> {ck}")


def depth_like_colors(P):
    d = P[:, 2] - P[:, 2].min()
    d = d / max(d.max(), 1e-9)
    return plt.cm.viridis(d)


if __name__ == "__main__":
    for sd in sys.argv[1:]:
        print(f"== checks {sd}")
        make_checks(sd)
