# GALLERY_V31.md — v3.1 图件索引

所有图件位于 `figures_v31/` 与 `four_path_v4/`。前缀说明：`overlay_fg_*` = 前景点云叠加（ref绿/VGGT橙）；
`fg_error_*` = 前景误差着色图（颜色阈值 mm）；`depth_montage_real` = 真实深度 montage（m colorbars）；
`*_routes_overlay` = 四路 v4 因式分解对比（5 路 × 4 视角）。

---

## 1. 修正后前景可视化（P0-1/2/3）

| 序列 | 3D | XY | XZ | YZ | error(pred→gt) | error(gt→pred) |
|------|----|----|----|----|----------------|----------------|
| 05-03-24 | [3d](figures_v31/plantview__langdon_4__05-03-24_overlay_fg_3d.png) | [xy](figures_v31/plantview__langdon_4__05-03-24_overlay_fg_xy.png) | [xz](figures_v31/plantview__langdon_4__05-03-24_overlay_fg_xz.png) | [yz](figures_v31/plantview__langdon_4__05-03-24_overlay_fg_yz.png) | [p2g](figures_v31/plantview__langdon_4__05-03-24_fg_error_pred2gt.png) | [g2p](figures_v31/plantview__langdon_4__05-03-24_fg_error_gt2pred.png) |
| 12-03-24 | [3d](figures_v31/plantview__langdon_4__12-03-24_overlay_fg_3d.png) | [xy](figures_v31/plantview__langdon_4__12-03-24_overlay_fg_xy.png) | [xz](figures_v31/plantview__langdon_4__12-03-24_overlay_fg_xz.png) | [yz](figures_v31/plantview__langdon_4__12-03-24_overlay_fg_yz.png) | [p2g](figures_v31/plantview__langdon_4__12-03-24_fg_error_pred2gt.png) | [g2p](figures_v31/plantview__langdon_4__12-03-24_fg_error_gt2pred.png) |
| 13-02-24 | [3d](figures_v31/plantview__langdon_4__13-02-24_overlay_fg_3d.png) | [xy](figures_v31/plantview__langdon_4__13-02-24_overlay_fg_xy.png) | [xz](figures_v31/plantview__langdon_4__13-02-24_overlay_fg_xz.png) | [yz](figures_v31/plantview__langdon_4__13-02-24_overlay_fg_yz.png) | [p2g](figures_v31/plantview__langdon_4__13-02-24_fg_error_pred2gt.png) | [g2p](figures_v31/plantview__langdon_4__13-02-24_fg_error_gt2pred.png) |
| 20-02-24 | [3d](figures_v31/plantview__langdon_4__20-02-24_overlay_fg_3d.png) | [xy](figures_v31/plantview__langdon_4__20-02-24_overlay_fg_xy.png) | [xz](figures_v31/plantview__langdon_4__20-02-24_overlay_fg_xz.png) | [yz](figures_v31/plantview__langdon_4__20-02-24_overlay_fg_yz.png) | [p2g](figures_v31/plantview__langdon_4__20-02-24_fg_error_pred2gt.png) | [g2p](figures_v31/plantview__langdon_4__20-02-24_fg_error_gt2pred.png) |
| plot_463 | [3d](figures_v31/wheat3dgs__plot_463_overlay_fg_3d.png) | [xy](figures_v31/wheat3dgs__plot_463_overlay_fg_xy.png) | [xz](figures_v31/wheat3dgs__plot_463_overlay_fg_xz.png) | [yz](figures_v31/wheat3dgs__plot_463_overlay_fg_yz.png) | [p2g](figures_v31/wheat3dgs__plot_463_fg_error_pred2gt.png) | [g2p](figures_v31/wheat3dgs__plot_463_fg_error_gt2pred.png) |

**修复验证**：metric 输入点 ≡ figure 输入点（同一 `P_fore`/`Q_fore` array，已对齐），符合 P0-1。

---

## 2. 真实深度 Montage（P0-4）

列：RGB | GT depth(m) | VGGT raw(m) | VGGT aligned(m) | Abs error(m) | Rel error（含 m colorbars）

| 序列 | 文件 |
|------|------|
| 05-03-24 | [depth_montage_real](figures_v31/plantview__langdon_4__05-03-24_depth_montage_real.png) |
| 12-03-24 | [depth_montage_real](figures_v31/plantview__langdon_4__12-03-24_depth_montage_real.png) |
| 13-02-24 | [depth_montage_real](figures_v31/plantview__langdon_4__13-02-24_depth_montage_real.png) |
| 20-02-24 | [depth_montage_real](figures_v31/plantview__langdon_4__20-02-24_depth_montage_real.png) |

**废弃图（v3）**：`figures_v3/*_depth_montage.png`（仅 validity mask，非深度值）→ 见 `VISUAL_AUDIT_v3_DEPRECATED.md`

---

## 3. Four-Path v4 因式分解对比（P1）

每文件 = 5 路（A/B-K/B-E/B-KE/C）× 4 视角（3D/XY/XZ/YZ）。参考点云叠加比较。

| 序列 | n=8 | n=16 | n=24 | n=36 |
|------|-----|------|------|------|
| success_05-03-24 | [n8](four_path_v4/success_05-03-24_n8_routes_overlay.png) | [n16](four_path_v4/success_05-03-24_n16_routes_overlay.png) | [n24](four_path_v4/success_05-03-24_n24_routes_overlay.png) | [n36](four_path_v4/success_05-03-24_n36_routes_overlay.png) |
| fail_12-03-24 | [n8](four_path_v4/fail_12-03-24_n8_routes_overlay.png) | [n16](four_path_v4/fail_12-03-24_n16_routes_overlay.png) | [n24](four_path_v4/fail_12-03-24_n24_routes_overlay.png) | [n36](four_path_v4/fail_12-03-24_n36_routes_overlay.png) |

**数值**：`four_path_v4/verdict_v4.json`；**定义**：`four_path_v4/metric_definitions.json`

---

## 4. Scanner-GT 验证图（P2）

| 序列 | 文件 |
|------|------|
| langdon_4/19-03-24 | `figures_v31/scanner_gt_*.png`（camera-sim3 + icp 对齐对比）|

来源：`scanner_gt/SCANNER_GT_MANIFEST.json` + `scanner_gt/SCANNER_GT_GEOMETRY_TABLE.csv`

---

## 5. CSV 数据表

- `FOREGROUND_METRICS_V31.csv` — 严格 5/10/20/50mm 前景指标
- `PHENOTYPE_OUTLIER_SENSITIVITY.csv` — 5 种 outlier 处理下的 robust 表型
- `GEOMETRY_AUDIT_TABLE.csv` — 全序列汇总（v3.1 修正）
- `DEPTH_UNIT_AUDIT.json` — 参考深度单位 VERIFIED（scale=0.001）
- `SCANNER_GT_GEOMETRY_TABLE.csv` — scanner-GT 几何表（单株）

---

## 6. 弃用图（v3 缺陷遗留）

`VISUAL_AUDIT_v3_DEPRECATED.md` 记录 v3 图件为何弃用：
1. overlay_fg 用未对齐点（画图 ≠ 指标）
2. depth montage 仅 validity mask（非深度值）
3. 3DGS 被误称 GT（实为 pseudo-reference）
4. 参考深度单位未审计（uint16 当米用）
