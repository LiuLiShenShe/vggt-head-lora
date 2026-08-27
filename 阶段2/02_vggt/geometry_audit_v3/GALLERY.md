# GALLERY.md — Geometry Audit v3 关键产物索引

> 所有 PNG 已由 `git add -f` 强制纳入（全局 `.gitignore` 忽略 `*.png`），GitHub 可见。
> 坐标约定：overlay/error-colored 图均两云并集 **统一轴范围、同坐标系、同视角**。

## 目录结构

```
阶段2/02_vggt/geometry_audit_v3/
├── unproject_v3.py              # 修正后反投影（≡ 官方，P0 门禁）
├── geometry_metrics_v3.py      # 双向指标 Chamfer/F/P/R@τ/D
├── foreground_v3.py            # 前景 mask 提取
├── align_v3.py                 # 参考云对齐（保存 before/after）
├── depth_audit_v3.py           # 单帧深度审计
├── phenotype_v3.py            # 表型指标
├── four_path_v3.py             # 修正反投影四路重算
├── run_geometry_audit_v3.py    # 主驱动
├── UNPROJECTION_AUDIT.json     # Q1 证据：旧公式错误
├── GEOMETRY_AUDIT_TABLE.csv    # 指标分布（不反向设计 gate）
├── four_path_v3/               # 修正后四路 verdict + grid 图
├── figures_v3/                 # 6 类图 × 6 代表序列（35 PNG）
├── per_seq/                    # 每序列 .geo_v3.json + before/after/ref .npy
├── tests/                      # 6 测试文件（全部 PASS）
├── VISUAL_AUDIT.md
├── GALLERY.md
└── PHASE22_GEOMETRY_AUDIT_V3.md
```

## 关键图像（figures_v3/）

### plant_view_3d — PASS 序列（前景几何可信）

| 序列 | overlay_full | overlay_foreground | error_colored | nn_hist | fscore_curve | depth_montage |
|------|--------------|-------------------|---------------|---------|--------------|---------------|
| 05-03-24 | [png](figures_v3/plantview__langdon_4__05-03-24_overlay_full.png) | [png](figures_v3/plantview__langdon_4__05-03-24_overlay_foreground.png) | [png](figures_v3/plantview__langdon_4__05-03-24_error_colored.png) | [png](figures_v3/plantview__langdon_4__05-03-24_nn_hist.png) | [png](figures_v3/plantview__langdon_4__05-03-24_fscore_curve.png) | [png](figures_v3/plantview__langdon_4__05-03-24_depth_montage.png) |
| 13-02-24 | [png](figures_v3/plantview__langdon_4__13-02-24_overlay_full.png) | [png](figures_v3/plantview__langdon_4__13-02-24_overlay_foreground.png) | [png](figures_v3/plantview__langdon_4__13-02-24_error_colored.png) | [png](figures_v3/plantview__langdon_4__13-02-24_nn_hist.png) | [png](figures_v3/plantview__langdon_4__13-02-24_fscore_curve.png) | [png](figures_v3/plantview__langdon_4__13-02-24_depth_montage.png) |
| 20-02-24 | [png](figures_v3/plantview__langdon_4__20-02-24_overlay_full.png) | [png](figures_v3/plantview__langdon_4__20-02-24_overlay_foreground.png) | [png](figures_v3/plantview__langdon_4__20-02-24_error_colored.png) | [png](figures_v3/plantview__langdon_4__20-02-24_nn_hist.png) | [png](figures_v3/plantview__langdon_4__20-02-24_fscore_curve.png) | [png](figures_v3/plantview__langdon_4__20-02-24_depth_montage.png) |

### plant_view_3d — FAIL 序列（相机头失败，几何随之崩溃）

| 序列 | overlay_full | overlay_foreground | error_colored | nn_hist | fscore_curve | depth_montage |
|------|--------------|-------------------|---------------|---------|--------------|---------------|
| 12-03-24 | [png](figures_v3/plantview__langdon_4__12-03-24_overlay_full.png) | [png](figures_v3/plantview__langdon_4__12-03-24_overlay_foreground.png) | [png](figures_v3/plantview__langdon_4__12-03-24_error_colored.png) | [png](figures_v3/plantview__langdon_4__12-03-24_nn_hist.png) | [png](figures_v3/plantview__langdon_4__12-03-24_fscore_curve.png) | [png](figures_v3/plantview__langdon_4__12-03-24_depth_montage.png) |

### 不可审计数据集（非米制 / 地理偏移参考）

| 序列 | overlay_full | error_colored | nn_hist | fscore_curve | depth_montage |
|------|--------------|---------------|---------|--------------|---------------|
| wheat3dgs__plot_463 | [png](figures_v3/wheat3dgs__plot_463_overlay_full.png) | [png](figures_v3/wheat3dgs__plot_463_error_colored.png) | [png](figures_v3/wheat3dgs__plot_463_nn_hist.png) | [png](figures_v3/wheat3dgs__plot_463_fscore_curve.png) | [png](figures_v3/wheat3dgs__plot_463_depth_montage.png) |
| mustc__pos00 | [png](figures_v3/mustc__plot198__230613__ugv__pos00_overlay_full.png) | [png](figures_v3/mustc__plot198__230613__ugv__pos00_error_colored.png) | [png](figures_v3/mustc__plot198__230613__ugv__pos00_nn_hist.png) | [png](figures_v3/mustc__plot198__230613__ugv__pos00_fscore_curve.png) | [png](figures_v3/mustc__plot198__230613__ugv__pos00_depth_montage.png) |

> wheat/mustc 无 `overlay_foreground`（无逐帧前景 mask，`reference_frame_auditable=False`）。

### 四路修正重算（four_path_v3/）

- [success_05-03-24_grid.png](four_path_v3/success_05-03-24_grid.png) — A/B/C 三路到参考距离（修正反投影）
- [fail_12-03-24_grid.png](four_path_v3/fail_12-03-24_grid.png) — 12-03-24 A/C 全部超截断（相机头失败）
- [verdict.json](four_path_v3/verdict.json) — 数值 verdict
- [metric_definitions.json](four_path_v3/metric_definitions.json) — 指标定义

## 每序列 JSON（per_seq/）

| 序列 | geo_v3.json | align_sim3 | pred_before | pred_aligned | ref |
|------|-------------|-----------|-------------|--------------|-----|
| 05-03-24 | [json](per_seq/plantview__langdon_4__05-03-24.geo_v3.json) | [json](per_seq/plantview__langdon_4__05-03-24_align_sim3.json) | [npy](per_seq/plantview__langdon_4__05-03-24_pred_before.npy) | [npy](per_seq/plantview__langdon_4__05-03-24_pred_aligned.npy) | [npy](per_seq/plantview__langdon_4__05-03-24_ref.npy) |
| 12-03-24 | [json](per_seq/plantview__langdon_4__12-03-24.geo_v3.json) | [json](per_seq/plantview__langdon_4__12-03-24_align_sim3.json) | [npy](per_seq/plantview__langdon_4__12-03-24_pred_before.npy) | [npy](per_seq/plantview__langdon_4__12-03-24_pred_aligned.npy) | [npy](per_seq/plantview__langdon_4__12-03-24_ref.npy) |
| 13-02-24 | [json](per_seq/plantview__langdon_4__13-02-24.geo_v3.json) | [json](per_seq/plantview__langdon_4__13-02-24_align_sim3.json) | [npy](per_seq/plantview__langdon_4__13-02-24_pred_before.npy) | [npy](per_seq/plantview__langdon_4__13-02-24_pred_aligned.npy) | [npy](per_seq/plantview__langdon_4__13-02-24_ref.npy) |
| 20-02-24 | [json](per_seq/plantview__langdon_4__20-02-24.geo_v3.json) | [json](per_seq/plantview__langdon_4__20-02-24_align_sim3.json) | [npy](per_seq/plantview__langdon_4__20-02-24_pred_before.npy) | [npy](per_seq/plantview__langdon_4__20-02-24_pred_aligned.npy) | [npy](per_seq/plantview__langdon_4__20-02-24_ref.npy) |
| wheat3dgs__plot_463 | [json](per_seq/wheat3dgs__plot_463.geo_v3.json) | [json](per_seq/wheat3dgs__plot_463_align_sim3.json) | [npy](per_seq/wheat3dgs__plot_463_pred_before.npy) | [npy](per_seq/wheat3dgs__plot_463_pred_aligned.npy) | [npy](per_seq/wheat3dgs__plot_463_ref.npy) |
| mustc__pos00 | [json](per_seq/mustc__plot198__230613__ugv__pos00.geo_v3.json) | [json](per_seq/mustc__plot198__230613__ugv__pos00_align_sim3.json) | [npy](per_seq/mustc__plot198__230613__ugv__pos00_pred_before.npy) | [npy](per_seq/mustc__plot198__230613__ugv__pos00_pred_aligned.npy) | [npy](per_seq/mustc__plot198__230613__ugv__pos00_ref.npy) |

## Pose Gate 状态（冻结，未重算）

| 序列 | dataset | pose_gate | reference_auditable | geometry_gate |
|------|---------|-----------|---------------------|---------------|
| 05-03-24 | plant_view_3d | PASS (True) | True | not_yet_established |
| 12-03-24 | plant_view_3d | **FAIL** (False) | True | not_yet_established |
| 13-02-24 | plant_view_3d | PASS (True) | True | not_yet_established |
| 20-02-24 | plant_view_3d | PASS (True) | True | not_yet_established |
| wheat3dgs__plot_463 | wheat3dgs | PASS (True) | **False** | not_yet_established |
| mustc__pos00 | mustc | PASS (True) | **False** | not_yet_established |

> ⚠️ Pose Gate（14/17）与 Geometry Gate **严格分离**：前者冻结于 `gate_stats_clean_rerun.json`，
> 后者 **NOT YET ESTABLISHED**——本次仅给出指标分布与可视化，未反向设计通过率（遵守禁止动作 #2/#6）。

## 测试状态

`pytest tests/` → **26 passed**。关键守卫测试：
- `test_unprojection_vs_official.py` — custom ≡ 官方（P0 门禁）
- `test_geometry_gate_no_v1_fields.py` — 无 v1 字段、无 "几何 14/17" 措辞
- `test_same_axis_visualization.py` — overlay 同轴范围
