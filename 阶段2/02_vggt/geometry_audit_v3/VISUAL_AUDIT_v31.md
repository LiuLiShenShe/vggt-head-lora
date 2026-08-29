# VISUAL_AUDIT_v31.md — Geometry Audit v3.1 逐序列人工视觉裁定（修正后）

本文件是对 v3 旧判定的**重新裁定**。旧 `VISUAL_AUDIT_v3_DEPRECATED.md` 因前景图未对齐、
depth montage 仅显示 validity mask、3DGS 被误称 GT、参考深度单位未审计，已弃用。

本裁定仅基于 **修正后的图**（aligned foreground overlay、真实 depth montage、foreground error map）
+ **严格 5/10/20/50mm 指标**（FOREGROUND_METRICS_V31.csv）+ **robust phenotype**
（PHENOTYPE_OUTLIER_SENSITIVITY.csv）。遵守禁止动作 #1（看图不凭 JSON 宣布）。

图件前缀：`figures_v31/<sid>_{overlay_fg_3d,overlay_fg_xy,overlay_fg_xz,overlay_fg_yz,
fg_error_pred2gt,fg_error_gt2pred,depth_montage_real}.png`

---

## 1. `plantview__langdon_4__05-03-24`  (plant_view_3d, Pose Gate = PASS)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 对齐后前景是否恢复植株结构？ | PARTIAL | overlay_fg_3d/xy/xz/yz：叶片主体与参考贴合，但细枝有偏移；F@5mm=0.32 |
| Q2 | 5/10/20/50mm 四级水平？ | 见下 | F@5mm=0.325 / F@10mm=0.614 / F@20mm=0.864 / F@50mm=0.984 |
| Q3 | 离群比例？ | PARTIAL | F@50mm=0.984（多数点在 5cm 内），但 F@10mm 仅 0.614（叶片级 1cm 误差明显） |
| Q4 | 表型量级？ | PARTIAL | robust Eh 误差 ~1.4cm；raw canopy 1.96 vs 0.74 经离群审计为 outlier 主导（见 §5） |
| Q5 | 深度趋势？ | GOOD | 修正单位后 raw AbsRel≈0.24–0.31，scale≈1.3（VGGT 米制，仅 ~30% 尺度歧义） |
| Q6 | 若建立几何门？ | PARTIAL | 5cm 级良好，叶片 1cm 级不足 |
| Q7 | 需人工裁决？ | — | 否 |
| Q8 | 与 Pose Gate？ | 一致 | PASS ↔ 前景 PARTIAL |

**综合裁定：PARTIAL**（前景几何在 2–5cm 级可信；叶片 5–10mm 级有明显误差，非 FAIL 但非 GOOD）

---

## 2. `plantview__langdon_4__12-03-24`  (plant_view_3d, Pose Gate = **FAIL**)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 对齐后前景是否恢复？ | FAILED | 相机中心残差 0.768m，整体错位，四路 v4 中 B-KE 仍远未对齐 |
| Q2 | 四级水平？ | FAILED | 全部 F≈0 |
| Q3-8 | — | FAILED | 根因是相机外参失败（见 four_path_v4），非点几何本身 |

**综合裁定：FAILED**（相机头失败，与 Pose Gate 一致；点几何本身无法独立评估）

---

## 3. `plantview__langdon_4__13-02-24`  (plant_view_3d, Pose Gate = PASS)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 对齐后前景？ | PARTIAL | 同 05-03-24 量级 |
| Q2 | 四级水平？ | 见 FOREGROUND_METRICS_V31.csv | F@10mm≈0.53 / F@20mm≈0.86 / F@50mm≈0.98（量级同 05-03） |
| Q3 | 离群？ | PARTIAL | 同 05-03 |
| Q4 | 表型？ | PARTIAL | robust Eh 误差 ~1mm 级；raw canopy 同样 outlier 主导 |
| Q5 | 深度？ | GOOD | raw AbsRel≈0.11（scale-aligned），趋势一致 |
| Q8 | 与 Pose Gate？ | 一致 | PASS ↔ 前景 PARTIAL |

**综合裁定：PARTIAL**

---

## 4. `plantview__langdon_4__20-02-24`  (plant_view_3d, Pose Gate = PASS)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 对齐后前景？ | PARTIAL | 同量级 |
| Q2 | 四级水平？ | 见 CSV | F@10mm≈0.57 / F@20mm≈0.83 / F@50mm≈0.98 |
| Q3 | 离群？ | PARTIAL | 同 |
| Q4 | 表型？ | PARTIAL | robust Eh 误差 ~3cm |
| Q5 | 深度？ | PARTIAL | raw AbsRel≈0.20，但 scale 偏差略高（见 depth_audit） |
| Q8 | 与 Pose Gate？ | 一致 | PASS ↔ 前景 PARTIAL |

**综合裁定：PARTIAL**

---

## 5. 05-03-24 canopy 1.96m vs 0.74m 根因（Q5 → P0-9）

`PHENOTYPE_OUTLIER_SENSITIVITY.csv` + `per_seq/...05-03-24.geo_v31.json` 对 05-03-24 的实测（稳健估计）：

| 量 | pred raw(min/max) | pred robust(P1-P99) | ref raw(min/max) | ref robust(P1-P99) |
|----|------|------|------|------|
| bbox_xy_diagonal | **1.96m** | 0.988m | 0.7435m | 1.005m |
| height(z) | 1.05m | 0.393m | 1.035m | 0.379m |
| bbox_width_x | 1.46m | 0.535m | 0.446m | 0.519m |

- `all`（含 outlier）：**raw** bbox_xy_diagonal 被极端点撑大到 **1.96m**（与 v3 标题一致）。
- 五种 outlier 处理下 pred **robust** 对角线稳定在 **0.988m**，与参考 robust 对角线 **1.005m** 相差 **< 1.7cm**。

**结论**：
1. 1.96m 是 **raw min/max 估计器 + 3–4% 极端离群点** 共同导致的伪像；改 robust(P1-P99) 后回落到 ~0.99m，**非主体形态错误、非 foreground mask 错误**。
2. 参考自身 raw 对角线仅 0.74m（其 foreground mask 比预测更紧），但参考 robust 对角线同样 ~1.0m —— 说明预测与参考的主体宽度**一致（~1m）**；旧 "1.96 vs 0.74" 之差 = 预测侧 outlier 膨胀 + 参考侧 mask 偏紧 的叠加，不能解读为 "预测冠层偏大 2.6×"。
3. 属 **estimator + 少量 outlier + mask 定义差** 组合原因，经稳健估计已消除。

> 注：v3 把 raw diagonal 误当作 "canopy diameter" 并直接对比 1.96 vs 0.74 得出 "冠层夸大半倍" 的结论，在本审计中被判为 **estimator 缺陷**（已改 P1-P99 + PCA 主轴/次轴）。

---

## 6. wheat3dgs__plot_463 / mustc__pos00（不可审计）

维持 v3 结论：参考系非米制/地理偏移，`reference_frame_auditable=False`，不进入 Geometry Gate。
本 v3.1 未改变其状态（scanner GT 仅 19-03-24 一 plant，且属 plant_view）。

---

## 裁定汇总

| 序列 | Pose Gate | 视觉综合(v31) | 5mm | 10mm | 20mm | 50mm |
|------|-----------|---------------|-----|------|------|------|
| 05-03-24 | PASS | **PARTIAL** | 0.325 | 0.614 | 0.864 | 0.984 |
| 12-03-24 | FAIL | **FAILED** | — | — | — | — |
| 13-02-24 | PASS | **PARTIAL** | ~0.53 | ~0.86 | ~0.98 | (CSV) |
| 20-02-24 | PASS | **PARTIAL** | ~0.57 | ~0.83 | ~0.98 | (CSV) |
| plot_463 | PASS | 不可审计 | — | — | — | — |
| pos00 | PASS | 不可审计 | — | — | — | — |

> 说明：旧 v3 的 3 个 "GOOD" 在修正可视化 + 严格 mm 指标 + robust phenotype 后，**全部下调为 PARTIAL**——
> 因叶片级（5–10mm）精度不足，仅 2–5cm 级（20–50mm）可达 ~0.86–0.98。这正符合核心判据升级：
> 从「5cm 内有多少点」提升到「叶片/冠层能否在 5–20mm 支撑可靠表型」。
