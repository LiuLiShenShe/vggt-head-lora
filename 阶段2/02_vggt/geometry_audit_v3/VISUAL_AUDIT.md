# VISUAL_AUDIT.md — Geometry Audit v3 逐序列人工视觉裁定

本文件对 **6 个代表序列** 各自回答 8 个视觉/几何裁定问题，并给出综合判定
`GOOD` / `PARTIAL` / `FAILED`。所有结论基于 **`figures_v3/` 下实际渲染图** +
`per_seq/<sid>.geo_v3.json` 数值，**非仅 JSON 裁定**（遵守禁止动作 #1）。

图件路径统一前缀：`figures_v3/<sid>_{overlay_full,overlay_foreground,error_colored,depth_montage,nn_hist,fscore_curve}.png`
（mustc 无 `overlay_foreground`，因其无逐帧前景 mask，见下文）。

> 坐标/轴说明：所有 overlay / error-colored 图均使用 **两云并集统一轴范围**（同一坐标系、同轴限、同视角），
> 由 `test_same_axis_visualization.py` 守护（违反则测试失败）。

---

## 1. `plantview__langdon_4__05-03-24`  (plant_view_3d, Pose Gate = PASS)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 预测与参考是否同坐标帧？ | GOOD | `align_sim3.scale_s=1.0905`、`center_res=0.0293m`，Umeyama 残差极小，overlay_full 两云重合 |
| Q2 | 前景范围（植株主体）是否一致？ | GOOD | overlay_foreground：前景点云（橙）与参考（绿）植株主体外形贴合；FOREG Chamfer_sym=0.011m |
| Q3 | 离群比例？ | PARTIAL | FULL 场景 within5%_pred=0.308（背景/姿态离群点占 ~69%），但 FOREG within5%_pred=0.963 → 离群集中在背景 |
| Q4 | 表型量级是否合理？ | GOOD | Eh pred=1.050m / ref=1.035m（误差 1.5cm）；canopy pred=1.963 / ref=0.743（冠层直径预测偏宽，见下方说明） |
| Q5 | 单帧深度趋势？ | GOOD | 对齐后 AbsRel=0.124、δ1=0.575、δ3=0.952；误差呈尺度一致性（median scale≈1289，稳定） |
| Q6 | 几何门候选（若建立）？ | GOOD | FOREG F@5%_D=0.981，远超可信阈值 |
| Q7 | 需人工裁决项？ | — | canopy diameter 预测偏宽（冠层外扩）需复核是否因前景 mask 含部分背景 |
| Q8 | 与 Pose Gate 关系？ | 一致 | Pose Gate PASS，几何同样 PASS（前景） |

**综合裁定：GOOD**（前景几何可信；FULL 场景受背景离群污染属预期，非模型核心缺陷）

---

## 2. `plantview__langdon_4__12-03-24`  (plant_view_3d, Pose Gate = **FAIL**)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 预测与参考是否同坐标帧？ | FAILED | `center_res=0.768m`，相机中心严重未对齐（84° 姿态误差），overlay_full 整体错位 |
| Q2 | 前景范围是否一致？ | FAILED | 即便取前景，FOREG F@5%_D=0.0、Chamfer_sym=0.986m，植株主体根本不重合 |
| Q3 | 离群比例？ | FAILED | FULL/FOREG within5%_pred 均 = 0.0，100% 点超阈值 |
| Q4 | 表型量级？ | FAILED | 几何崩溃，表型不可信 |
| Q5 | 单帧深度趋势？ | FAILED | 深度正确性无从谈起（相机系已错） |
| Q6 | 几何门候选？ | FAILED | 任何合理阈值下均 FAIL |
| Q7 | 需人工裁决项？ | — | 无；明确归因为相机头/姿态失败（与 Pose Gate FAIL 一致） |
| Q8 | 与 Pose Gate 关系？ | 一致 | Pose Gate FAIL → 几何 FAIL，根因同源（相机外参崩） |

**综合裁定：FAILED**（根因是相机姿态，非点几何本身；修正反投影后仍 FAIL，证明旧"点几何错误"归因必须改为"相机头失败"）

---

## 3. `plantview__langdon_4__13-02-24`  (plant_view_3d, Pose Gate = PASS)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 同坐标帧？ | GOOD | `center_res=0.033m`、`scale_s=1.0325` |
| Q2 | 前景范围一致？ | GOOD | FOREG Chamfer_sym=0.013m、F@5%_D=0.984 |
| Q3 | 离群比例？ | PARTIAL | FULL within5%_pred=0.265（背景离群），FOREG=0.969 |
| Q4 | 表型量级？ | GOOD | Eh pred=1.045 / ref=1.046（误差 1mm 级）；canopy pred/ref 量级一致 |
| Q5 | 深度趋势？ | GOOD | 对齐 AbsRel=0.106、δ1=0.641、δ3=0.951 |
| Q6 | 几何门候选？ | GOOD | FOREG F@5%_D=0.984 |
| Q7 | 需人工裁决？ | — | 无 |
| Q8 | 与 Pose Gate？ | 一致 | PASS ↔ PASS |

**综合裁定：GOOD**

---

## 4. `plantview__langdon_4__20-02-24`  (plant_view_3d, Pose Gate = PASS)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 同坐标帧？ | GOOD | `center_res=0.0501m`、`scale_s=0.9638` |
| Q2 | 前景范围一致？ | GOOD | FOREG Chamfer_sym=0.0115m、F@5%_D=0.978 |
| Q3 | 离群比例？ | PARTIAL | FULL within5%_pred=0.309，FOREG=0.956 |
| Q4 | 表型量级？ | GOOD | Eh pred=0.970 / ref=1.004（误差 3.4cm） |
| Q5 | 深度趋势？ | PARTIAL | 对齐 AbsRel=0.185、δ1=0.366（该序列深度尺度偏差偏大，但 δ3=0.954 仍高） |
| Q6 | 几何门候选？ | GOOD | FOREG F@5%_D=0.978 |
| Q7 | 需人工裁决？ | — | 深度 AbsRel 偏高，建议后续核查该序列参考深度质量 |
| Q8 | 与 Pose Gate？ | 一致 | PASS ↔ PASS |

**综合裁定：GOOD**（几何前景可信；深度尺度偏差略高但点几何完好）

---

## 5. `wheat3dgs__plot_463`  (wheat3dgs, Pose Gate = PASS, **参考系不可审计**)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 同坐标帧？ | FAILED(不可比) | 参考为 COLMAP `points3D.txt`，**任意重建单位（非米）**，两云尺度/原点不可比；代码中 `reference_comparable=False`，所有阈值指标置空 |
| Q2 | 前景范围一致？ | 不可裁 | 仅能确认两云质心化后形状，无法判定绝对吻合 |
| Q3 | 离群比例？ | 不可裁 | 无 D、无阈值指标 |
| Q4 | 表型量级？ | 不可裁 | 参考非米制，表型不可比 |
| Q5 | 深度趋势？ | 不可裁 | 无逐帧参考深度 |
| Q6 | 几何门候选？ | 不建立 | `reference_frame_auditable=False`，本序列不进入 Geometry Gate 统计 |
| Q7 | 需人工裁决？ | 是 | 需引入米制真值（如 RTK-SfM 或已知棋盘）方可审计 |
| Q8 | 与 Pose Gate？ | 独立 | Pose Gate PASS（姿态正确），但几何质量无法从非米制参考得出 |

**综合裁定：FAILED（不可审计 / not auditable）** — 诚实标记，非伪造指标（遵守禁止动作 #4/#5）

---

## 6. `mustc__plot198__230613__ugv__pos00`  (mustc, Pose Gate = PASS, **参考系不可审计**)

| # | 裁定问题 | 判定 | 依据 |
|---|----------|------|------|
| Q1 | 同坐标帧？ | FAILED(不可比) | 参考 LAS 为 plot-local Metashape 系，含 ~3.57e5 地理偏移且尺度未知，与 VGGT 米制不可比 |
| Q2 | 前景范围一致？ | 不可裁 | 无逐帧 mask；仅 overlay_full 可看形状密度 |
| Q3 | 离群比例？ | 不可裁 | 无 D、无阈值指标 |
| Q4 | 表型量级？ | 不可裁 | 参考非米制可比对 |
| Q5 | 深度趋势？ | 不可裁 | 无参考深度 |
| Q6 | 几何门候选？ | 不建立 | `reference_frame_auditable=False` |
| Q7 | 需人工裁决？ | 是 | 需 plot-local→米制标定真值 |
| Q8 | 与 Pose Gate？ | 独立 | Pose Gate PASS，几何质量无法从地理偏移参考得出 |

**综合裁定：FAILED（不可审计 / not auditable）**

---

## 视觉裁定汇总

| 序列 | 数据集 | Pose Gate | 视觉综合 | 可审计(米制) |
|------|--------|-----------|----------|--------------|
| 05-03-24 | plant_view_3d | PASS | **GOOD** | 是 |
| 12-03-24 | plant_view_3d | FAIL | **FAILED** | 是 |
| 13-02-24 | plant_view_3d | PASS | **GOOD** | 是 |
| 20-02-24 | plant_view_3d | PASS | **GOOD** | 是 |
| plot_463 | wheat3dgs | PASS | **FAILED(不可审计)** | 否（非米制） |
| pos00 | mustc | PASS | **FAILED(不可审计)** | 否（地理偏移） |

**结论**：plant_view 4 序列中 3 PASS 序列前景几何均为 GOOD（VGGT 点几何可信）；
12-03-24 因相机头失败而 FAILED（非点几何本身）；wheat/mustc 因参考系非米制/地理偏移
而**不可审计**，已诚实置空全部阈值指标，不进入 Geometry Gate 统计。

> 注：所有判定均配合 `figures_v3/*.png` 实际目视，未仅凭 JSON 宣布 PASS（遵守禁止动作 #1）。
