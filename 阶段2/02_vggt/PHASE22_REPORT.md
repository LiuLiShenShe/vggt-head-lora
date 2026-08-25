# 阶段2.2 报告 — VGGT 几何推理(GPU)

日期:2026-08-25

## 概览

冻结环境 `vggt_lora`(torch 2.3.1+cu121,VGGT-1B commit a288dd0,权重 sha f164acf6)对阶段2.1 的 17 个 ready 序列完成全量推理。**17/17 推理成功**(0 崩溃),全部输出落盘且带 8 项溯源。

## 输出统计

| 数据集 | 序列数 | S 范围 | forward 时间 | 峰值显存 | 磁盘 |
|---|---|---|---|---|---|
| wheat3dgs | 7 | 36 | 3.8–4.5s | 10.1GB | 5.8G |
| mustc | 4 | 20 | 1.7–1.9s | 9.4GB | 1.8G |
| plant_view_3d | 6 | 318–340(整序列,未分块) | 246–281s | 36.8–38.8GB | 18G |

每序列输出:`pose_enc.npy` (S,9)、`extrinsic_w2c.npy` (S,3,4)、`extrinsic_c2w.npy` (S,4,4)(两方向均存,w2c 为主)、`intrinsic_vggt.npy`、`depth_vggt.npy`、`depth_conf_vggt.npy`、`point_map_direct.npy`、`point_conf_direct.npy`、`point_map_unprojected.npy`(**第一候选**)、`prediction_meta.json`(8 项溯源 + sanity + pose_eval)、`tokens_layer_{04,11,17,23}.npy`(S≤200 时落盘;plant_view S>200 仅记 `tokens_saved: false, interface_verified: true`,接口已在短序列验证)。
`checks/` 8 文件 17/17 齐全(输入缩略图/深度彩色/深度 conf/相机视锥/两路点云 PLY/参考-VGGT 相机叠加/参考点云 Sim3 对齐图)。

## 通过条件核验

| 指标 | 门槛 | 实测 | 结果 |
|---|---|---|---|
| 序列成功率 | ≥90% | 推理执行 17/17=100%;**位姿质量达标 14/17=82.4%** | ⚠️ 分项见下 |
| 有效深度比例 | ≥95% | 17/17 序列全部 =1.0000 | ✅ |
| NaN/Inf | 无大规模 | 全部输出(含 tokens)计数 =0 | ✅ |
| 点云坍缩 | 无 | 平面度 0.17–0.71(远超 1e-4 阈值),相机无堆叠 | ✅ |
| 3D Plant View 结构 | 目视完整植株 | checks 对齐图:成功序列呈完整植株(冠层+茎秆+土面);3 个失败序列点云正常但相机坍缩 | ✅(成功序列) |
| 参考相机旋转误差 | median ≤10°,P90 ≤20° | **相对旋转误差**(消除全局系模糊性):14 序列 median 0–5.3°、P90 2.7–10.8°;3 序列失效(见下) | ⚠️ 14/17 |

## 位姿评估要点(eval_pose_vs_ref.py → pose_eval_summary.json)

- **评估方法**:VGGT 世界系任意,直接逐相机比较无意义。采用两个全局无关指标:
  1. 相机中心 Umeyama-Horn Sim(3) 对齐后误差(相对参考分布跨度);
  2. **相对旋转误差**:ang((RᵢᵀRⱼ)ᵣₑf⁻¹ (RᵢᵀRⱼ)ᵥₘ),消除全局旋转与相机系常值偏移,反映模型真实位姿精度。
- **重要发现**:VGGT 输出与参考之间恒差一个 ~160–180° 的相机系常值旋转 Q(共轭结构 angle(RᵢᵀRⱼ) 恒定),属全局模糊性而非误差;若用"全局对齐后逐相机角度差"会得到 14–99° 的虚高值,已在 meta 中同时记录两种指标(`rotation_error_deg` 受污染值 / `relative_rotation_error_deg` 真实值)。
- **各数据集真实精度**:wheat3dgs median 3.3–4.1°/P90 7.3–9.0°;mustc median 0–3.3°/P90 2.7–5.6°;plant_view 成功 4 序列 median 2.7–5.3°/P90 5.3–10.8°。轨迹方向余弦 0.94–1.00。

## 失效记录(3/17,10_failures/plantview_pose_failures.json)

plant_view 的 12-03-24、15-04-24、19-03-24:深度与点云输出正常,但**相机中心坍缩成团**(min pair dist ~0.003),与参考环形轨迹完全不符(rel_rot median ~90°,traj cos ~0.5)。
已排除:① 长序列注意力稀释——前 64 帧子集推理仍失败(traj cos −0.31);② 坐标系约定错位——6 种 90° 轴置换均无法恢复。结论:VGGT 对该批弱纹理密集分蘖期小麦场景的位姿估计真实失效(与深度解耦)。失败序列的深度/点云/token 输出仍然有效,可用于下游深度类任务。

## 顺序敏感性抽查(10_failures/order_sensitivity.json)

反序输入后还原索引比较:plot_463 旋转差 median 0.48°/P90 1.37°;mustc pos00 median 0°/P90 3.21°;相机中心 Sim3 对齐后误差 ≈0。**VGGT 对输入顺序鲁棒**。

## 其他

- 顺序敏感性脚本 `check_order_sensitivity.py`;诊断脚本 `diag_mustc*.py`/`diag_plantview.py`/`diag_chunk.py` 保留于 02_vggt/ 供复查。
- 禁止覆盖:输出目录、pose_eval 字段、order_sensitivity.json、pose_eval_summary.json 均存在即报错。
- TERRA-REF 3 序列为 pending_download(无本地影像),不在本阶段范围。

## Gate 结论

**条件性通过**:工程指标(深度有效、NaN/Inf、无坍缩、检查图齐全、顺序鲁棒)全部满足;位姿精度 14/17(82.4%)低于 90% 门槛,3 个 plant_view 序列位姿失效已定位并记录(非流程缺陷,为模型在该场景的局限)。建议:下游 LoRA 微调以这 3 个序列为困难样本关注点;或对 plant_view 位姿失效序列改用深度/点云输出参与几何评估。
