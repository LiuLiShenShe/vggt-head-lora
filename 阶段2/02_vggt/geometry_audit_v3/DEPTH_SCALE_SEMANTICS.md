# DEPTH_SCALE_SEMANTICS.md — VGGT 深度语义声明

> **Policy**: v3.2 强制修正 — 禁止将 VGGT raw depth 称为 "metric depth"

---

## 1. VGGT 深度的真实语义

VGGT-1B（1B 参数 transformer）在训练时采用 **canonical / scene-normalized scale**。
其输出深度值并非以物理米为锚点，而是在模型内部归一化坐标系中的数值。

### 在 Plant View 数据集上的观察

| 观测项 | 数值 | 说明 |
|--------|------|------|
| depth_scale_to_meter（参考PNG）| 0.001 | uint16 毫米（NeRFStudio 配置）|
| VGGT raw depth vs 参考 median scale | 1.17–1.28 | VGGT 输出大约在 ~1.2m 量级，与参考 1.0m 接近 |
| raw AbsRel（无 scaling）| 0.11–0.28 | 修正深度单位后，直接比较的相对误差 |
| scale-aligned AbsRel | 0.07–0.12 | median scaling 后的相对误差 |

**结论**：VGGT 在此特定数据集上的输出**近似**米制（scale ratio ~1.2），但这不等于 **guaranteed metric depth**。

---

## 2. 三套深度报告必须区分

| 层级 | 名称 | 含义 | VGGT 是否满足 |
|------|------|------|--------------|
| **raw numerical scale** | raw-scale depth | 模型输出的原始数值，未经任何缩放 | ✅ 有（绝对值取决于训练归一化） |
| **scale-aligned relative depth** | median-scaled depth | 逐帧 median scaling 后的相对深度 | ✅ 有（AbsRel 0.07–0.12）|
| **true externally anchored metric depth** | absolute metric depth | 以物理米为锚点、可直接部署的深度 | ❌ 不保证 |

---

## 3. 为什么不能称 "VGGT is metric depth"

1. **scale ratio 1.17–1.28 ≠ 1.0**：即使在"近似米制"的数据集上，VGGT 输出仍有 17–30% 的全局尺度偏差
2. **scale 因序列而异**：不同日期（05-03、13-02、20-02）的 scale ratio 分别不同，说明尺度不是固定映射
3. **无外部锚点**：VGGT 训练未强制输出以物理米为单位；其"近米制"输出是训练数据分布的巧合，非设计保证
4. **部署时不可依赖**：在新场景（不同植物、不同相机距离、不同光照）下，scale ratio 可能完全不同

---

## 4. 本审计中的正确用语

✅ 允许：
- "VGGT raw-scale depth" / "VGGT approximately metric-scale on this dataset"
- "raw numerical scale ~1.2× reference"
- "scale-aligned relative depth (median scaling)"

❌ 禁止：
- "VGGT depth is metric" / "VGGT produces metric depth"
- "VGGT absolute metric depth accuracy = X"
- 任何暗示 VGGT 深度可直接用于物理测量的表述

---

## 5. Metric Depth 的正确角色

Metric Depth（如 DA3、UniDepthV2）是**候选修复模块**：
- 目标：将 VGGT 的 raw-scale depth 转化为 true anchored metric depth
- 评价标准：转化后是否改善 absolute metric accuracy（F@10mm、height error 等）
- 不能因为 baseline 有 ~1.2× scale 就跳过 metric depth 测试

---

*Updated: v3.2 | See also: DEPTH_VALIDITY_AUDIT.json, DEPTH_UNIT_AUDIT.json*
