# 阶段0完成报告

版本：`v0.1.0-phase0`  
日期：`2026-08-24`

## 1. 已完成

- 冻结植株级与叶片级表型定义；
- 冻结世界坐标、地面、重力和行向定义；
- 冻结样本身份、plant-date统计单位和leaf ID规则；
- 冻结按plot/plant/sequence分组的数据划分；
- 冻结数据目录、清单字段和模型派生数据版本字段；
- 冻结L1–L5数据等级、QA–QR质量等级和拒收规则；
- 完成MuST-C、Wheat3DGS、3D Plant View、TERRA-REF的数据集级准入审计；
- 建立阶段1逐文件审计模板和许可硬门禁；
- 提供JSON Schema、CSV模板和静态验证脚本。

## 2. 四数据集结论

### MuST-C：通过

- 官方提供5GB单plot样例；
- 数据为CC BY 4.0；
- 适合田间、多传感器、跨平台和plot级几何；
- 不能承担统一四表型监督。

### Wheat3DGS：有条件通过

- 官方提供七个plot、每plot约36张图像、COLMAP相机和麦穗实例相关文件；
- 官方仓库明确代码MIT许可，但不能由此自动推导Google Drive数据许可；
- 仓库仍将包含laser scans的数据发布列为TODO；
- 在取得数据独立许可证据前，不得进入正式训练或发布派生数据。

### 3D Plant View：通过

- 20株小麦、112个plant-date实例，具有机器人变换、COLMAP和部分扫描真值；
- GigaDB关联数据原则上按CC0发布，阶段1仍需保存102661条目的具体许可快照；
- 适合受控几何、相机、视角数和扫描消融，不代表田间泛化。

### TERRA-REF：通过

- 公开数据政策为CC0；
- Dryad约800MB元数据/trait catalog可作为低成本审计入口；
- 适合plot级RGB、3D扫描和外部田间验证；
- 全量传感器数据非常大，应先由元数据反查最小plot/date子集。

## 3. 当前尚未完成

这些属于阶段1，不是阶段0缺失：

- 实际下载并计算压缩包SHA256；
- 文件数量和损坏检查；
- RGB与相机一一对应；
- RGB与点云/扫描在坐标和时间上的对应；
- 坐标系、单位和尺度来源核验；
- Wheat3DGS独立数据许可确认；
- TERRA-REF最小RGB+scanner子集定位；
- 最终可运行VGGT的序列清单。

## 4. 阶段1建议下载顺序

1. MuST-C官方5GB sample；
2. Wheat3DGS一个plot，仅做条件审计；
3. 3D Plant View官方sample或一个plant-date；
4. TERRA-REF Dryad metadata、trait_data和sensor catalogs；
5. 根据TERRA元数据再下载一个小plot/date的RGB与scanner数据。

该顺序优先获得最清晰的多传感器样例，同时避免直接下载TERRA-REF大规模原始数据。

## 5. 阶段1退出条件

至少满足：

- 三个数据集存在可读RGB；
- 每个用于VGGT的数据序列都能固定图像顺序与sequence ID；
- 至少两个数据集具有可核验相机信息；
- 至少两个数据集具有物理尺度或独立点云参考；
- 所有正式使用的数据许可为VERIFIED；
- 形成可直接输入阶段2冒烟测试的最小序列清单。

## 6. 阶段0最终状态

规范内容：`COMPLETE`。  
项目负责人签署：`PENDING`。  
阶段1允许状态：项目负责人确认`P0-15`后允许启动。

