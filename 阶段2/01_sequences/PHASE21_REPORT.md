# 阶段2.1 报告 — 统一数据适配器(sequence.json)

日期:2026-08-25

## 概览

四个 RGB 数据集已统一适配为 `sequence.json` 清单,位于
`阶段2/01_sequences/sequences/{plant_view,wheat3dgs,mustc,terraref}/`。
BonnBeetClouds3D 为纯点云数据集,按决策排除(留给纯点云评估阶段)。

## 生成结果

| 数据集 | 序列数 | 状态 | 视图/序列 | 相机来源 | 参考点云 |
|---|---|---|---|---|---|
| plant_view (3D Plant View langdon_4) | 6(每日期 1 个) | ready | 318–340 | adjusted transforms.json,OpenGL c2w→OpenCV w2c | GS splat.ply |
| wheat3dgs | 7(plot_461–467) | ready | 36 | COLMAP cameras.txt/images.txt(PINHOLE,w2c 原生) | sparse/0/points3D.txt |
| mustc | 4(pos00–03) | ready | 20 同步机位 | cam_params.xml(Metashape sensor 内参 + camera c2w→w2c) | UGV-LMI las(+Ouster 备用记录于 extra) |
| terraref | 3(season_6 示例,max-seqs 限制) | pending_download | — (stereo left/right 目录级) | 无本地影像,catalog 记录远程 path+md5 | laser3d per-plot catalog |

**校验:validation_report.json — 20/20 全部通过。**

## Gate 通过条件核验

| 条件 | 核验方式 | 结果 |
|---|---|---|
| 每个序列图像顺序固定 | 每个 sequence 的 extra.sort_key 记录排序规则(natural/COLMAP 序/nikon 序号/frame_path 序);适配器确定性输出 | ✅ |
| RGB 数量 = 相机数量 | validate_sequences.py 断言 len(rgb_paths)==len(camera_ids) | ✅ |
| 相机编号可映射到图像 | 校验器逐对检查 camera_id 出现在文件名中(或 plant_view 的 view_%04d 索引规则) | ✅ |
| 坐标系和单位有明确字段 | 所有序列含 linear_unit(meter / unknown_colmap_units)+ camera_convention=opencv_w2c;MuST-C extra 注明 plot-local 坐标系非 UTM | ✅ |
| 不依赖手工改文件名 | build_all_sequences.sh 一键运行;write_sequence 禁止覆盖(重复运行报错,需显式删除) | ✅ |

## 几何正确性抽查

- **wheat3dgs**:36 个 R 正交(det=1);points3D 全部点在全部抽查相机前方(100%,202278/202278);相机中心距质心 0.33–1.00(COLMAP 归一化单位)。注:该数据集 images.txt 导出时未含 2D 观测点(npts=0),无法做重投影像素级比对,已用前视约束+正交性替代验证。
- **mustc**:20 机位环绕 plot,中心距质心 1.08–2.89 m,量纲合理(米制 plot-local)。
- **plant_view**:R det=1;相机中心 Y 跨度 0.57m、Z 0.72m,符合机械臂环形采集轨迹(capture_radius=0.9 配置一致)。

## 已知事项

1. **TERRA-REF**:本地无影像。season_4 无 per-plot laser 目录(laser3d_las_plots 为空),只有 season_6 可配对;当前仅生成 3 条示例(max_seqs=5 上限控制),需要更多时运行:
   `python3 adapters/adapt_terraref.py --seasons season_6 --max-seqs 0`(不限量,预计数千条)。
2. **禁止覆盖**:所有写入(write_sequence / dump_intrinsics / dump_extrinsics)遇已存在文件直接报错;重建需先删除对应目录/文件。
3. wheat3dgs 的 points3D 单位为 COLMAP 归一化单位(非真实米),linear_unit 如实标注 `unknown_colmap_units`。
