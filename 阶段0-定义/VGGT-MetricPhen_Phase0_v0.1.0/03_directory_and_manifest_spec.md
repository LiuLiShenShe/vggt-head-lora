# 统一目录与清单规范

协议版本：`v0.1.0-phase0`

## 1. 目录结构

```text
data/
├── raw_readonly/
│   ├── public/
│   │   ├── must_c/
│   │   ├── wheat3dgs/
│   │   ├── 3d_plant_view/
│   │   └── terra_ref/
│   └── self_capture/
├── manifests/
│   ├── dataset_registry.yaml
│   ├── sample_manifest.csv
│   ├── leaf_measurements.csv
│   ├── file_audits/
│   ├── licenses/
│   └── splits/
├── interim/
│   ├── decoded_images/
│   ├── calibrated_cameras/
│   ├── plot_clips/
│   └── aligned_geometry/
├── derived/
│   ├── masks/
│   ├── vggt/
│   ├── metric_depth/
│   ├── msam/
│   └── phenotype_targets/
└── qc/
    ├── reports/
    ├── thumbnails/
    ├── rejected/
    └── revision_log/
```

`raw_readonly`解压完成后应设置只读；任何裁剪、重命名、格式转换或人工修订进入`interim`或`derived`。

## 2. 单样本推荐目录

```text
{dataset_id}/{sample_id}/
├── rgb/
│   ├── view_000.jpg
│   └── ...
├── camera/
│   ├── intrinsics.json
│   ├── extrinsics.json
│   └── distortion.json
├── scale/
│   ├── anchors.json
│   └── calibration.json
├── geometry/
│   ├── reference.ply
│   └── ground_plane.json
├── labels/
│   ├── phenotype.json
│   ├── leaves.json
│   └── masks/
├── metadata/
│   └── sample.json
└── qc/
    └── qc.json
```

不要求复制超大公开数据；可以使用相对路径或软链接，但清单必须能追溯到原始archive和SHA256。

## 3. 路径规则

- 清单中使用相对于项目数据根目录的POSIX路径；
- 不保存用户主目录、Windows盘符或临时下载URL；
- 原始文件名保留在`source_relative_path`；
- 标准化路径与原始路径建立一对一映射；
- 路径大小写固定，不依赖大小写不敏感文件系统；
- 空值使用空字符串或JSON `null`，不得使用`0`代替缺失表型。

## 4. 必需清单

### 4.1 dataset registry

记录数据集级：官方名称、版本、下载入口、许可、用途、禁止用途和分组键。

### 4.2 file audit

每个下载包/压缩包一行，记录：

- 预期与实测大小；
- SHA256；
- 文件数量；
- RGB、相机、点云、表型与许可文件数量；
- RGB—camera—pointcloud对应证据；
- 坐标系与单位；
- 审计状态。

### 4.3 sample manifest

每个`plant_date`、plot-date或sequence窗口一行。视角路径可以JSON数组字符串保存。

### 4.4 leaf measurements

每个`leaf_obs_id`一行。多次重复测量可以多行保存，通过`repeat_id`区分，汇总值另存为`consensus`行。

## 5. 模型派生数据命名

```text
derived/vggt/{vggt_checkpoint}/{preprocess_version}/{sample_id}/
derived/metric_depth/{model_name}/{model_version}/{sample_id}/
derived/msam/{msam_version}/{sample_id}/
```

缓存键至少包含：

- 输入图像SHA256列表；
- 图像排序；
- resize/crop参数；
- 相机内参缩放方式；
- 模型名称与checkpoint哈希；
- 软件提交哈希；
- 精度模式。

## 6. 文件版本和来源

所有人工修订保存：

- `previous_value`；
- `new_value`；
- `reason`；
- `operator`；
- `timestamp`；
- `source_evidence`。

不能直接覆盖金标而不留修订记录。

## 7. 许可文件

每个公开数据集目录至少保存：

```text
licenses/{dataset_id}/
├── license_snapshot.txt或pdf
├── source_url.txt
├── accessed_at.txt
└── citation.bib
```

如果许可只出现在网页上，应保存网页PDF/截图和访问日期。代码许可与数据许可分开保存。

