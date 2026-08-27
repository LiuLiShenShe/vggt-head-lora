"""Clean rerun P0-7:迁移 prediction_meta.json 到版本化评价字段格式。

对每个 meta:
- 备份原文件 -> <seq_dir>/prediction_meta.pre_migration.json(若备份已存在则跳过该目录,禁止重复迁移)
- 旧 pose_eval 逐字改名为 pose_eval_v1;pose_eval_v2 保持
- 写 active_evaluation_version="v2" / gate_source="pose_eval_v2" / deprecated_fields=["pose_eval_v1"]
- clean rerun 新 meta 若无 pose_eval(v1 从未产生),pose_eval_v1 置 None

用法: python enforce_eval_version.py <seq_dir> [...]
"""
import json
import os
import shutil
import sys

sys.path.insert(0, "/fj/VGGT+head+lora实验/阶段2/00_environment")
from eval_version import apply_version_fields, get_active_pose_eval


def migrate(seq_dir):
    mp = os.path.join(seq_dir, "prediction_meta.json")
    backup = os.path.join(seq_dir, "prediction_meta.pre_migration.json")
    if os.path.exists(backup):
        print(f"[skip] {mp} 已迁移过(备份存在)")
        return
    meta = json.load(open(mp))
    assert "active_evaluation_version" not in meta, f"{mp} 已含版本字段"
    pe_v1 = meta.pop("pose_eval", None)          # clean rerun 新 meta 无此键 -> None
    assert "pose_eval_v2" in meta, f"{mp} 缺 pose_eval_v2,先运行 eval_pose_v2.py"
    meta = apply_version_fields(meta, pe_v1)
    # 终检:迁移后必须能通过强制读取器
    get_active_pose_eval(meta)
    shutil.copy2(mp, backup)
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[migrated] {mp} (pose_eval_v1={'继承' if pe_v1 is not None else 'None(本次运行未产生v1)'})")


def main():
    for sd in sys.argv[1:]:
        migrate(sd)


if __name__ == "__main__":
    main()
