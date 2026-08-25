"""阶段2.1 gate 校验:检查所有 sequence.json 是否满足通过条件。

条件:
1. rgb_paths 非空且顺序固定(有 sort_key 说明)
2. len(rgb_paths) == len(camera_ids)
3. camera_ids 可映射到图像文件名
4. camera_convention / linear_unit 非空
5. ready:rgb_paths 全部存在于磁盘;pending:必须有 remote_paths+checksum
6. 汇总 validation_report.json;失败序列记录到 10_failures/
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "00_environment"))
from provenance import sha256_file  # noqa: E402

SEQ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequences")
FAIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "10_failures")


def validate(seq_path: str):
    errs = []
    with open(seq_path) as f:
        seq = json.load(f)
    for field in ("dataset_id", "sequence_id", "status", "camera_ids",
                  "linear_unit", "camera_convention", "group_key"):
        if not seq.get(field) and field != "status":
            errs.append(f"missing/empty field: {field}")
    if seq["status"] not in ("ready", "pending_download"):
        errs.append(f"bad status: {seq['status']}")

    if seq["status"] == "ready":
        if not seq["rgb_paths"]:
            errs.append("ready but rgb_paths empty")
        if len(seq["rgb_paths"]) != len(seq["camera_ids"]):
            errs.append(f"n_rgb={len(seq['rgb_paths'])} != n_cameras={len(seq['camera_ids'])}")
        # camera_id 映射检查:camera_id 出现在对应文件名中,或为 view_%04d 索引规则
        for cid, p in zip(seq["camera_ids"], seq["rgb_paths"]):
            base = os.path.basename(p)
            ok = (cid in base) or (seq["dataset_id"] == "plant_view_3d"
                                   and cid.startswith("view_"))
            if not ok:
                errs.append(f"camera_id {cid} not mappable to {base}")
                break
        # 磁盘存在性
        for p in seq["rgb_paths"]:
            if not os.path.exists(p):
                errs.append(f"missing file: {p}")
                break
        # 排序规则说明
        if not seq.get("extra", {}).get("sort_key"):
            errs.append("ready sequence missing extra.sort_key")
    else:  # pending
        rp = seq.get("extra", {}).get("remote_paths") or []
        if not rp or not all(e.get("checksum_md5") for e in rp):
            errs.append("pending sequence must carry remote_paths with checksums")

    return errs, seq


def main():
    report = {"sequences": {}, "summary": {}}
    failures = []
    # 只把顶层 sequence.json 当序列;camera/*.json 是附属相机文件,不校验
    all_paths = sorted(glob.glob(os.path.join(SEQ_ROOT, "*", "*.json")))
    n_ok = 0
    for p in all_paths:
        rel = os.path.relpath(p, SEQ_ROOT)
        try:
            errs, seq = validate(p)
        except Exception as e:
            errs, seq = [f"exception: {e}"], {}
        report["sequences"][rel] = {"ok": not errs, "errors": errs,
                                    "sequence_id": seq.get("sequence_id"),
                                    "status": seq.get("status"),
                                    "n_views": len(seq.get("rgb_paths", []))}
        if errs:
            failures.append({"sequence": rel, "errors": errs})
        else:
            n_ok += 1

    by_ds = {}
    for rel, r in report["sequences"].items():
        ds = rel.split("/")[0]
        by_ds.setdefault(ds, [0, 0])
        by_ds[ds][0] += 1
        by_ds[ds][1] += int(r["ok"])
    report["summary"] = {
        "total": len(all_paths), "passed": n_ok,
        "by_dataset": {k: {"total": v[0], "passed": v[1]} for k, v in by_ds.items()},
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if failures:
        os.makedirs(FAIL_DIR, exist_ok=True)
        with open(os.path.join(FAIL_DIR, "sequence_validation_failures.json"), "w") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)

    print(json.dumps(report["summary"], indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
