#!/usr/bin/env python3
"""Static validation for the VGGT-MetricPhen Phase 0 protocol package."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "00_FREEZE_MANIFEST.yaml",
    "01_phenotype_protocol.md",
    "02_identity_leaf_split_protocol.md",
    "03_directory_and_manifest_spec.md",
    "04_quality_and_grading.md",
    "05_phase0_completion_report.md",
    "config/dataset_registry.yaml",
    "config/controlled_vocabularies.yaml",
    "schemas/sample_manifest.schema.json",
    "schemas/leaf_measurement.schema.json",
    "templates/dataset_admission_audit.csv",
    "templates/source_evidence.csv",
    "templates/dataset_file_audit_template.csv",
    "templates/sample_manifest_template.csv",
    "templates/leaf_measurement_template.csv",
    "templates/split_group_template.csv",
    "templates/source_evidence_template.csv",
    "templates/phase0_freeze_checklist.csv",
]

REQUIRED_DATASETS = {"must_c", "wheat3dgs", "plant_view_3d", "terra_ref"}

CSV_REQUIRED_COLUMNS = {
    "templates/dataset_admission_audit.csv": {
        "dataset_id", "data_license_status", "phase0_admission",
        "expected_max_level", "split_group_key", "last_checked",
    },
    "templates/dataset_file_audit_template.csv": {
        "dataset_id", "package_id", "sha256", "license_status",
        "rgb_camera_mapping_status", "audit_status",
    },
    "templates/sample_manifest_template.csv": {
        "sample_id", "dataset_id", "site_id", "plot_id", "sequence_id",
        "rgb_paths_json", "data_level", "quality_grade", "split",
        "group_key", "license_id",
    },
    "templates/leaf_measurement_template.csv": {
        "leaf_obs_id", "leaf_uid", "plant_id", "plant_date_id",
        "leaf_match_status", "visibility_ratio", "measurable", "label_source",
    },
}


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def validate_files() -> None:
    missing = [item for item in REQUIRED_FILES if not (ROOT / item).is_file()]
    if missing:
        fail(f"missing required files: {missing}")


def validate_json() -> None:
    for relative in (
        "schemas/sample_manifest.schema.json",
        "schemas/leaf_measurement.schema.json",
    ):
        with (ROOT / relative).open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if parsed.get("type") != "object" or not parsed.get("required"):
            fail(f"invalid schema structure: {relative}")


def validate_csv_headers() -> None:
    for relative, expected in CSV_REQUIRED_COLUMNS.items():
        with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        absent = expected.difference(header)
        if absent:
            fail(f"{relative} missing columns: {sorted(absent)}")


def validate_dataset_rows() -> None:
    path = ROOT / "templates/dataset_admission_audit.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = {row["dataset_id"] for row in rows if row.get("dataset_id")}
    if ids != REQUIRED_DATASETS:
        fail(f"dataset IDs differ: expected={sorted(REQUIRED_DATASETS)} actual={sorted(ids)}")
    for row in rows:
        if not row.get("official_landing_url") or not row.get("phase0_admission"):
            fail(f"incomplete admission row: {row.get('dataset_id')}")


def validate_freeze_markers() -> None:
    manifest = (ROOT / "00_FREEZE_MANIFEST.yaml").read_text(encoding="utf-8")
    for marker in ("v0.1.0-phase0", "FROZEN_FOR_PHASE1_AUDIT", "Wheat3DGS"):
        if marker not in manifest:
            fail(f"freeze manifest missing marker: {marker}")


def main() -> int:
    validate_files()
    validate_json()
    validate_csv_headers()
    validate_dataset_rows()
    validate_freeze_markers()
    print("PASS: VGGT-MetricPhen Phase 0 package is internally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
