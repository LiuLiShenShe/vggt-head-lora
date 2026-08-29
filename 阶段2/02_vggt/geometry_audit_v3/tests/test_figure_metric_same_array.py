"""P1-2/P3-5 (v3.2.1): PR figures must be generated from the REAL foreground arrays, and the
FIGURE_INPUT_MANIFEST.json must record the exact point-cloud paths + sha256 so a figure's metric
is traceable to a specific array (no full-scene mislabeling like v3.2).
"""
import os, json, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_figure_manifest_records_real_foreground():
    m = os.path.join(ROOT, "FIGURE_INPUT_MANIFEST.json")
    assert os.path.exists(m), "先运行 precision_recall_visual_v321.py"
    d = json.load(open(m))
    assert len(d["figure_inputs"]) > 0
    for fi in d["figure_inputs"]:
        # pred/ref must be the foreground arrays, NOT full-scene *_pred_aligned.npy
        assert fi["pred_foreground_npy"].endswith("_pred_foreground_aligned.npy"), \
            f"figure used wrong array: {fi['pred_foreground_npy']}"
        assert fi["reference_foreground_npy"].endswith("_reference_foreground.npy")
        # files exist and sha256 matches
        pp = os.path.join(ROOT, fi["pred_foreground_npy"])
        rp = os.path.join(ROOT, fi["reference_foreground_npy"])
        assert os.path.exists(pp) and os.path.exists(rp)
        import hashlib
        assert hashlib.sha256(open(pp, "rb").read()).hexdigest() == fi["pred_sha256"]
        assert hashlib.sha256(open(rp, "rb").read()).hexdigest() == fi["ref_sha256"]


def test_figure_files_exist():
    figdir = os.path.join(ROOT, "figures_v321")
    man = json.load(open(os.path.join(ROOT, "FIGURE_INPUT_MANIFEST.json")))
    for fi in man["figure_inputs"]:
        sid = fi["sequence_id"]
        p = os.path.join(figdir, f"{sid}_precision_recall_explanation.png")
        assert os.path.exists(p), f"figure missing: {p}"


if __name__ == "__main__":
    test_figure_manifest_records_real_foreground()
    test_figure_files_exist()
    print("ALL figure-metric-same-array tests passed")
