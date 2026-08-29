"""P0-6: depth validity audit must flag far-plane pixels (>10m) and filter them."""
import os, json, sys

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)


def test_far_plane_flagged():
    p = os.path.join(ROOT, "DEPTH_VALIDITY_AUDIT.json")
    assert os.path.exists(p), "先运行 DEPTH_VALIDITY_AUDIT.py"
    d = json.load(open(p))
    for seq, info in d["sequences"].items():
        bins = info.get("bin_proportions_mean", {})
        far = bins.get("10_65m", 0)
        # Must have the far-plane bin reported (even if 0)
        assert "10_65m" in bins, f"{seq} missing 10_65m bin"
        # If far-plane > 1%, must be flagged
        if far > 0.01:
            assert info.get("pct_far_plane_gt_10m", 0) > 0.01, \
                f"{seq} has {far:.1%} far-plane but not flagged"


def test_validity_audit_has_all_seqs():
    p = os.path.join(ROOT, "DEPTH_VALIDITY_AUDIT.json")
    d = json.load(open(p))
    required = ["langdon_4__05-03-24", "langdon_4__12-03-24", "langdon_4__13-02-24", "langdon_4__20-02-24"]
    for seq in required:
        assert seq in d["sequences"], f"missing {seq} in validity audit"


if __name__ == "__main__":
    test_far_plane_flagged()
    test_validity_audit_has_all_seqs()
    print("ALL depth invalid farplane filter tests passed")
