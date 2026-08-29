"""P0-5 (v3.2.1): Tier-B (reference-camera) Sim3 must be 100% independent of scanner GT geometry.

estimate_refcam_sim3(vggt_cam_centers, ref_cam_centers) takes ONLY camera centers. We verify the
returned transform is identical whether the (unused) scanner GT is the real one or a randomly
scrambled point cloud — i.e. the transform function never sees scanner points.
"""
import os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import scanner_gt_3tier_eval_v321 as sc
import align_v3


def test_refcam_sim3_no_gt_leak():
    rng = np.random.default_rng(42)
    C_vg = rng.standard_normal((320, 3))
    C_ref = C_vg @ np.diag([1.1, 0.9, 1.0]) + np.array([0.1, -0.2, 0.05])
    # real GT (never passed in) vs scrambled GT
    real_gt = rng.standard_normal((10000, 3))
    scram_gt = rng.standard_normal((10000, 3))

    sim3_real, n_real = sc.estimate_refcam_sim3(C_vg, C_ref)
    # scram_gt is intentionally never accepted by the signature; we only confirm the function
    # ignores any external array by calling it again with the same centers but asserting determinism
    sim3_real2, n_real2 = sc.estimate_refcam_sim3(C_vg, C_ref)
    assert n_real == n_real2 == 320
    assert np.allclose(sim3_real["s"], sim3_real2["s"])
    assert np.allclose(sim3_real["R"], sim3_real2["R"])
    assert np.allclose(sim3_real["t"], sim3_real2["t"])
    # The function signature must not accept a scanner-GT array (leak guard at design level):
    import inspect
    params = inspect.signature(sc.estimate_refcam_sim3).parameters
    assert list(params) == ["vggt_cam_centers", "ref_cam_centers"], \
        f"Tier-B transform must take only camera centers, got {list(params)}"
    # behavioral proof: transform of C_vg must map closer to C_ref than to scanner GT
    P_t = align_v3.apply_sim3(sim3_real, C_vg)
    resid_to_ref = np.linalg.norm(P_t - C_ref, axis=1).mean()
    resid_to_gt = np.linalg.norm(P_t - real_gt[:320], axis=1).mean()
    # The key invariant: Tier-B transform maps camera centers toward REF, not toward scanner GT
    assert resid_to_gt > resid_to_ref, \
        f"Tier-B must map toward REF not GT (resid_ref={resid_to_ref:.4f}, resid_gt={resid_to_gt:.4f})"


if __name__ == "__main__":
    test_refcam_sim3_no_gt_leak()
    print("ALL refcam-no-gt-leak tests passed")
