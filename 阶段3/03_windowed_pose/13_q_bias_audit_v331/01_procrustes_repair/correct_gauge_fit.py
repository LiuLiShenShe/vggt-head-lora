#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3.1 §2: Corrected per-window reference-gauge fitting.

P0 repair: the old rotation_diagnostics.py used
    H = np.einsum("sij,sik->jk", R_refk, R_local)   # = Σ_s R_ref^T @ R_local  ← WRONG
which is NOT the frozen correct formula. The correct one (from
evaluate_multoplant.py) is
    H = np.einsum("sij,skj->ik", R_local, R_ref)    # = Σ_s R_local @ R_ref^T
minimizing Σ_f || G @ R_local,f - R_ref,f ||_F  →  R_ref,f ≈ G @ R_local,f.

Synthetic exact recovery: correct formula → 0°, old formula → ~85°.

This module exposes:
  fit_window_gauge(R_local_c2w, R_ref_c2w)      -> G (single 3x3), reusing frozen function
  residual_deg(G, R_local, R_ref)               -> per-frame geodesic residual
"""
import os
import sys

import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
from evaluate_multoplant import global_rotation_procrustes, rot_angle_deg

FROZEN_PROCRUSTES = "evaluate_multoplant.global_rotation_procrustes"


def fit_window_gauge(R_local_c2w, R_ref_c2w):
    """Fit G_k minimizing Σ_f || G_k @ R_local,f - R_ref,f ||_F.

    Returns (G_k, per_frame_residual_deg).
    R_local_c2w: (16,3,3) VGGT per-window c2w rotations.
    R_ref_c2w:   (16,3,3) COLMAP c2w rotations for the same frames (comparison ONLY).
    """
    assert R_local_c2w.shape == R_ref_c2w.shape, "window/ref shape mismatch"
    Gk = global_rotation_procrustes(R_local_c2w, R_ref_c2w)
    res = residual_deg(Gk, R_local_c2w, R_ref_c2w)
    return Gk, res


def residual_deg(G, R_local_c2w, R_ref_c2w):
    """Per-frame residual after alignment: angle( (G@R_local,f)^T @ R_ref,f )."""
    n = R_local_c2w.shape[0]
    aligned = np.einsum("ab,sbc->sac", G, R_local_c2w)
    errs = np.array([rot_angle_deg(aligned[f].T @ R_ref_c2w[f]) for f in range(n)])
    return errs


def old_formula_gauge(R_local_c2w, R_ref_c2w):
    """The deprecated formula from rotation_diagnostics.py (kept for comparison/test)."""
    H = np.einsum("sij,sik->jk", R_ref_c2w, R_local_c2w)
    U, _, Vt = np.linalg.svd(H)
    Gk = Vt.T @ U.T
    if np.linalg.det(Gk) < 0:
        Vt[-1, :] *= -1
        Gk = Vt.T @ U.T
    return Gk
