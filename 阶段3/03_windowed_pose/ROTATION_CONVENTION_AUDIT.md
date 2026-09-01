# Rotation Convention Audit — Phase 3C.1

## Stored Extrinsic Convention

All VGGT outputs store **world-to-camera (w2c)** extrinsic matrices:

```
ext_w2c[i] = [R_w2c | t]   shape (3,4) or (4,4)
```

where `R_w2c` is a 3×3 rotation matrix and `t` is a 3×1 translation vector.

A 3D point `X_w` in world coordinates is projected to camera coordinates as:

```
x_c = R_w2c @ X_w + t
```

The camera center in world coordinates is:

```
C_w = -R_w2c^T @ t
```

Derivation: at the camera center, `x_c = 0`, so `0 = R_w2c @ C_w + t`, giving `C_w = -R_w2c^T @ t`.

## Camera-to-World (c2w) Convention

The c2w rotation is the inverse of w2c:

```
R_c2w = R_w2c^T
```

The c2w matrix maps camera-frame axes to world-frame axes:

```
X_w = R_c2w @ x_c + C_w
```

## Cross-Window Transform Convention

Consider two windows A and B with a shared gauge transformation:

```
x_A = Q @ x_B + t          (1)
```

where `Q ∈ SO(3)` is the gauge rotation and `t ∈ R³` is the gauge translation.
This means: a point expressed in B's world frame, when rotated by Q and shifted by t,
gives the same point in A's world frame.

### Camera Center Transform

From the w2c convention:

```
C_A = -R_w2c_A^T @ t_w2c_A
C_B = -R_w2c_B^T @ t_w2c_B
```

Applying Eq (1) to the camera center:

```
C_A = Q @ C_B + t           (2)
```

### c2w Rotation Transform

For an overlap frame `i` appearing in both windows:

```
R_c2w_A_i = Q @ R_c2w_B_i   (3)
```

Proof: A point in camera frame `x_c` maps to A-world as:
```
X_A = R_c2w_A_i @ x_c + C_A
     = R_c2w_A_i @ x_c + Q @ C_B + t
```

But also via B-world:
```
X_A = Q @ X_B + t
     = Q @ (R_c2w_B_i @ x_c + C_B) + t
     = Q @ R_c2w_B_i @ x_c + Q @ C_B + t
```

Comparing: `R_c2w_A_i = Q @ R_c2w_B_i`. ∎

### w2c Rotation Transform

Since `R_w2c = R_c2w^T`:

```
R_w2c_A_i = R_c2w_A_i^T = (Q @ R_c2c_B_i)^T = R_c2w_B_i^T @ Q^T = R_w2c_B_i @ Q^T   (4)
```

### Relative Gauge Rotation from Overlap

From Eq (3):

```
Q_i = R_c2w_A_i @ R_c2w_B_i^T     (5)
```

Equivalently using w2c (Eq 4):

```
Q_i = (R_w2c_A_i @ Q_i^T)^T ... 
```

More directly: from `R_w2c_A = R_w2c_B @ Q^T`:

```
Q^T = R_w2c_B^T @ R_w2c_A = R_c2w_B @ R_c2w_A^T
Q   = R_c2w_A @ R_c2w_B^T          (same as Eq 5)
```

### Gauge-Anchor Transform

When composing pairwise transforms `G_{k→k+1}` through a chain:

```
G_0 = I (identity — Window 0 is the reference gauge)
G_{k+1} = G_k ∘ S_{k→k+1}
```

where `S_{k→k+1}` maps from window `k+1`'s gauge to window `k`'s gauge:

```
S_{k→k+1} = (Q, s, t)
```

Applying to window `k+1` cameras:

```
C_global = s * Q_global @ C_local + t_global     (6)
R_c2w_global = Q_global @ R_c2w_local             (7)
```

## Summary Table

| Quantity | Formula | Notes |
|---|---|---|
| Camera center | `C = -R_w2c^T @ t` | Derived from w2c |
| c2w rotation | `R_c2w = R_w2c^T` | Inverse of w2c |
| Gauge transform | `x_A = Q x_B + t` | World B → World A |
| Center transform | `C_A = Q C_B + t` | Direct from gauge |
| c2w rotation transform | `R_c2w_A = Q R_c2w_B` | Left-multiply by Q |
| w2c rotation transform | `R_w2c_A = R_w2c_B Q^T` | Right-multiply by Q^T |
| Overlap Q estimation | `Q_i = R_c2w_A_i @ R_c2w_B_i^T` | Per-overlap-frame |
| Composed center | `C_g = s Q_g C_l + t_g` | Scale + rotate + shift |
| Composed c2w | `R_c2w_g = Q_g @ R_c2w_l` | Left-multiply |

## Convention Audit Test

A synthetic test must verify all formulas above (see `test_orientation_transform_convention.py`).
