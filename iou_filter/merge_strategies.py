#!/usr/bin/env python3
"""
Merge strategies for combining BEV and 3D segmentation predictions.

Provides:
- project_3d_logits_to_bev: project per-point 3D logits to BEV grid
- merge_weighted_softmax: α·softmax(BEV) + (1-α)·softmax(3D)
- merge_confidence_selection: per-pixel pick higher-confidence model
- merge_3d_priority: BEV base, override where 3D confidence > threshold
"""

import numpy as np


def project_3d_logits_to_bev(tile_points, logits_3d, grid_scale, grid_size):
    """
    Project per-point 3D logits to BEV grid using highest-Z-wins.

    Uses the same projection logic as project_3d_bev_img3 in ply_tile_processor.py:
    for each BEV pixel, the point with the highest Z coordinate wins.

    Class mapping: 3D has 13 classes (0-12), BEV has 14 (0=unlabeled, 1-13).
    3D class i maps to BEV channel i+1.

    Args:
        tile_points: (N, 7) [x, y, z, r, g, b, class], coords in [0, tile_size]
        logits_3d: (N, 13) per-point logits from Concerto
        grid_scale: meters per pixel
        grid_size: tile size in meters

    Returns:
        (14, H, W) BEV logits. Channel 0 = unlabeled (zero), channels 1-13 from 3D.
        Empty pixels have all-zero logits.
    """
    gs = int(grid_size / grid_scale)
    N = tile_points.shape[0]

    bev_logits = np.zeros((14, gs, gs), dtype=np.float32)
    alt = np.full((gs, gs), -1e9, dtype=np.float32)

    xs = np.clip((tile_points[:, 0] / grid_scale).astype(np.int32), 0, gs - 1)
    ys = np.clip((tile_points[:, 1] / grid_scale).astype(np.int32), 0, gs - 1)
    zs = tile_points[:, 2]

    for i in range(N):
        if zs[i] > alt[ys[i], xs[i]]:
            alt[ys[i], xs[i]] = zs[i]
            bev_logits[0, ys[i], xs[i]] = 0.0  # unlabeled channel stays zero
            bev_logits[1:, ys[i], xs[i]] = logits_3d[i]  # 13 class logits

    return bev_logits


def _softmax(logits, axis=0):
    """Numerically stable softmax along given axis."""
    e = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def merge_weighted_softmax(bev_logits, logits_3d_bev, alpha=0.5):
    """
    Weighted softmax merge: merged = α * softmax(BEV) + (1-α) * softmax(3D).

    Empty BEV pixels (all-zero logits) produce uniform softmax,
    so the BEV model naturally dominates there.

    Args:
        bev_logits: (C, H, W) BEV model logits
        logits_3d_bev: (C, H, W) 3D model logits projected to BEV
        alpha: weight for BEV model (default 0.5)

    Returns:
        (H, W) merged prediction (class indices)
    """
    prob_bev = _softmax(bev_logits, axis=0)
    prob_3d = _softmax(logits_3d_bev, axis=0)
    merged = alpha * prob_bev + (1 - alpha) * prob_3d
    return merged.argmax(axis=0).astype(np.int64)


def merge_confidence_selection(bev_logits, logits_3d_bev):
    """
    Per-pixel, select prediction from the model with higher max(softmax).

    Args:
        bev_logits: (C, H, W) BEV model logits
        logits_3d_bev: (C, H, W) 3D model logits projected to BEV

    Returns:
        (H, W) merged prediction (class indices)
    """
    prob_bev = _softmax(bev_logits, axis=0)
    prob_3d = _softmax(logits_3d_bev, axis=0)

    conf_bev = prob_bev.max(axis=0)  # (H, W)
    conf_3d = prob_3d.max(axis=0)    # (H, W)

    pred_bev = bev_logits.argmax(axis=0)
    pred_3d = logits_3d_bev.argmax(axis=0)

    use_3d = conf_3d > conf_bev
    result = pred_bev.copy()
    result[use_3d] = pred_3d[use_3d]
    return result.astype(np.int64)


def merge_3d_priority(bev_logits, logits_3d_bev, threshold=0.7):
    """
    BEV as base, override where 3D confidence > threshold.

    Args:
        bev_logits: (C, H, W) BEV model logits
        logits_3d_bev: (C, H, W) 3D model logits projected to BEV
        threshold: confidence threshold for 3D override (default 0.7)

    Returns:
        (H, W) merged prediction (class indices)
    """
    prob_3d = _softmax(logits_3d_bev, axis=0)
    conf_3d = prob_3d.max(axis=0)  # (H, W)

    pred_bev = bev_logits.argmax(axis=0)
    pred_3d = logits_3d_bev.argmax(axis=0)

    use_3d = conf_3d > threshold
    result = pred_bev.copy()
    result[use_3d] = pred_3d[use_3d]
    return result.astype(np.int64)
