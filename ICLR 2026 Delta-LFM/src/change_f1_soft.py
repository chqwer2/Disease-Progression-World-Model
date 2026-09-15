"""cF1_soft: a low-variance version of cF1, and the maths behind the --fm_cf1_w training loss.

Why: cF1 splits voxels with hard thresholds, R = {|delta_gt| > tau} and P = {|delta_pred| > tau}.
P depends on the prediction itself, so a small change in the prediction moves a batch of voxels
across the threshold and the reading jumps.

Fix: replace the hard indicators with continuous weights of the same meaning; nothing else changes.
    1[|delta_gt| > tau]    ->  g = |delta_gt|   / (|delta_gt|   + tau)   in (0, 1)
    1[|delta_pred| > tau]  ->  p = |delta_pred| / (|delta_pred| + tau)
    cRecall    = clip( sum g * delta_gt * delta_pred / sum g * delta_gt^2,   0, 1 )
    cPrecision = clip( sum p * delta_gt * delta_pred / sum p * delta_pred^2, 0, 1 )
    cF1_soft   = 2 * cP * cR / (cP + cR)

Anchors are preserved (analytically):
    exact      delta_pred = delta_gt        -> 1
    copy       delta_pred = 0               -> 0
    reversed   delta_pred = -delta_gt       -> 0
    x0.5 / x2  of the ground truth          -> 0.667 (charged to recall / precision)
"""
import numpy as np

_E = 1e-9


def change_f1_soft(pred, target, input, mask=None, frac=0.25):
    pred, target, input = map(lambda x: np.asarray(x, np.float64), (pred, target, input))
    brain = (np.asarray(mask) > 0.5) if mask is not None else ((input > 0.05) | (target > 0.05))
    if brain.sum() == 0:
        brain = np.ones_like(input, dtype=bool)
    dgt = (target - input)[brain]
    dpr = (pred - input)[brain]
    tau = max(frac * float(np.percentile(np.abs(dgt), 99)), 1e-6)
    g = np.abs(dgt) / (np.abs(dgt) + tau)
    p = np.abs(dpr) / (np.abs(dpr) + tau)
    rec_raw = float((g * dgt * dpr).sum() / ((g * dgt * dgt).sum() + _E))
    pre_raw = float((p * dgt * dpr).sum() / ((p * dpr * dpr).sum() + _E))
    rec, pre = float(np.clip(rec_raw, 0, 1)), float(np.clip(pre_raw, 0, 1))
    return {"cF1s": round(2 * pre * rec / (pre + rec + _E), 4),
            "cRecall_s": round(rec, 4), "cPrecision_s": round(pre, 4),
            "cRecall_s_raw": round(rec_raw, 4), "cPrecision_s_raw": round(pre_raw, 4)}
