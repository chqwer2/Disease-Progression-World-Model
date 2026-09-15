"""delta_f1_metrics: magnitude-weighted, direction-aware F1 for progression change,
plus a synthetic self-test of the four failure modes. Change field = followup - baseline."""
import numpy as np


def delta_f1_metrics(pred, target, input, mask=None, eps=1e-6):
    """pred/target/input: volumes in [0,1] (decoded followup-pred / real followup / baseline).
    Returns dF1 (main), Recall (miss-sensitive), Precision (FP-sensitive), RMAE_miss."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    input = np.asarray(input, dtype=np.float64)
    if mask is not None:
        brain = np.asarray(mask) > 0.5
    else:
        brain = (input > 0.05) | (target > 0.05)
    if brain.sum() == 0:
        brain = np.ones_like(input, dtype=bool)
    dgt = (target - input)[brain]     # signed GT change
    dgn = (pred - input)[brain]       # signed predicted change
    agt, agn = np.abs(dgt), np.abs(dgn)
    a = 1.0 - np.abs(dgt - dgn) / (agt + agn + eps)          # per-voxel agreement in [0,1]
    R = float((agt * a).sum() / (agt.sum() + eps))           # recall: |Δgt|-weighted -> misses
    P = float((agn * a).sum() / (agn.sum() + eps))           # precision: |Δgen|-weighted -> FPs
    dF1 = float(2 * P * R / (P + R + eps))
    rmae_miss = float((agt * (np.abs(dgt - dgn) / (0.5 * (agt + agn) + eps))).sum() / (agt.sum() + eps))
    return {"dF1": round(dF1, 4), "Recall": round(R, 4), "Precision": round(P, 4), "RMAE_miss": round(rmae_miss, 4)}


def region_f1_metrics(pred, target, input, mask=None, frac=0.25, eps=1e-6):
    """rF1: HARD, direction-aware REGION F1 of the change map (binary complement to dF1).
    Change field = pred-input vs target-input, restricted to brain. A per-volume threshold
    tau = frac * P99(|Dgt|) binarizes both change regions; a voxel is a true positive only
    if it is above tau in BOTH pred and GT change AND the change directions agree (same sign).
    Returns rF1 (main), rRecall (region miss-rate), rPrecision (region false-positive-rate)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    input = np.asarray(input, dtype=np.float64)
    if mask is not None:
        brain = np.asarray(mask) > 0.5
    else:
        brain = (input > 0.05) | (target > 0.05)
    if brain.sum() == 0:
        brain = np.ones_like(input, dtype=bool)
    dgt = (target - input)[brain]
    dgn = (pred - input)[brain]
    agt, agn = np.abs(dgt), np.abs(dgn)
    tau = frac * float(np.percentile(agt, 99))
    if tau < eps:
        tau = eps
    gt_pos = agt > tau
    pr_pos = agn > tau
    agree = np.sign(dgt) == np.sign(dgn)
    tp = float(np.sum(gt_pos & pr_pos & agree))
    Pr = tp / (float(pr_pos.sum()) + eps)
    Re = tp / (float(gt_pos.sum()) + eps)
    rF1 = 2 * Pr * Re / (Pr + Re + eps)
    return {"rF1": round(float(rF1), 4), "rRecall": round(float(Re), 4), "rPrecision": round(float(Pr), 4)}


# ---------- synthetic self-test ----------
rng = np.random.RandomState(0)
shape = (16, 16, 16)
input = np.clip(rng.rand(*shape) * 0.3 + 0.3, 0, 1)              # baseline
# true change: localized atrophy (negative) in a small ROI
true_change = np.zeros(shape); true_change[4:8, 4:8, 4:8] = -0.2
