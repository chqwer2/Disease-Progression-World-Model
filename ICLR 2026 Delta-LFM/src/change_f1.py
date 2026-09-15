"""cF1: an F1 score over the change field. Direction, location and magnitude all have to
match, and the value range is interpretable without a null-model correction.
Definition
  tau = frac * P99(|delta_gt|)               absolute threshold (the same one used by rF1)
  R = {|delta_gt| > tau}   true change region      P = {|delta_pred| > tau}   predicted change region
  Recall    = clip( sum_R delta_gt.delta_pred / sum_R delta_gt^2,     0, 1 )   projection of the prediction onto the true change
  Precision = clip( sum_P delta_gt.delta_pred / sum_P delta_pred^2,   0, 1 )   share of the prediction supported by true change


Interpretation of the value range
  exact       delta_pred=delta_gt       R=1, P=1     -> cF1 = 1
  reversed    delta_pred=-delta_gt      negative projection, clipped -> cF1 = 0
  copy        delta_pred=0              R=0          -> cF1 = 0
  pure noise (independent of truth)     sum delta_gt.noise approx 0 -> cF1 approx 0
  under-scaled delta_pred=0.5*delta_gt  R=0.5, P=1   -> cF1 = 0.667, attributed to recall
  over-scaled  delta_pred=2*delta_gt    R=1, P=0.5   -> cF1 = 0.667, attributed to precision
Amplitude error is penalized symmetrically, so the metric peaks at the correct amplitude and
cannot be improved by rescaling. Unlike rF1, which is a region-overlap score, cF1 charges
under-scaling to recall and over-scaling to precision.
"""
import numpy as np
_E = 1e-9

def change_f1(pred, target, input, mask=None, frac=0.25):
    pred, target, input = map(lambda x: np.asarray(x, np.float64), (pred, target, input))
    brain = (np.asarray(mask) > 0.5) if mask is not None else ((input > 0.05) | (target > 0.05))
    if brain.sum() == 0: brain = np.ones_like(input, dtype=bool)
    dgt = (target - input)[brain]; dpr = (pred - input)[brain]
    tau = max(frac * float(np.percentile(np.abs(dgt), 99)), 1e-6)
    R = np.abs(dgt) > tau; P = np.abs(dpr) > tau
    rec_raw = float((dgt[R] * dpr[R]).sum() / ((dgt[R] ** 2).sum() + _E)) if R.sum() > 0 else 0.0
    pre_raw = float((dgt[P] * dpr[P]).sum() / ((dpr[P] ** 2).sum() + _E)) if P.sum() > 0 else 0.0
    rec, pre = float(np.clip(rec_raw, 0, 1)), float(np.clip(pre_raw, 0, 1))
    return {"cF1": round(2 * pre * rec / (pre + rec + _E), 4),
            "cRecall": round(rec, 4), "cPrecision": round(pre, 4),
            "cRecall_raw": round(rec_raw, 4), "cPrecision_raw": round(pre_raw, 4),  # unclipped; may be negative or above 1
            "change_frac": round(float(R.mean()), 4)}
