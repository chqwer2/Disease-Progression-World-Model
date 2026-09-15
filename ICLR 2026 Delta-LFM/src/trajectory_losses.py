"""
Order-based linear-trajectory losses for the Step-1 AE (Δ-LFM target).

Target (per patient p, visits time-ordered z_1..z_n, full flattened latent):

    z_i ≈ z̄_p + s_i · v_p ,   s_1 < s_2 < ... < s_n

i.e. the visits lie on a STRAIGHT line (collinear), MONOTONE in *visit order*
(older = later = farther along +v_p), with the spacing s_i FREE — so slow-
progression intervals move little, fast ones move a lot.  Age/follow_up is used
ONLY to establish the order (argsort), never the magnitude.

Design of the terms:
  (1) the SVD is taken over the TRAJECTORY (visits x full flattened latent),
      not over spatial dimensions of a channel-averaged latent;
  (2) WITHIN-patient only (no cross-patient InfoNCE / identity term);
  (3) the line direction is SIGN-PINNED by visit order (no SVD sign ambiguity);
  (4) monotonicity is ORDER/RANK based, not proportional to chronological age.

All SVD math runs in fp32; bf16 SVD is numerically unstable.

Terms (compose via `weights`):
  T1  collinearity, scale-free rank-1 ratio  = 1 - line_r2                (SVD)
  T3  spectral-gap hinge  relu(sigma2 - eps*sigma1)                       (SVD, soft)
  C1  order-monotone hinge on the PC1 projection (min gap `margin`)       (SVD dir)
  C3  step-direction consistency  mean(1 - cos(dz_i, dz_{i-1}))           (deltas)
  LINE  non-SVD control: residual of the rank-regressed line (equal-spaced)

Every term returns a scalar (mean over patients in the batch); a term with too
few valid visits contributes a differentiable 0.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _group_by_patient(z: torch.Tensor, order: torch.Tensor, n_visits: int):
    """z:[n_visits*B, ...] stacked as [v1(B), v2(B), ..., vN(B)]; order:[n_visits*B].
    Returns Z:[B, n, D] (fp32, full flattened latent) and R:[B, n] (visit order)."""
    N = z.shape[0]
    B = N // n_visits
    D = z.reshape(N, -1).shape[1]
    Z = z.reshape(n_visits, B, D).permute(1, 0, 2).contiguous().float()   # [B, n, D]
    R = order.reshape(n_visits, B).permute(1, 0).contiguous().float()     # [B, n]
    return Z, R, B


def _centered(Z: torch.Tensor):
    return Z - Z.mean(dim=1, keepdim=True)          # [B, n, D]


def linear_trajectory_loss(
    z: torch.Tensor,
    order: torch.Tensor,
    n_visits: int = 3,
    weights: dict | None = None,
    margin: float = 0.1,
    gap_eps: float = 1.0,
    move_floor: float = 0.0,
    eps: float = 1e-8,
):
    """Compute the requested order-based trajectory terms.

    Args:
        z:        latents, shape [n_visits*B, C, ...] stacked visit-major.
        order:    per-sample visit order key (e.g. follow_up); only its RANK is used.
        n_visits: visits per patient in the stack (triplet -> 3).
        weights:  dict subset of {"T1","T3","C1","C3","LINE"} -> float. Only the
                  keys present are computed and returned (others skipped for speed).
        margin:   min normalized gap between consecutive projections (C1).
        gap_eps:  T3 wants sigma2 < gap_eps*sigma1; hinge = relu(sigma2 - gap_eps*sigma1).

    Returns:
        (total, parts) where total is the weighted sum (scalar tensor) and parts is
        a dict term->unweighted scalar (for logging).
    """
    weights = weights or {"T1": 1.0, "C1": 1.0, "C3": 1.0}
    dev = z.device
    zero = torch.zeros((), device=dev)
    if n_visits < 3:                                 # collinearity/step need >=3
        return zero, {k: 0.0 for k in weights}

    Z, R, B = _group_by_patient(z, order, n_visits)  # [B,n,D], [B,n]
    Zc = _centered(Z)                                # [B,n,D]
    dR = R - R.mean(dim=1, keepdim=True)             # [B,n] centered order

    need_svd = any(k in weights for k in ("T1", "T3", "C1"))
    S = Vh = None
    if need_svd:
        # batched SVD over the [n, D] trajectory matrix; core is tiny (n x n).
        U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)   # S:[B,n]  Vh:[B,n,D]

    parts: dict[str, torch.Tensor] = {}

    # ---- T1: collinearity (scale-free rank-1 ratio) = 1 - line_r2 ----
    if "T1" in weights:
        s2 = S.pow(2)                                # [B,n]
        line_r2 = s2[:, 0] / s2.sum(dim=1).clamp_min(eps)
        parts["T1"] = (1.0 - line_r2).mean()

    # ---- T3: spectral-gap hinge (soft collinearity) ----
    if "T3" in weights:
        parts["T3"] = F.relu(S[:, 1] - gap_eps * S[:, 0]).mean() \
            if S.shape[1] > 1 else zero

    # ---- C1: order-monotone hinge on PC1 projection (sign-pinned by order) ----
    if "C1" in weights:
        v1 = Vh[:, 0, :]                             # [B,D] top right singular vector
        a = torch.einsum("bnd,bd->bn", Zc, v1)       # [B,n] projections on the line
        sign = torch.sign((a * dR).sum(dim=1, keepdim=True))   # pin +v to increasing order
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        a = a * sign
        norm = Zc.pow(2).sum(dim=(1, 2)).sqrt().clamp_min(eps)  # scale-free
        a = a / norm.unsqueeze(1)
        a_ord = torch.gather(a, 1, R.argsort(dim=1))            # sort by visit order
        gaps = a_ord[:, 1:] - a_ord[:, :-1]                     # want each >= margin
        parts["C1"] = F.relu(margin - gaps).mean()

    # ---- C3: step-direction consistency (straight + forward for triplets) ----
    if "C3" in weights:
        d = F.normalize(Z[:, 1:, :] - Z[:, :-1, :], dim=-1)     # [B,n-1,D]
        cos = (d[:, 1:, :] * d[:, :-1, :]).sum(-1)              # [B,n-2]
        parts["C3"] = (1.0 - cos).mean() if cos.numel() else zero

    # ---- MOVE: minimum-motion (anti-collapse) — each step must be a real size ----
    if "MOVE" in weights:
        dn = (Z[:, 1:, :] - Z[:, :-1, :]).norm(dim=-1)         # [B,n-1] raw step magnitudes
        parts["MOVE"] = F.relu(move_floor - dn).mean()

    # ---- LINE: non-SVD control — residual of the rank-regressed line ----
    if "LINE" in weights:
        denom = (dR * dR).sum(dim=1, keepdim=True).clamp_min(eps)   # [B,1]
        v = (dR.unsqueeze(-1) * Zc).sum(dim=1) / denom             # [B,D] rank-slope
        Zhat = dR.unsqueeze(-1) * v.unsqueeze(1)                   # [B,n,D]
        resid = (Zc - Zhat).pow(2).sum(dim=(1, 2))
        total_var = Zc.pow(2).sum(dim=(1, 2)).clamp_min(eps)
        parts["LINE"] = (resid / total_var).mean()

    total = sum(weights[k] * parts[k] for k in parts)
    return total, {k: float(v.detach()) for k, v in parts.items()}
