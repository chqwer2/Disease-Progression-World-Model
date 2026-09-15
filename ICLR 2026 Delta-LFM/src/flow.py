"""Latent flow-matching interpolation + velocity target for Δ-LFM (Step-3).

Implements the two time-schemes and the optional Brownian-noise path from the
paper (arXiv:2512.09185):

  * Interpolation (Eq.1, rectified/OT linear path):   x_t = (1-s) x0 + s x1
  * Standard velocity target (Eq.2):                  u* = x1 - x0
  * Δ-LFM real-time velocity target (Eq.12):          u* = (x1 - x0) / Δt
  * Optional stochastic extension (Eq.4):             + σ · dw   (Brownian bridge)

`s` is always the PATH FRACTION in [0,1]; the caller decides what timestep to
feed the network (s for the standard scheme, s·Δt for the real-time scheme).
"""
import torch


def _bcast(t, ref):
    """Reshape a per-sample [B] tensor to broadcast over ref's dims [B,1,1,...]."""
    return t.view(-1, *([1] * (ref.dim() - 1)))


def _noise_envelope(tt, schedule):
    """t-dependent noise envelope, all zero at t=0 and t=1 (endpoints preserved)."""
    tc = tt.clamp(0.0, 1.0)
    if schedule == "cosine":          # smooth, peak at t=0.5
        import math
        return torch.sin(math.pi * tc)
    if schedule == "earlyhigh":       # more noise in the FIRST half (peak ~t=0.25)
        return (tc.sqrt() * (1.0 - tc)) / 0.385           # /max so peak≈1
    if schedule == "latehigh":        # more noise in the SECOND half (peak ~t=0.75)
        return (tc * (1.0 - tc).sqrt()) / 0.385
    if schedule == "flat":            # near-constant plateau, tapered at ends
        return (4.0 * tc * (1.0 - tc)).clamp_max(1.0)     # =1 across the middle, 0 at ends
    # default "bridge": Brownian bridge sqrt(t(1-t))
    return torch.sqrt((tc * (1.0 - tc)).clamp_min(0.0))


def compute_xt(x0, x1, t, sigma_min=0.0, noise_schedule="bridge"):
    """Point on the flow path at fraction ``t`` in [0,1].

    Rectified/OT linear interpolation ``(1-t)x0 + t x1``. When ``sigma_min>0``
    a stochastic term ``σ·env(t)·z`` is added where ``env(t)`` is a t-dependent
    NOISE SCHEDULE (all schedules vanish at t=0 and t=1 so endpoints x0,x1 are
    preserved). schedules: bridge=√(t(1-t)) [default], cosine=sin(πt),
    earlyhigh (peak ~0.25), latehigh (peak ~0.75), flat (plateau across middle).
    ``sigma_min=0`` → deterministic rectified path.
    """
    tt = _bcast(t, x0)
    xt = (1.0 - tt) * x0 + tt * x1
    if sigma_min and float(sigma_min) > 0.0:
        env = _noise_envelope(tt, noise_schedule)
        xt = xt + float(sigma_min) * env * torch.randn_like(xt)
    return xt


_RATE_POW = 1.0   # exponent p in u* = (x1-x0)/dt**p, set by --fm_rate_pow; 1.0 = plain real-time scheme


def compute_ut(x0, x1, t=None, dt=None):
    """Target velocity along the path.

    Standard rectified (Eq.2): constant ``u* = x1 - x0`` (``dt`` is None).
    Δ-LFM real-time (Eq.12): ``u* = (x1 - x0) / Δt`` where ``dt`` is the real
    per-sample inter-visit gap [B] (years) — velocity is change-per-real-time,
    so integrating for the real Δt reproduces the observed change. ``t`` is
    unused for the linear path (velocity is constant in s) but kept for API
    symmetry / future non-linear schedules.
    """
    ut = x1 - x0
    if dt is not None:
        _d = _bcast(dt, x0).clamp_min(1e-3)
        _p = float(globals().get("_RATE_POW", 1.0))
        ut = ut / (_d if _p == 1.0 else _d.pow(_p))
    return ut
