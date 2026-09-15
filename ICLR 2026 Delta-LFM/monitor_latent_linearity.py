"""
Latent-trajectory linearity probe for the Step-1 AE.

Measures whether a patient's images lie on a straight line in latent space. This
module encodes every distinct visit of every patient with at least 3 visits using
the current autoencoder and measures, per patient, how linear and how
monotone the sequence of latents z_{t0}..z_{tn} is along real time (`follow_up`):

  * line_r2      : fraction of latent variance captured by the best-fit line
                   (largest PCA singular value^2 / total).  1.0 == perfectly collinear.
  * mono_rho     : |Spearman(proj_on_PC1, follow_up)|.  1.0 == monotone along the line.
  * step_cos     : mean cosine between consecutive latent deltas (z_{t+1}-z_t).
                   1.0 == constant direction (a straight, non-zig-zag path).
  * vel_r2       : R^2 of ||z_t - z_0|| vs follow_up (constant-speed progression).

These are the properties the trajectory losses are meant to induce, so the probe
doubles as their success metric.
"""
import numpy as np
import torch
from scipy.stats import spearmanr


def _apply_norm(x, mode):
    """Match the training encoder-input normalization so the probe latent lines up."""
    if mode == "pm1":
        return x * 2.0 - 1.0
    if mode == "std":
        m = x.mean(dim=(-1, -2, -3), keepdim=True)
        s = x.std(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-5)
        return (x - m) / s
    return x


@torch.no_grad()
def encode_visits(autoencoder, loader, device, max_batches=None, channels=0, norm_mode="01"):
    """Run the AE encoder over the visit loader -> {subject: [(follow_up, z_flat)]}.

    channels>0 restricts the latent to its first-k channels — the PROGRESSION
    SUBSPACE that the subspace-design loss actually constrains (and that Δ-LFM
    would flow-match). channels=0 uses the full latent.
    """
    autoencoder.eval()
    seqs = {}
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        img = batch["image"].to(device).float()
        out = autoencoder(_apply_norm(img, norm_mode))
        z = out[1] if isinstance(out, (tuple, list)) else out      # z_mu [B,C,...]
        if channels > 0 and z.dim() > 2:
            z = z[:, :channels]
        z = z.flatten(1).float().cpu().numpy()
        sids = batch["subject_str"]
        fus = batch["follow_up"]
        fus = fus.tolist() if torch.is_tensor(fus) else list(fus)
        for k in range(len(sids)):
            s = sids[k] if isinstance(sids[k], str) else str(sids[k])
            seqs.setdefault(s, []).append((float(fus[k]), z[k]))
    return seqs


def _patient_metrics(fu, Z):
    """fu: (n,) follow_up ; Z: (n,d) latents sorted by time. Returns dict or None.

    Target is ORDER-based (not age-proportional): mono_rho measures monotone
    ordering (rank corr, already order-based); vel_r2 is a DIAGNOSTIC only
    (constant-speed-vs-age, which we do NOT enforce). mono_rho_shuf is the
    order-permutation control (chance floor); v1 is the PC1 direction used for
    cross-patient direction-diversity.
    """
    n = Z.shape[0]
    if n < 3:
        return None
    Zc = Z - Z.mean(0, keepdims=True)
    # PCA via SVD
    try:
        _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    total = float((S ** 2).sum()) + 1e-12
    line_r2 = float(S[0] ** 2 / total)                       # collinearity
    proj = Zc @ Vt[0]                                        # position along PC1
    rho = spearmanr(proj, fu).correlation
    mono_rho = float(abs(rho)) if rho == rho else 0.0        # nan-safe (ORDER metric)
    # order-permutation control: shuffle the time labels -> chance monotonicity
    perm = np.random.default_rng(0).permutation(n)
    rho_s = spearmanr(proj, fu[perm]).correlation
    mono_rho_shuf = float(abs(rho_s)) if rho_s == rho_s else 0.0
    # consecutive-delta direction consistency
    d = np.diff(Z, axis=0)
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    step_norm = float(nrm.mean())                            # mean ||dz|| — motion size (anti-collapse read-out)
    dn = d / (nrm + 1e-12)
    step_cos = float((dn[1:] * dn[:-1]).sum(1).mean()) if n >= 3 else 1.0
    # constant-speed vs AGE (DIAGNOSTIC ONLY -- not a target under order-based loss)
    disp = np.linalg.norm(Z - Z[0:1], axis=1)
    if np.std(fu) > 1e-9 and np.std(disp) > 1e-9:
        r = np.corrcoef(disp, fu)[0, 1]
        vel_r2 = float(r ** 2) if r == r else 0.0
    else:
        vel_r2 = 0.0
    v1 = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)             # PC1 unit dir (sign-arbitrary)
    return dict(line_r2=line_r2, mono_rho=mono_rho, mono_rho_shuf=mono_rho_shuf,
                step_cos=step_cos, vel_r2=vel_r2, step_norm=step_norm, n=n, v1=v1)


def evaluate_linearity(autoencoder, loader, device, max_batches=None, channels=0, norm_mode="01"):
    """Aggregate per-patient linearity metrics into mean/median summaries.

    channels>0 evaluates the progression subspace (first-k latent channels).
    """
    seqs = encode_visits(autoencoder, loader, device, max_batches=max_batches, channels=channels, norm_mode=norm_mode)
    per = []
    for s, lst in seqs.items():
        lst.sort(key=lambda x: x[0])
        fu = np.array([a for a, _ in lst], dtype=np.float64)
        Z = np.stack([b for _, b in lst], 0).astype(np.float64)
        m = _patient_metrics(fu, Z)
        if m is not None:
            per.append(m)
    if not per:
        return {}
    keys = ["line_r2", "mono_rho", "mono_rho_shuf", "step_cos", "vel_r2", "step_norm"]
    out = {"n_patients": len(per)}
    for k in keys:
        vals = np.array([p[k] for p in per])
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_median"] = float(np.median(vals))
    # HONESTY CONTROL 1 -- order-monotonicity above chance (real - shuffled).
    # A straight line NOT tied to visit order gives ~0 here even at line_r2=1.
    out["mono_gap"] = out["mono_rho_mean"] - out["mono_rho_shuf_mean"]
    # HONESTY CONTROL 2 -- cross-patient direction diversity. Mean pairwise |cos|
    # of the per-patient PC1 directions; ~1 => all patients share ONE global axis
    # (degenerate clock), low => genuine per-patient trajectories. Report 1-|cos|.
    V = np.stack([p["v1"] for p in per], 0)                 # (P, d), sign-arbitrary
    if len(V) >= 2:
        C = np.abs(V @ V.T)                                 # |cos| pairwise
        iu = np.triu_indices(len(V), k=1)
        out["dir_collapse"] = float(C[iu].mean())           # 1 => collapsed
        out["dir_diversity"] = float(1.0 - C[iu].mean())    # 1 => diverse
    return out


def format_summary(m):
    if not m:
        return "linearity: (no patients)"
    return (f"linearity[{m['n_patients']}pt] "
            f"line_r2={m['line_r2_mean']:.3f} "
            f"mono_rho={m['mono_rho_mean']:.3f} "
            f"mono_gap={m.get('mono_gap', 0):.3f} "
            f"step_cos={m['step_cos_mean']:.3f} "
            f"dir_div={m.get('dir_diversity', 0):.3f} "
            f"(vel_r2={m['vel_r2_mean']:.3f} diag)")
