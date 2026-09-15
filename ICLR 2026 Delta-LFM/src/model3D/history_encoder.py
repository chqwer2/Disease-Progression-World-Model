import torch
import torch.nn as nn


class HistoryEncoder(nn.Module):
    """Isolated, opt-in patient-history conditioner for Delta-LFM Step-3 flow-matching.

    Encodes the patient's PREVIOUS scan (prior-visit latent) into a set of
    cross-attention context tokens (B, n_tokens, ctx_dim). These are concatenated
    onto the model's existing covariate context (B, 1, ctx_dim) so they pass through
    the SAME cross-attention projection -> the flow generator is steered by where the
    patient came from (recent trajectory), not just the single baseline snapshot.

    A learned null token covers samples whose starting scan is the first visit
    (no prior). Toggled entirely off by the caller (fm_history=0) -> zero effect.

    ctx_dim MUST match the model's context feature dim (here 10 = 8 covariates +
    age-diff + diagnosis-diff).
    """

    def __init__(self, in_ch: int = 4, ctx_dim: int = 10, n_tokens: int = 64, width: int = 128):
        super().__init__()
        self.n_tokens = int(n_tokens)
        side = round(self.n_tokens ** (1.0 / 3.0))
        assert side ** 3 == self.n_tokens, f"n_tokens must be a perfect cube, got {n_tokens}"
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, width // 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, width // 2), nn.SiLU(),
            nn.Conv3d(width // 2, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv3d(width, width, 3, stride=1, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((side, side, side))
        self.proj = nn.Linear(width, ctx_dim)
        self.null = nn.Parameter(torch.randn(1, self.n_tokens, ctx_dim) * 0.02)

    def forward(self, prior_latent: torch.Tensor, has_prior: torch.Tensor | None = None) -> torch.Tensor:
        # prior_latent: (B, in_ch, D, H, W) ; has_prior: (B,) in {0,1} or None
        B = prior_latent.shape[0]
        h = self.enc(prior_latent)               # (B, width, d, h, w)
        h = self.pool(h)                         # (B, width, s, s, s)
        h = h.flatten(2).transpose(1, 2)         # (B, n_tokens, width)
        h = self.proj(h)                         # (B, n_tokens, ctx_dim)
        if has_prior is not None:
            m = has_prior.reshape(B, 1, 1).to(h.dtype)
            h = m * h + (1.0 - m) * self.null.to(h.dtype).expand(B, -1, -1)
        return h
