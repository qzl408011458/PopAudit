"""
FlowSyn v2 — faithful port of the OFFICIAL FlowSyn generator pieces that
materially shape the generated joint distribution, onto our 13-attr PUMS wrapper.

Ported from flowsyn/official/clean-source-code/:
  - Velocity field = token-mixing gMLP (GatedMLP_Continuous + GatedMLPBlock with
    Conv1d spatial gating over attribute tokens) — model/mlp.py:487-676.  *** key fix ***
  - TimeStepEmbedding (sinusoidal + SiLU MLP)                            — model/mlp.py:19-67.
  - Beta(5,5)->[-1,1] source prior                                       — utils/pytorch_utils.py:291-310.
Kept from our v1: Transformer VAE tokenizer (Fourier continuous + discrete embed),
per-attribute latent [B, d, k_lat].  Training tricks (masked VAE, beta-anneal,
quantile scaling, latent normalization) live in train_flowsyn_v2.py.

Stays self-contained (no Hydra / Accelerate / official package import).
"""
from __future__ import annotations
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the VAE tokenizer/encoder/decoder + loss from v1
from model.flowsyn import FlowSynVAE, vae_loss  # noqa: F401


# ----------------------------------------------------------------------------
# Masked-denoising wrapper around FlowSynVAE (official mask_ratio~0.4 objective).
# Adds one learnable mask token per attribute position; during training a random
# fraction of tokens is replaced by their mask token, and the decoder must
# reconstruct ALL attributes (including the masked ones) -> forces the latent to
# capture inter-attribute dependencies (the joint structure the audit cares about).
# ----------------------------------------------------------------------------
class MaskedFlowSynVAE(FlowSynVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_tok = nn.Parameter(torch.randn(1, self.d, self.k) * 0.02)

    def encode_masked(self, x_cont, x_cat, mask_ratio: float):
        tok = self.tokenize(x_cont, x_cat)                # [B, d, k]
        if mask_ratio > 0:
            B = tok.shape[0]
            m = (torch.rand(B, self.d, 1, device=tok.device) < mask_ratio).float()
            tok = (1 - m) * tok + m * self.mask_tok       # swap masked tokens
        h = self.encoder(tok)
        return self.to_mu(h), self.to_lv(h)

    def forward_masked(self, x_cont, x_cat, mask_ratio: float):
        mu, lv = self.encode_masked(x_cont, x_cat, mask_ratio)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
        num_pred, cat_logits = self.decode(z)
        return num_pred, cat_logits, mu, lv


# ----------------------------------------------------------------------------
# Beta source prior:  2*Beta(a,a) - 1  on [-1,1]   (official BetaSource, a=5)
# ----------------------------------------------------------------------------
class BetaSource:
    def __init__(self, alpha: float = 5.0, device="cpu"):
        self.dist = torch.distributions.Beta(
            torch.tensor(float(alpha)), torch.tensor(float(alpha)))
        self.device = device

    def sample(self, shape):
        return (2.0 * self.dist.sample(shape) - 1.0).to(self.device)


# ----------------------------------------------------------------------------
# Time embedding (official TimeStepEmbedding, non-fourier branch)
# ----------------------------------------------------------------------------
class TimeStepEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 100, n_layers: int = 2):
        super().__init__()
        assert dim % 2 == 0
        self.dim, self.max_period = dim, max_period
        layers = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(dim, dim), nn.SiLU()]
        self.fc = nn.Sequential(*layers, nn.Linear(dim, dim))

    def forward(self, t):                       # t: [B]
        mid = self.dim // 2
        fs = torch.exp(-math.log(self.max_period) / mid *
                       torch.arange(mid, device=t.device))
        x = t[:, None] * fs[None]
        emb = torch.cat([x.cos(), x.sin()], dim=-1)
        return self.fc(emb)


# ----------------------------------------------------------------------------
# Gated-MLP block with Conv1d token-mixing (official GatedMLPBlock)
# operates on token sequence [B, n_tokens, emb_dim] -> cross-attribute mixing
# ----------------------------------------------------------------------------
class GatedMLPBlock(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, n_tokens: int, act=nn.GELU):
        super().__init__()
        self.x_act = act()
        self.x_proj = nn.Linear(in_dim, hid_dim * 2)
        self.v_proj = nn.Linear(hid_dim, in_dim)
        self.spatial_proj = nn.Conv1d(n_tokens, n_tokens, kernel_size=1)
        nn.init.ones_(self.spatial_proj.bias)

    def spatial_gating(self, x):
        u, v = x.chunk(2, dim=-1)
        v = F.normalize(v, dim=-1)
        v = self.spatial_proj(v)        # mixes across the n_tokens (attribute) axis
        return u * v

    def forward(self, x):
        res = x
        x = F.normalize(x, dim=-1)
        x = self.x_proj(x)
        x = self.x_act(x)
        x = self.spatial_gating(x)
        x = self.v_proj(x)
        return x + res


class GatedMLPVelocity(nn.Module):
    """Official GatedMLP_Continuous: token-sequence velocity field v_theta(z_t; t)."""
    def __init__(self, n_tokens: int, in_dim: int, emb_dim: int = 32,
                 n_layers: int = 5, n_units: int = 1280, max_period: int = 100):
        super().__init__()
        self.time_emb = TimeStepEmbedding(emb_dim, max_period=max_period)
        self.proj = nn.Linear(in_dim, emb_dim)
        self.gmlp = nn.ModuleList(
            [GatedMLPBlock(emb_dim, n_units, n_tokens) for _ in range(n_layers)])
        self.out_proj = nn.Linear(emb_dim, in_dim)

    def forward(self, x, t):                # x: [B, n_tokens, in_dim]; t: [B]
        te = self.time_emb(t)               # [B, emb_dim]
        x = self.proj(x) + te.unsqueeze(1)
        for blk in self.gmlp:
            x = blk(x)
        return self.out_proj(x)


# ----------------------------------------------------------------------------
# CFM on the TOKEN latent [B, d, k_lat]  (linear-interp flow, velocity target)
# ----------------------------------------------------------------------------
def cfm_loss_tokens(vfield, z1, source: BetaSource):
    z0 = source.sample(z1.shape).to(z1.device)
    t = torch.rand(z1.shape[0], device=z1.device)
    tb = t[:, None, None]
    z_t = (1 - tb) * z0 + tb * z1
    return F.mse_loss(vfield(z_t, t), z1 - z0)


@torch.no_grad()
def ode_sample_tokens(vfield, n, n_tokens, k_lat, source: BetaSource, device, steps=25):
    """Midpoint ODE solve from Beta source, t: 0->1, on token latent."""
    z = source.sample((n, n_tokens, k_lat)).to(device)
    dt = 1.0 / steps
    for i in range(steps):
        t0 = torch.full((n,), i * dt, device=device)
        k1 = vfield(z, t0)
        zm = z + 0.5 * dt * k1
        tm = torch.full((n,), (i + 0.5) * dt, device=device)
        z = z + dt * vfield(zm, tm)
    return z
