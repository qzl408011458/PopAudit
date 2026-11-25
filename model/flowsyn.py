"""
FlowSyn — simple faithful re-implementation (audit-only).

Two stages (Lyu, Ch.6 "Unconditional tabular data generation"):
  Stage 1: masked Transformer-VAE attribute embedding
           - continuous attrs -> learnable Fourier features (eq 6.1)
           - discrete attrs    -> learnable embedding
           - Transformer encoder/decoder, per-attribute latent (NOT compressed)
           - loss = MSE(cont) + CE(disc) + beta*KL              (eq 6.2)
  Stage 2: conditional flow matching in the VAE latent space
           - linear-interpolation conditional flow z_t=(1-t)z0+t z1 (eq 6.8)
           - gated-MLP velocity field + sinusoidal time embed     (eq 6.12-6.14)
           - sampling: solve ODE dz/dt=v_theta from N(0,I) -> decode (Alg 8)

This is a SIMPLE reproduction for use as an audit object; no fine-tuning.
No external flow-matching/ODE libraries required.
"""
from __future__ import annotations
import math
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Tokenizer: per-attribute embedding -> tokens [B, d, k]
# ----------------------------------------------------------------------------
class FourierContinuousEmbed(nn.Module):
    """Learnable Fourier features for continuous scalars (eq 6.1)."""
    def __init__(self, n_cont: int, k: int, n_freq: int = 16):
        super().__init__()
        self.n_cont = n_cont
        self.freq = nn.Parameter(torch.randn(n_cont, n_freq) * 2.0)  # learnable w
        self.proj = nn.ModuleList([nn.Linear(2 * n_freq, k) for _ in range(n_cont)])

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:  # [B, n_cont] -> [B, n_cont, k]
        toks = []
        for j in range(self.n_cont):
            xj = x_cont[:, j:j + 1]                       # [B,1]
            ang = 2.0 * math.pi * xj * self.freq[j][None, :]  # [B, n_freq]
            feat = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)
            toks.append(F.relu(self.proj[j](feat)))       # [B,k]
        return torch.stack(toks, dim=1)                   # [B, n_cont, k]


class DiscreteEmbed(nn.Module):
    def __init__(self, cat_dims: List[int], k: int):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(K, k) for K in cat_dims])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:  # [B, n_cat] int -> [B, n_cat, k]
        return torch.stack([emb(x_cat[:, j]) for j, emb in enumerate(self.embs)], dim=1)


# ----------------------------------------------------------------------------
# Stage 1: Transformer VAE
# ----------------------------------------------------------------------------
class FlowSynVAE(nn.Module):
    def __init__(self, n_cont: int, cat_dims: List[int], k: int = 32, k_lat: int = 8,
                 n_layers: int = 3, n_heads: int = 4, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_cont = n_cont
        self.cat_dims = cat_dims
        self.d = n_cont + len(cat_dims)
        self.k = k
        self.k_lat = k_lat

        self.cont_embed = FourierContinuousEmbed(n_cont, k) if n_cont else None
        self.disc_embed = DiscreteEmbed(cat_dims, k) if cat_dims else None
        self.pos = nn.Parameter(torch.randn(1, self.d, k) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(k, n_heads, ff, dropout,
                                               batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.to_mu = nn.Linear(k, k_lat)
        self.to_lv = nn.Linear(k, k_lat)

        self.from_lat = nn.Linear(k_lat, k)
        self.dec_pos = nn.Parameter(torch.randn(1, self.d, k) * 0.02)
        dec_layer = nn.TransformerEncoderLayer(k, n_heads, ff, dropout,
                                               batch_first=True, activation="gelu")
        self.decoder = nn.TransformerEncoder(dec_layer, n_layers)
        self.num_head = nn.Linear(k, 1) if n_cont else None
        self.cat_heads = nn.ModuleList([nn.Linear(k, K) for K in cat_dims])

    def tokenize(self, x_cont, x_cat):
        toks = []
        if self.cont_embed is not None:
            toks.append(self.cont_embed(x_cont))
        if self.disc_embed is not None:
            toks.append(self.disc_embed(x_cat))
        return torch.cat(toks, dim=1) + self.pos          # [B, d, k]

    def encode(self, x_cont, x_cat):
        h = self.encoder(self.tokenize(x_cont, x_cat))    # [B, d, k]
        return self.to_mu(h), self.to_lv(h)               # [B, d, k_lat] each

    def decode(self, z):                                  # z [B, d, k_lat]
        h = self.from_lat(z) + self.dec_pos
        h = self.decoder(h)                               # [B, d, k]
        num_pred = None
        if self.num_head is not None:
            num_pred = self.num_head(h[:, :self.n_cont, :]).squeeze(-1)  # [B, n_cont]
        cat_logits = [head(h[:, self.n_cont + j, :]) for j, head in enumerate(self.cat_heads)]
        return num_pred, cat_logits

    def forward(self, x_cont, x_cat):
        mu, lv = self.encode(x_cont, x_cat)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
        num_pred, cat_logits = self.decode(z)
        return num_pred, cat_logits, mu, lv


def vae_loss(x_cont, x_cat, num_pred, cat_logits, mu, lv, beta=1e-3):
    loss_num = F.mse_loss(num_pred, x_cont) if num_pred is not None else x_cont.new_tensor(0.0)
    loss_cat = x_cont.new_tensor(0.0)
    for j, logits in enumerate(cat_logits):
        loss_cat = loss_cat + F.cross_entropy(logits, x_cat[:, j])
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return loss_num + loss_cat + beta * kl, loss_num.detach(), loss_cat.detach(), kl.detach()


# ----------------------------------------------------------------------------
# Stage 2: conditional flow matching with a gated-MLP velocity field
# ----------------------------------------------------------------------------
def sinusoidal_time_embed(t, dim):           # t [B] -> [B, dim]
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class GatedMLPBlock(nn.Module):
    """eq 6.13-6.14: LayerNorm -> FC split -> ReLU gate -> linear -> residual."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 2 * hidden)
        self.gate_norm = nn.LayerNorm(hidden)
        self.gate_proj = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, h):
        x = self.norm(h)
        a, b = self.fc(x).chunk(2, dim=-1)
        gated = F.relu(a) * self.gate_proj(self.gate_norm(b))
        return h + self.out(gated)


class VelocityField(nn.Module):
    def __init__(self, z_dim, hidden=256, n_blocks=4, t_dim=64):
        super().__init__()
        self.in_proj = nn.Linear(z_dim, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.t_dim = t_dim
        self.blocks = nn.ModuleList([GatedMLPBlock(hidden, hidden) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden, z_dim)

    def forward(self, z, t):                  # z [B, Z], t [B]
        h = self.in_proj(z) + self.t_mlp(sinusoidal_time_embed(t, self.t_dim))
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)


def cfm_loss(vfield, z1):
    """Linear-interpolation conditional flow matching (eq 6.8)."""
    z0 = torch.randn_like(z1)
    t = torch.rand(z1.shape[0], device=z1.device)
    z_t = (1 - t)[:, None] * z0 + t[:, None] * z1
    target = z1 - z0
    return F.mse_loss(vfield(z_t, t), target)


@torch.no_grad()
def ode_sample(vfield, n, z_dim, device, steps=15):
    """Midpoint ODE solve dz/dt = v(z,t), t: 0 -> 1."""
    z = torch.randn(n, z_dim, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t0 = torch.full((n,), i * dt, device=device)
        k1 = vfield(z, t0)
        zm = z + 0.5 * dt * k1
        tm = torch.full((n,), (i + 0.5) * dt, device=device)
        z = z + dt * vfield(zm, tm)
    return z
