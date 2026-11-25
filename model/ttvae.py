
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: MMD
# -----------------------------
def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    # x: [B, D], y: [B, D]
    # return: [B, B]
    x2 = (x * x).sum(dim=1, keepdim=True)  # [B,1]
    y2 = (y * y).sum(dim=1, keepdim=True).t()  # [1,B]
    dist2 = x2 + y2 - 2.0 * (x @ y.t())
    return torch.exp(-dist2 / (2.0 * sigma * sigma + 1e-12))


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """
    Unbiased-ish MMD^2 with mixture of RBF kernels.
    x, y: [B, D]
    """
    assert x.dim() == 2 and y.dim() == 2 and x.size(0) == y.size(0)
    B = x.size(0)

    k_xx = 0.0
    k_yy = 0.0
    k_xy = 0.0
    for s in sigmas:
        k_xx = k_xx + _rbf_kernel(x, x, s)
        k_yy = k_yy + _rbf_kernel(y, y, s)
        k_xy = k_xy + _rbf_kernel(x, y, s)

    # remove diagonal for xx/yy
    diag = torch.eye(B, device=x.device, dtype=x.dtype)
    k_xx = (k_xx * (1.0 - diag)).sum() / (B * (B - 1) + 1e-12)
    k_yy = (k_yy * (1.0 - diag)).sum() / (B * (B - 1) + 1e-12)
    k_xy = k_xy.mean()
    return k_xx + k_yy - 2.0 * k_xy


# -----------------------------
# Schema: after preprocessing
# -----------------------------
@dataclass
class TTVAESchema:
    """
    processed row = concat(
        [alpha_i (1) + beta_i (Mi onehot)] for each numerical column i
        + [onehot(cat_j)] for each categorical column j
    )
    We also keep a per-field token grouping to build "column tokens".
    """
    # For each numerical column: (alpha_dim=1, mode_dim=Mi)
    num_mode_dims: List[int]
    # For each categorical column: num_classes
    cat_dims: List[int]

    @property
    def n_num(self) -> int:
        return len(self.num_mode_dims)

    @property
    def n_cat(self) -> int:
        return len(self.cat_dims)

    @property
    def processed_dim(self) -> int:
        return sum(1 + m for m in self.num_mode_dims) + sum(self.cat_dims)

    @property
    def n_fields(self) -> int:
        # each original column becomes one token
        return self.n_num + self.n_cat

    def field_slices(self) -> List[Tuple[int, int]]:
        """
        Returns slices in processed vector corresponding to each field token.
        For numerical field i: alpha+beta block.
        For cat field j: onehot block.
        """
        sl = []
        s = 0
        for m in self.num_mode_dims:
            e = s + (1 + m)
            sl.append((s, e))
            s = e
        for k in self.cat_dims:
            e = s + k
            sl.append((s, e))
            s = e
        return sl


# -----------------------------
# Model: Transformer VAE w/ MMD
# -----------------------------
class TTVAE(nn.Module):
    """
    TTVAE-like:
      - encode column tokens with self-attn -> contextual embeddings h
      - pool -> mu/logvar -> z
      - decode with transformer decoder:
          tgt = learned field query tokens
          memory = contextual h + z injection (broadcast)
      - heads reconstruct each field block
    """

    def __init__(
        self,
        schema: TTVAESchema,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        z_dim: int = 32,
    ):
        super().__init__()
        self.schema = schema
        self.d_model = d_model
        self.z_dim = z_dim

        # ---- field projection (each field has variable width) ----
        # We'll project each field-block vector -> d_model using individual Linear layers.
        self.field_proj = nn.ModuleList()
        for (s, e) in schema.field_slices():
            self.field_proj.append(nn.Linear(e - s, d_model))

        # positional embedding over fields
        self.pos_emb = nn.Parameter(torch.zeros(schema.n_fields, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # pool -> latent params
        self.fc_mu = nn.Linear(d_model, z_dim)
        self.fc_logvar = nn.Linear(d_model, z_dim)

        # ---- decoder ----
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # learned field query tokens as tgt
        self.field_queries = nn.Parameter(torch.randn(schema.n_fields, 1, d_model) * 0.02)

        # inject z into memory tokens
        self.z_to_mem = nn.Linear(z_dim, d_model)

        # ---- heads: reconstruct each field block ----
        # numeric field i: output alpha (1) + mode logits (Mi)
        # cat field j: output logits (K)
        self.field_heads = nn.ModuleList()
        for m in schema.num_mode_dims:
            self.field_heads.append(nn.Linear(d_model, 1 + m))
        for k in schema.cat_dims:
            self.field_heads.append(nn.Linear(d_model, k))

        # init
        nn.init.zeros_(self.pos_emb)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def split_to_field_tokens(self, x_proc: torch.Tensor) -> torch.Tensor:
        """
        x_proc: [B, D_proc]
        return tokens: [T=n_fields, B, d_model]
        """
        B = x_proc.size(0)
        slices = self.schema.field_slices()
        toks = []
        for i, (s, e) in enumerate(slices):
            block = x_proc[:, s:e]  # [B, width_i]
            toks.append(self.field_proj[i](block).unsqueeze(0))  # [1,B,D]
        h = torch.cat(toks, dim=0)  # [T,B,D]
        h = h + self.pos_emb  # broadcast [T,1,D]
        return h

    def encode(self, x_proc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return:
          h_ctx: [T,B,D] contextual embeddings
          mu/logvar: [B,z]
        """
        h0 = self.split_to_field_tokens(x_proc)  # [T,B,D]
        h_ctx = self.encoder(h0)  # [T,B,D]

        # pool across tokens (mean)
        pooled = h_ctx.mean(dim=0)  # [B,D]
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return h_ctx, mu, logvar

    def decode(self, z: torch.Tensor, h_ctx: torch.Tensor) -> torch.Tensor:
        """
        z:     [B,z]
        h_ctx: [T,B,D] (contextual embeddings)
        return logits per field token: [T,B,width(field)]
        """
        B = z.size(0)
        z_mem = self.z_to_mem(z).unsqueeze(0)  # [1,B,D]
        memory = h_ctx + z_mem  # broadcast to [T,B,D]

        tgt = self.field_queries.repeat(1, B, 1)  # [T,B,D]
        out = self.decoder(tgt=tgt, memory=memory)  # [T,B,D]

        # project each token with its head
        logits_blocks = []
        for t in range(self.schema.n_fields):
            logits_blocks.append(self.field_heads[t](out[t]))  # [B,width_t]
        # pad into list-like tensor is inconvenient; return as a list in training helper
        # but we keep as list->stack with max width if needed.
        return logits_blocks  # List[Tensor], length T

    def forward(self, x_proc: torch.Tensor) -> Dict[str, object]:
        """
        returns dict:
          h_ctx, mu, logvar, z, logits_blocks(list)
        """
        h_ctx, mu, logvar = self.encode(x_proc)
        z = self.reparameterize(mu, logvar)
        logits_blocks = self.decode(z, h_ctx)
        return {"h_ctx": h_ctx, "mu": mu, "logvar": logvar, "z": z, "logits_blocks": logits_blocks}
