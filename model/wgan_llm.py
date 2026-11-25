import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TabularSchema:
    """
    Schema describing processed tabular data layout:
      [ num_1 ... num_M | cat1(one-hot) | cat2(one-hot) | ... ]
    """
    num_dim: int
    cat_dims: List[int]

    @property
    def data_dim(self) -> int:
        return self.num_dim + sum(self.cat_dims)

    @property
    def cat_slices(self) -> List[Tuple[int, int]]:
        """
        Returns index slices for each categorical one-hot block
        e.g. [(num_dim, num_dim+k1), (num_dim+k1, num_dim+k1+k2), ...]
        """
        slices = []
        s = self.num_dim
        for k in self.cat_dims:
            slices.append((s, s + k))
            s += k
        return slices

# -------------------------
# Persona adapter: 4096 -> cond_dim
# -------------------------
class PersonaAdapter(nn.Module):
    def __init__(self, in_dim: int = 4096, cond_dim: int = 256, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, cond_dim),
            nn.LayerNorm(cond_dim),
        )

    def forward(self, e4096: torch.Tensor) -> torch.Tensor:
        return self.net(e4096)


# -------------------------
# FiLM block
# -------------------------
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * feat_dim)

        # init ~identity for stability: gamma=0, beta=0
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gb = self.to_gb(c)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


# -------------------------
# Conditional Generator: concat(z,c) + FiLM in hidden layers
# -------------------------
class CondWGANGeneratorFiLM(nn.Module):
    def __init__(
        self,
        data_dim: int,
        noise_dim: int = 128,
        persona_dim: int = 4096,
        cond_dim: int = 256,
        hidden=(256, 256, 256),
        dropout: float = 0.0,
        use_tanh: bool = False,
        adapter_hidden: int = 1024,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.use_tanh = use_tanh

        self.adapter = PersonaAdapter(persona_dim, cond_dim, hidden=adapter_hidden, dropout=adapter_dropout)

        dims = [noise_dim + cond_dim] + list(hidden) + [data_dim]
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.films = nn.ModuleList([FiLM(cond_dim, h) for h in hidden])
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else None

    def forward(self, z: torch.Tensor, e4096: torch.Tensor) -> torch.Tensor:
        c = self.adapter(e4096)                 # [B, cond_dim]
        h = torch.cat([z, c], dim=1)            # [B, noise_dim+cond_dim]

        for i in range(len(self.films)):
            h = self.fcs[i](h)
            h = self.films[i](h, c)
            h = F.relu(h, inplace=True)
            if self.drop is not None:
                h = self.drop(h)

        x = self.fcs[-1](h)
        if self.use_tanh:
            x = torch.tanh(x)
        return x


# -------------------------
# Conditional Critic: projection discriminator style
# score(x,c) = w^T phi(x) + <psi(c), phi(x)>
# -------------------------
class CondWGANCriticProj(nn.Module):
    def __init__(
        self,
        data_dim: int,
        persona_dim: int = 4096,
        cond_dim: int = 256,
        hidden=(256, 256, 256),
        feat_dim: int = 256,
        dropout: float = 0.0,
        adapter_hidden: int = 1024,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = PersonaAdapter(persona_dim, cond_dim, hidden=adapter_hidden, dropout=adapter_dropout)

        layers = []
        d = data_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, feat_dim))
        self.phi = nn.Sequential(*layers)

        self.out = nn.Linear(feat_dim, 1)
        self.psi = nn.Linear(cond_dim, feat_dim)

    def forward(self, x: torch.Tensor, e4096: torch.Tensor) -> torch.Tensor:
        c = self.adapter(e4096)           # [B, cond_dim]
        hx = self.phi(x)                  # [B, feat_dim]
        base = self.out(hx).view(-1)      # [B]
        proj = (self.psi(c) * hx).sum(dim=1)
        return base + proj


# -------------------------
# WGAN-GP gradient penalty (conditional; keep persona fixed)
# -------------------------
def gradient_penalty_cond(
    critic: CondWGANCriticProj,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    e4096: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
):
    B = real_x.size(0)
    alpha = torch.rand(B, 1, device=device).expand_as(real_x)
    inter = alpha * real_x + (1.0 - alpha) * fake_x
    inter.requires_grad_(True)

    score = critic(inter, e4096)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=inter,
        grad_outputs=torch.ones_like(score, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp


# -------------------------
# Your existing categorical projection (keep)
# -------------------------
@torch.no_grad()
def project_onehot_blocks(x: torch.Tensor, schema, cat_mode: str = "argmax", temperature: float = 1.0):
    x = x.clone()
    for (s, e), k in zip(schema.cat_slices, schema.cat_dims):
        logits = x[:, s:e] / max(temperature, 1e-6)
        if cat_mode == "softmax_sample":
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            idx = logits.argmax(dim=1)
        x[:, s:e] = F.one_hot(idx, num_classes=k).float()
    return x


# -------------------------
# Dataset: (x, persona)
# -------------------------
class TabularPersonaDataset(Dataset):
    def __init__(self, x_proc: np.ndarray, e_pers: np.ndarray):
        assert len(x_proc) == len(e_pers), "processed 和 persona embeddings 必须一一对应且长度相等"
        self.x = x_proc.astype(np.float32)
        self.e = e_pers.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.e[idx]

@torch.no_grad()
def sample_wgan_processed_cond(
    gen: CondWGANGeneratorFiLM,
    schema,
    embeddings_pers: np.ndarray,     # shape [N, 4096]
    device: str = "cuda",
    noise_dim: int = 128,
    batch_size: int = 4096,
    cat_mode: str = "argmax",
    cat_temperature: float = 1.0,
):
    gen.eval()
    dev = torch.device(device)

    N = embeddings_pers.shape[0]
    outs = []
    start = 0
    while start < N:
        b = min(batch_size, N - start)
        e = torch.from_numpy(embeddings_pers[start:start+b].astype(np.float32)).to(dev)
        z = torch.randn(b, noise_dim, device=dev)
        x = gen(z, e)
        x = project_onehot_blocks(x, schema, cat_mode=cat_mode, temperature=cat_temperature)
        outs.append(x.detach().cpu().numpy())
        start += b
    return np.concatenate(outs, axis=0)

def processed_to_label_df(
    samples_processed: np.ndarray,
    schema,
    num_cols: list,
    cat_cols: list,
):
    """
    Convert processed samples back to dataframe with numeric columns
    and categorical label columns.

    samples_processed: np.ndarray [N, D]
      - numeric part: first schema.num_dim columns
      - categorical part: consecutive one-hot blocks

    Returns: pd.DataFrame with columns = num_cols + cat_cols
    """
    import pandas as pd

    N, D = samples_processed.shape
    assert D == schema.data_dim, f"Dim mismatch: got {D}, expect {schema.data_dim}"

    out = {}

    # ---- numeric columns ----
    num_part = samples_processed[:, :schema.num_dim]
    for i, c in enumerate(num_cols):
        out[c] = num_part[:, i]

    # ---- categorical columns ----
    start = schema.num_dim
    for col, k in zip(cat_cols, schema.cat_dims):
        block = samples_processed[:, start:start + k]   # [N, k]
        start += k

        # one-hot -> label index
        out[col] = block.argmax(axis=1).astype(np.int64)

    return pd.DataFrame(out)

###########################################################################################
## For ablation study
class CondWGANCriticFiLM(nn.Module):
    """
    score(x,c) = head( FiLM( f_L( ... FiLM(f_1(x), c) ... ), c) )
    """
    def __init__(
        self,
        data_dim: int,
        persona_dim: int = 4096,
        cond_dim: int = 256,
        hidden=(256, 256, 256),
        dropout: float = 0.0,
        adapter_hidden: int = 1024,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = PersonaAdapter(persona_dim, cond_dim, hidden=adapter_hidden, dropout=adapter_dropout)

        dims = [data_dim] + list(hidden)
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.films = nn.ModuleList([FiLM(cond_dim, h) for h in hidden])
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else None

        self.out = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor, e4096: torch.Tensor) -> torch.Tensor:
        c = self.adapter(e4096)  # [B, cond_dim]
        h = x
        for i in range(len(self.fcs)):
            h = self.fcs[i](h)
            h = self.films[i](h, c)
            h = F.leaky_relu(h, 0.2, inplace=True)
            if self.drop is not None:
                h = self.drop(h)
        score = self.out(h).view(-1)
        return score

class CondWGANGeneratorConcat(nn.Module):
    def __init__(
        self,
        data_dim: int,
        noise_dim: int = 128,
        persona_dim: int = 4096,
        cond_dim: int = 256,
        hidden=(256, 256, 256),
        dropout: float = 0.0,
        use_tanh: bool = False,
        adapter_hidden: int = 1024,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.use_tanh = use_tanh

        self.adapter = PersonaAdapter(persona_dim, cond_dim, hidden=adapter_hidden, dropout=adapter_dropout)

        dims = [noise_dim + cond_dim] + list(hidden) + [data_dim]
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else None

    def forward(self, z: torch.Tensor, e4096: torch.Tensor) -> torch.Tensor:
        c = self.adapter(e4096)
        h = torch.cat([z, c], dim=1)
        for i in range(len(self.fcs) - 1):
            h = self.fcs[i](h)
            h = F.relu(h, inplace=True)
            if self.drop is not None:
                h = self.drop(h)
        x = self.fcs[-1](h)
        if self.use_tanh:
            x = torch.tanh(x)
        return x
