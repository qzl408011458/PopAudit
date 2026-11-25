from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TabularSchema:
    num_dim: int
    cat_dims: List[int]

    @property
    def data_dim(self) -> int:
        return self.num_dim + sum(self.cat_dims)

    @property
    def cat_slices(self) -> List[Tuple[int, int]]:
        slices = []
        s = self.num_dim
        for k in self.cat_dims:
            slices.append((s, s + k))
            s += k
        return slices


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, dropout: float = 0.0, act="relu"):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if act == "lrelu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WGANGenerator(nn.Module):
    """
    z -> x (processed space)
    For tabular: output is continuous for all dims; cat blocks will be projected to one-hot at sampling time.
    """
    def __init__(
        self,
        data_dim: int,
        noise_dim: int = 128,
        hidden: List[int] = (256, 256, 256),
        dropout: float = 0.0,
        use_tanh: bool = False,  # only if your whole processed vector is scaled to [-1,1]
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.use_tanh = use_tanh
        self.mlp = MLP(noise_dim, list(hidden), data_dim, dropout=dropout, act="relu")

    def forward(self, z):
        x = self.mlp(z)
        if self.use_tanh:
            x = torch.tanh(x)
        return x


class WGANCritic(nn.Module):
    """
    x -> scalar score (no sigmoid)
    """
    def __init__(
        self,
        data_dim: int,
        hidden: List[int] = (256, 256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp = MLP(data_dim, list(hidden), 1, dropout=dropout, act="lrelu")

    def forward(self, x):
        return self.mlp(x).view(-1)


def gradient_penalty(
    critic: WGANCritic,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
):
    """
    WGAN-GP gradient penalty on interpolated samples.
    """
    B = real_x.size(0)
    alpha = torch.rand(B, 1, device=device)
    alpha = alpha.expand_as(real_x)
    inter = alpha * real_x + (1.0 - alpha) * fake_x
    inter.requires_grad_(True)

    score = critic(inter)
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


@torch.no_grad()
def project_onehot_blocks(
    x: torch.Tensor,
    schema: TabularSchema,
    cat_mode: str = "argmax",          # "argmax" or "softmax_sample"
    temperature: float = 1.0,
):
    """
    x: [N, D] continuous
    return: [N, D] with categorical blocks projected to one-hot
    """
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


@torch.no_grad()
def sample_wgan_processed(
    gen: WGANGenerator,
    schema: TabularSchema,
    n: int,
    device: torch.device,
    noise_dim: int = 128,
    batch_size: int = 4096,
    cat_mode: str = "argmax",
    cat_temperature: float = 1.0,
):
    gen.eval()
    chunks = []
    remain = n
    while remain > 0:
        b = min(batch_size, remain)
        z = torch.randn(b, noise_dim, device=device)
        x = gen(z)
        x = project_onehot_blocks(x, schema, cat_mode=cat_mode, temperature=cat_temperature)
        chunks.append(x.detach().cpu().numpy())
        remain -= b
    return np.concatenate(chunks, axis=0)


def processed_to_label_df(samples_processed: np.ndarray, schema: TabularSchema, num_cols: List[str], cat_cols: List[str]):
    import pandas as pd
    num_part = samples_processed[:, :schema.num_dim]
    out = {c: num_part[:, i] for i, c in enumerate(num_cols)}
    start = schema.num_dim
    for col, k in zip(cat_cols, schema.cat_dims):
        block = samples_processed[:, start:start + k]
        start += k
        out[col] = block.argmax(axis=1).astype(int)
    return pd.DataFrame(out)
