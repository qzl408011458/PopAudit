
# model/tvae_llm.py
# ------------------------------------------------------------
# SemaPop-VAE (TVAE MLP + FiLM + PriorNet) utilities
# Designed to match the coding style of your wgan_llm training script:
#   - TabularSchema / TabularPersonaDataset
#   - SemaPopTVAE_OneHot model (outputs numeric + per-cat logits)
#   - sampling helper returning "processed space" samples
#   - processed_to_label_df helper (onehot -> label indices)
# ------------------------------------------------------------
# <other_repo>/modelSave
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# -------------------------
# Schema
# -------------------------
@dataclass
class TabularSchema:
    num_dim: int
    cat_dims: List[int]

    @property
    def data_dim(self) -> int:
        return int(self.num_dim + sum(self.cat_dims))

    @property
    def cat_slices(self) -> List[Tuple[int, int]]:
        s = self.num_dim
        out = []
        for k in self.cat_dims:
            out.append((s, s + int(k)))
            s += int(k)
        return out


# -------------------------
# Dataset (processed tabular + persona embedding)
# -------------------------
class TabularPersonaDataset(Dataset):
    def __init__(self, x_processed, persona_emb):
        """
        x_processed: np.ndarray or scipy sparse [N, D]  (processed: num_scaled + cat_onehot)
        persona_emb: np.ndarray [N, P]
        """
        if hasattr(x_processed, "toarray"):
            x_processed = x_processed.toarray()
        self.x = torch.tensor(np.asarray(x_processed), dtype=torch.float32)
        self.e = torch.tensor(np.asarray(persona_emb), dtype=torch.float32)
        assert self.x.shape[0] == self.e.shape[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.e[idx]


# -------------------------
# Model blocks
# -------------------------
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gb = self.to_gb(c)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class PriorNet(nn.Module):
    def __init__(self, persona_dim: int, z_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(persona_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_logvar = nn.Linear(hidden, z_dim)

    def forward(self, c: torch.Tensor):
        h = self.net(c)
        return self.fc_mu(h), self.fc_logvar(h)


class PersonaAdapter(nn.Module):
    def __init__(self, in_dim: int = 4096, cond_dim: int = 512, hidden: int = 1024, dropout: float = 0.1):
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


class SemaPopTVAE_OneHot(nn.Module):
    """
    Input x_processed = [num_scaled | cat_onehot]
    Output heads:
      - numeric_mu: (B, num_dim)
      - numeric_log_sigma: (B, num_dim)
      - cat_logits_i: (B, Ki) for each categorical feature i
    """
    def __init__(
        self,
        input_dim: int,
        num_dim: int,
        cat_sizes: List[int],
        persona_dim: int = 4096,
        cond_dim: int = 512,
        latent_dim: int = 32,
        enc_hidden: int = 512,
        dec_hidden: int = 512,
        enc_depth: int = 3,
        dec_depth: int = 3,
        dropout: float = 0.1,
        adapter_hidden: int = 1024,
        adapter_dropout: float = 0.1,
        prior_hidden: int = 512,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_dim = int(num_dim)
        self.cat_sizes = [int(k) for k in cat_sizes]
        self.z_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)

        self.ca_net = PersonaAdapter(
            in_dim=int(persona_dim),
            cond_dim=self.cond_dim,
            hidden=int(adapter_hidden),
            dropout=float(adapter_dropout),
        )
        self.prior_net = PriorNet(self.cond_dim, self.z_dim, hidden=int(prior_hidden))

        # encoder
        layers = []
        d = self.input_dim
        for _ in range(int(enc_depth)):
            layers += [nn.Linear(d, int(enc_hidden)), nn.ReLU(), nn.Dropout(float(dropout))]
            d = int(enc_hidden)
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(int(enc_hidden), self.z_dim)
        self.fc_logvar = nn.Linear(int(enc_hidden), self.z_dim)

        # decoder
        self.dec_in = nn.Linear(self.z_dim, int(dec_hidden))
        self.film = FiLM(self.cond_dim, int(dec_hidden))
        d_layers = []
        for _ in range(max(int(dec_depth) - 1, 0)):
            d_layers += [nn.Linear(int(dec_hidden), int(dec_hidden)), nn.ReLU(), nn.Dropout(float(dropout))]
        self.decoder = nn.Sequential(*d_layers)

        # --- numeric heads (Gaussian likelihood in standardized space) ---
        self.numeric_mu = nn.Linear(int(dec_hidden), self.num_dim)
        self.numeric_log_sigma = nn.Linear(int(dec_hidden), self.num_dim)
        self.cat_heads = nn.ModuleList([nn.Linear(int(dec_hidden), k) for k in self.cat_sizes])

        # Keep numeric log-sigma initialized near 0 (sigma ~ 1) for stability
        nn.init.zeros_(self.numeric_log_sigma.weight)
        nn.init.zeros_(self.numeric_log_sigma.bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x_processed: torch.Tensor):
        h = self.encoder(x_processed)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        h = self.dec_in(z)
        h = self.film(h, c)
        h = self.decoder(h) if len(self.decoder) > 0 else h
        num_mu = self.numeric_mu(h)
        num_log_sigma = self.numeric_log_sigma(h)
        cat_logits = [head(h) for head in self.cat_heads]
        # Keep backward-compatible alias: many downstream utilities expect "numeric pred".
        # For Gaussian likelihood, the mean is the point prediction.
        num_pred = num_mu
        return [num_mu, num_log_sigma, num_pred] + cat_logits

    def forward(self, x_processed: torch.Tensor, persona_emb: torch.Tensor):
        c = self.ca_net(persona_emb)
        mu_prior, logvar_prior = self.prior_net(c)

        mu, logvar = self.encode(x_processed)
        z = self.reparameterize(mu, logvar)

        outputs = self.decode(z, c)
        return outputs, mu, logvar, mu_prior, logvar_prior

    @torch.no_grad()
    def sample_processed(self, persona_emb: torch.Tensor, cat_mode: str = "argmax", cat_temperature: float = 1.0):
        """
        Sample in processed space (num_scaled + cat_onehot_hard).
        Returns torch.Tensor [B, input_dim].
        """
        self.eval()
        c = self.ca_net(persona_emb)
        mu_p, logvar_p = self.prior_net(c)
        z = self.reparameterize(mu_p, logvar_p)

        outs = self.decode(z, c)
        # outs layout: [num_mu, num_log_sigma, num_pred, cat_logits...]
        num_mu = outs[0]
        num_log_sigma = outs[1]
        num_log_sigma = num_log_sigma.clamp(-7.0, 3.0)
        sigma = torch.exp(num_log_sigma)
        num_hat = num_mu + torch.randn_like(num_mu) * sigma
        onehots = []
        for logits in outs[3:]:
            if cat_temperature != 1.0:
                logits = logits / float(cat_temperature)

            if cat_mode == "argmax":
                idx = torch.argmax(logits, dim=-1)
            elif cat_mode == "sample":
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                raise ValueError(f"Unknown cat_mode: {cat_mode}")

            oh = F.one_hot(idx, num_classes=logits.size(-1)).float()
            onehots.append(oh)

        x_hat = torch.cat([num_hat] + onehots, dim=-1)
        return x_hat


# -------------------------
# VAE loss (ELBO with conditional prior)
# -------------------------
def kl_gaussians(q_mu, q_logvar, p_mu, p_logvar):
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    return 0.5 * ((p_logvar - q_logvar) + (q_var + (q_mu - p_mu) ** 2) / (p_var + 1e-8) - 1.0).sum(dim=-1)


def split_x_processed(x_processed: torch.Tensor, num_dim: int, cat_sizes: List[int]):
    x_num = x_processed[:, :num_dim]
    x_cat = x_processed[:, num_dim:]
    cat_onehots = []
    s = 0
    for k in cat_sizes:
        cat_onehots.append(x_cat[:, s:s+k])
        s += k
    return x_num, cat_onehots


import numpy as np
import torch
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F

def semapop_vae_loss_onehot(
    outputs,
    x_processed,
    num_dim,
    cat_sizes,
    mu,
    logvar,
    mu_prior,
    logvar_prior,
    beta: float = 1.0,
    lambda_nll: float = 0.001,      # NLL regularizer weight (e.g., 0.0005/0.0001) 0.005 / 0.003 / 0.001
    w_num: float = 120.0,             # overall scaling to match old: nll + 50*mse == 50*(mse + 0.02*nll) 200 / 300 / 400
    use_learned_sigma: bool = True,  # heteroscedastic NLL
    sigma_const: float = 1.0,        # used only if use_learned_sigma=False
    sigma_reg_weight: float = 0.0,   # optional: discourage sigma inflation (e.g., 1e-4~1e-3)
    log_sigma_min: float = -7.0,
    log_sigma_max: float = 3.0,
):
    """
    outputs layout:
      [0] num_mu:        (B, num_dim)
      [1] num_log_sigma: (B, num_dim)  # optional if use_learned_sigma=False
      [2] num_pred:      (unused here)
      [3:] cat_logits...

    Numeric loss (MSE main, NLL as regularizer) with explicit overall scaling:
      num_loss = w_num * ( MSE(mu, x) + lambda_nll * NLL(x | mu, sigma) ) + sigma_reg_weight * sigma_reg

    Notes:
      - If you previously used: num_loss_old = NLL + 50*MSE,
        then an equivalent re-parameterization is:
          w_num = 50, lambda_nll = 1/50 (=0.02)
        because: NLL + 50*MSE = 50*(MSE + 0.02*NLL)
      - With argmax/MAP decoding for numeric heads, this aligns well with your precision metric.
    """
    x_num, x_cat_onehots = split_x_processed(x_processed, num_dim, cat_sizes)

    # -----------------------------
    # numeric recon
    # -----------------------------
    num_mu = outputs[0]

    if use_learned_sigma:
        num_log_sigma = outputs[1].clamp(log_sigma_min, log_sigma_max)
        sigma = torch.exp(num_log_sigma)
        # per-dim NLL: 0.5*log(2π) + logσ + 0.5*((x-μ)/σ)^2
        nll_num_mat = (
            0.5 * np.log(2.0 * np.pi)
            + num_log_sigma
            + 0.5 * ((x_num - num_mu) / (sigma + 1e-8)) ** 2
        )
        # optional sigma regularization (discourage extreme log_sigma values)
        sigma_reg = (num_log_sigma ** 2).mean() if sigma_reg_weight > 0 else num_mu.new_tensor(0.0)
    else:
        sigma = torch.as_tensor(sigma_const, device=num_mu.device, dtype=num_mu.dtype).clamp_min(1e-8)
        nll_num_mat = (
            0.5 * np.log(2.0 * np.pi)
            + torch.log(sigma)
            + 0.5 * ((x_num - num_mu) / sigma) ** 2
        )
        sigma_reg = num_mu.new_tensor(0.0)

    # average per-dimension
    nll_num = nll_num_mat.sum(dim=1).mean() / max(int(num_dim), 1)

    # μ-only MSE (average per-dimension)
    mse_num = ((x_num - num_mu) ** 2).sum(dim=1).mean() / max(int(num_dim), 1)

    # MSE main + NLL regularizer (with overall scaling w_num)
    num_loss = float(w_num) * (mse_num + float(lambda_nll) * nll_num) + float(sigma_reg_weight) * sigma_reg

    # -----------------------------
    # categorical recon (CE)
    # -----------------------------
    recon_cat = 0.0
    for logits, onehot in zip(outputs[3:], x_cat_onehots):
        target = onehot.argmax(dim=1)
        recon_cat = recon_cat + F.cross_entropy(logits, target, reduction="mean")
    recon_cat = recon_cat / max(len(cat_sizes), 1)

    # -----------------------------
    # KL(q||p) prior-conditioned
    # -----------------------------
    kl = kl_gaussians(mu, logvar, mu_prior, logvar_prior).mean()

    loss = num_loss + recon_cat + float(beta) * kl

    # -------- logging formatting --------
    def _f6(x):
        return f"{float(x):.6f}"

    log_sigma_mean = (
        outputs[1].detach().mean().cpu()
        if (use_learned_sigma and len(outputs) > 1)
        else float("nan")
    )

    return loss, {
        "loss": _f6(loss.detach().cpu()),
        "num_loss": _f6(num_loss.detach().cpu()),
        "nll_num": _f6(nll_num.detach().cpu()),
        "mse_num": _f6(mse_num.detach().cpu()),
        "w_num": _f6(w_num),
        "lambda_nll": _f6(lambda_nll),
        "sigma_reg": _f6(sigma_reg.detach().cpu()) if torch.is_tensor(sigma_reg) else _f6(sigma_reg),
        "sigma_reg_weight": _f6(sigma_reg_weight),
        "recon_cat": _f6(recon_cat.detach().cpu()),
        "kl": _f6(kl.detach().cpu()),
        "log_sigma_mean": _f6(log_sigma_mean) if np.isfinite(float(log_sigma_mean)) else "nan",
    }


# -------------------------
# Sampling helper (style aligned with wgan)
# -------------------------
@torch.no_grad()
def sample_tvae_processed_cond(
    model: SemaPopTVAE_OneHot,
    schema: TabularSchema,
    embeddings_pers: np.ndarray,
    device: str = "cuda",
    batch_size: int = 512,
    cat_mode: str = "argmax",
    cat_temperature: float = 1.0,
):
    """
    Returns:
      samples_proc: np.ndarray [B, D] in processed space (num_scaled + cat_onehot_hard)
    """
    dev = torch.device(device)
    e = torch.tensor(embeddings_pers, dtype=torch.float32, device=dev)
    x = model.sample_processed(e, cat_mode=cat_mode, cat_temperature=cat_temperature)
    return x.detach().cpu().numpy()


def processed_to_label_df(samples_proc: np.ndarray, schema: TabularSchema, num_cols: List[str], cat_cols: List[str]):
    """
    Convert processed samples (num_scaled + cat_onehot) to a DataFrame:
      - numeric columns: standardized floats
      - categorical columns: label indices (int) via argmax on each one-hot block
    """
    if hasattr(samples_proc, "toarray"):
        samples_proc = samples_proc.toarray()
    X = np.asarray(samples_proc)

    num_dim = schema.num_dim
    out = {}

    # numeric
    for i, name in enumerate(num_cols):
        out[name] = X[:, i].astype(np.float32)

    # categorical: onehot -> argmax
    s = num_dim
    for name, k in zip(cat_cols, schema.cat_dims):
        block = X[:, s:s+k]
        out[name] = np.argmax(block, axis=1).astype(int)
        s += k

    import pandas as pd
    return pd.DataFrame(out)
