#!/usr/bin/env python
"""
model/tabddpm_v2.py
===================
A **faithful, compact** re-implementation of TabDDPM (Kotelnikov, Baranchuk,
Rubachev & Babenko, ICML 2023, "TabDDPM: Modelling Tabular Data with Diffusion
Models") for the v2 baseline experiments.

Why this file exists
--------------------
``model/tabddpm.py`` (an earlier in-house implementation, not shipped) is
**not** TabDDPM.  It runs a single *Gaussian* DDPM over the whole
row, i.e. over a continuous **relaxation** of the one-hot blocks, adds a
cross-entropy term on the categorical logits, and projects the categorical
slices to one-hot with an ``argmax`` *at every reverse step*
(``model/tabddpm.py::p_sample_mixed`` / ``_project_cat_in_chain``).  That is a
"Gaussian-relaxed tabular DDPM", not the multinomial diffusion of the paper:
the forward process it is trained on never matches the reverse process it is
sampled with, and the argmax projection is not a sample from any posterior.

This module implements the real thing:

* **Gaussian diffusion** on the numeric block (eps-prediction MSE, exactly the
  usual DDPM objective);
* **Multinomial diffusion** (Hoogeboom et al. 2021) *per categorical block*:
  the uniform-mixing forward kernel
  ``q(x_t | x_{t-1}) = Cat(alpha_t x_{t-1} + (1-alpha_t)/K)``,
  the closed-form ``q(x_t | x_0)``, the true posterior ``q(x_{t-1} | x_t, x_0)``
  and the KL-based categorical variational bound (with the categorical decoder
  NLL at ``t == 0``);
* a shared MLP denoiser over ``[numeric | per-block log-probabilities]`` with a
  sinusoidal time embedding, plus an **optional FiLM conditioning hook** (the
  ``cond`` argument threaded through ``forward`` / ``p_sample`` / ``sample``)
  so that SemaPop-DDPM (persona conditioning) can reuse this module unchanged.
  For the baseline ``cond`` is ``None`` and ``cond_dim == 0``.

Provenance
----------
The diffusion math was transcribed from the official implementation,
https://github.com/yandex-research/tab-ddpm , files
``tab_ddpm/gaussian_multinomial_diffsuion.py`` and ``tab_ddpm/utils.py``
(fetched 2026-08-18).  Specifically the following were taken and adapted:

* ``get_named_beta_schedule`` / ``betas_for_alpha_bar``  (cosine + linear);
* the Gaussian block: ``gaussian_q_sample``,
  ``gaussian_q_posterior_mean_variance``, ``gaussian_p_mean_variance`` (with
  their ``model_variance = cat([posterior_variance[1], (1-alphas)[1:]])``
  choice), ``_predict_xstart_from_eps``, ``gaussian_p_sample``,
  ``_gaussian_loss`` (mse / vb);
* the multinomial block: ``q_pred_one_timestep``, ``q_pred``, ``q_posterior``,
  ``predict_start``, ``p_pred``, ``p_sample``, ``log_sample_categorical``,
  ``kl_prior``, ``compute_Lt``, ``sample_time``, ``_multinomial_loss`` and the
  ``mixed_loss`` / ``sample`` composition (including the ``loss_multi /
  n_blocks`` normalisation and the ``kl / pt`` importance re-weighting);
* helpers ``log_1_min_a``, ``log_add_exp``, ``extract``,
  ``index_to_log_onehot``, ``sum_except_batch``, ``mean_flat``,
  ``log_categorical``, ``normal_kl``.

Deliberate differences from the official code (all cosmetic / numerical, none
change the objective):

1. ``sliced_logsumexp`` (their ``logcumsumexp`` + ``log_sub_exp`` trick) is
   replaced by a plain per-block ``torch.logsumexp`` loop.  Identical values,
   no catastrophic cancellation, and it works for a handful of blocks.
2. In-place writes into ``torch.empty_like`` buffers are replaced by
   ``torch.cat`` over per-block tensors (autograd-friendlier).
3. The class-label conditioning (``out_dict['y']``, ``y_dist``) of the official
   code is replaced by the generic ``cond`` FiLM hook; the baseline is
   unconditional.
4. Their Gaussian VB path uses ``discretized_gaussian_log_likelihood``, which
   assumes uint8-scaled images; here the ``t = 0`` term is a continuous
   Gaussian decoder NLL instead.  Only relevant for ``gaussian_loss_type='kl'``
   (not the default, which is ``'mse'`` as in the paper).
5. Optional ``generator=`` arguments so sampling is reproducible from a seed
   that is independent of the training seed.

Status (measured, not claimed)
------------------------------
The implementation is validated *without any training* by two oracle reverse
chains in ``tests/test_tabddpm_v2.py`` (A5 / A6): plugging the analytically
optimal predictor into ``p_sample`` reproduces the data law to Monte-Carlo
accuracy (categorical TV < 2e-3 at T=50 and T=1000; Gaussian mean/std within
0.003).  On real data the faithful defaults are *numerically fragile* -- see
``cat_input`` below and ``clip_x0`` in
:class:`GaussianMultinomialDiffusion` -- so the recommended baseline invocation
is ``--cat-input onehot --clip-x0 5``.

Data convention
---------------
``x_num``: ``[B, num_dim]`` float, already scaled (z-score or quantile-normal).
``x_cat``: ``[B, n_cat]`` int64 **category codes** (not one-hot).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TabularSchema",
    "SinusoidalTimeEmbedding",
    "TabDenoiserV2",
    "GaussianMultinomialDiffusion",
    "index_to_log_onehot",
    "log_onehot_to_index",
]

#: official TabDDPM clamps the one-hot at 1e-30 before taking the log
ONEHOT_LOG_EPS = 1e-30


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
@dataclass
class TabularSchema:
    """``x = [numeric | onehot(cat_1) | ... | onehot(cat_m)]`` layout."""

    num_dim: int
    cat_dims: List[int]

    @property
    def cat_total(self) -> int:
        return int(sum(self.cat_dims))

    @property
    def data_dim(self) -> int:
        return int(self.num_dim) + self.cat_total

    @property
    def n_cat(self) -> int:
        return len(self.cat_dims)

    @property
    def cat_slices(self) -> List[Tuple[int, int]]:
        """Slices **inside the categorical part only** (offset 0 = first block)."""
        out, start = [], 0
        for k in self.cat_dims:
            out.append((start, start + int(k)))
            start += int(k)
        return out


# --------------------------------------------------------------------------- #
# small helpers (ported from tab_ddpm/utils.py)
# --------------------------------------------------------------------------- #
def log_1_min_a(a: torch.Tensor) -> torch.Tensor:
    return torch.log((1.0 - a.exp()).clamp_min(0.0) + 1e-40)


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.max(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m))


def extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    """``a[t]`` broadcast to ``shape`` (``a`` is a [T] buffer, ``t`` is [B])."""
    out = a.to(t.device).gather(-1, t)
    while out.dim() < len(shape):
        out = out[..., None]
    return out.expand(shape)


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).sum(-1)


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=list(range(1, x.dim())))


def log_categorical(log_x_start: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x: torch.Tensor, cat_dims: Sequence[int]) -> torch.Tensor:
    """``[B, m]`` int codes -> ``[B, sum K]`` log one-hot (``log 1e-30`` for zeros)."""
    parts = [F.one_hot(x[:, i].long(), int(k)) for i, k in enumerate(cat_dims)]
    onehot = torch.cat(parts, dim=1).float()
    return torch.log(onehot.clamp_min(ONEHOT_LOG_EPS))


def log_onehot_to_index(log_x: torch.Tensor, cat_dims: Sequence[int]) -> torch.Tensor:
    """``[B, sum K]`` log-probs -> ``[B, m]`` argmax codes."""
    out, s = [], 0
    for k in cat_dims:
        out.append(log_x[:, s:s + int(k)].argmax(dim=1))
        s += int(k)
    return torch.stack(out, dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2) -> torch.Tensor:
    return 0.5 * (
        -1.0 + logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def betas_for_alpha_bar(T: int, alpha_bar, max_beta: float = 0.999) -> np.ndarray:
    betas = []
    for i in range(T):
        t1, t2 = i / T, (i + 1) / T
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.asarray(betas, dtype=np.float64)


def get_named_beta_schedule(name: str, T: int) -> np.ndarray:
    """``cosine`` (Nichol & Dhariwal) or ``linear`` (Ho et al.), as in TabDDPM."""
    if name == "linear":
        scale = 1000.0 / T
        return np.linspace(scale * 1e-4, scale * 2e-2, T, dtype=np.float64)
    if name == "cosine":
        return betas_for_alpha_bar(
            T, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
    raise NotImplementedError(f"unknown beta schedule {name!r}")


def block_log_softmax(x: torch.Tensor, slices: Sequence[Tuple[int, int]]) -> torch.Tensor:
    return torch.cat([F.log_softmax(x[:, s:e], dim=1) for s, e in slices], dim=1)


def block_logsumexp(x: torch.Tensor, slices: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Per-block ``logsumexp``, expanded back to the block width (normaliser)."""
    return torch.cat(
        [torch.logsumexp(x[:, s:e], dim=1, keepdim=True).expand(-1, e - s)
         for s, e in slices],
        dim=1,
    )


# --------------------------------------------------------------------------- #
# denoiser
# --------------------------------------------------------------------------- #
class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer/DDPM sinusoidal embedding of the integer timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class TabDenoiserV2(nn.Module):
    """MLP denoiser over ``[x_num_t | log q(x_cat_t)]``.

    Output layout is the same as the input: ``[eps_num | cat_logits]``, with
    ``cat_logits`` interpreted per block by
    :meth:`GaussianMultinomialDiffusion.predict_start`.

    ``cond`` hook (SemaPop-DDPM).  When ``cond_dim > 0`` a small adapter maps
    the conditioning vector (e.g. a persona embedding) to per-layer FiLM
    ``(gamma, beta)`` pairs applied *before* the non-linearity:
    ``h <- (1 + gamma) * h + beta``.  Initialised to the identity (the last
    adapter layer is zeroed), so an untrained conditional model is exactly the
    unconditional one.  For the unconditional baseline ``cond_dim == 0`` and ``cond`` must
    be ``None``.

    ``cat_input``.  The categorical part of ``x`` is *always* an exact log
    one-hot (``log_sample_categorical`` -> ``index_to_log_onehot``), i.e. the
    vector ``log(1e-30) * (1 - onehot)``.  ``'logprob'`` (default) feeds it as
    is, exactly like the official TabDDPM; ``'onehot'`` feeds ``exp`` of it.
    The two are an affine re-parameterisation of the *same* information and the
    diffusion process is identical -- but ``'logprob'`` presents the MLP with
    inputs of magnitude 69 next to numeric inputs of magnitude 1, which makes
    the sinusoidal time embedding (magnitude ~1, added to ``proj(x)``)
    numerically negligible at initialisation.  Use ``'onehot'`` if a run
    collapses; whichever was used is recorded in ``metrics.json``.
    """

    def __init__(
        self,
        num_dim: int,
        cat_dims: Sequence[int],
        hidden: Sequence[int] = (512, 512, 512),
        dim_t: int = 128,
        dropout: float = 0.0,
        cond_dim: int = 0,
        cond_hidden: int = 256,
        cat_input: str = "logprob",
        persona_dim: int = 0,
        adapter_hidden: int = 1024,
        adapter_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if cat_input not in ("logprob", "onehot"):
            raise ValueError(f"unknown cat_input {cat_input!r}")
        self.cat_input = cat_input
        self.num_dim = int(num_dim)
        self.cat_dims = [int(k) for k in cat_dims]
        self.d_in = self.num_dim + sum(self.cat_dims)
        self.dim_t = int(dim_t)
        self.cond_dim = int(cond_dim)
        self.hidden = [int(h) for h in hidden]

        # SemaPop-DDPM: the SAME PersonaAdapter the GAN arm uses, so the two backbones
        # differ in the generator and nothing else.  Without it `cond_dim` would have to
        # equal the 4096-d embedding width and the searched cond_dim / adapter_hidden /
        # adapter_dropout would be inert -- the arms would then differ in conditioning
        # capacity as well as in backbone, which is the one thing the comparison must not
        # confound.  persona_dim == 0 leaves the unconditional baseline untouched.
        self.persona_adapter = None
        if int(persona_dim) > 0:
            from model.wgan_llm import PersonaAdapter
            self.persona_adapter = PersonaAdapter(int(persona_dim), self.cond_dim,
                                                  hidden=int(adapter_hidden),
                                                  dropout=float(adapter_dropout))

        self.time_emb = SinusoidalTimeEmbedding(self.dim_t)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.dim_t, self.dim_t), nn.SiLU(),
            nn.Linear(self.dim_t, self.dim_t),
        )
        # official MLPDiffusion: x = proj(x) + time_emb, then the MLP trunk
        self.proj = nn.Linear(self.d_in, self.dim_t)

        layers, d = [], self.dim_t
        for h in self.hidden:
            layers.append(nn.Linear(d, h))
            d = h
        self.layers = nn.ModuleList(layers)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(d, self.d_in)

        self.film: Optional[nn.Module] = None
        if self.cond_dim > 0:
            out_dim = 2 * sum(self.hidden)
            self.film = nn.Sequential(
                nn.Linear(self.cond_dim, cond_hidden), nn.SiLU(),
                nn.Linear(cond_hidden, out_dim),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond is not None and self.film is None:
            raise ValueError(
                "TabDenoiserV2 was built with cond_dim=0 but a cond tensor was "
                "passed; rebuild with cond_dim=<persona dim> for SemaPop-DDPM"
            )
        if self.cat_input == "onehot" and self.num_dim < x.shape[1]:
            # log q(x_cat_t) is always an exact log ONE-HOT ({0, log 1e-30}), so
            # exponentiating is an exact affine re-parameterisation of the same
            # input -- it changes nothing about the diffusion, only the scale the
            # MLP sees (|-69| vs 1).  See the class docstring.
            x = torch.cat([x[:, : self.num_dim], x[:, self.num_dim:].exp()], dim=1)
        emb = self.time_mlp(self.time_emb(t))
        h = self.proj(x) + emb

        gammas_betas = None
        if self.film is not None and cond is not None:
            if self.persona_adapter is not None:
                cond = self.persona_adapter(cond)
            gammas_betas = self.film(cond)

        off = 0
        for lin, width in zip(self.layers, self.hidden):
            h = lin(h)
            if gammas_betas is not None:
                gamma = gammas_betas[:, off:off + width]
                beta = gammas_betas[:, off + width:off + 2 * width]
                h = (1.0 + gamma) * h + beta
            off += 2 * width
            h = self.drop(self.act(h))
        return self.head(h)


# --------------------------------------------------------------------------- #
# Gaussian + multinomial diffusion
# --------------------------------------------------------------------------- #
class GaussianMultinomialDiffusion(nn.Module):
    """Gaussian diffusion on the numeric block, multinomial diffusion on cats.

    Parameters
    ----------
    denoise_fn
        module with signature ``(x, t, cond) -> [B, num_dim + sum K]``.
    num_numerical_features
        width of the numeric block.
    num_classes
        ``[K_1, ..., K_m]`` categories per categorical column.
    num_timesteps
        ``T``.
    scheduler
        ``'cosine'`` (default, as in the paper) or ``'linear'``.
    gaussian_loss_type
        ``'mse'`` (paper default) or ``'kl'`` (full Gaussian VB term).
    gaussian_parametrization
        ``'eps'`` (default) or ``'x0'``.
    parametrization
        multinomial reverse parametrization, ``'x0'`` (default; predict
        ``x_0`` logits then apply the exact posterior) or ``'direct'``.
    clip_x0
        if ``> 0``, clamp the reconstructed ``x_0`` of the *Gaussian* block to
        ``[-clip_x0, +clip_x0]`` in the scaled space.  **Off by default**: the
        official TabDDPM never clips (``clip_denoised`` is accepted but
        ignored).  At ``t = T-1`` the eps -> x_0 map has gain
        ``1/sqrt(abar_T) ~ 2e4`` for a cosine schedule with ``T = 1000``, so an
        under-trained denoiser can make the numeric chain diverge; this switch
        is the escape hatch and is recorded in ``metrics.json``.
    """

    def __init__(
        self,
        denoise_fn: nn.Module,
        num_numerical_features: int,
        num_classes: Sequence[int],
        num_timesteps: int = 1000,
        scheduler: str = "cosine",
        gaussian_loss_type: str = "mse",
        gaussian_parametrization: str = "eps",
        parametrization: str = "x0",
        multinomial_loss_type: str = "vb_stochastic",
        clip_x0: float = 0.0,
    ) -> None:
        super().__init__()
        if parametrization not in ("x0", "direct"):
            raise ValueError(f"unknown parametrization {parametrization!r}")
        if gaussian_parametrization not in ("eps", "x0"):
            raise ValueError(f"unknown gaussian_parametrization {gaussian_parametrization!r}")
        if gaussian_loss_type not in ("mse", "kl"):
            raise ValueError(f"unknown gaussian_loss_type {gaussian_loss_type!r}")
        if multinomial_loss_type not in ("vb_stochastic",):
            raise ValueError(f"unknown multinomial_loss_type {multinomial_loss_type!r}")

        self._denoise_fn = denoise_fn
        self.num_numerical_features = int(num_numerical_features)
        self.num_classes = [int(k) for k in num_classes]
        self.num_timesteps = int(num_timesteps)
        self.scheduler = scheduler
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.parametrization = parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.clip_x0 = float(clip_x0)

        self.schema = TabularSchema(self.num_numerical_features, list(self.num_classes))
        self.cat_slices = self.schema.cat_slices
        self.cat_total = self.schema.cat_total

        # ---- schedules (float64 -> float32 buffers) ------------------------
        betas_np = get_named_beta_schedule(scheduler, self.num_timesteps)
        alphas_np = 1.0 - betas_np
        alphas = torch.from_numpy(alphas_np)
        betas = torch.from_numpy(betas_np)

        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=0)
        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas.dtype),
                                         alphas_cumprod[:-1]], dim=0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # official TabDDPM: replace the (zero) t=0 entry by the t=1 entry
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]], dim=0)
        )
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas)
                                / (1.0 - alphas_cumprod))
        # variance actually used by p (official: sigma_t^2 = beta_t except t=0)
        model_variance = torch.cat([posterior_variance[1:2], betas[1:]], dim=0)

        def reg(name: str, v: torch.Tensor) -> None:
            self.register_buffer(name, v.float())

        reg("betas", betas)
        reg("alphas", alphas)
        reg("log_alpha", log_alpha)
        reg("log_1_min_alpha", log_1_min_alpha)
        reg("log_cumprod_alpha", log_cumprod_alpha)
        reg("log_1_min_cumprod_alpha", log_1_min_cumprod_alpha)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        reg("posterior_variance", posterior_variance)
        reg("posterior_log_variance_clipped", posterior_log_variance_clipped)
        reg("posterior_mean_coef1", posterior_mean_coef1)
        reg("posterior_mean_coef2", posterior_mean_coef2)
        reg("model_variance", model_variance)
        reg("model_log_variance", torch.log(model_variance))

        # [sum K] vector holding K for every one-hot slot (the "1/K" of the
        # uniform mixing kernel)
        if self.cat_total:
            expanded = np.concatenate(
                [np.full(k, k, dtype=np.float64) for k in self.num_classes]
            )
        else:
            expanded = np.zeros(0, dtype=np.float64)
        self.register_buffer("num_classes_expanded",
                             torch.from_numpy(expanded).float())
        self.register_buffer("log_num_classes_expanded",
                             torch.log(torch.from_numpy(expanded).float().clamp_min(1.0)))

        # importance sampling of t (Nichol & Dhariwal / multinomial-diffusion)
        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))

        # sanity checks from the official code
        with torch.no_grad():
            assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1e-5
            assert (log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha)
                    .abs().sum().item() < 1e-5)

    # ------------------------------------------------------------------ #
    # device helper
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return self.betas.device

    # ------------------------------------------------------------------ #
    # Gaussian part
    # ------------------------------------------------------------------ #
    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var = extract(self.posterior_variance, t, x_t.shape)
        logvar = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, logvar

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps)

    def gaussian_p_mean_variance(self, model_output, x, t) -> Dict[str, torch.Tensor]:
        var = extract(self.model_variance, t, x.shape)
        logvar = extract(self.model_log_variance, t, x.shape)
        if self.gaussian_parametrization == "eps":
            pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
        else:
            pred_xstart = model_output
        if self.clip_x0 > 0:
            pred_xstart = pred_xstart.clamp(-self.clip_x0, self.clip_x0)
        mean, _, _ = self.gaussian_q_posterior_mean_variance(pred_xstart, x, t)
        return {"mean": mean, "variance": var, "log_variance": logvar,
                "pred_xstart": pred_xstart}

    def gaussian_p_sample(self, model_out, x, t, generator=None) -> torch.Tensor:
        out = self.gaussian_p_mean_variance(model_out, x, t)
        noise = _randn(x.shape, x.device, generator)
        nonzero = (t != 0).float().view(-1, *([1] * (x.dim() - 1)))
        return out["mean"] + nonzero * torch.exp(0.5 * out["log_variance"]) * noise

    def _gaussian_vb(self, model_out, x_start, x_t, t) -> torch.Tensor:
        true_mean, _, true_logvar = self.gaussian_q_posterior_mean_variance(x_start, x_t, t)
        out = self.gaussian_p_mean_variance(model_out, x_t, t)
        kl = mean_flat(normal_kl(true_mean, true_logvar, out["mean"], out["log_variance"]))
        kl = kl / math.log(2.0)
        # continuous Gaussian decoder NLL (the official code uses the uint8
        # discretised likelihood, which is meaningless for scaled tabular data)
        nll = 0.5 * (
            math.log(2 * math.pi) + out["log_variance"]
            + (x_start - out["mean"]) ** 2 * torch.exp(-out["log_variance"])
        )
        nll = mean_flat(nll) / math.log(2.0)
        return torch.where(t == 0, nll, kl)

    def _gaussian_loss(self, model_out, x_start, x_t, t, noise) -> torch.Tensor:
        if self.gaussian_loss_type == "mse":
            return mean_flat((noise - model_out) ** 2)
        return self._gaussian_vb(model_out, x_start, x_t, t)

    # ------------------------------------------------------------------ #
    # Multinomial part
    # ------------------------------------------------------------------ #
    @staticmethod
    def multinomial_kl(log_prob1, log_prob2) -> torch.Tensor:
        return (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)

    def q_pred_one_timestep(self, log_x_t, t):
        """``log q(x_t | x_{t-1}) = log[ alpha_t x_{t-1} + (1 - alpha_t)/K ]``."""
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        return log_add_exp(log_x_t + log_alpha_t,
                           log_1_min_alpha_t - self.log_num_classes_expanded)

    def q_pred(self, log_x_start, t):
        """``log q(x_t | x_0) = log[ abar_t x_0 + (1 - abar_t)/K ]``."""
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        return log_add_exp(log_x_start + log_cumprod_alpha_t,
                           log_1_min_cumprod_alpha - self.log_num_classes_expanded)

    def predict_start(self, model_out, log_x_t=None, t=None):
        """Per-block log-softmax of the denoiser's categorical logits."""
        if model_out.shape[1] != self.cat_total:
            raise ValueError(f"expected {self.cat_total} cat logits, got {model_out.shape[1]}")
        return block_log_softmax(model_out, self.cat_slices)

    def q_posterior(self, log_x_start, log_x_t, t):
        """``log q(x_{t-1} | x_t, x_0)`` (exact, per block)."""
        t_minus_1 = (t - 1).clamp_min(0)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)
        t_broadcast = t.view(-1, 1).expand_as(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)
        unnormed = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        return unnormed - block_logsumexp(unnormed, self.cat_slices)

    def p_pred(self, model_out, log_x, t):
        if self.parametrization == "x0":
            log_x_recon = self.predict_start(model_out, log_x, t)
            return self.q_posterior(log_x_recon, log_x, t)
        return self.predict_start(model_out, log_x, t)

    def log_sample_categorical(self, logits, generator=None) -> torch.Tensor:
        """Gumbel-max sample per block, returned as a log one-hot."""
        idx = []
        for s, e in self.cat_slices:
            block = logits[:, s:e]
            u = _rand(block.shape, block.device, generator)
            g = -torch.log(-torch.log(u + 1e-30) + 1e-30)
            idx.append((g + block).argmax(dim=1, keepdim=True))
        return index_to_log_onehot(torch.cat(idx, dim=1), self.num_classes)

    def q_sample(self, log_x_start, t, generator=None):
        return self.log_sample_categorical(self.q_pred(log_x_start, t), generator)

    def p_sample(self, model_out, log_x, t, generator=None):
        return self.log_sample_categorical(self.p_pred(model_out, log_x, t), generator)

    def kl_prior(self, log_x_start) -> torch.Tensor:
        """``KL(q(x_T | x_0) || Uniform)`` -- constant in the parameters."""
        b = log_x_start.shape[0]
        ones = torch.ones(b, device=log_x_start.device, dtype=torch.long)
        log_qxT = self.q_pred(log_x_start, (self.num_timesteps - 1) * ones)
        log_uniform = -self.log_num_classes_expanded.expand_as(log_qxT)
        return sum_except_batch(self.multinomial_kl(log_qxT, log_uniform))

    def compute_Lt(self, model_out, log_x_start, log_x_t, t) -> torch.Tensor:
        """``L_{t-1}`` for ``t>0`` and the decoder NLL at ``t == 0``."""
        log_true_prob = self.q_posterior(log_x_start, log_x_t, t)
        log_model_prob = self.p_pred(model_out, log_x_t, t)
        kl = sum_except_batch(self.multinomial_kl(log_true_prob, log_model_prob))
        decoder_nll = sum_except_batch(-log_categorical(log_x_start, log_model_prob))
        mask = (t == 0).float()
        return mask * decoder_nll + (1.0 - mask) * kl

    def sample_time(self, b: int, device, method: str = "uniform", generator=None):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, "uniform", generator)
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 1e-4
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)
            t = torch.multinomial(pt_all, num_samples=b, replacement=True,
                                  generator=generator).long()
            return t, pt_all.gather(0, t)
        if method == "uniform":
            t = _randint(self.num_timesteps, b, device, generator)
            return t, torch.full((b,), 1.0 / self.num_timesteps, device=device)
        raise ValueError(f"unknown time sampling method {method!r}")

    def _update_Lt_history(self, t: torch.Tensor, loss: torch.Tensor) -> None:
        with torch.no_grad():
            Lt2 = loss.detach().float() ** 2
            for ti, v in zip(t.tolist(), Lt2.tolist()):
                self.Lt_history[ti] = 0.1 * self.Lt_history[ti] + 0.9 * v
                self.Lt_count[ti] = self.Lt_count[ti] + 1

    # ------------------------------------------------------------------ #
    # combined objective
    # ------------------------------------------------------------------ #
    def mixed_loss(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        time_method: str = "uniform",
        return_parts: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """TabDDPM's combined loss.

        Returns a dict with ``loss_multi`` (categorical VB / #blocks),
        ``loss_gauss`` (eps-MSE) and the raw diagnostics ``kl_multi`` (the
        un-reweighted per-row VB term) and ``kl_prior``.
        """
        b = x_num.shape[0] if self.num_numerical_features else x_cat.shape[0]
        device = x_num.device if self.num_numerical_features else x_cat.device
        t, pt = self.sample_time(b, device, time_method)

        noise = None
        x_num_t = x_num
        if self.num_numerical_features:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)

        log_x_cat = log_x_cat_t = torch.zeros((b, 0), device=device)
        if self.cat_total:
            log_x_cat = index_to_log_onehot(x_cat, self.num_classes)
            log_x_cat_t = self.q_sample(log_x_cat, t)

        model_out = self._denoise_fn(
            torch.cat([x_num_t, log_x_cat_t], dim=1).float(), t, cond
        )
        out_num = model_out[:, : self.num_numerical_features]
        out_cat = model_out[:, self.num_numerical_features:]

        zero = torch.zeros((), device=device)
        loss_multi = zero
        kl_raw = zero
        kl_prior = zero
        if self.cat_total:
            kl = self.compute_Lt(out_cat, log_x_cat, log_x_cat_t, t)
            if time_method == "importance":
                self._update_Lt_history(t, kl)
            kl_prior = self.kl_prior(log_x_cat)
            vb = kl / pt + kl_prior
            loss_multi = vb.mean() / len(self.num_classes)
            kl_raw = kl.mean()
            kl_prior = kl_prior.mean()

        loss_gauss = zero
        if self.num_numerical_features:
            loss_gauss = self._gaussian_loss(out_num, x_num, x_num_t, t, noise).mean()

        out = {"loss_multi": loss_multi, "loss_gauss": loss_gauss,
               "kl_multi": kl_raw, "kl_prior": kl_prior}
        if return_parts:
            # SemaPop-DDPM's marginal regulariser needs a DIFFERENTIABLE x0 estimate.
            # A GAN gets one for free from a forward pass; a diffusion model would have
            # to run the whole reverse chain, so it reads the denoiser's own x0
            # prediction at the sampled timestep instead.  Additive and off by default,
            # so the unconditional baseline is untouched.
            out.update({"out_cat": out_cat, "out_num": out_num,
                        "log_x_cat_t": log_x_cat_t, "x_num_t": x_num_t, "t": t})
        return out

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        n: int,
        cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        progress_cb=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ancestral sampling.  Returns ``(x_num [n, num_dim], x_cat [n, m])``."""
        device = self.device
        was_training = self.training
        self.eval()

        z_norm = _randn((n, self.num_numerical_features), device, generator)
        if self.cat_total:
            uniform_logits = torch.zeros((n, self.cat_total), device=device)
            log_z = self.log_sample_categorical(uniform_logits, generator)
        else:
            log_z = torch.zeros((n, 0), device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(), t, cond
            )
            out_num = model_out[:, : self.num_numerical_features]
            out_cat = model_out[:, self.num_numerical_features:]
            if self.num_numerical_features:
                z_norm = self.gaussian_p_sample(out_num, z_norm, t, generator)
            if self.cat_total:
                log_z = self.p_sample(out_cat, log_z, t, generator)
            if progress_cb is not None:
                progress_cb(self.num_timesteps - i)

        x_cat = (log_onehot_to_index(log_z, self.num_classes) if self.cat_total
                 else torch.zeros((n, 0), dtype=torch.long, device=device))
        if was_training:
            self.train()
        return z_norm, x_cat

    @torch.no_grad()
    def sample_batched(
        self,
        n: int,
        cond: Optional[torch.Tensor] = None,
        batch_size: int = 4096,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nums, cats = [], []
        done = 0
        while done < n:
            b = min(batch_size, n - done)
            # the slice is moved per batch, never the whole conditioning matrix: at the
            # national tier n=3,051,604 and a 4096-wide cond is 50 GB, which no card holds.
            # Values and generator consumption are unchanged -- self.sample sees the same
            # rows in the same order as when the caller pre-moved the full tensor.
            c = None if cond is None else cond[done:done + b].to(self.device,
                                                                 non_blocking=True)
            xn, xc = self.sample(b, cond=c, generator=generator)
            nums.append(xn.cpu())
            cats.append(xc.cpu())
            done += b
        return torch.cat(nums, 0), torch.cat(cats, 0)


# --------------------------------------------------------------------------- #
# rng helpers (so sampling can be driven by an explicit generator)
# --------------------------------------------------------------------------- #
def _randn(shape, device, generator=None) -> torch.Tensor:
    return torch.randn(*shape, device=device, generator=generator)


def _rand(shape, device, generator=None) -> torch.Tensor:
    return torch.rand(*shape, device=device, generator=generator)


def _randint(high: int, b: int, device, generator=None) -> torch.Tensor:
    return torch.randint(0, high, (b,), device=device, generator=generator).long()
