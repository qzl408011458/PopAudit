"""Borysov, Rich & Pereira (2019) population-synthesis VAE -- PyTorch transcription.

WHAT THIS IS AND IS NOT.  This is OUR PyTorch transcription of the model, loss and sampler
of ``pop-synth-vae.ipynb`` cells 36-38 from github.com/stasmix/popsynth @089df3e (MIT,
Stanislav Borysov, 18 Oct 2018), the code released with

    S. S. Borysov, J. Rich, F. C. Pereira, "How to generate micro-agents?  A deep generative
    modeling approach to population synthesis", Transportation Research Part C 106:73-97
    (2019); preprint arXiv:1808.06910.

It is NOT their code and it was NOT run.  Their notebook cannot be executed: it is a
dehydrated skeleton whose dataset-specific names are placeholders and which carries at least
four NameErrors (``cols`` undefined inside prepare_data's get_dummies; ``year_ids`` undefined
in get_split_ids), and it targets TF1.x + standalone Keras, which caps at Python 3.7 and
builds against CUDA 10.0 -- no sm_86 support, so it cannot run on sm_86 GPUs such as the A6000 either.
Any row produced by this file must be labelled "Borysov-style VAE (our implementation)" and
must never be attributed to their code.

WHERE THE RELEASED CODE AND THE PAPER DISAGREE -- the code wins here, because "their
implementation" is what a baseline row claims to be, and the discrepancies are recorded so a
reader can flip them:

  numeric term   CODE  K.sum(K.square(pred - true))            -- NO 1/2
                 PAPER Eq. (2) has 1/2.  The code therefore doubles the numeric weight
                 relative to the categorical term of Eq. (1).  ``half_sse=True`` restores
                 the paper's form.
  KL sigma       PAPER Eq. (4) reads -(1/2) sum(1 + log s - mu^2 - s) with the prose calling
                 s a standard deviation, which is only the correct closed form if s is the
                 VARIANCE.  The code settles it: z_log_var is log-variance
                 (reparameterisation uses exp(z_log_var/2)), so the textbook KL is right.
  cat weight     The code carries a ``cat_loss_weight`` multiplier with no counterpart in
                 Eq. (1).  Its grid is the single value [1.0], so it is inert; kept here at
                 1.0 for transcription fidelity, not as a knob to tune.

THE REDUCTION IS THE PORT'S MAIN TRAP.  Keras computes, per sample, a SUM over numeric
features, a SUM over categorical groups of that group's cross-entropy, and a SUM over latent
dimensions for the KL -- then ONE mean over the minibatch:
``K.mean(recon_num + cat_loss_weight * recon_cat + beta * kl)``.  A port that averages any of
those over features, groups or latent dims silently rescales beta and the numeric:categorical
ratio, which are exactly the two quantities the paper grid-searches.  Every reduction below
is therefore sum-then-mean, matching cell 37 line by line.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BorysovVAE(nn.Module):
    """Encoder n-A-B-D_Z, decoder the exact mirror D_Z-B-A-n (cell 37).

    Hidden activation is tanh throughout; the paper names it and the released grid searches
    only ``activation_grid = ['tanh']``.  z_mean and z_log_var are LINEAR heads.  The decoder
    emits one linear head over all numeric columns and one softmax head PER categorical
    attribute -- the "multinomial logit form" of the paper's Sec. 3.1.  No dropout, no batch
    norm (the notebook imports BatchNormalization at cell 34 and never uses it).
    """

    def __init__(self, num_dim: int, cat_dims: Sequence[int],
                 hidden_layers_shape: Sequence[int], latent_dim: int) -> None:
        super().__init__()
        self.num_dim = int(num_dim)
        self.cat_dims = [int(d) for d in cat_dims]
        self.latent_dim = int(latent_dim)
        self.in_dim = self.num_dim + sum(self.cat_dims)

        enc: List[nn.Module] = []
        d = self.in_dim
        for w in hidden_layers_shape:
            enc += [nn.Linear(d, int(w)), nn.Tanh()]
            d = int(w)
        self.enc = nn.Sequential(*enc)
        self.z_mean = nn.Linear(d, self.latent_dim)
        self.z_log_var = nn.Linear(d, self.latent_dim)

        dec: List[nn.Module] = []
        d = self.latent_dim
        for w in reversed(list(hidden_layers_shape)):
            dec += [nn.Linear(d, int(w)), nn.Tanh()]
            d = int(w)
        self.dec = nn.Sequential(*dec)
        self.num_head = nn.Linear(d, self.num_dim) if self.num_dim > 0 else None
        self.cat_heads = nn.ModuleList([nn.Linear(d, k) for k in self.cat_dims])

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.z_mean(h), self.z_log_var(h)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (numeric outputs, per-attribute LOGITS).

        Logits rather than probabilities: the loss below needs log-softmax for numerical
        stability, and the sampler needs an argmax, which is invariant to the softmax.  The
        released code applies softmax inside the Dense layer; the two are equivalent here.
        """
        h = self.dec(z)
        num = self.num_head(h) if self.num_head is not None else h.new_zeros((h.shape[0], 0))
        return num, [head(h) for head in self.cat_heads]

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        # cell 37: z_mean + K.exp(z_log_var / 2) * epsilon, epsilon ~ N(0, 1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        num, cat_logits = self.decode(z)
        return num, cat_logits, mu, logvar


def borysov_loss(num_pred: torch.Tensor, cat_logits: Sequence[torch.Tensor],
                 x_num: torch.Tensor, x_cat_onehots: Sequence[torch.Tensor],
                 mu: torch.Tensor, logvar: torch.Tensor, beta: float,
                 cat_loss_weight: float = 1.0, half_sse: bool = False):
    """K.mean(recon_num + cat_loss_weight * recon_cat + beta * kl), cell 37 verbatim."""
    B = mu.shape[0]
    if num_pred.shape[1] > 0:
        se = (num_pred - x_num) ** 2
        recon_num = se.sum(dim=1)                       # per-sample SUM over numeric features
        if half_sse:                                    # paper Eq. (2); the code omits the 1/2
            recon_num = 0.5 * recon_num
    else:
        recon_num = mu.new_zeros(B)

    # Keras categorical_crossentropy on a softmax block is -sum(y_true * log p), per sample;
    # the released code SUMS that over attribute groups without averaging.
    recon_cat = mu.new_zeros(B)
    for logits, onehot in zip(cat_logits, x_cat_onehots):
        recon_cat = recon_cat - (onehot * F.log_softmax(logits, dim=1)).sum(dim=1)

    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)   # SUM over latent dims

    total = (recon_num + float(cat_loss_weight) * recon_cat + float(beta) * kl).mean()
    return total, {"recon_num": float(recon_num.mean()),
                   "recon_cat": float(recon_cat.mean()),
                   "kl": float(kl.mean())}


@torch.no_grad()
def borysov_sample(model: BorysovVAE, n: int, device, generator=None) -> np.ndarray:
    """cell 38 vae_get_samples: z ~ N(0,1), decode, ARGMAX each categorical group.

    The categorical output is argmaxed back to a one-hot, NOT drawn from the softmax -- the
    released sampler is deterministic given z.  Numeric inverse-scaling, integer rounding and
    min/max clipping happen in the caller, because this repo's F6 contract (clip THEN round,
    against the measured training support) differs from the notebook's round-then-clip against
    author-declared bins; the difference is recorded as a departure rather than reproduced.
    """
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device, generator=generator)
    num, cat_logits = model.decode(z)
    parts = [num.cpu().numpy()] if model.num_dim > 0 else []
    for logits in cat_logits:
        idx = logits.argmax(dim=1).cpu().numpy()
        oh = np.zeros((n, logits.shape[1]), dtype=np.float32)
        oh[np.arange(n), idx] = 1.0
        parts.append(oh)
    return np.concatenate(parts, axis=1) if parts else np.zeros((n, 0), np.float32)
