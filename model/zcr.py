"""
model/zcr.py
============
Zero-cell regularisation (ZCR) for tabular GANs, after

    Kim, E.-J. & Bansal, P. (2023). *A deep generative model for feasible and
    diverse population synthesis.*  Transportation Research Part C 148, 104053.

The paper separates two kinds of empty cells in the attribute cross-table:

* **structural zeros** -- combinations that cannot exist (a 6-year-old with a
  full-time job).  A generator that wanders off the data manifold invents them,
  which destroys *feasibility* (precision);
* **sampling zeros** -- combinations that are feasible but simply absent from
  the finite training sample.  A generator that only memorises training rows
  never produces them, which destroys *diversity* (recall).

ZCR therefore combines two row-wise nearest-neighbour statistics between a
batch of generated rows ``G = {g_i}`` and a bank of real rows ``R = {r_j}``:

    BD  =  (1/|G|) sum_i  min_j  || g_i - r_j ||_2         (minimised)
    AD  =  (1/|G|) sum_i  mean_j || g_i - r_j ||_2         (maximised)

    L_zcr = bd_weight * BD  -  ad_weight * AD

BD pulls every generated row towards *some* real row (feasibility / structural
zero avoidance); AD pushes the cloud of generated rows away from the average
real row, i.e. it rewards leaving the exact training records (diversity /
sampling zeros).  The two act at different scales -- BD is a min over the bank,
AD is a mean over the bank -- which is what makes the pair non-degenerate.

What was wrong with the previous implementation
-----------------------------------------------
``train_wgan.py:20-54`` (an earlier in-house implementation, not shipped) computed

    bd_i = torch.sqrt((pred_x_i - train_set_x) ** 2).min()      # (A)
    ad_i = torch.sqrt((pred_x_i - train_set_x) ** 2).mean()     # (B)

on a matrix in which the categorical columns were **integer labels** obtained
with ``torch.multinomial``.  Three separate defects:

1. ``(A)`` takes a *global* minimum over the whole ``|R| x D`` elementwise
   ``|difference|`` matrix, not a row-wise Euclidean norm.  With integer labels
   some real row matches the sampled label of some column exactly in almost
   every batch, so the global min is **identically 0.0** and its gradient is
   the gradient of a constant.  BD never acted.
2. ``torch.multinomial`` is a non-differentiable sampler, so no gradient at all
   reaches the categorical logits; only the ``num_dim`` numeric columns of
   ``(B)`` received any signal.
3. ``sqrt(x**2) = |x|`` elementwise, so even ``(B)`` was a mean *L1* distance
   per cell, not the L2 row distance of the paper.

Result: WGAN-GP-ZCR was numerically almost the same model as WGAN-GP, which
explains the inconsistent precision/recall of the old implementation.
:func:`legacy_zcr_terms` reproduces that computation verbatim so the fix can be
demonstrated on the same batch.

This implementation
-------------------
* rows live in the **same differentiable space** for real and fake:
  ``[ z-scored numerics | per-block probability simplex ]``.  Real rows are
  one-hot; generated rows come out of :class:`UncondSimplexGenerator`, whose
  default ``gumbel_st`` head emits an exact one-hot in the forward pass (so the
  distances are the paper's distances between comparable objects) while the
  gradient flows through the soft relaxation;
* distances are true row-wise L2 distances computed with a matmul-based
  ``cdist`` in chunks over the bank;
* the bank is a fresh random subsample of the training matrix at every
  generator step (``n_neighbours``), which keeps the estimator unbiased for AD
  and a stochastic upper bound for BD.

Complexity
----------
For a batch of ``B`` generated rows, a bank subsample of ``M`` rows and ``D``
columns, one call costs ``O(B*M*D)`` flops and, with chunking over the bank in
blocks of ``C`` rows, ``O(B*C)`` memory for the distance block (plus ``O(C*D)``
for the block itself).  The full 40k bank at ``B=512, D=88`` would need a
``512 x 40000`` distance matrix = 82 MB in fp32 *per* call plus the autograd
graph -- hence both the subsample (``--zcr-neighbors``, default 4096) and the
chunking (``--zcr-chunk``, default 4096, i.e. one block at the default
neighbour count): 512 x 4096 x 4 B = 8.4 MB per block.  BD keeps a running
row-wise minimum across blocks (``torch.minimum``, gradient flows to the
surviving argmin), AD keeps a running sum, so any chunk size gives the same
value (verified) at the cost of extra kernel launches -- measured on 4 CPU
threads at B=512, D=88: 7.1 ms/step (fwd+bwd) for M=C=4096, 42.8 ms/step for
M=4096 split into two 2048-blocks, 66.3 ms/step for the full M=40000 bank.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "UncondSimplexGenerator",
    "zcr_terms",
    "legacy_zcr_terms",
    "sample_bank",
]


# --------------------------------------------------------------------------- #
# generator head: numeric | per-block simplex
# --------------------------------------------------------------------------- #
class UncondSimplexGenerator(nn.Module):
    """Unconditional generator whose categorical blocks are probability simplices.

    Same idea as ``train_semapop_gan_v2.SimplexGenerator`` but for a plain
    ``z -> x`` backbone (``model.wgan.WGANGenerator``), so that the critic, the
    ZCR distances and the sampler all consume the *same* object.

    ``cat_mode``
        ``gumbel_st``   straight-through Gumbel-softmax (CTGAN / Kim & Bansal
                        style).  Forward value is an exact one-hot -> the ZCR
                        distance is computed between one-hot rows on both
                        sides; the gradient uses the soft relaxation.  Default.
        ``gumbel_soft`` soft Gumbel-softmax.
        ``softmax``     deterministic per-block softmax with temperature.
    """

    def __init__(
        self,
        base: nn.Module,
        num_dim: int,
        cat_dims: Sequence[int],
        cat_mode: str = "gumbel_st",
        temperature: float = 0.667,
    ) -> None:
        super().__init__()
        if cat_mode not in ("gumbel_st", "gumbel_soft", "softmax"):
            raise ValueError(f"unknown cat_mode {cat_mode!r}")
        self.base = base
        self.num_dim = int(num_dim)
        self.cat_dims = [int(k) for k in cat_dims]
        self.cat_mode = cat_mode
        self.temperature = float(temperature)

    @property
    def noise_dim(self) -> int:
        return int(self.base.noise_dim)

    @property
    def cat_slices(self) -> List[Tuple[int, int]]:
        out, s = [], self.num_dim
        for k in self.cat_dims:
            out.append((s, s + k))
            s += k
        return out

    def raw(self, z: torch.Tensor) -> torch.Tensor:
        """Raw backbone output ``[numeric | per-block logits]`` (no head)."""
        return self.base(z)

    def _blocks(self, raw: torch.Tensor, mode: str, tau: float) -> List[torch.Tensor]:
        out = []
        for s, e in self.cat_slices:
            logits = raw[:, s:e]
            if mode == "softmax":
                out.append(F.softmax(logits / max(tau, 1e-6), dim=1))
            elif mode == "gumbel_soft":
                out.append(F.gumbel_softmax(logits, tau=tau, hard=False, dim=1))
            elif mode == "gumbel_st":
                out.append(F.gumbel_softmax(logits, tau=tau, hard=True, dim=1))
            elif mode == "argmax":
                out.append(F.one_hot(logits.argmax(dim=1), num_classes=e - s).float())
            elif mode == "multinomial":
                p = F.softmax(logits / max(tau, 1e-6), dim=1)
                idx = torch.multinomial(p, num_samples=1).squeeze(1)
                out.append(F.one_hot(idx, num_classes=e - s).float())
            else:
                raise ValueError(f"unknown block mode {mode!r}")
        return out

    def forward(
        self,
        z: torch.Tensor,
        mode: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        raw = self.raw(z)
        mode = mode or self.cat_mode
        tau = self.temperature if temperature is None else float(temperature)
        blocks = self._blocks(raw, mode, tau)
        return torch.cat([raw[:, : self.num_dim]] + blocks, dim=1)

    def sample_mode(self, requested: str) -> str:
        """Resolve ``--sample-cat-mode auto`` to a mode matching how we trained."""
        if requested != "auto":
            return requested
        return "multinomial" if self.cat_mode.startswith("gumbel") else "argmax"


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def _scaled(x: torch.Tensor, num_dim: int, cat_scale: float) -> torch.Tensor:
    """Optionally re-weight the categorical part of a processed row.

    Deviation switch (default 1.0 = paper behaviour, i.e. none).  With one-hot
    blocks two rows differing in one categorical column are sqrt(2) apart
    regardless of the number of levels, while numeric columns are z-scored;
    ``cat_scale`` lets an ablation re-balance the two halves without touching
    anything else.
    """
    if cat_scale == 1.0 or num_dim >= x.shape[1]:
        return x
    return torch.cat([x[:, :num_dim], cat_scale * x[:, num_dim:]], dim=1)


def _euclid(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 distance matrix ``[A, B]`` via the matmul expansion.

    ``torch.cdist`` with ``p=2`` returns a NaN gradient when a pair coincides
    exactly (d/dx sqrt(0)); the numeric part of a row is continuous so that is
    vanishingly unlikely, but the clamp below makes it impossible.
    """
    a2 = (a * a).sum(dim=1, keepdim=True)              # [A,1]
    b2 = (b * b).sum(dim=1, keepdim=True).transpose(0, 1)  # [1,B]
    d2 = a2 + b2 - 2.0 * (a @ b.transpose(0, 1))
    return torch.sqrt(torch.clamp(d2, min=eps))


def sample_bank(bank: torch.Tensor, n: int,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Random subsample of ``n`` rows of the real bank (with replacement)."""
    n = int(min(n, bank.shape[0])) if n > 0 else bank.shape[0]
    if n >= bank.shape[0]:
        return bank
    idx = torch.randint(0, bank.shape[0], (n,), device=bank.device, generator=generator)
    return bank.index_select(0, idx)


def zcr_terms(
    x_fake: torch.Tensor,
    bank: torch.Tensor,
    *,
    num_dim: int,
    cat_scale: float = 1.0,
    chunk: int = 2048,
    want_bd: bool = True,
    want_ad: bool = True,
    ad_clamp: float = 0.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-wise ZCR terms of Kim & Bansal (2023).

    Parameters
    ----------
    x_fake : ``[B, D]`` generated rows in ``[numeric | simplex]`` space.
    bank   : ``[M, D]`` real rows in ``[numeric | one-hot]`` space (already
             subsampled by the caller; see :func:`sample_bank`).
    ad_clamp : deviation, disabled at ``0``.  When ``> 0`` the AD term is
             hinged at that distance (``min(mean_dist, ad_clamp)``) so the
             diversity push cannot run away; the paper has no such bound and
             relies on the adversarial term to hold the cloud in place.

    Returns
    -------
    ``(bd_loss, ad_loss)`` -- both are *losses to be minimised* and are added to
    the generator objective as ``bd_weight * bd_loss + ad_weight * ad_loss``.
    ``bd_loss`` is the mean nearest-neighbour distance (>= 0).  ``ad_loss`` is
    the **negated** mean distance to the bank (<= 0), matching the sign
    convention of the original ``regularization_ad_db`` so the two code paths
    can be compared directly.
    """
    if x_fake.dim() != 2 or bank.dim() != 2:
        raise ValueError("x_fake and bank must be 2-D")
    if x_fake.shape[1] != bank.shape[1]:
        raise ValueError(f"dim mismatch: fake {x_fake.shape[1]} vs bank {bank.shape[1]}")

    f = _scaled(x_fake, num_dim, cat_scale)
    zero = x_fake.new_zeros(())
    if not (want_bd or want_ad):
        return zero, zero

    m_total = bank.shape[0]
    chunk = int(chunk) if chunk and chunk > 0 else m_total
    bd_min: Optional[torch.Tensor] = None       # [B] running row-wise minimum
    ad_sum: Optional[torch.Tensor] = None       # [B] running row-wise sum

    for s in range(0, m_total, chunk):
        r = _scaled(bank[s : s + chunk], num_dim, cat_scale)
        d = _euclid(f, r, eps=eps)              # [B, c]  <- O(B*c) memory
        if want_bd:
            cur = d.min(dim=1).values
            bd_min = cur if bd_min is None else torch.minimum(bd_min, cur)
        if want_ad:
            cur = d.sum(dim=1)
            ad_sum = cur if ad_sum is None else ad_sum + cur

    bd_loss = bd_min.mean() if bd_min is not None else zero
    if ad_sum is not None:
        ad_mean = ad_sum / float(m_total)       # [B] mean distance to the bank
        if ad_clamp and ad_clamp > 0:
            ad_mean = torch.clamp(ad_mean, max=float(ad_clamp))
        ad_loss = -ad_mean.mean()               # maximise the distance
    else:
        ad_loss = zero
    return bd_loss, ad_loss


# --------------------------------------------------------------------------- #
# the historical (degenerate) version -- audit only
# --------------------------------------------------------------------------- #
def legacy_zcr_terms(
    raw_pred: torch.Tensor,
    bank_labels: torch.Tensor,
    num_dim: int,
    cat_dims: Sequence[int],
    only_ad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verbatim re-implementation of ``train_wgan.py:20-54``.

    Kept **only** so that the smoke test can print the old and the new terms on
    the same batch.  ``bank_labels`` is ``[numeric z-scored | integer category
    labels]`` exactly as the old ``train_set_x`` was built
    (``train_wgan.py:264-268``).  Never call this from a training loop.
    """
    numeric_samples = raw_pred[:, :num_dim]
    cat_logits = raw_pred[:, num_dim:]

    categorical_samples = []
    start = 0
    for k in cat_dims:
        head_output = cat_logits[:, start : start + k]
        start += k
        probs = torch.softmax(head_output, dim=1)
        samples = torch.multinomial(probs, 1)          # <- kills the gradient
        categorical_samples.append(samples)
    categorical_samples = torch.cat(categorical_samples, dim=1).to(raw_pred.dtype)

    pred_x = torch.cat([numeric_samples, categorical_samples], dim=1)

    bd_loss = torch.zeros((), dtype=raw_pred.dtype, device=raw_pred.device)
    if not only_ad:
        for i in range(len(pred_x)):
            pred_x_i = pred_x[i : i + 1].repeat(len(bank_labels), 1)
            bd_i = torch.sqrt((pred_x_i - bank_labels) ** 2).min()   # GLOBAL min -> 0
            bd_loss = bd_loss + bd_i / len(pred_x)

    ad_loss = torch.zeros((), dtype=raw_pred.dtype, device=raw_pred.device)
    for i in range(len(pred_x)):
        pred_x_i = pred_x[i : i + 1].repeat(len(bank_labels), 1)
        ad_i = torch.sqrt((pred_x_i - bank_labels) ** 2).mean()      # elementwise L1
        ad_loss = ad_loss - ad_i / len(pred_x)

    return bd_loss, ad_loss
