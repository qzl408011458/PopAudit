"""Persona embedding priors p_psi(e) for SemaPop.

Why this module exists
----------------------
Persona embeddings produced by ``model.EmbeddingLLM.QwenPersonaEmbeddingB`` are
4096-d, LayerNorm-ed vectors whose norm is *exactly* ``sqrt(4096) == 64`` and
whose raw pairwise cosine similarity is ~0.99 for every pair.  Neither the norm
nor the raw-space nearest-neighbour cosine carries information, so a prior must
be fitted -- and diagnosed -- in a *centred + whitened PCA subspace*.

Two rules encoded here (both are deliberate, do not "simplify" them away):

1. **Subspace consistency.**  The generator must be trained on the same subspace
   it is sampled from.  Use :meth:`SubspaceTransform.project_and_invert` on the
   train/val embeddings before feeding the adapter (``in_dim=4096`` variant), or
   feed the whitened coordinates directly (``in_dim=n_components`` variant).
   Mixing "raw 4096-d for training" with "prior samples for sampling" is the bug
   this module is meant to prevent.
2. **Model selection by validation likelihood only.**  ``K``, covariance type,
   flow capacity, ... are chosen with :meth:`score_val` on held-out *validation*
   embeddings.  Never select a prior by downstream fidelity (SRMSE / F1 / E1).

Uniform prior API
-----------------
Every prior implements::

    prior.fit(E_train, E_val)      # E_* : (n, 4096) float arrays
    prior.sample(n, rng)           # -> (n, 4096) float32, in the projected subspace
    prior.score_val(E_val)         # -> float, HIGHER IS BETTER
    prior.save(path) / Prior.load(path)

``score_val`` is comparable *within* a family (GMM K/covariance grid, flow
checkpoints) for a **fixed** :class:`SubspaceTransform`.  It is NOT comparable
across families (GMM returns an exact log-density, the flow/diffusion priors
return a negative training-objective proxy) and not comparable across different
subspaces (a change of basis changes the density by a Jacobian factor).

CLI
---
    python -m model.persona_prior --emb CA/embPers.pkl --emb-val CA/embPers_val.pkl \
        --out <semapop_repo>/results/prior_ca --family gmm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, Sequence[np.ndarray], Dict[str, np.ndarray]]


def load_embeddings(path: str) -> np.ndarray:
    """Load a (n, d) embedding matrix from .pkl / .npy / .npz."""
    if path.endswith(".npy"):
        E = np.load(path)
    elif path.endswith(".npz"):
        z = np.load(path)
        E = z[z.files[0]]
    else:
        with open(path, "rb") as f:
            E = pickle.load(f)
    E = np.asarray(E)
    if E.ndim != 2:
        raise ValueError(f"expected a 2-D embedding matrix, got shape {E.shape} from {path}")
    return np.ascontiguousarray(E, dtype=np.float32)


def _as_list(E: ArrayLike) -> Tuple[List[np.ndarray], List[str]]:
    """Normalise ndarray / list / dict input into (arrays, names)."""
    if isinstance(E, dict):
        names = list(E.keys())
        return [np.asarray(E[k]) for k in names], names
    if isinstance(E, np.ndarray):
        return [E], ["all"]
    if isinstance(E, (list, tuple)):
        return [np.asarray(e) for e in E], [f"src{i}" for i in range(len(E))]
    raise TypeError(f"unsupported embedding container: {type(E)}")


# ---------------------------------------------------------------------------
# 1. subspace
# ---------------------------------------------------------------------------

class SubspaceTransform:
    """Centred PCA (+ optional whitening) of persona embeddings.

    ``fit`` accepts a single ``(n, d)`` array, a list of arrays or a
    ``{state: array}`` dict; multiple sources are **pooled** so the basis is
    shared across states (required by E8 for cross-state direction comparison).
    Use :func:`fit_subspaces` with ``mode="per_state"`` for per-state bases.

    Attributes set by ``fit``
    -------------------------
    mean_          (d,)             pooled mean
    components_    (n_spectrum, d)  principal directions, rows orthonormal
    eigvals_       (n_spectrum,)    variance along each direction (s^2/(n-1))
    total_variance_ float           trace of the full covariance (all d dims)
    """

    def __init__(
        self,
        n_components: int = 128,
        whiten: bool = True,
        n_spectrum: int = 256,
        random_state: int = 0,
        eps: float = 1e-8,
    ) -> None:
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.n_spectrum = max(int(n_spectrum), int(n_components))
        self.random_state = int(random_state)
        self.eps = float(eps)

        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.eigvals_: Optional[np.ndarray] = None
        self.total_variance_: Optional[float] = None
        self.n_samples_: int = 0
        self.sources_: List[str] = []

    # -- fitting ------------------------------------------------------------
    def fit(self, E: ArrayLike) -> "SubspaceTransform":
        from sklearn.utils.extmath import randomized_svd

        mats, names = _as_list(E)
        d = mats[0].shape[1]
        for m in mats:
            if m.shape[1] != d:
                raise ValueError("all embedding sources must share the same dimension")
        X = np.concatenate([np.asarray(m, dtype=np.float64) for m in mats], axis=0)
        n = X.shape[0]
        if n <= self.n_spectrum:
            self.n_spectrum = max(1, min(self.n_spectrum, n - 1))
            self.n_components = min(self.n_components, self.n_spectrum)

        self.mean_ = X.mean(axis=0)
        X -= self.mean_  # X is always a fresh copy (np.concatenate); centre in place
        # trace of the covariance over ALL d dims (denominator for explained var)
        self.total_variance_ = float(np.einsum("ij,ij->", X, X) / (n - 1))

        _, S, Vt = randomized_svd(
            X, n_components=self.n_spectrum, n_oversamples=16, n_iter=5,
            random_state=self.random_state,
        )
        self.components_ = np.ascontiguousarray(Vt, dtype=np.float64)
        self.eigvals_ = (S ** 2) / (n - 1)
        self.n_samples_ = int(n)
        self.sources_ = names
        return self

    # -- transforms ---------------------------------------------------------
    def _check(self) -> None:
        if self.components_ is None:
            raise RuntimeError("SubspaceTransform is not fitted")

    @property
    def scale_(self) -> np.ndarray:
        """Per-component whitening scale (sqrt of the eigenvalue), length k."""
        self._check()
        k = self.n_components
        if self.whiten:
            return np.sqrt(self.eigvals_[:k] + self.eps)
        return np.ones(k, dtype=np.float64)

    def transform(self, E: np.ndarray) -> np.ndarray:
        """(n, d) embeddings -> (n, k) whitened PCA coordinates."""
        self._check()
        k = self.n_components
        X = np.asarray(E, dtype=np.float64) - self.mean_
        Z = X @ self.components_[:k].T
        return np.ascontiguousarray(Z / self.scale_, dtype=np.float32)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """(n, k) whitened coordinates -> (n, d) embeddings."""
        self._check()
        k = self.n_components
        Zc = np.asarray(Z, dtype=np.float64) * self.scale_
        X = Zc @ self.components_[:k] + self.mean_
        return np.ascontiguousarray(X, dtype=np.float32)

    def project_and_invert(self, E: np.ndarray) -> np.ndarray:
        """Drop the residual energy outside the top-k subspace.

        **This is what the generator must be trained on** so that the training
        space and the prior sampling space coincide.
        """
        return self.inverse_transform(self.transform(E))

    def reconstruction_error(self, E: np.ndarray) -> Dict[str, float]:
        """Relative L2 error of ``project_and_invert`` (per-row and global)."""
        E = np.asarray(E, dtype=np.float64)
        R = np.asarray(self.project_and_invert(E), dtype=np.float64)
        num = np.linalg.norm(E - R, axis=1)
        den = np.linalg.norm(E, axis=1) + 1e-12
        cen = np.linalg.norm(E - self.mean_, axis=1) + 1e-12
        return {
            "rel_l2_mean": float(np.mean(num / den)),
            "rel_l2_max": float(np.max(num / den)),
            "rel_l2_centred_mean": float(np.mean(num / cen)),
            "global_rel_l2": float(np.linalg.norm(E - R) / (np.linalg.norm(E) + 1e-12)),
        }

    # -- spectrum -----------------------------------------------------------
    def explained_variance_ratio_at(self, ks: Iterable[int] = (16, 32, 64, 128, 256)) -> Dict[int, float]:
        """Cumulative share of the *centred* variance captured by the top-k PCs."""
        self._check()
        c = np.cumsum(self.eigvals_)
        out: Dict[int, float] = {}
        for k in ks:
            k = int(k)
            if k <= len(c):
                out[k] = float(c[k - 1] / self.total_variance_)
        return out

    def effective_rank(self) -> Dict[str, float]:
        """Entropy-based effective rank and participation ratio of the spectrum."""
        self._check()
        ev = np.clip(self.eigvals_, 0.0, None)
        p = ev / max(ev.sum(), 1e-30)
        p = p[p > 0]
        return {
            "effective_rank_entropy": float(np.exp(-(p * np.log(p)).sum())),
            "participation_ratio": float((ev.sum() ** 2) / max((ev ** 2).sum(), 1e-30)),
            "n_spectrum": int(len(ev)),
        }

    # -- persistence --------------------------------------------------------
    def state_dict(self) -> Dict[str, np.ndarray]:
        self._check()
        return {
            "mean": self.mean_.astype(np.float64),
            "components": self.components_.astype(np.float64),
            "eigvals": self.eigvals_.astype(np.float64),
            "meta": np.array(json.dumps({
                "n_components": self.n_components,
                "whiten": self.whiten,
                "n_spectrum": self.n_spectrum,
                "random_state": self.random_state,
                "eps": self.eps,
                "total_variance": self.total_variance_,
                "n_samples": self.n_samples_,
                "sources": self.sources_,
            })),
        }

    @classmethod
    def from_state_dict(cls, sd: Dict[str, np.ndarray]) -> "SubspaceTransform":
        meta = json.loads(str(sd["meta"]))
        obj = cls(
            n_components=meta["n_components"], whiten=meta["whiten"],
            n_spectrum=meta["n_spectrum"], random_state=meta["random_state"],
            eps=meta["eps"],
        )
        obj.mean_ = np.asarray(sd["mean"], dtype=np.float64)
        obj.components_ = np.asarray(sd["components"], dtype=np.float64)
        obj.eigvals_ = np.asarray(sd["eigvals"], dtype=np.float64)
        obj.total_variance_ = float(meta["total_variance"])
        obj.n_samples_ = int(meta["n_samples"])
        obj.sources_ = list(meta["sources"])
        return obj

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if not path.endswith(".npz"):
            path = path + ".npz"
        np.savez_compressed(path, **self.state_dict())
        return path

    @classmethod
    def load(cls, path: str) -> "SubspaceTransform":
        z = np.load(path, allow_pickle=False)
        return cls.from_state_dict({k: z[k] for k in z.files})


def fit_subspaces(
    E_by_state: Dict[str, np.ndarray],
    mode: str = "pooled",
    n_components: int = 128,
    **kwargs,
) -> Union[SubspaceTransform, Dict[str, SubspaceTransform]]:
    """Fit a shared (``mode='pooled'``) or per-state basis.

    ``pooled`` is the plan default: one basis fitted on the concatenated train
    embeddings of all states, so directions are comparable across states (E8).
    """
    if mode == "pooled":
        return SubspaceTransform(n_components=n_components, **kwargs).fit(E_by_state)
    if mode == "per_state":
        return {
            s: SubspaceTransform(n_components=n_components, **kwargs).fit(E)
            for s, E in E_by_state.items()
        }
    raise ValueError("mode must be 'pooled' or 'per_state'")


# ---------------------------------------------------------------------------
# 2. base class
# ---------------------------------------------------------------------------

class BasePrior:
    """Uniform prior interface; see the module docstring."""

    family: str = "base"

    def __init__(self, transform: Optional[SubspaceTransform] = None) -> None:
        self.transform = transform if transform is not None else SubspaceTransform()
        self.fitted_ = False

    # subclasses implement _fit / _sample_z / _score_z
    def _fit(self, Z_train: np.ndarray, Z_val: Optional[np.ndarray]) -> None:
        raise NotImplementedError

    def _sample_z(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _score_z(self, Z: np.ndarray) -> float:
        raise NotImplementedError

    def fit(self, E_train: ArrayLike, E_val: Optional[np.ndarray] = None,
            fit_transform: bool = True) -> "BasePrior":
        if fit_transform and self.transform.components_ is None:
            self.transform.fit(E_train)
        mats, _ = _as_list(E_train)
        Z_train = self.transform.transform(np.concatenate(mats, axis=0))
        Z_val = self.transform.transform(E_val) if E_val is not None else None
        self._fit(Z_train, Z_val)
        self.fitted_ = True
        return self

    def sample(self, n: int, rng: Union[int, np.random.Generator, None] = None) -> np.ndarray:
        """Draw ``n`` persona embeddings, returned in the 4096-d projected space."""
        if not self.fitted_:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        Z = self._sample_z(int(n), rng)
        return self.transform.inverse_transform(Z)

    def sample_z(self, n: int, rng: Union[int, np.random.Generator, None] = None) -> np.ndarray:
        """Draw ``n`` samples in whitened coordinates (for ``in_dim=k`` adapters)."""
        rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        return self._sample_z(int(n), rng)

    def score_val(self, E_val: np.ndarray) -> float:
        """Validation score, HIGHER IS BETTER. Family-internal comparisons only."""
        return self._score_z(self.transform.transform(E_val))

    # persistence -----------------------------------------------------------
    def save(self, path: str) -> str:
        raise NotImplementedError

    @staticmethod
    def load(path: str) -> "BasePrior":
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. GMM prior
# ---------------------------------------------------------------------------

class GMMPrior(BasePrior):
    """Gaussian mixture in whitened PCA space, selected by validation log-lik.

    The ``(K, covariance_type)`` grid is scanned and the configuration with the
    highest mean per-sample validation log-likelihood wins.  ``selection_table_``
    keeps every candidate for the appendix table.
    """

    family = "gmm"

    def __init__(
        self,
        transform: Optional[SubspaceTransform] = None,
        k_grid: Sequence[int] = (4, 8, 16, 32),
        covariance_types: Sequence[str] = ("full", "diag"),
        reg_covar: float = 1e-6,
        n_init: int = 2,
        max_iter: int = 300,
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(transform)
        self.k_grid = tuple(int(k) for k in k_grid)
        self.covariance_types = tuple(covariance_types)
        self.reg_covar = float(reg_covar)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self.gmm_ = None
        self.best_k_: Optional[int] = None
        self.best_cov_: Optional[str] = None
        self.best_val_ll_: Optional[float] = None
        self.selection_table_: List[Dict[str, float]] = []
        self._chol_: Optional[np.ndarray] = None

    def _fit(self, Z_train: np.ndarray, Z_val: Optional[np.ndarray]) -> None:
        from sklearn.mixture import GaussianMixture

        if Z_val is None:
            raise ValueError("GMMPrior needs validation embeddings for model selection")
        best = None
        for cov in self.covariance_types:
            for k in self.k_grid:
                t0 = time.time()
                gm = GaussianMixture(
                    n_components=k, covariance_type=cov, reg_covar=self.reg_covar,
                    n_init=self.n_init, max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                gm.fit(Z_train)
                row = {
                    "K": int(k), "covariance_type": cov,
                    "train_ll": float(gm.score(Z_train)),
                    "val_ll": float(gm.score(Z_val)),
                    "bic_train": float(gm.bic(Z_train)),
                    "converged": bool(gm.converged_),
                    "fit_sec": round(time.time() - t0, 2),
                }
                self.selection_table_.append(row)
                if self.verbose:
                    print(f"  [gmm] K={k:3d} cov={cov:5s} val_ll={row['val_ll']:.3f} "
                          f"train_ll={row['train_ll']:.3f} ({row['fit_sec']}s)")
                if best is None or row["val_ll"] > best[0]:
                    best = (row["val_ll"], k, cov, gm)

        self.best_val_ll_, self.best_k_, self.best_cov_, self.gmm_ = best
        self._chol_ = self._make_chol(self.gmm_)

    @staticmethod
    def _make_chol(gm) -> np.ndarray:
        """Per-component lower-triangular sampling factor, shape (K, k, k)."""
        k = gm.means_.shape[1]
        K = gm.means_.shape[0]
        L = np.zeros((K, k, k), dtype=np.float64)
        ct = gm.covariance_type
        for j in range(K):
            if ct == "full":
                C = gm.covariances_[j]
            elif ct == "diag":
                C = np.diag(gm.covariances_[j])
            elif ct == "tied":
                C = gm.covariances_
            elif ct == "spherical":
                C = np.eye(k) * gm.covariances_[j]
            else:
                raise ValueError(f"unsupported covariance_type {ct}")
            L[j] = np.linalg.cholesky(C + 1e-10 * np.eye(k))
        return L

    def _sample_z(self, n: int, rng: np.random.Generator) -> np.ndarray:
        w = self.gmm_.weights_
        comp = rng.choice(len(w), size=n, p=w / w.sum())
        eps = rng.standard_normal((n, self.gmm_.means_.shape[1]))
        Z = self.gmm_.means_[comp] + np.einsum("nij,nj->ni", self._chol_[comp], eps)
        return Z.astype(np.float32)

    def _score_z(self, Z: np.ndarray) -> float:
        return float(self.gmm_.score(Z))

    # persistence -----------------------------------------------------------
    def save(self, path: str) -> str:
        if not path.endswith(".npz"):
            path = path + ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        sd = {f"tf_{k}": v for k, v in self.transform.state_dict().items()}
        sd["weights"] = self.gmm_.weights_
        sd["means"] = self.gmm_.means_
        sd["covariances"] = np.asarray(self.gmm_.covariances_)
        sd["chol"] = self._chol_
        sd["meta"] = np.array(json.dumps({
            "family": self.family,
            "covariance_type": self.gmm_.covariance_type,
            "best_k": self.best_k_, "best_cov": self.best_cov_,
            "best_val_ll": self.best_val_ll_,
            "selection_table": self.selection_table_,
            "k_grid": list(self.k_grid),
            "covariance_types": list(self.covariance_types),
            "random_state": self.random_state,
        }))
        np.savez_compressed(path, **sd)
        return path

    @staticmethod
    def load(path: str) -> "GMMPrior":
        from sklearn.mixture import GaussianMixture

        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        tf = SubspaceTransform.from_state_dict({k[3:]: z[k] for k in z.files if k.startswith("tf_")})
        obj = GMMPrior(transform=tf, k_grid=meta["k_grid"],
                       covariance_types=meta["covariance_types"],
                       random_state=meta["random_state"])
        gm = GaussianMixture(n_components=int(meta["best_k"]),
                             covariance_type=meta["covariance_type"])
        gm.weights_ = z["weights"]
        gm.means_ = z["means"]
        gm.covariances_ = z["covariances"]
        gm.precisions_cholesky_ = _precisions_cholesky(z["covariances"], meta["covariance_type"])
        obj.gmm_ = gm
        obj._chol_ = z["chol"]
        obj.best_k_ = int(meta["best_k"])
        obj.best_cov_ = meta["best_cov"]
        obj.best_val_ll_ = float(meta["best_val_ll"])
        obj.selection_table_ = meta["selection_table"]
        obj.fitted_ = True
        return obj


def _precisions_cholesky(cov: np.ndarray, covariance_type: str) -> np.ndarray:
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    return _compute_precision_cholesky(cov, covariance_type)


# ---------------------------------------------------------------------------
# 4. shared time-conditioned MLP (flow / diffusion)
# ---------------------------------------------------------------------------

def _sinusoidal_time_embed(t, dim: int):
    import torch
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    ang = t[:, None] * freqs[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


def _make_time_mlp(z_dim: int, hidden: int, n_layers: int, t_dim: int):
    """4x512 SiLU residual MLP with a sinusoidal time embedding."""
    import torch
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.act = nn.SiLU()

        def forward(self, h):
            return h + self.fc2(self.act(self.fc1(self.norm(h))))

    class TimeMLP(nn.Module):
        """v_theta(z, t) (flow matching) or eps_theta(z, t) (diffusion)."""

        def __init__(self):
            super().__init__()
            self.t_dim = t_dim
            self.in_proj = nn.Linear(z_dim, hidden)
            self.t_mlp = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.blocks = nn.ModuleList([_Block(hidden) for _ in range(n_layers)])
            self.out_norm = nn.LayerNorm(hidden)
            self.out_proj = nn.Linear(hidden, z_dim)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        def forward(self, z, t):
            h = self.in_proj(z) + self.t_mlp(_sinusoidal_time_embed(t, self.t_dim))
            for blk in self.blocks:
                h = blk(h)
            return self.out_proj(self.out_norm(h))

    return TimeMLP()


class _TorchPriorMixin:
    """Shared torch plumbing: device handling, training loop, checkpointing."""

    def _device(self):
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _train_loop(self, Z_train: np.ndarray, Z_val: Optional[np.ndarray], loss_fn) -> None:
        import torch

        dev = self._device()
        torch.manual_seed(self.seed)
        self.net_ = _make_time_mlp(Z_train.shape[1], self.hidden, self.n_layers, self.t_dim).to(dev)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        Xtr = torch.as_tensor(Z_train, dtype=torch.float32, device=dev)
        n = Xtr.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        # deterministic validation batch: fixed noise + fixed t so the val curve
        # is a clean early-stopping signal rather than Monte-Carlo noise.
        val_pack = None
        if Z_val is not None:
            Xva = torch.as_tensor(Z_val, dtype=torch.float32, device=dev)
            g2 = torch.Generator(device="cpu").manual_seed(self.seed + 1)
            eps = torch.randn(Xva.shape, generator=g2).to(dev)
            tv = torch.rand(Xva.shape[0], generator=g2).to(dev)
            val_pack = (Xva, eps, tv)

        best = math.inf
        best_state = None
        bad = 0
        self.history_: List[Dict[str, float]] = []
        for ep in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=gen).to(dev)
            tot, nb = 0.0, 0
            for i in range(0, n, self.batch_size):
                xb = Xtr[perm[i:i + self.batch_size]]
                eps = torch.randn(xb.shape, generator=gen).to(dev)
                t = torch.rand(xb.shape[0], generator=gen).to(dev)
                loss = loss_fn(xb, eps, t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach()) * xb.shape[0]
                nb += xb.shape[0]
            tr = tot / max(nb, 1)

            if val_pack is not None:
                self.net_.eval()
                with torch.no_grad():
                    vl = float(loss_fn(*val_pack))
            else:
                vl = tr
            self.history_.append({"epoch": ep, "train_loss": tr, "val_loss": vl})
            if self.verbose and (ep % max(1, self.max_epochs // 20) == 0 or ep == self.max_epochs - 1):
                print(f"  [{self.family}] epoch {ep:4d} train {tr:.5f} val {vl:.5f}")

            if vl < best - self.min_delta:
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.best_val_loss_ = float(best)
        self.epochs_run_ = len(self.history_)

    def _state_blob(self) -> dict:
        return {
            "family": self.family,
            "net": {k: v.cpu() for k, v in self.net_.state_dict().items()},
            "transform": self.transform.state_dict(),
            "cfg": self._cfg(),
            "history": self.history_,
            "best_val_loss": self.best_val_loss_,
        }

    def save(self, path: str) -> str:
        import torch
        if not path.endswith(".pt"):
            path = path + ".pt"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self._state_blob(), path)
        return path


# ---------------------------------------------------------------------------
# 5. flow-matching prior
# ---------------------------------------------------------------------------

class FlowPrior(_TorchPriorMixin, BasePrior):
    """Conditional flow matching (linear path) in whitened PCA space.

    Path ``z_t = (1-t) z0 + t z1`` with ``z0 ~ N(0, I)`` and ``z1`` a real
    whitened persona coordinate; the network regresses the constant conditional
    velocity ``z1 - z0``.  Sampling integrates ``dz/dt = v(z, t)`` from t=0 to
    t=1 with a midpoint or RK4 solver.

    ``score_val`` returns ``-val_cfm_loss`` (higher is better) computed on a
    fixed noise/time draw, so it is a deterministic early-stopping / capacity
    selection signal.  It is *not* a log-likelihood and must not be compared
    against :class:`GMMPrior` scores.
    """

    family = "flow"

    def __init__(
        self,
        transform: Optional[SubspaceTransform] = None,
        hidden: int = 512,
        n_layers: int = 4,
        t_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 512,
        max_epochs: int = 300,
        patience: int = 20,
        min_delta: float = 1e-5,
        steps: int = 50,
        solver: str = "midpoint",
        device: str = "auto",
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        BasePrior.__init__(self, transform)
        self.hidden, self.n_layers, self.t_dim = int(hidden), int(n_layers), int(t_dim)
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.batch_size, self.max_epochs = int(batch_size), int(max_epochs)
        self.patience, self.min_delta = int(patience), float(min_delta)
        self.steps, self.solver = int(steps), str(solver)
        self.device, self.seed, self.verbose = device, int(seed), bool(verbose)
        self.net_ = None
        self.history_: List[Dict[str, float]] = []
        self.best_val_loss_: float = float("nan")
        self.epochs_run_: int = 0

    def _cfg(self) -> dict:
        return {k: getattr(self, k) for k in
                ("hidden", "n_layers", "t_dim", "lr", "weight_decay", "batch_size",
                 "max_epochs", "patience", "min_delta", "steps", "solver", "device", "seed")}

    def _cfm_loss_factory(self):
        import torch
        import torch.nn.functional as F

        def loss_fn(z1, z0, t):
            z_t = (1.0 - t)[:, None] * z0 + t[:, None] * z1
            return F.mse_loss(self.net_(z_t, t), z1 - z0)

        return loss_fn

    def _fit(self, Z_train: np.ndarray, Z_val: Optional[np.ndarray]) -> None:
        self._train_loop(Z_train, Z_val, self._cfm_loss_factory())

    def train(self, E_train: ArrayLike, E_val: Optional[np.ndarray] = None) -> "FlowPrior":
        """Alias of :meth:`fit` (kept because the plan calls it ``train()``)."""
        return self.fit(E_train, E_val)

    def _sample_z(self, n: int, rng: np.random.Generator) -> np.ndarray:
        import torch

        dev = next(self.net_.parameters()).device
        k = self.net_.out_proj.out_features
        z = torch.as_tensor(rng.standard_normal((n, k)), dtype=torch.float32, device=dev)
        dt = 1.0 / self.steps
        self.net_.eval()
        with torch.no_grad():
            for i in range(self.steps):
                t0 = torch.full((n,), i * dt, device=dev)
                if self.solver == "euler":
                    z = z + dt * self.net_(z, t0)
                elif self.solver == "midpoint":
                    k1 = self.net_(z, t0)
                    tm = torch.full((n,), (i + 0.5) * dt, device=dev)
                    z = z + dt * self.net_(z + 0.5 * dt * k1, tm)
                elif self.solver == "rk4":
                    tm = torch.full((n,), (i + 0.5) * dt, device=dev)
                    t1 = torch.full((n,), (i + 1) * dt, device=dev)
                    k1 = self.net_(z, t0)
                    k2 = self.net_(z + 0.5 * dt * k1, tm)
                    k3 = self.net_(z + 0.5 * dt * k2, tm)
                    k4 = self.net_(z + dt * k3, t1)
                    z = z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                else:
                    raise ValueError(f"unknown solver {self.solver}")
        return z.cpu().numpy().astype(np.float32)

    def _score_z(self, Z: np.ndarray) -> float:
        import torch
        dev = next(self.net_.parameters()).device
        g = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        z1 = torch.as_tensor(Z, dtype=torch.float32, device=dev)
        z0 = torch.randn(z1.shape, generator=g).to(dev)
        t = torch.rand(z1.shape[0], generator=g).to(dev)
        self.net_.eval()
        with torch.no_grad():
            return -float(self._cfm_loss_factory()(z1, z0, t))

    @staticmethod
    def load(path: str) -> "FlowPrior":
        import torch
        blob = torch.load(path, map_location="cpu")
        obj = FlowPrior(transform=SubspaceTransform.from_state_dict(blob["transform"]), **blob["cfg"])
        k = obj.transform.n_components
        obj.net_ = _make_time_mlp(k, obj.hidden, obj.n_layers, obj.t_dim)
        obj.net_.load_state_dict(blob["net"])
        obj.net_.to(obj._device())
        obj.history_ = blob.get("history", [])
        obj.best_val_loss_ = blob.get("best_val_loss", float("nan"))
        obj.fitted_ = True
        return obj


# ---------------------------------------------------------------------------
# 6. diffusion prior (optional third family)
# ---------------------------------------------------------------------------

class DiffPrior(_TorchPriorMixin, BasePrior):
    """Small epsilon-prediction DDPM in whitened PCA space (optional, §7.1 P-Diff).

    Same backbone as :class:`FlowPrior`; cosine beta schedule, ``T`` steps of
    ancestral sampling.  ``score_val`` returns the negative denoising MSE on a
    fixed noise/time draw (higher is better, family-internal only).
    """

    family = "diff"

    def __init__(
        self,
        transform: Optional[SubspaceTransform] = None,
        T: int = 200,
        hidden: int = 512,
        n_layers: int = 4,
        t_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 512,
        max_epochs: int = 300,
        patience: int = 20,
        min_delta: float = 1e-5,
        device: str = "auto",
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        BasePrior.__init__(self, transform)
        self.T = int(T)
        self.hidden, self.n_layers, self.t_dim = int(hidden), int(n_layers), int(t_dim)
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.batch_size, self.max_epochs = int(batch_size), int(max_epochs)
        self.patience, self.min_delta = int(patience), float(min_delta)
        self.device, self.seed, self.verbose = device, int(seed), bool(verbose)
        self.net_ = None
        self.history_: List[Dict[str, float]] = []
        self.best_val_loss_: float = float("nan")
        self.epochs_run_: int = 0
        self.steps = self.T  # for reporting symmetry with FlowPrior

    def _cfg(self) -> dict:
        return {k: getattr(self, k) for k in
                ("T", "hidden", "n_layers", "t_dim", "lr", "weight_decay", "batch_size",
                 "max_epochs", "patience", "min_delta", "device", "seed")}

    def _alphas(self):
        """Cosine schedule (Nichol & Dhariwal), returned as numpy alpha_bar."""
        s = 0.008
        t = np.arange(self.T + 1) / self.T
        f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        ab = f / f[0]
        betas = np.clip(1 - ab[1:] / ab[:-1], 1e-8, 0.999)
        alphas = 1.0 - betas
        return betas, alphas, np.cumprod(alphas)

    def _loss_factory(self):
        import torch
        import torch.nn.functional as F

        dev = self._device()
        _, _, ab = self._alphas()
        ab_t = torch.as_tensor(ab, dtype=torch.float32, device=dev)

        def loss_fn(x0, eps, u):
            idx = torch.clamp((u * self.T).long(), 0, self.T - 1)
            a = ab_t[idx][:, None]
            x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps
            tnorm = idx.float() / self.T
            return F.mse_loss(self.net_(x_t, tnorm), eps)

        return loss_fn

    def _fit(self, Z_train: np.ndarray, Z_val: Optional[np.ndarray]) -> None:
        self._train_loop(Z_train, Z_val, self._loss_factory())

    def _sample_z(self, n: int, rng: np.random.Generator) -> np.ndarray:
        import torch

        dev = next(self.net_.parameters()).device
        k = self.net_.out_proj.out_features
        betas, alphas, ab = self._alphas()
        x = torch.as_tensor(rng.standard_normal((n, k)), dtype=torch.float32, device=dev)
        self.net_.eval()
        with torch.no_grad():
            for i in reversed(range(self.T)):
                t = torch.full((n,), i / self.T, device=dev)
                eps = self.net_(x, t)
                a, abar = float(alphas[i]), float(ab[i])
                mean = (x - (betas[i] / math.sqrt(1 - abar)) * eps) / math.sqrt(a)
                if i > 0:
                    z = torch.as_tensor(rng.standard_normal((n, k)), dtype=torch.float32, device=dev)
                    x = mean + math.sqrt(betas[i]) * z
                else:
                    x = mean
        return x.cpu().numpy().astype(np.float32)

    def _score_z(self, Z: np.ndarray) -> float:
        import torch
        dev = next(self.net_.parameters()).device
        g = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        x0 = torch.as_tensor(Z, dtype=torch.float32, device=dev)
        eps = torch.randn(x0.shape, generator=g).to(dev)
        u = torch.rand(x0.shape[0], generator=g).to(dev)
        self.net_.eval()
        with torch.no_grad():
            return -float(self._loss_factory()(x0, eps, u))

    @staticmethod
    def load(path: str) -> "DiffPrior":
        import torch
        blob = torch.load(path, map_location="cpu")
        obj = DiffPrior(transform=SubspaceTransform.from_state_dict(blob["transform"]), **blob["cfg"])
        k = obj.transform.n_components
        obj.net_ = _make_time_mlp(k, obj.hidden, obj.n_layers, obj.t_dim)
        obj.net_.load_state_dict(blob["net"])
        obj.net_.to(obj._device())
        obj.history_ = blob.get("history", [])
        obj.best_val_loss_ = blob.get("best_val_loss", float("nan"))
        obj.fitted_ = True
        return obj


PRIOR_FAMILIES = {"gmm": GMMPrior, "flow": FlowPrior, "diff": DiffPrior}


def build_prior(family: str, transform: Optional[SubspaceTransform] = None, **kwargs) -> BasePrior:
    if family not in PRIOR_FAMILIES:
        raise ValueError(f"unknown prior family {family!r}; choose from {sorted(PRIOR_FAMILIES)}")
    return PRIOR_FAMILIES[family](transform=transform, **kwargs)


def load_prior(path: str) -> BasePrior:
    """Dispatch on the file extension (.npz -> GMM, .pt -> flow/diff)."""
    if path.endswith(".npz"):
        return GMMPrior.load(path)
    import torch
    fam = torch.load(path, map_location="cpu").get("family", "flow")
    return PRIOR_FAMILIES[fam].load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(description="Fit a persona embedding prior p_psi(e).")
    ap.add_argument("--emb", nargs="+", required=True, help="train embedding file(s) (.pkl/.npy)")
    ap.add_argument("--emb-val", required=True, help="validation embedding file")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--family", default="gmm", choices=sorted(PRIOR_FAMILIES))
    ap.add_argument("--n-components", type=int, default=128)
    ap.add_argument("--n-spectrum", type=int, default=256)
    # Refitting the PCA basis is NOT equivalent to reusing one, even on identical data:
    # components are defined up to sign, so a refit can flip an axis and silently put every
    # sample in a different coordinate system from the one the generator was trained on.
    # Any prior meant to be swapped into an EXISTING generator must be fitted in that
    # generator's subspace -- pass its subspace.npz here.
    ap.add_argument("--subspace", default="",
                    help="load this SubspaceTransform instead of fitting a new one")
    ap.add_argument("--k-grid", type=int, nargs="+", default=[4, 8, 16, 32])
    # The shipped grid [4,8,16,32] selected its own BOUNDARY (val LL still rising at
    # K=32), so the optimum was never located.  Worse, the selection CRITERION is
    # wrong: over a 16-config sweep the within-dimension correlation between held-out
    # log-likelihood and downstream probe SRMSE-1 is r=+0.50 -- HIGHER likelihood goes
    # with WORSE downstream.  At k=128 the config with by far the worst val LL (K=256,
    # -240.5 vs K=32's -142.3) had the BEST downstream score.  What does predict
    # downstream is the 1-NN distance to the training manifold (r=+0.57 overall,
    # +0.84 at k=64).  Use --cov-types/--k-grid to pin a config chosen that way
    # instead of letting val LL choose.
    ap.add_argument("--cov-types", nargs="+", default=["full", "diag"],
                    choices=["full", "diag", "tied", "spherical"],
                    help="candidate GMM covariance types (val-LL selected among them)")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-sample", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    E_tr = {os.path.basename(os.path.dirname(p)) or os.path.basename(p): load_embeddings(p)
            for p in args.emb}
    E_va = load_embeddings(args.emb_val)
    print(f"[prior] train sources: { {k: v.shape for k, v in E_tr.items()} }  val: {E_va.shape}")

    if args.subspace:
        tf = SubspaceTransform.load(args.subspace)
        print(f"[prior] reusing subspace {args.subspace} (k={tf.n_components})")
    else:
        tf = SubspaceTransform(n_components=args.n_components, n_spectrum=args.n_spectrum,
                               random_state=args.seed).fit(E_tr)
    evr = tf.explained_variance_ratio_at((16, 32, 64, 128, 256))
    print(f"[prior] explained variance: {evr}")
    tf.save(os.path.join(args.out, "subspace.npz"))

    kw = {"verbose": True}
    if args.family == "gmm":
        kw["k_grid"] = args.k_grid
        kw["covariance_types"] = tuple(args.cov_types)
    else:
        kw["max_epochs"] = args.max_epochs
        kw["device"] = args.device
        kw["seed"] = args.seed
    prior = build_prior(args.family, transform=tf, **kw)
    prior.fit(E_tr, E_va, fit_transform=False)
    path = prior.save(os.path.join(args.out, f"prior_{args.family}"))
    print(f"[prior] saved {path}  val_score={prior.score_val(E_va):.4f}")

    rng = np.random.default_rng(args.seed)
    S = prior.sample(args.n_sample, rng)
    np.save(os.path.join(args.out, f"samples_{args.family}.npy"), S)
    summary = {
        "family": args.family,
        "explained_variance_ratio": {str(k): v for k, v in evr.items()},
        "effective_rank": tf.effective_rank(),
        "reconstruction_val": tf.reconstruction_error(E_va[:2000]),
        "val_score": prior.score_val(E_va),
        "sample_norm_mean": float(np.linalg.norm(S, axis=1).mean()),
    }
    if args.family == "gmm":
        summary["best_k"] = prior.best_k_
        summary["best_cov"] = prior.best_cov_
        summary["selection_table"] = prior.selection_table_
    with open(os.path.join(args.out, f"prior_{args.family}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "selection_table"}, indent=2))


if __name__ == "__main__":
    _main()
