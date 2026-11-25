# train_ttv ae.py
from __future__ import annotations
import os
import sys
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from model.ttvae import TTVAE, TTVAESchema, mmd_rbf
from utils.evals import evaluate_marg, evaluate_bivar, eval_zerocells, eval_zerocells_travel, eval_travelattrs
warnings.filterwarnings('ignore')

# -----------------------------
# Preprocessing (GMM per numeric + onehot categorical)
# -----------------------------
@dataclass
class NumGMMInfo:
    gmm: GaussianMixture
    # per component mean/std for inverse
    means: np.ndarray  # [M]
    stds: np.ndarray   # [M]
    # to normalize alpha roughly to [-1,1]
    # we store 3*std as scale to clip
    scales: np.ndarray  # [M]


class TTVAEPreprocessor:
    """
    Fit:
      - each numeric col: GMM with <= max_components, choose via BIC
      - each cat col: one-hot encoder
    Transform numeric x -> alpha (1) + beta (M onehot)
    Transform cat -> onehot
    """

    def __init__(self, num_cols: List[str], cat_cols: List[str], max_components: int = 10, random_state: int = 42):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.max_components = max_components
        self.random_state = random_state

        self.num_infos: Dict[str, NumGMMInfo] = {}
        self.ohe: Optional[OneHotEncoder] = None
        self.cat_dims: List[int] = []

    def fit(self, df: pd.DataFrame) -> "TTVAEPreprocessor":
        # numeric GMMs
        for c in self.num_cols:
            x = df[c].to_numpy().reshape(-1, 1).astype(np.float64)
            best_gmm = None
            best_bic = float("inf")
            # components 1..max
            for k in range(1, self.max_components + 1):
                gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=self.random_state)
                gmm.fit(x)
                bic = gmm.bic(x)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            assert best_gmm is not None
            means = best_gmm.means_.reshape(-1)
            # full cov -> take sqrt of diag
            cov = best_gmm.covariances_
            if cov.ndim == 3:
                stds = np.sqrt(cov[:, 0, 0])
            else:
                stds = np.sqrt(cov.reshape(-1))
            stds = np.clip(stds, 1e-6, None)
            scales = 3.0 * stds
            self.num_infos[c] = NumGMMInfo(best_gmm, means=means, stds=stds, scales=scales)

        # categorical OHE
        if len(self.cat_cols) > 0:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.ohe.fit(df[self.cat_cols])
            self.cat_dims = [len(cats) for cats in self.ohe.categories_]
        else:
            self.ohe = None
            self.cat_dims = []

        return self

    def build_schema(self) -> TTVAESchema:
        num_mode_dims = [self.num_infos[c].gmm.n_components for c in self.num_cols]
        return TTVAESchema(num_mode_dims=num_mode_dims, cat_dims=self.cat_dims)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []

        # numeric -> alpha + beta
        for c in self.num_cols:
            info = self.num_infos[c]
            x = df[c].to_numpy().astype(np.float64)
            x2 = x.reshape(-1, 1)
            comp = info.gmm.predict(x2)  # [N]
            M = info.gmm.n_components
            beta = np.eye(M, dtype=np.float32)[comp]  # [N,M]

            # alpha: normalized residual within chosen component, clipped to [-1,1]
            mu = info.means[comp]
            scale = info.scales[comp]
            alpha = (x - mu) / scale
            alpha = np.clip(alpha, -1.0, 1.0).astype(np.float32).reshape(-1, 1)
            parts.append(alpha)
            parts.append(beta.astype(np.float32))

        # cat -> onehot
        if self.ohe is not None and len(self.cat_cols) > 0:
            cat_oh = self.ohe.transform(df[self.cat_cols]).astype(np.float32)
            parts.append(cat_oh)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def inverse_transform(self, x_proc: np.ndarray) -> pd.DataFrame:
        """
        Convert processed vectors back to original-ish space:
          numeric: pick component by argmax(beta), recover x = mu + alpha*scale
          cat: argmax each onehot block -> category
        """
        N = x_proc.shape[0]
        out = {}

        s = 0
        for c in self.num_cols:
            info = self.num_infos[c]
            M = info.gmm.n_components
            alpha = x_proc[:, s:s+1].reshape(-1)
            s += 1
            beta = x_proc[:, s:s+M]
            s += M
            comp = beta.argmax(axis=1)
            mu = info.means[comp]
            scale = info.scales[comp]
            x = mu + alpha * scale
            out[c] = x

        # categorical
        if self.ohe is not None and len(self.cat_cols) > 0:
            cat_vals = []
            for k in self.cat_dims:
                block = x_proc[:, s:s+k]
                s += k
                idx = block.argmax(axis=1)
                cat_vals.append(idx)

            cat_vals = np.stack(cat_vals, axis=1)  # [N, n_cat]
            # map back using categories_
            for j, c in enumerate(self.cat_cols):
                cats = self.ohe.categories_[j]
                out[c] = cats[cat_vals[:, j]]

        return pd.DataFrame(out)


# -----------------------------
# Dataset
# -----------------------------
class NumpyDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


# -----------------------------
# Reconstruction loss
# -----------------------------
def recon_loss_from_logits(
    logits_blocks: List[torch.Tensor],  # length T, each [B,width_t]
    x_proc: torch.Tensor,
    schema: TTVAESchema,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    numeric field i:
      - alpha target: x_proc has 1 dim in [-1,1] (MSE)
      - beta target: onehot (CE with target index)
    cat field:
      - onehot target (CE)
    """
    B = x_proc.size(0)
    slices = schema.field_slices()

    losses = []
    for t, (s, e) in enumerate(slices):
        target_block = x_proc[:, s:e]  # [B,width]

        if t < schema.n_num:
            M = schema.num_mode_dims[t]
            # target alpha
            alpha_t = target_block[:, 0]  # [B]
            beta_t = target_block[:, 1:1+M]  # [B,M]
            beta_idx = beta_t.argmax(dim=1)  # [B]

            pred = logits_blocks[t]  # [B,1+M]
            pred_alpha = torch.tanh(pred[:, 0])  # keep consistent with [-1,1]
            pred_beta_logits = pred[:, 1:1+M]

            losses.append(F.mse_loss(pred_alpha, alpha_t, reduction="none").mean(dim=0))
            losses.append(F.cross_entropy(pred_beta_logits, beta_idx, reduction="none").mean(dim=0))
        else:
            # categorical
            pred_logits = logits_blocks[t]  # [B,K]
            tgt_oh = target_block
            tgt_idx = tgt_oh.argmax(dim=1)
            losses.append(F.cross_entropy(pred_logits, tgt_idx, reduction="none").mean(dim=0))

    loss = torch.stack(losses).sum()
    if reduction == "mean":
        return loss
    elif reduction == "sum":
        return loss * B
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")


# -----------------------------
# Latent interpolation (triangular)
# -----------------------------
@torch.no_grad()
def triangular_interpolation(z: torch.Tensor, n_out: int, k: int = 3) -> torch.Tensor:
    """
    z: [N, D] latent codes from encoder
    produce n_out new codes:
      pick k points, sample Dirichlet weights, convex combo
    """
    N, D = z.shape
    out = []
    for _ in range(n_out):
        idx = torch.randint(0, N, (k,), device=z.device)
        pts = z[idx]  # [k,D]
        w = torch.distributions.Dirichlet(torch.ones(k, device=z.device)).sample()  # [k]
        out.append((w[:, None] * pts).sum(dim=0))
    return torch.stack(out, dim=0)  # [n_out,D]


# -----------------------------
# Train
# -----------------------------
def train_ttvae(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    save_dir: str = "ttvae_runs",
    device: str = "cuda",
    # model hp
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 2,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    z_dim: int = 32,
    # optim hp
    lr: float = 2e-4,
    weight_decay: float = 1e-6,
    batch_size: int = 512,
    epochs: int = 200,
    beta_mmd: float = 1.0,
    # preproc hp
    max_gmm_components: int = 10,
    seed: int = 42,
    PATIENCE: int = 10,
):
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # fit preprocessor
    pre = TTVAEPreprocessor(num_cols, cat_cols, max_components=max_gmm_components, random_state=seed).fit(df_train)
    schema = pre.build_schema()

    x_train = pre.transform(df_train)
    x_val = pre.transform(df_val)

    train_loader = DataLoader(NumpyDataset(x_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(NumpyDataset(x_val), batch_size=batch_size, shuffle=False, drop_last=False)

    model = TTVAE(
        schema=schema,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        z_dim=z_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_path = os.path.join(save_dir, "best.pt")
    patience = 0
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_rec = 0.0
        tr_mmd = 0.0
        n_tr = 0

        pbar = tqdm(train_loader, desc=f"Epoch {ep:03d} [train]", leave=False)
        for xb in pbar:
            xb = xb.to(device)
            out = model(xb)

            # recon
            rec = recon_loss_from_logits(out["logits_blocks"], xb, schema, reduction="mean")

            # MMD: sample z_prior ~ N(0,1)
            z_post = out["z"]
            z_prior = torch.randn_like(z_post)
            mmd = mmd_rbf(z_post, z_prior)

            loss = rec + beta_mmd * mmd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            B = xb.size(0)
            tr_loss += loss.item() * B
            tr_rec += rec.item() * B
            tr_mmd += mmd.item() * B
            n_tr += B

            pbar.set_postfix(
                loss=f"{tr_loss / max(n_tr, 1):.4f}",
                rec=f"{tr_rec / max(n_tr, 1):.4f}",
                mmd=f"{tr_mmd / max(n_tr, 1):.4f}",
            )

        # val
        model.eval()
        va_loss = 0.0
        va_rec = 0.0
        va_mmd = 0.0
        n_va = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {ep:03d} [val ]", leave=False)
            for xb in pbar:
                xb = xb.to(device)
                out = model(xb)
                rec = recon_loss_from_logits(out["logits_blocks"], xb, schema, reduction="mean")
                z_post = out["z"]
                z_prior = torch.randn_like(z_post)
                mmd = mmd_rbf(z_post, z_prior)
                loss = rec + beta_mmd * mmd

                B = xb.size(0)
                va_loss += loss.item() * B
                va_rec += rec.item() * B
                va_mmd += mmd.item() * B
                n_va += B

                pbar.set_postfix(
                    loss=f"{va_loss / max(n_va, 1):.4f}",
                    rec=f"{va_rec / max(n_va, 1):.4f}",
                    mmd=f"{va_mmd / max(n_va, 1):.4f}",
                )

        tr_loss /= max(n_tr, 1)
        tr_rec /= max(n_tr, 1)
        tr_mmd /= max(n_tr, 1)
        va_loss /= max(n_va, 1)
        va_rec /= max(n_va, 1)
        va_mmd /= max(n_va, 1)

        print(
            f"Epoch {ep:03d} | "
            f"train: loss={tr_loss:.4f} rec={tr_rec:.4f} mmd={tr_mmd:.4f} | "
            f"val: loss={va_loss:.4f} rec={va_rec:.4f} mmd={va_mmd:.4f}"
        )

        if va_loss < best_val:
            patience = 0
            best_val = va_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "schema": schema.__dict__,
                    "num_cols": num_cols,
                    "cat_cols": cat_cols,
                    "preprocessor": _serialize_preprocessor(pre),
                    "hp": dict(
                        d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
                        dropout=dropout, z_dim=z_dim, lr=lr, weight_decay=weight_decay, beta_mmd=beta_mmd,
                        batch_size=batch_size, max_gmm_components=max_gmm_components
                    ),
                },
                best_path,
            )
        else:
            patience += 1
        if patience > PATIENCE:
            break

    print(f"Best val={best_val:.4f} saved to {best_path}")
    return model, pre, schema, best_path


def _serialize_preprocessor(pre: TTVAEPreprocessor) -> dict:
    """
    Save enough info to reload inverse transform.
    GaussianMixture itself is not trivially JSON-serializable; we save its sklearn params + learned arrays.
    For production you should joblib.dump(pre), but here we keep it portable.
    """
    d = {
        "num_cols": pre.num_cols,
        "cat_cols": pre.cat_cols,
        "max_components": pre.max_components,
        "random_state": pre.random_state,
        "num_infos": {},
        "ohe_categories": None,
    }
    for c, info in pre.num_infos.items():
        gmm = info.gmm
        d["num_infos"][c] = {
            "n_components": gmm.n_components,
            "weights_": gmm.weights_.tolist(),
            "means_": gmm.means_.reshape(-1).tolist(),
            "covariances_": gmm.covariances_.reshape(gmm.n_components, -1).tolist(),
            "precisions_cholesky_": gmm.precisions_cholesky_.reshape(gmm.n_components, -1).tolist(),
            "means": info.means.tolist(),
            "stds": info.stds.tolist(),
            "scales": info.scales.tolist(),
        }
    if pre.ohe is not None:
        d["ohe_categories"] = [cats.tolist() for cats in pre.ohe.categories_]
    return d


@torch.no_grad()
def sample_ttvae(
    ckpt_path: str,
    n: int,
    device: str = "cuda",
    method: str = "triangular",  # "triangular" or "prior"
    batch_size: int = 2048,
    k_tri: int = 3,
) -> pd.DataFrame:
    """
    Sampling consistent with paper idea:
      - encode a pool of real rows -> (h,z)
      - generate z_hat by interpolation on z
      - decode using corresponding contextual h (same batch) + z_hat
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    schema = TTVAESchema(**ckpt["schema"])
    num_cols = ckpt["num_cols"]
    cat_cols = ckpt["cat_cols"]

    pre = _deserialize_preprocessor(ckpt["preprocessor"])
    hp = ckpt["hp"]

    model = TTVAE(
        schema=schema,
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_layers=hp["num_layers"],
        dim_feedforward=hp["dim_feedforward"],
        dropout=hp["dropout"],
        z_dim=hp["z_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # We need some real processed rows to provide contextual h if method=="triangular".
    # Here we assume you saved/loaded train processed data yourself in your pipeline.
    # Minimal fallback: sample from prior and set h_ctx as zeros (works but deviates from paper).
    # Recommended: pass in a cached array of processed train rows and use that.
    raise_if_no_real_context = (method == "triangular")

    # ---- user hook: load your cached processed training data ----
    train_proc_path = os.path.join(os.path.dirname(ckpt_path), "train_processed.npy")
    if os.path.exists(train_proc_path):
        x_train = np.load(train_proc_path).astype(np.float32)
    else:
        x_train = None

    if raise_if_no_real_context and x_train is None:
        raise FileNotFoundError(
            "triangular sampling needs cached train_processed.npy next to checkpoint. "
            "Save it once after preprocessing: np.save('train_processed.npy', x_train)"
        )

    outs = []
    pbar = tqdm(
        total=n,
        desc=f"Sampling TTVAE ({method})",
        unit="rows",
    )
    remain = n
    while remain > 0:
        b = min(batch_size, remain)

        if method == "prior":
            # no contextual info; use a dummy h_ctx = 0
            z = torch.randn(b, hp["z_dim"], device=device)
            T = schema.n_fields
            h_ctx = torch.zeros(T, b, hp["d_model"], device=device)
        else:
            # sample a batch of real rows -> encode -> get h,z then interpolate z
            idx = np.random.randint(0, x_train.shape[0], size=b)
            xb = torch.from_numpy(x_train[idx]).to(device)
            h_ctx, mu, logvar = model.encode(xb)
            z_post = model.reparameterize(mu, logvar)
            z = triangular_interpolation(z_post, n_out=b, k=k_tri)

        logits_blocks = model.decode(z, h_ctx)  # list length T

        # convert logits_blocks -> processed x
        x_proc = blocks_to_processed(logits_blocks, schema)  # [B,D_proc]
        x_np = x_proc.detach().cpu().numpy()
        df = pre.inverse_transform(x_np)
        outs.append(df)

        remain -= b
        pbar.update(b)
    pbar.close()
    return pd.concat(outs, axis=0, ignore_index=True)


def blocks_to_processed(logits_blocks: List[torch.Tensor], schema: TTVAESchema) -> torch.Tensor:
    """
    Convert per-field logits into a processed vector.
    numeric:
      alpha = tanh(pred_alpha)
      beta  = onehot(argmax(beta_logits))
    cat:
      onehot(argmax(logits))
    """
    B = logits_blocks[0].size(0)
    parts = []
    # numeric fields
    for i in range(schema.n_num):
        M = schema.num_mode_dims[i]
        pred = logits_blocks[i]  # [B,1+M]
        alpha = torch.tanh(pred[:, 0:1])
        beta_idx = pred[:, 1:1+M].argmax(dim=1)
        beta = F.one_hot(beta_idx, num_classes=M).float()
        parts.append(alpha)
        parts.append(beta)
    # cat fields
    for j in range(schema.n_cat):
        t = schema.n_num + j
        k = schema.cat_dims[j]
        logits = logits_blocks[t]
        idx = logits.argmax(dim=1)
        oh = F.one_hot(idx, num_classes=k).float()
        parts.append(oh)

    return torch.cat(parts, dim=1)


def _deserialize_preprocessor(d: dict) -> TTVAEPreprocessor:
    num_cols = d["num_cols"]
    cat_cols = d["cat_cols"]
    pre = TTVAEPreprocessor(num_cols, cat_cols, max_components=d["max_components"], random_state=d["random_state"])

    # rebuild GMMs
    for c, info_d in d["num_infos"].items():
        n_components = info_d["n_components"]
        gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=pre.random_state)
        # set learned params
        gmm.weights_ = np.array(info_d["weights_"], dtype=np.float64)
        gmm.means_ = np.array(info_d["means_"], dtype=np.float64).reshape(n_components, 1)
        cov = np.array(info_d["covariances_"], dtype=np.float64).reshape(n_components, 1, 1)
        gmm.covariances_ = cov
        gmm.precisions_cholesky_ = np.array(info_d["precisions_cholesky_"], dtype=np.float64).reshape(n_components, 1, 1)

        pre.num_infos[c] = NumGMMInfo(
            gmm=gmm,
            means=np.array(info_d["means"], dtype=np.float64),
            stds=np.array(info_d["stds"], dtype=np.float64),
            scales=np.array(info_d["scales"], dtype=np.float64),
        )

    # rebuild OHE categories
    if d["ohe_categories"] is not None and len(cat_cols) > 0:
        pre.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # hack: fit on dummy data matching categories
        cats = [np.array(c, dtype=object) for c in d["ohe_categories"]]
        pre.ohe.categories_ = cats
        pre.cat_dims = [len(c) for c in cats]
    else:
        pre.ohe = None
        pre.cat_dims = []

    return pre


# -----------------------------
# Ensemble sampling helpers
# -----------------------------

def set_sampling_seed(seed: int) -> None:
    """Set Python / NumPy / Torch seeds before each realization."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def postprocess_generated_population(
    df_syn: pd.DataFrame,
    *,
    cat_cols: List[str],
    num_cols: List[str],
    dic_les: Dict[str, LabelEncoder],
    num_bounds: Dict[str, tuple],
) -> pd.DataFrame:
    """Map encoded categories back to labels and clip/round numeric attributes."""
    df_out = df_syn.copy()

    for c in cat_cols:
        df_out[c] = dic_les[c].inverse_transform(df_out[c].astype(int).values)

    for col in num_cols:
        lo, hi = num_bounds[col]
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").clip(lo, hi)

    df_out[num_cols] = df_out[num_cols].round().astype(int)
    return df_out


def sample_ttvae_ensemble(
    *,
    best_path: str,
    n_samples: int,
    device: str,
    cat_cols: List[str],
    num_cols: List[str],
    dic_les: Dict[str, LabelEncoder],
    num_bounds: Dict[str, tuple],
    save_path: str,
    ensemble_seeds: List[int],
    method: str = "triangular",
    batch_size: int = 2048,
) -> List[pd.DataFrame]:
    """
    Generate multiple full-population realizations and save them under
    save_path/generated_ensembles/.
    """
    ensemble_dir = os.path.join(save_path, "generated_ensembles")
    os.makedirs(ensemble_dir, exist_ok=True)

    realizations: List[pd.DataFrame] = []

    for realization_id, seed in enumerate(ensemble_seeds):
        print(f"[Ensemble Sampling] realization_id={realization_id}, seed={seed}")
        set_sampling_seed(seed)

        df_k = sample_ttvae(
            best_path,
            n=n_samples,
            device=device,
            method=method,
            batch_size=batch_size,
        )
        df_k = postprocess_generated_population(
            df_k,
            cat_cols=cat_cols,
            num_cols=num_cols,
            dic_les=dic_les,
            num_bounds=num_bounds,
        )
        # v3 audit-layer contract (adaptation spec, not shipped): each realization
        # CSV holds EXACTLY the modelled columns of cols_config_v3, num then cat
        # order — no realization_id / sampling_seed bookkeeping columns.
        df_k = df_k[num_cols + cat_cols]

        out_path = os.path.join(ensemble_dir, f"final_generated_samples_{realization_id}.csv")
        df_k.to_csv(out_path, index=False)
        print(f"[OK] Saved realization {realization_id} to: {out_path}")

        realizations.append(df_k)

    # combined = pd.concat(realizations, axis=0, ignore_index=True)
    # combined_path = os.path.join(ensemble_dir, "final_generated_samples_all.csv")
    # combined.to_csv(combined_path, index=False)
    # print(f"[OK] Saved combined ensemble to: {combined_path}")

    # Backward-compatible single-population output used by the existing evaluation block.
    first = realizations[0].drop(columns=["realization_id", "sampling_seed"], errors="ignore").copy()
    first.to_csv(os.path.join(save_path, "final_generated_samples.csv"), index=False)

    return realizations


def eval_v2_val_metrics(df_val_raw: pd.DataFrame, df_gen: pd.DataFrame,
                        eval_config_path: str) -> Dict[str, float]:
    """v3 selection metrics (frozen protocol, adaptation spec, not shipped):
    SRMSE-1/2 + per-block exact-tuple precision/recall/F1 of a len(val)-row
    generated sample against the held-out val split, via utils.eval_v2.
    The reference split is never touched here."""
    import utils.eval_v2 as ev
    cfg = ev.EvalConfig.from_yaml(eval_config_path)
    if not cfg.is_fitted:
        cfg.fit(df_val_raw)
    m: Dict[str, float] = {}
    m["srmse_1"] = float(ev.srmse_kway(df_val_raw, df_gen, cfg, k=1)["mean"])
    m["srmse_2"] = float(ev.srmse_kway(df_val_raw, df_gen, cfg, k=2)["mean"])
    pr = ev.block_precision_recall(df_val_raw, df_gen, cfg)
    f1s = []
    for b, v in pr.items():
        if not isinstance(v, dict) or "f1" not in v:
            continue
        if b == "all":
            m["block_f1_all"] = float(v["f1"])
            continue
        m[f"block_f1_{b}"] = float(v["f1"])
        m[f"block_precision_{b}"] = float(v["precision"])
        m[f"block_recall_{b}"] = float(v["recall"])
        f1s.append(float(v["f1"]))
    if f1s:
        m["block_f1"] = float(np.mean(f1s))
    return m


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Third-party TTVAE trainer, O1 adaptation (cols_config_v3, "
                    "dual sampling arms, eval_v2 val metrics).")
    parser.add_argument("state", nargs="?", default="CA", choices=["CA", "OH", "TX"],
                        help="state (positional, backward compatible with the HPO harness)")
    parser.add_argument("--cols-config", default="cols_config_v3.yaml",
                        help="cols config yaml; a bare filename is resolved inside "
                             "data/pums_states/<ST>/ (default: cols_config_v3.yaml, "
                             "12 modelled columns, Region_Category excluded)")
    parser.add_argument("--sample-method", default="triangular",
                        choices=["triangular", "prior"],
                        help="sampling arm for BOTH the final sample and the ensemble: "
                             "triangular = paper's published procedure (encode real rows, "
                             "Dirichlet convex combos of posterior latents, contextual h); "
                             "prior = z~N(0,I) with h_ctx=0 (first-class comparison arm)")
    parser.add_argument("--eval-config", default="configs/eval_v3_pums12.yaml",
                        help="frozen utils.eval_v2 EvalConfig yaml for the val metrics")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="rows of the single final sample (0 = len(testdata))")
    parser.add_argument("--out-root", default="modelSave",
                        help="root dir for the run dir (smoke runs: session scratchpad)")
    # ---- HPO-harness integration (hpo/spaces.py::_ttvae_argv) -------------
    # Hyperparameter precedence: flag > TTVAE_* env var > built-in default.
    # argparse defaults are None so "flag not given" is distinguishable from any
    # real value; the built-in default VALUES below are unchanged.
    hp = parser.add_argument_group(
        "hyperparameters (flag > TTVAE_* env > built-in default)")
    hp.add_argument("--d-model", type=int, default=None)          # TTVAE_D_MODEL, 128
    hp.add_argument("--nhead", type=int, default=None)            # TTVAE_NHEAD, 8
    hp.add_argument("--num-layers", type=int, default=None)       # TTVAE_NUM_LAYERS, 3
    hp.add_argument("--dim-feedforward", type=int, default=None)  # TTVAE_DIM_FEEDFORWARD, 512
    hp.add_argument("--dropout", type=float, default=None)        # TTVAE_DROPOUT, 0.1
    hp.add_argument("--z-dim", type=int, default=None)            # TTVAE_Z_DIM, 128
    hp.add_argument("--lr", type=float, default=None)             # TTVAE_LR, 2e-5
    hp.add_argument("--weight-decay", type=float, default=None)   # TTVAE_WEIGHT_DECAY, 1e-6
    hp.add_argument("--batch-size", type=int, default=None)       # TTVAE_BATCH_SIZE, 512
    hp.add_argument("--epochs", type=int, default=None)           # TTVAE_EPOCHS, 1000
    hp.add_argument("--beta-mmd", type=float, default=None)       # TTVAE_BETA_MMD, 1.0
    hp.add_argument("--patience", type=int, default=None)         # TTVAE_PATIENCE, 10
    hp.add_argument("--max-gmm-components", type=int, default=None)  # TTVAE_MAX_GMM_COMPONENTS, 10
    parser.add_argument("--seed", type=int, default=42,
                        help="training seed, wired through ALL training-time seeding "
                             "(torch/numpy/random in train_ttvae + GMM/OHE "
                             "random_state); default 42 = the historical fixed value")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="seed of the eval_v2 val-metrics sample only (default: "
                             "the historical EVAL_SAMPLE_SEED=4242). Ensemble "
                             "realizations ALWAYS use audit_config.ENSEMBLE_SEEDS.")
    parser.add_argument("--run-name", default=None,
                        help="run dir name under --out-root; the HPO harness reads "
                             "<out-root>/<run-name>/metrics.json. Default: "
                             "ttvae_<ST>_<timestamp> (legacy behaviour)")
    parser.add_argument("--hpo-out", default="",
                        help="LEGACY old-harness quick mode: write the score json here "
                             "and DELETE the run dir (equivalent to TTVAE_HPO_OUT env; "
                             "flag wins). The new harness uses --run-name instead and "
                             "keeps the run dir.")
    parser.add_argument("--subsample", type=int, default=0,
                        help="subsample train to N rows (flag wins over HPO_MAX_TRAIN env)")
    parser.add_argument("--subsample-val", type=int, default=0,
                        help="subsample val to N rows (default: subsample/4, mirroring "
                             "the HPO_MAX_TRAIN env behaviour)")
    parser.add_argument("--device", default=None,
                        help="torch device (flag wins over TTVAE_DEVICE env; default auto)")
    parser.add_argument("--f1-weight", type=float, default=0.0,
                        help="accepted for harness compatibility; recorded in "
                             "metrics.json, wired into nothing (no composite "
                             "objective inside this trainer)")
    parser.add_argument("--select-on", default="val_srmse",
                        help="accepted for harness compatibility; recorded in "
                             "metrics.json, wired into nothing: this trainer has no "
                             "per-epoch eval_v2 checkpoint selection — its original "
                             "val-loss early stopping (patience) is untouched")
    from utils.ensemble_v2 import add_ensemble_args
    add_ensemble_args(parser)  # --ensemble / --ensemble-k / --ensemble-n
    args = parser.parse_args()

    STATE = args.state
    data_name = f'pums_states/{STATE}'

    # ---- column selection (change A): cols_config_v3, read with usecols ----
    cols_cfg_path = args.cols_config
    if not os.path.exists(cols_cfg_path):
        cols_cfg_path = os.path.join("data", data_name, args.cols_config)
    with open(cols_cfg_path, 'r', encoding='utf-8') as f:
        cols = yaml.safe_load(f)
    cat_cols = cols["cat_cols"]
    num_cols = cols["num_cols"]
    model_cols = num_cols + cat_cols  # v3 order: num then cat

    df_train = pd.read_csv(f"data/{data_name}/traindata_pers.csv", usecols=model_cols)[model_cols]
    df_val = pd.read_csv(f"data/{data_name}/valdata_pers.csv", usecols=model_cols)[model_cols]
    # NOTE: testdata_pers.csv (= the reference split) is deliberately NOT read
    # here any more: an HPO trial must never open it (hpo/guard/sitecustomize.py
    # raises on its basename). It is loaded lazily in the --ensemble / --hpo-out
    # branch at the bottom, which trials never enter.

    # HPO subsample: --subsample/--subsample-val flags win over the legacy
    # HPO_MAX_TRAIN env (which subsamples train to N and val to N/4); the
    # random_state=42 draw is the same as the original env logic either way.
    _hpo_max = int(os.environ.get("HPO_MAX_TRAIN", "0"))
    _sub_n = args.subsample if args.subsample and args.subsample > 0 else _hpo_max
    _sub_v = (args.subsample_val if args.subsample_val and args.subsample_val > 0
              else _sub_n // 4)
    if _sub_n > 0 and len(df_train) > _sub_n:
        df_train = df_train.sample(_sub_n, random_state=42).reset_index(drop=True)
    if _sub_v > 0 and len(df_val) > _sub_v:
        df_val = df_val.sample(_sub_v, random_state=42).reset_index(drop=True)

    device = args.device or os.environ.get(
        "TTVAE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or \
        f'ttvae_{STATE}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    save_path = os.path.join(args.out_root, run_name)

    # ---- hyperparameter resolution: flag > TTVAE_* env > built-in default ----
    # (default values = the original hardcoded ones, unchanged)
    def _hpi(flag, name, d): return int(flag) if flag is not None else int(os.environ.get(name, d))
    def _hpf(flag, name, d): return float(flag) if flag is not None else float(os.environ.get(name, d))
    D_MODEL = _hpi(args.d_model, "TTVAE_D_MODEL", 128)
    NHEAD = _hpi(args.nhead, "TTVAE_NHEAD", 8)
    NUM_LAYERS = _hpi(args.num_layers, "TTVAE_NUM_LAYERS", 3)
    DIM_FF = _hpi(args.dim_feedforward, "TTVAE_DIM_FEEDFORWARD", 512)
    DROPOUT = _hpf(args.dropout, "TTVAE_DROPOUT", 0.1)
    Z_DIM = _hpi(args.z_dim, "TTVAE_Z_DIM", 128)
    LR = _hpf(args.lr, "TTVAE_LR", 2e-5)
    WEIGHT_DECAY = _hpf(args.weight_decay, "TTVAE_WEIGHT_DECAY", 1e-6)
    BATCH_SIZE = _hpi(args.batch_size, "TTVAE_BATCH_SIZE", 512)
    EPOCHS = _hpi(args.epochs, "TTVAE_EPOCHS", 1000)
    BETA_MMD = _hpf(args.beta_mmd, "TTVAE_BETA_MMD", 1.0)
    MAX_GMM = _hpi(args.max_gmm_components, "TTVAE_MAX_GMM_COMPONENTS", 10)
    PATIENCE_HP = _hpi(args.patience, "TTVAE_PATIENCE", 10)
    SEED = int(args.seed)
    HPO_OUT = args.hpo_out or os.environ.get("TTVAE_HPO_OUT", "")

    # raw-label copy of the val split for the frozen eval_v2 protocol
    # (df_val itself is label-encoded in place just below)
    df_val_raw = df_val.copy(deep=True)

    # smoke/HPO-subsample robustness: LabelEncoder is fit on train only (original
    # behavior), so drop val rows whose categories the (subsampled) train never
    # shows. At full scale this is a no-op — the original script would have
    # crashed on any case this filter touches.
    _seen_mask = pd.Series(True, index=df_val.index)
    for col in cat_cols:
        _seen_mask &= df_val[col].isin(set(df_train[col].unique()))
    if (~_seen_mask).any():
        print(f"[warn] dropping {int((~_seen_mask).sum())} val rows with "
              f"categories unseen in the train subsample")
        df_val = df_val[_seen_mask].reset_index(drop=True)
        df_val_raw = df_val_raw[_seen_mask.values].reset_index(drop=True)

    dic_les = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        dic_les[col] = le
        df_val[col] = le.transform(df_val[col])

    model, pre, schema, best_path = train_ttvae(
        df_train, df_val, num_cols, cat_cols,
        save_dir=save_path,
        device=device,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF, dropout=DROPOUT, z_dim=Z_DIM,
        lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE, epochs=EPOCHS, beta_mmd=BETA_MMD,
        max_gmm_components=MAX_GMM, seed=SEED, PATIENCE=PATIENCE_HP
    )

    # IMPORTANT: cache processed train rows for triangular sampling
    x_train = pre.transform(df_train)
    np.save(os.path.join(save_path, "train_processed.npy"), x_train)

    df_all = pd.concat([df_train, df_val], ignore_index=True)
    num_bounds = {
        col: (df_all[col].min(), df_all[col].max())
        for col in num_cols
    }

    # ---------- v3 val evaluation (change C): frozen utils.eval_v2 protocol ----------
    # len(val)-row sample vs the held-out val split -> metrics.json; the HPO
    # quick mode below re-uses these numbers (merged into the HPO_OUT json).
    # --sample-seed overrides the EVAL-sample seed only; the audit ensemble
    # always draws its seeds from audit_config.ENSEMBLE_SEEDS. Historical
    # default 4242 kept when the flag is absent. (The old "disjoint from
    # ENSEMBLE_SEEDS" claim was wrong: 4242 = ENSEMBLE_SEEDS[42]; the harness
    # passes --sample-seed 1234, which genuinely is disjoint.)
    EVAL_SAMPLE_SEED = int(args.sample_seed) if args.sample_seed is not None else 4242
    set_sampling_seed(EVAL_SAMPLE_SEED)
    df_val_gen = sample_ttvae(best_path, n=len(df_val_raw), device=device,
                              method=args.sample_method)
    df_val_gen = postprocess_generated_population(
        df_val_gen, cat_cols=cat_cols, num_cols=num_cols,
        dic_les=dic_les, num_bounds=num_bounds,
    )[num_cols + cat_cols]
    val_m = eval_v2_val_metrics(df_val_raw, df_val_gen, args.eval_config)
    metrics_payload = {
        "model": "ttvae",
        "state": STATE,
        "sample_method": args.sample_method,
        "cols_config": cols_cfg_path,
        "eval_config": args.eval_config,
        "eval_sample_seed": EVAL_SAMPLE_SEED,
        "seed": SEED,
        "device": device,
        "run_name": run_name,
        # accepted for harness compatibility, recorded, wired into nothing
        # (see the argparse help for --f1-weight / --select-on)
        "f1_weight": float(args.f1_weight),
        "select_on": args.select_on,
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val_raw)),
        "n_val_generated": int(len(df_val_gen)),
        "val_metrics": val_m,
        # hpo/objective.py contract: tail_val_metrics -> final_val_metrics ->
        # selection.best_metrics. This trainer keeps no per-epoch eval_v2
        # history, so final_val_metrics IS the scored block, with the same key
        # names train_tabddpm_v2.py emits (srmse_1/srmse_2/block_f1*).
        # NO reference_metrics key is ever written here: trials are val-only.
        "final_val_metrics": val_m,
        # flat copies for harness convenience
        "srmse_1": val_m.get("srmse_1"),
        "srmse_2": val_m.get("srmse_2"),
        "block_f1": val_m.get("block_f1"),
        "hp": dict(d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                   dim_feedforward=DIM_FF, dropout=DROPOUT, z_dim=Z_DIM, lr=LR,
                   weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE, epochs=EPOCHS,
                   beta_mmd=BETA_MMD, max_gmm_components=MAX_GMM,
                   patience=PATIENCE_HP, seed=SEED, hpo_max_train=_hpo_max,
                   subsample=_sub_n, subsample_val=_sub_v),
        "faithfulness": (
            "training loop, preprocessing and model code are the untouched "
            "third-party TTVAE trainer; O1 additions only: cols_config_v3 column "
            "selection, --sample-method arm (triangular|prior), ensemble k/n "
            "controls, and this post-hoc eval_v2 val evaluation; P1 harness "
            "additions: argv hyperparameter flags (flag > TTVAE_* env > built-in "
            "default, default values unchanged), --seed/--run-name/--subsample "
            "plumbing, and the audit ensemble is now written ONLY when "
            "--ensemble is passed (HPO trials never ensemble and never open the "
            "reference split)"
        ),
    }
    with open(os.path.join(save_path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[ttvae/{STATE}] val ({args.sample_method}): "
          f"srmse1={val_m.get('srmse_1', float('nan')):.4f} "
          f"srmse2={val_m.get('srmse_2', float('nan')):.4f} "
          f"blockF1={val_m.get('block_f1', float('nan')):.4f} "
          f"-> {save_path}/metrics.json")

    # ---------- HPO-trial early exit (behavioural fix, required) ----------
    # Ensemble generation used to be unconditional ("always on, legacy
    # behaviour").  Under the new harness that would draw K x len(reference)
    # rows per trial and, worse, would open testdata_pers.csv, which the trial
    # guard (hpo/guard/sitecustomize.py) forbids.  Aligned with the other v2
    # trainers (utils/ensemble_v2.add_ensemble_args semantics): the audit
    # ensemble is written ONLY when --ensemble is passed (finalize), and the
    # legacy quick mode (json + rmtree of the run dir) ONLY when
    # --hpo-out/TTVAE_HPO_OUT is explicitly set.  A harness trial (--run-name,
    # neither of the two) stops here with the run dir + metrics.json KEPT and
    # the reference split never opened.
    # Every other v2 trainer writes generated_samples.csv UNCONDITIONALLY, sized
    # from --n-samples and never from the reference split.  TTVAE used to exit
    # before sampling whenever --ensemble was absent, so finalize's non-certificate
    # seeds (142, 242) produced no sample file at all and finalize's
    # reference_metrics_for() skipped them with only a [warn] -- the whole TTVAE
    # row was missing from the fidelity table.  Sampling here needs no reference.
    df_std = None
    if args.n_samples and args.n_samples > 0:
        df_std = sample_ttvae(best_path, n=args.n_samples, device=device,
                              method=args.sample_method)
        for c in cat_cols:
            df_std[c] = dic_les[c].inverse_transform(df_std[c].astype(int).values)
        df_std = df_std[model_cols]
        df_std.to_csv(os.path.join(save_path, "generated_samples.csv"), index=False)
        print(f"[ttvae/{STATE}] generated_samples.csv <- {len(df_std)} rows "
              f"(method={args.sample_method})")

    if not args.ensemble and not HPO_OUT:
        print(f"[ttvae/{STATE}] trial mode (no --ensemble/--hpo-out): "
              f"run dir kept at {save_path}")
        sys.exit(0)

    # reference split: opened ONLY past this point (finalize / legacy quick mode)
    real_test_data = pd.read_csv(f"data/{data_name}/testdata_pers.csv",
                                 usecols=model_cols)[model_cols]

    n_final = args.n_samples if args.n_samples and args.n_samples > 0 else len(real_test_data)
    if df_std is not None and len(df_std) == n_final:
        df_final = df_std.copy()          # reuse the sample already written above
    else:
        df_final = sample_ttvae(best_path, n=n_final, device=device, method=args.sample_method)
        for c in cat_cols:
            df_final[c] = dic_les[c].inverse_transform(df_final[c].astype(int).values)

    # ---------- Ensemble sampling ----------
    # Generate multiple population realizations for Layer-1/Layer-2 ensemble audits.
    # The first realization is also saved as final_generated_samples.csv for backward compatibility.
    from audit_config import ENSEMBLE_SEEDS as ensemble_seeds  # K=50 fixed (audit_config.py)
    ensemble_seeds = list(ensemble_seeds)
    if HPO_OUT:
        ensemble_seeds = ensemble_seeds[:1]  # legacy quick mode: 1 realization only
    elif args.ensemble_k and args.ensemble_k > 0:
        ensemble_seeds = ensemble_seeds[:args.ensemble_k]  # smoke mode (spec: k=2)
    ens_n = args.ensemble_n if args.ensemble_n and args.ensemble_n > 0 else len(real_test_data)
    generated_ensembles = sample_ttvae_ensemble(
        best_path=best_path,
        n_samples=ens_n,
        device=device,
        cat_cols=cat_cols,
        num_cols=num_cols,
        dic_les=dic_les,
        num_bounds=num_bounds,
        save_path=save_path,
        ensemble_seeds=ensemble_seeds,
        method=args.sample_method,
        batch_size=2048,
    )

    # Keep the existing evaluation block unchanged by evaluating realization 0.
    df_final = generated_ensembles[0].drop(columns=["realization_id", "sampling_seed"], errors="ignore").copy()

    eval_SRMSE_marg = evaluate_marg(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)
    eval_SRMSE_bivar = evaluate_bivar(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)
    eval_SRMSE_joint_travel = eval_travelattrs(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)
    eval_zc = eval_zerocells_travel(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)

    if HPO_OUT:
        m, bv = float(eval_SRMSE_marg), float(eval_SRMSE_bivar)
        pr, rc, f1 = float(eval_zc[0]), float(eval_zc[1]), float(eval_zc[2])
        score = f1 / 100.0 - 0.5 * (m + bv)
        # legacy keys first (old harness reads "score"); v3 eval_v2 val metrics
        # merged in with a val_ prefix so the new harness can score this mode too
        # (save_path incl. metrics.json is deleted below; HPO_OUT is what survives)
        hpo_payload = {"SRMSE_M": m, "SRMSE_B": bv, "Precision": pr, "Recall": rc,
                       "F1": f1, "score": score,
                       "sample_method": args.sample_method}
        hpo_payload.update({f"val_{k}": v for k, v in val_m.items()})
        json.dump(hpo_payload, open(HPO_OUT, "w"))
        print(f"[ttvae/{STATE}] HPO quick: F1={f1:.2f} M={m:.4f} B={bv:.4f} score={score:.4f}")
        import shutil as _sh; _sh.rmtree(save_path, ignore_errors=True)
        sys.exit(0)

    eval_zc_sa = eval_zerocells(real_test_data.copy(deep=True), df_final.copy(deep=True), cat_cols, num_cols)

    config = {}
    config['sample_method'] = args.sample_method
    config['cols_config'] = cols_cfg_path
    config['ensemble_seeds'] = ensemble_seeds
    config['n_ensemble_realizations'] = len(ensemble_seeds)
    config['n_ensemble_rows'] = int(ens_n)
    config['eval_SRMSE_marg'] = str(float(eval_SRMSE_marg))
    config['eval_SRMSE_bivar'] = str(float(eval_SRMSE_bivar))
    config['eval_SRMSE_joint(travel)'] = str(float(eval_SRMSE_joint_travel))
    config['eval_precision(travel)'] = str(float(eval_zc[0]))
    config['eval_recall(travel)'] = str(float(eval_zc[1]))
    config['eval_f1(travel)'] = str(float(eval_zc[2]))

    config['eval_precision(spatial_age)'] = str(float(eval_zc_sa[0]))
    config['eval_recall(spatial_age)'] = str(float(eval_zc_sa[1]))
    config['eval_f1(spatial_age)'] = str(float(eval_zc_sa[2]))

    with open(f'{save_path}/model_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)