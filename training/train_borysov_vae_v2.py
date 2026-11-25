#!/usr/bin/env python
"""Borysov-style population-synthesis VAE on this repo's v2 protocol.

WHAT THIS ROW IS.  Our PyTorch transcription of the model, loss and sampler of
``pop-synth-vae.ipynb`` cells 36-38 (github.com/stasmix/popsynth @089df3e, MIT), the code
released with Borysov, Rich & Pereira, Transp. Res. C 106:73-97 (2019).  It is NOT their
code: their notebook cannot execute (placeholders for every dataset-specific name, four
NameErrors, TF1/Keras which caps at Python 3.7 and CUDA 10.0 with no sm_86 support).  Label
the row "Borysov-style VAE (our implementation)"; never "Borysov et al. (2019)" unqualified.
model/borysov_vae.py records the code-vs-paper discrepancies and the reduction convention.

WHY THIS ROW EXISTS.  The paper's VAE-family baseline was this project's own model with the
conditioning zeroed, labelled "TVAE (Xu et al. 2019)" -- a misattribution, and to the wrong
lineage besides: our numeric objective is weighted MSE + cross-entropy + beta*KL, which is
Borysov's Eq. (1) family, not Xu's (Xu explicitly argues against exactly that construction and
replaces the hand weights with a learned per-column variance).  With that row corrected to a
reference ctgan.TVAE, the table still had no population-synthesis-lineage VAE -- the natural
baseline for this method.

TWO ENCODINGS, AND THE PAPER PREFERS THE SECOND.  ``--encoding mixed`` standardises the eight
numeric columns and one-hots the categoricals; ``--encoding discretised`` bins every numeric
column and one-hots everything, which is what Sec. 5.2 recommends ("it is possible to avoid
this problem by converting numerical variables to categorical using a sufficiently fine
grid") and what wins in Tables 2-3.  Reporting only ``mixed`` would silently pick his loser.

DEPARTURES FROM THE RELEASED CODE, all recorded into metrics.json:
  * splits -- our pre-built 64,562/16,629 national split, not his internal re-split of a 20%
    sample (code train_dev_ratio=0.8; paper Sec. 4.2 says 75/25; ours is 79.5/20.5 -- three
    different numbers, none of them papered over)
  * scaler fit on the train split only
  * F6 parity: this repo clips to the measured training support and THEN rounds; his sampler
    rounds and then clips to author-declared bins.  We keep the repo's contract so the row is
    scored on the same footing as every other row, and record the difference here.
  * no model selection, matching his protocol (no EarlyStopping/ModelCheckpoint in the
    notebook; vae_train returns the final weights).  --select-on is available for diagnosis
    but --load-final is the faithful default.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release bootstrap: repository root first on sys.path (see popaudit_paths.py)

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

REPO_DIR = _PA.R("")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "scripts"))

import utils.eval_v2 as ev                                        # noqa: E402
from model.borysov_vae import (BorysovVAE, borysov_loss,          # noqa: E402
                               borysov_sample)
from train_ctgan_v2 import (_read_split, cat_vocabulary,          # noqa: E402
                            frozen_rng, integer_flags,
                            load_eval_config, numeric_support,
                            postprocess, selection_score,
                            set_torch_seed, val_metrics)

DEFAULT_EVAL_CONFIG = os.path.join(REPO_DIR, "configs", "eval_v3_pums12.yaml")


# --------------------------------------------------------------------------- #
# encoding: his prepare_data, plus the discretised variant Sec. 5.2 recommends
# --------------------------------------------------------------------------- #
class Encoding:
    """One-hot categoricals (+ optionally binned numerics) and a numeric scaler."""

    def __init__(self, df_fit: pd.DataFrame, num_cols: Sequence[str],
                 cat_cols: Sequence[str], mode: str, n_bins: int) -> None:
        self.mode, self.num_cols, self.cat_cols = mode, list(num_cols), list(cat_cols)
        self.bin_edges: Dict[str, np.ndarray] = {}
        self.is_int: Dict[str, bool] = {}
        self.exact_cols: List[str] = []
        if mode == "discretised":
            # "converting numerical variables to categorical using a sufficiently fine grid"
            # (Sec. 5.2).  For a small-integer COUNT variable (household size, children,
            # vehicles) the finest grid is its own values, so those are one-hot on the exact
            # value -- quantile bins would collapse to a handful of edges and the uniform
            # within-bin decode then smears a point mass such as 'no children' across
            # {0, 1} (measured: TV 0.59 on Number_of_children before this rule).  Wide
            # numerics (Age) keep n_bins quantile edges on the FIT split only; ties collapse,
            # which is why the per-column bin count is recorded rather than assumed.
            for c in self.num_cols:
                v = pd.to_numeric(df_fit[c], errors="coerce").to_numpy(float)
                fin = v[np.isfinite(v)]
                self.is_int[c] = bool(np.all(np.isclose(fin, np.round(fin))))
                if self.is_int[c] and len(np.unique(fin)) <= n_bins:
                    self.exact_cols.append(c)          # exact-value categorical
                    continue
                e = np.unique(np.nanquantile(v, np.linspace(0, 1, n_bins + 1)))
                e[0], e[-1] = -np.inf, np.inf
                self.bin_edges[c] = e
            self.enc_cat = self.cat_cols + self.num_cols
            self.enc_num: List[str] = []
        else:
            self.enc_cat, self.enc_num = list(self.cat_cols), list(self.num_cols)

        self.levels: Dict[str, List[str]] = {}
        for c in self.enc_cat:
            self.levels[c] = sorted(self._as_cat(df_fit, c).unique().tolist())
        self.scaler = StandardScaler().fit(
            df_fit[self.enc_num].to_numpy(float)) if self.enc_num else None

    def _as_cat(self, df: pd.DataFrame, c: str) -> pd.Series:
        if c in self.exact_cols:
            return pd.to_numeric(df[c], errors="coerce").round().astype("Int64").astype(str)
        if c in self.bin_edges:
            return pd.Series(np.digitize(pd.to_numeric(df[c], errors="coerce").to_numpy(float),
                                         self.bin_edges[c][1:-1]), index=df.index).astype(str)
        return df[c].astype(str)

    @property
    def cat_dims(self) -> List[int]:
        return [len(self.levels[c]) for c in self.enc_cat]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.enc_num:
            parts.append(self.scaler.transform(df[self.enc_num].to_numpy(float)))
        for c in self.enc_cat:
            lv = self.levels[c]
            idx = pd.Categorical(self._as_cat(df, c), categories=lv).codes
            oh = np.zeros((len(df), len(lv)), np.float32)
            ok = idx >= 0
            oh[np.nonzero(ok)[0], idx[ok]] = 1.0
            parts.append(oh)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def inverse(self, arr: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
        """Back to a labelled frame.  Binned numerics are drawn UNIFORMLY inside their bin --
        the bin is all the model knows, and picking the centre would put a point mass at 25
        values per column and make every dispersion metric meaningless."""
        out: Dict[str, Any] = {}
        off = 0
        if self.enc_num:
            out.update({c: v for c, v in zip(
                self.enc_num, self.scaler.inverse_transform(arr[:, :len(self.enc_num)]).T)})
            off = len(self.enc_num)
        for c in self.enc_cat:
            k = len(self.levels[c])
            code = arr[:, off:off + k].argmax(axis=1)
            off += k
            if c in self.exact_cols:
                out[c] = pd.to_numeric(pd.Series(np.asarray(self.levels[c], dtype=object)[code]),
                                       errors="coerce").to_numpy()
            elif c in self.bin_edges:
                e = self.bin_edges[c].copy()
                lo_all = np.nanmin([x for x in e if np.isfinite(x)])
                hi_all = np.nanmax([x for x in e if np.isfinite(x)])
                e[0], e[-1] = lo_all, hi_all
                lo, hi = e[code], e[np.minimum(code + 1, len(e) - 1)]
                if self.is_int.get(c, False):
                    # integer support: uniform over the integers the bin actually covers
                    # [ceil(lo), hi) -- the last bin is closed at hi_all
                    lo_i = np.ceil(lo); hi_i = np.where(code == len(e) - 2, np.floor(hi) + 1, np.ceil(hi))
                    hi_i = np.maximum(hi_i, lo_i + 1)
                    out[c] = lo_i + np.floor(rng.random(len(code)) * (hi_i - lo_i))
                else:
                    out[c] = lo + rng.random(len(code)) * np.maximum(hi - lo, 0.0)
            else:
                out[c] = np.asarray(self.levels[c], dtype=object)[code]
        return pd.DataFrame(out)[self.num_cols + self.cat_cols]


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    p = build_parser()
    a = p.parse_args(argv)
    t0 = time.time()
    set_torch_seed(a.seed, deterministic=True)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    cfgy = os.path.join(a.data_dir, a.cols_config)
    import yaml
    cc = yaml.safe_load(open(cfgy))
    num_cols = list(cc["num_cols"]); cat_cols = list(cc["cat_cols"])
    tr = _read_split(os.path.join(a.data_dir, a.train_file), num_cols, cat_cols, None,
                     a.subsample, a.data_seed)
    va = _read_split(os.path.join(a.data_dir, a.val_file), num_cols, cat_cols, None,
                     a.subsample_val, a.data_seed)
    print(f"[data] train {len(tr):,}  val {len(va):,}  "
          f"{len(num_cols)} numeric + {len(cat_cols)} categorical", flush=True)

    enc = Encoding(tr, num_cols, cat_cols, a.encoding, a.n_bins)
    X = torch.from_numpy(enc.transform(tr))
    print(f"[enc] {a.encoding}: input dim {X.shape[1]}  "
          f"({len(enc.enc_num)} scaled numeric + {len(enc.cat_dims)} one-hot blocks)", flush=True)

    model = BorysovVAE(len(enc.enc_num), enc.cat_dims,
                       a.hidden_layers_shape, a.latent_dim).to(dev)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"[model] {a.hidden_layers_shape} latent={a.latent_dim}  {n_par:,} params", flush=True)
    # cell 36: RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0).  Keras substitutes
    # K.epsilon()=1e-7 for epsilon=None; torch's default is 1e-8, so it is set explicitly --
    # rho is torch's `alpha`.
    opt = torch.optim.RMSprop(model.parameters(), lr=a.lr, alpha=a.rho, eps=1e-7)

    cfg = load_eval_config(a.eval_config, ev, fit_df=None)
    vocab, support = cat_vocabulary(tr, cat_cols), numeric_support(tr, num_cols)
    ints = integer_flags(tr, num_cols)
    med = {c: float(pd.to_numeric(tr[c], errors="coerce").median()) for c in num_cols}
    splits = [0] + list(np.cumsum([len(enc.enc_num)] + enc.cat_dims))

    def sample_df(n: int, seed: int) -> pd.DataFrame:
        g = torch.Generator(device=dev).manual_seed(int(seed))
        arr = borysov_sample(model, n, dev, generator=g)
        df = enc.inverse(arr, np.random.default_rng(int(seed) + 1))
        df, diag = postprocess(df, num_cols, cat_cols, support, ints, vocab, med)
        sample_df.last_diag = diag
        return df

    hist, best = [], {"score": float("inf"), "epoch": -1}
    ds = torch.utils.data.TensorDataset(X)
    gen_cpu = torch.Generator().manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                     drop_last=False, generator=gen_cpu)
    for ep in range(1, a.epochs + 1):
        model.train(); acc = []
        for (xb,) in dl:
            xb = xb.to(dev, non_blocking=True)
            nb = len(enc.enc_num)
            oh = [xb[:, splits[i + 1]:splits[i + 2]] for i in range(len(enc.cat_dims))]
            num_pred, logits, mu, lv = model(xb)
            loss, parts = borysov_loss(num_pred, logits, xb[:, :nb], oh, mu, lv,
                                       beta=a.beta, cat_loss_weight=a.cat_loss_weight,
                                       half_sse=a.half_sse)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            acc.append(float(loss))
        rec: Dict[str, Any] = {"epoch": ep, "loss": float(np.mean(acc)), **parts}
        if ep % a.eval_every == 0 or ep == a.epochs or ep == 1:
            with frozen_rng():
                m = val_metrics(va, sample_df(len(va), a.sample_seed + ep), cfg, ev,
                                use_weights=not a.no_weights)
            s = selection_score(m, a.f1_weight, a.select_on)
            rec.update({f"val_{k}": v for k, v in m.items()}); rec["val_score"] = s
            if s < best["score"]:
                best = {"score": s, "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                rec["is_best"] = True
            print(f"[ep {ep:04d}] loss={rec['loss']:.3f} val_srmse1={m['srmse_1']:.4f} "
                  f"val_srmse2={m['srmse_2']:.4f} score={s:.4f}"
                  f"{'  <- best' if rec.get('is_best') else ''}", flush=True)
        hist.append(rec)

    loaded = "final"
    if not a.load_final and best["epoch"] > 0:
        model.load_state_dict(best["state"]); loaded = f"best@{best['epoch']}"
    print(f"[select] using the {loaded} weights "
          f"({'faithful: the released code has no model selection' if a.load_final else 'DIAGNOSTIC ONLY'})",
          flush=True)

    save_dir = os.path.join(a.out_root, a.run_name); os.makedirs(save_dir, exist_ok=True)
    n_out = a.n_samples if a.n_samples > 0 else len(va)
    df_out = sample_df(n_out, a.sample_seed)
    csv = os.path.join(save_dir, "generated_samples.csv")
    df_out.to_csv(csv, index=False)
    diag = getattr(sample_df, "last_diag", {})
    print(f"[F6] rows out of numeric support before clipping: "
          f"{diag.get('n_rows_out_of_support_before_clip','?')}/{n_out}", flush=True)
    print(f"[done] samples -> {csv}", flush=True)

    # hpo/objective.py scores a run from top-level final_val_metrics (or
    # tail_val_metrics); score the FINAL sample against the val split exactly as the
    # per-epoch evaluations do, and average the last evaluations like train_ctgan_v2.
    with frozen_rng():
        final_val = val_metrics(va, df_out if n_out == len(va) else sample_df(len(va), a.sample_seed),
                                cfg, ev, use_weights=not a.no_weights)
    # ev.tail_metrics strips the "val_" prefix itself; hand it the raw history.
    tail_val = ev.tail_metrics([r for r in hist if "val_srmse_1" in r], k=3)
    metrics = {
        "final_val_metrics": final_val,
        "tail_val_metrics": tail_val,
        "run": {"name": a.run_name, "tag": a.tag, "finished": datetime.now().isoformat(),
                "wall_clock_s": round(time.time() - t0, 2), "n_params": n_par},
        "model": "borysov_vae_v2",
        "config": vars(a),
        "implementation": {
            "kind": "our PyTorch transcription -- their code was NOT run",
            "source_notebook": "github.com/stasmix/popsynth @089df3e, pop-synth-vae.ipynb cells 36-38 (MIT)",
            "paper": "Borysov, Rich & Pereira, Transp. Res. C 106:73-97 (2019); arXiv:1808.06910",
            "label_for_tables": "Borysov-style VAE (our implementation)",
            "why_not_their_code": ("the released notebook is a placeholder skeleton with four "
                                   "NameErrors and targets TF1/Keras (Python<=3.7, CUDA 10.0, "
                                   "no sm_86); it cannot execute even with data"),
            "code_vs_paper": {
                "numeric_term": ("code K.sum(K.square(.)) has NO 1/2; paper Eq.(2) has 1/2. "
                                 f"half_sse={a.half_sse} (False = follow the code)"),
                "cat_loss_weight": "code-only knob, released grid is the single value [1.0]",
                "kl_sigma": "code settles the paper's ambiguity: z_log_var is log-VARIANCE"},
            "departures": [
                f"splits: ours {len(tr)}/{len(va)} (79.5/20.5); code train_dev_ratio=0.8; paper Sec.4.2 75/25",
                "scaler fit on the train split only",
                "F6: this repo clips to measured training support THEN rounds; his sampler rounds then clips to declared bins",
                f"checkpoint: {loaded} ({'faithful -- no selection in the notebook' if a.load_final else 'val-selected, NOT his protocol'})",
                f"generated {n_out:,} rows to match the reference population",
                f"encoding={a.encoding}" + (f" with {a.n_bins} quantile bins per numeric column"
                                            if a.encoding == "discretised" else ""),
            ],
            "reduction": "per-sample SUM over features/groups/latent dims, then ONE mean over the batch (cell 37)",
        },
        "encoding": {"mode": a.encoding, "input_dim": int(X.shape[1]),
                     "cat_dims": enc.cat_dims, "n_scaled_numeric": len(enc.enc_num),
                     "bins_per_numeric": {c: int(len(e) - 1) for c, e in enc.bin_edges.items()},
                     "exact_value_numeric": list(enc.exact_cols),
                     "decode": "exact-value columns restored exactly; binned INTEGER columns "
                               "uniform over the integers in the bin; binned continuous "
                               "columns uniform within the bin"},
        "selection": {"loaded": loaded, "best_epoch": best["epoch"],
                      "best_score": best["score"] if best["epoch"] > 0 else None,
                      "rule": f"{a.select_on} (diagnostic; the published protocol has none)"},
        "postprocess_diag": diag,
        "history": hist,
    }
    out_json = os.path.join(save_dir, "metrics.json")
    json.dump(ev._jsonable(metrics), open(out_json, "w"), indent=2, default=str)
    print(f"[done] metrics -> {out_json}", flush=True)

    if getattr(a, "ensemble", False):
        from utils.ensemble_v2 import write_ensemble
        n_ens = a.ensemble_n if a.ensemble_n and a.ensemble_n > 0 else None
        if n_ens is None:
            raise SystemExit("--ensemble needs --ensemble-n > 0 (finalize passes |reference| explicitly)")
        paths = write_ensemble(lambda seed, n: sample_df(n, seed), n_samples=n_ens,
                               save_dir=save_dir, k=a.ensemble_k)
        metrics["ensemble"] = {"dir": os.path.dirname(paths[0]) if paths else None,
                               "k_realizations": len(paths), "n_samples": n_ens,
                               "seed_source": "audit_config.ENSEMBLE_SEEDS"}
        json.dump(ev._jsonable(metrics), open(out_json, "w"), indent=2, default=str)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Borysov-style VAE (our implementation)")
    p.add_argument("--data-dir", default=_PA.S("/data/pums_states/CA"))
    p.add_argument("--cols-config", default="cols_config_v3.yaml")
    p.add_argument("--train-file", default="traindata_pers.csv")
    p.add_argument("--val-file", default="valdata_pers.csv")
    p.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    p.add_argument("--out-root", default=_PA.S("/modelSave"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--tag", default="borysov_vae_v2")
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--subsample-val", type=int, default=0)
    # the two encodings of Sec. 4; the paper's Tables 2-3 prefer 'discretised'
    p.add_argument("--encoding", choices=("mixed", "discretised"), default="discretised")
    p.add_argument("--n-bins", type=int, default=25,
                   help="--encoding discretised: quantile bins per numeric column")
    # model -- grids from the paper Sec. 5.1: hidden in {25,50,100},{50-25,100-50},{100-50-25};
    # latent in {5,10,25}; beta in {0.01,0.05,0.1,0.5,1,10,100} (beta=0.5 selected in 5 of 6)
    p.add_argument("--hidden-layers-shape", type=int, nargs="+", default=[100, 50, 25])
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--cat-loss-weight", type=float, default=1.0)
    p.add_argument("--half-sse", action="store_true",
                   help="use the PAPER's 1/2 on the numeric SSE; default follows the released code")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rho", type=float, default=0.9, help="Keras rho == torch alpha")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--select-on", default="val_srmse")
    p.add_argument("--f1-weight", type=float, default=0.0)
    p.add_argument("--no-weights", action="store_true")
    p.add_argument("--load-final", action="store_true",
                   help="faithful: the released notebook has no EarlyStopping/ModelCheckpoint")
    p.add_argument("--n-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=1276)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    from utils.ensemble_v2 import add_ensemble_args
    add_ensemble_args(p)
    return p


if __name__ == "__main__":
    raise SystemExit(main() and 0)
