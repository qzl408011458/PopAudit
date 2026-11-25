#!/usr/bin/env python
"""SemaPop-VAE+ on the v2 protocol -- the ITSC-published model, this repo's plumbing.

WHAT IS THE PUBLISHED METHOD'S AND WHAT IS OURS, stated so the paper can state it.
The MODEL is model/tvae_llm_p.py, copied verbatim from pop_audit (the ITSC 2026
SemaPop-VAE+): a TVAE MLP with FiLM persona conditioning and a PriorNet conditional
prior p(z|c), trained on ELBO = numeric NLL + categorical BCE + beta*KL(q||p(z|c)),
plus the lambda_m marginal regulariser.  Everything AROUND it is this repo's v2
protocol, shared with the GAN arm by importing train_semapop_gan_v2's own functions
rather than re-implementing them: prepare_data (cols_config tables, persona
embeddings, subspace projection, subsample-with-keep), MarginalLossV2 (in place of
pop_audit's utils.MARG_LOSS -- same construction, one maintained implementation),
draw_cond / load_persona_prior (sampling-time conditions), decode_processed and
val_metrics (the frozen eval config), and selection_score (checkpoint selection on
--select-on, the same quantity Optuna ranks trials by).

ONE DELIBERATE DEPARTURE from the published trainer: sampling-time conditions.  The
ITSC code draws conditions from the TRAINING embeddings.  The argparse default here is
--cond-mode mu (GMM persona prior, the M-U arm).  NOTE: the arm audited in the paper
(SemaPop-VAE+, M-R) passes --cond-mode train, i.e. the published behaviour; M-U is the
diagnostic row of Supplementary Table S12 (see hpo/finalize.py --sampling-arm).
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release bootstrap: repository root first on sys.path (see popaudit_paths.py)

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, _PA.R(""))

from train_semapop_gan_v2 import (                       # noqa: E402
    set_torch_seed, prepare_data, val_metrics, selection_score,
    load_persona_prior, draw_cond, decode_processed, _try_eval_v2,
)
from utils.ensemble_v2 import add_ensemble_args, write_ensemble  # noqa: E402
from utils.marg_loss_v2 import build_marginal_loss       # noqa: E402


def _reference_rows(data_dir: str, probe_col: str) -> int:
    """N for the audit ensemble = len(reference population) when --ensemble-n is 0.
    Only the row count is needed, so a single modelled column is read (the
    reference split's content stays unopened, per the eval protocol)."""
    for name in ("referencedata_pers.csv", "testdata_pers.csv"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            n = int(len(pd.read_csv(path, usecols=[probe_col])))
            print(f"[ensemble] n_samples = {n} (rows of {name})", flush=True)
            return n
    raise SystemExit("[fatal] --ensemble with --ensemble-n 0 requires "
                     "referencedata_pers.csv or testdata_pers.csv in --data-dir "
                     "(or pass --ensemble-n explicitly)")
from model.tvae_llm_p import (                           # noqa: E402
    TabularSchema as VaeSchema, SemaPopTVAE_OneHot,
    semapop_vae_loss_onehot, sample_tvae_processed_cond,
)


def train(args: argparse.Namespace, data, save_dir: str) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    dev = torch.device(args.device)
    schema = data.schema

    marg, spec = build_marginal_loss(
        data.x_train, data.num_cols, data.cat_cols, data.cat_dims,
        data.num_mean, data.num_std, bw_scale=args.bw_scale,
        srmse_mode=args.srmse_mode, seed=args.seed)
    marg = marg.to(dev)
    integer_cols = {
        c: spec[c].kind in ("integer", "integer_tail", "time")
        or bool(np.allclose(data.df_train[c].values, np.rint(data.df_train[c].values)))
        for c in data.num_cols
    }

    vae_schema = VaeSchema(num_dim=len(data.num_cols), cat_dims=list(data.cat_dims))
    model = SemaPopTVAE_OneHot(
        input_dim=data.x_train.shape[1], num_dim=vae_schema.num_dim,
        cat_sizes=list(data.cat_dims),
        persona_dim=int(data.cond_train.shape[1]), cond_dim=args.cond_dim,
        latent_dim=args.latent_dim, enc_hidden=args.enc_hidden,
        dec_hidden=args.dec_hidden, enc_depth=args.enc_depth,
        dec_depth=args.dec_depth, dropout=args.dropout,
        adapter_hidden=args.adapter_hidden, adapter_dropout=args.adapter_dropout,
    ).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] SemaPop-VAE+ latent={args.latent_dim} cond={args.cond_dim} "
          f"enc/dec={args.enc_hidden}x{args.enc_depth} params={n_par}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    prior = None
    if args.cond_mode == "mu":
        prior = load_persona_prior(args.prior_path, data.cond_train, args.seed,
                                   family=args.prior_family)
    ev = _try_eval_v2()
    eval_cfg = None
    if ev is not None and args.eval_config:
        eval_cfg = ev.EvalConfig.from_yaml(args.eval_config)
        eval_cfg.fit(data.df_val)
        print(f"[eval] utils.eval_v2 config '{eval_cfg.name}' "
              f"blocks={list(eval_cfg.blocks)}", flush=True)

    ds = TensorDataset(torch.from_numpy(data.x_train),
                       torch.from_numpy(np.asarray(data.cond_train, np.float32)))
    gen_cpu = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, generator=gen_cpu)
    if len(loader) == 0:
        raise SystemExit(f"[fatal] batch_size={args.batch_size} exceeds the "
                         f"{len(ds):,}-row training split (drop_last leaves no batches)")

    metric_backend = "n/a"

    def evaluate(epoch: int) -> Dict[str, Any]:
        nonlocal metric_backend
        n = min(len(data.df_val), args.eval_n) if args.eval_n > 0 else len(data.df_val)
        cond = draw_cond(args.cond_mode, n, data.cond_train, data.cond_val, prior,
                         args.sample_seed + epoch)
        model.eval()
        xs = sample_tvae_processed_cond(model, vae_schema, cond, device=args.device,
                                        batch_size=args.sample_batch, cat_mode=args.sample_cat_mode)
        df_gen = decode_processed(xs, schema, data.num_cols, data.cat_cols,
                                  data.num_mean, data.num_std, spec, integer_cols,
                                  label_encoders=data.label_encoders)
        m, backend = val_metrics(data.df_val, df_gen, data.num_cols, data.cat_cols,
                                 spec, do_2way=not args.no_2way, ev=ev,
                                 eval_cfg=eval_cfg)
        metric_backend = backend
        return m

    history, best = [], {"score": float("inf"), "epoch": -1, "metrics": {}}
    best_path = os.path.join(save_dir, "vae_best.pt")
    last_path = os.path.join(save_dir, "vae_last.pt")

    def build_ckpt(epoch: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {"model_state": model.state_dict(), "config": vars(args),
                "schema": {"num_dim": vae_schema.num_dim,
                           "cat_dims": list(vae_schema.cat_dims)},
                "epoch": epoch, "val_metrics": metrics,
                "torch_version": torch.__version__}

    t0 = time.time()
    # --generate-only: skip training entirely and go straight to the checkpoint the
    # run already wrote.  Added because national runs died AT GENERATION with the
    # model fully trained -- retraining to recover a sampling bug would be waste.
    for ep in range(1, (0 if args.generate_only else args.epochs) + 1):
        model.train()
        beta_ep = (args.beta * min(1.0, ep / args.beta_warmup_epochs)
                   if args.beta_warmup_epochs > 0 else args.beta)
        el_ls, kl_ls, mg_ls = [], [], []
        for x_np, e_np in loader:
            x = x_np.to(dev, non_blocking=True)
            e = e_np.to(dev, non_blocking=True)
            outs, mu, logvar, mu_p, logvar_p = model(x, e)
            loss, logs = semapop_vae_loss_onehot(
                outs, x, vae_schema.num_dim, vae_schema.cat_dims,
                mu, logvar, mu_p, logvar_p, beta=beta_ep)
            if args.lambda_m > 0.0:
                # Published layout: reconstruction as a generated batch.  MarginalLossV2
                # wants [num_scaled | simplex], so the categorical logits are softmaxed
                # (pop_audit's cat_marginal helper did the same internally).
                num_pred = outs[2]
                cat_probs = [torch.softmax(lg, dim=1) for lg in outs[3:]]
                x_fake = torch.cat([num_pred] + cat_probs, dim=1)
                l_cat, l_cont, _ = marg(x_fake)
                mg = l_cat + l_cont
                loss = loss + args.lambda_m * mg
                mg_ls.append(float(mg.item()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            el_ls.append(float(loss.item()))
            kl_ls.append(float(logs.get("kl", 0.0)) if isinstance(logs, dict) else 0.0)

        rec: Dict[str, Any] = {"epoch": ep, "loss": float(np.mean(el_ls)),
                               "kl": float(np.mean(kl_ls)), "beta": float(beta_ep),
                               "marg": float(np.mean(mg_ls)) if mg_ls else 0.0,
                               "elapsed_s": round(time.time() - t0, 2)}
        if ep % args.eval_every == 0 or ep == args.epochs or ep == 1:
            m = evaluate(ep)
            score = selection_score(m, args.f1_weight, args.select_on)
            rec.update({f"val_{k}": v for k, v in m.items()})
            rec["val_score"] = score
            if score < best["score"]:
                best = {"score": score, "epoch": ep, "metrics": m}
                torch.save(build_ckpt(ep, m), best_path)
                rec["is_best"] = True
            print(f"[ep {ep:04d}] L={rec['loss']:.4f} kl={rec['kl']:.4f} "
                  f"marg={rec['marg']:.4f} | "
                  f"val_srmse1={m.get('srmse_1', float('nan')):.4f} "
                  f"val_srmse2={m.get('srmse_2', float('nan')):.4f} score={score:.4f}"
                  + ("  <- best" if rec.get("is_best") else ""), flush=True)
        else:
            print(f"[ep {ep:04d}] L={rec['loss']:.4f} kl={rec['kl']:.4f} "
                  f"marg={rec['marg']:.4f}", flush=True)
        history.append(rec)
        torch.save(build_ckpt(ep, rec), last_path)

    # ---- load best, final sample, metrics (the HPO contract) ---------------
    ck = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state"])
    if args.generate_only:
        # nothing was trained this process, so recover what the checkpoint knows about
        # itself rather than reporting the sentinel inf @ -1 into metrics.json
        best = {"score": float("nan"), "epoch": ck.get("epoch", -1),
                "metrics": ck.get("val_metrics", {})}
    print(f"[select] loaded vae_best.pt from epoch {ck['epoch']} "
          f"(best score={best['score']:.4f} @ epoch {best['epoch']})", flush=True)

    n_out = args.n_samples or len(data.df_val)
    cond = draw_cond(args.cond_mode, n_out, data.cond_train, data.cond_val, prior,
                     args.sample_seed)
    model.eval()
    # sample_tvae_processed_cond (the ITSC file, kept byte-identical) accepts batch_size and
    # ignores it: it puts the WHOLE conditioning matrix on the card.  At the national tier
    # that is 3,051,604 x 4096 = 50 GB and the run dies at generation with the model already
    # trained.  So the chunking lives here instead of in the ported file.  This DOES advance
    # the latent RNG differently from one single call, so national samples are not bit-
    # comparable with the state-tier ones; every national seed uses this path.
    step = max(int(args.sample_batch), 1) * 16
    xs = np.concatenate([
        sample_tvae_processed_cond(model, vae_schema, cond[i:i + step], device=args.device,
                                   batch_size=args.sample_batch,
                                   cat_mode=args.sample_cat_mode)
        for i in range(0, len(cond), step)], axis=0)
    df_out = decode_processed(xs, schema, data.num_cols, data.cat_cols,
                              data.num_mean, data.num_std, spec, integer_cols,
                              label_encoders=data.label_encoders)
    sample_csv = os.path.join(save_dir, "generated_samples.csv")
    df_out.to_csv(sample_csv, index=False)
    final_m, backend = val_metrics(data.df_val, df_out, data.num_cols, data.cat_cols,
                                   spec, do_2way=not args.no_2way, ev=ev,
                                   eval_cfg=eval_cfg)

    evals = [h for h in history if "val_score" in h]
    tail = evals[-max(1, args.tail_k):]
    if tail:
        keys = [k[4:] for k in tail[-1] if k.startswith("val_") and
                isinstance(tail[-1][k], (int, float))]
        tail_m = {k: float(np.mean([h["val_" + k] for h in tail if "val_" + k in h]))
                  for k in keys}
    else:
        # --generate-only trains nothing, so there is no in-training history to average.
        # The checkpoint's own recorded val metrics are the honest substitute; the paper's
        # numbers come from scripts/report_metrics_mixed.py on the samples either way.
        tail_m = {k: float(v) for k, v in (ck.get("val_metrics") or {}).items()
                  if isinstance(v, (int, float))}

    metrics = {
        "run": {"save_dir": save_dir, "started": datetime.now().isoformat(),
                "device": args.device},
        "impl": {"model": "model/tvae_llm_p.py (SemaPopTVAE_OneHot, ITSC 2026, "
                          "copied verbatim from pop_audit)",
                 "protocol": "train_semapop_gan_v2 pipeline (prepare/decode/eval)"},
        "config": vars(args),
        "prior": {"path": args.prior_path, "family": args.prior_family},
        "eval_config": (eval_cfg.to_dict() if eval_cfg is not None else None),
        "selection": {"rule": f"min selection_score on --select-on {args.select_on}",
                      "metric_backend": metric_backend,
                      "best_epoch": best["epoch"], "best_score": best["score"],
                      "best_metrics": best["metrics"],
                      "loaded_checkpoint": os.path.basename(best_path)},
        "tail_val_metrics": tail_m,
        "tail_k": int(args.tail_k),
        "final_val_metrics": final_m,
        "final_metric_backend": backend,
        "history": history,
        "outputs": {"samples_csv": sample_csv, "best_ckpt": best_path,
                    "last_ckpt": last_path},
    }
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[done] samples -> {sample_csv}")
    print(f"[done] metrics -> {os.path.join(save_dir, 'metrics.json')}")

    # ---- K=50 audit ensemble (utils/ensemble_v2 contract; finalize only) ----
    if getattr(args, "ensemble", False):
        modelled_cols = list(data.num_cols) + list(data.cat_cols)

        def _ensemble_sample(e_seed: int, n: int) -> pd.DataFrame:
            # Each realization must be a deterministic function of its seed alone:
            # reseed python/numpy/torch (+cuda), then follow EXACTLY the final-
            # sample path above -- same cond-mode (for 'train' a fresh uniform
            # resample of the TRAINING embeddings under this seed), same chunked
            # sampler, same decode -- returning the 12 modelled label-space
            # columns in cols_config order (num then cat).
            set_torch_seed(e_seed, deterministic=not args.no_deterministic)
            c = draw_cond(args.cond_mode, n, data.cond_train, data.cond_val,
                          prior, e_seed)
            model.eval()
            xs_e = np.concatenate([
                sample_tvae_processed_cond(model, vae_schema, c[i:i + step],
                                           device=args.device,
                                           batch_size=args.sample_batch,
                                           cat_mode=args.sample_cat_mode)
                for i in range(0, len(c), step)], axis=0)
            df_e = decode_processed(xs_e, schema, data.num_cols, data.cat_cols,
                                    data.num_mean, data.num_std, spec, integer_cols,
                                    label_encoders=data.label_encoders)
            return df_e[modelled_cols]

        n_ens = int(args.ensemble_n) if args.ensemble_n else \
            _reference_rows(args.data_dir, modelled_cols[0])
        write_ensemble(_ensemble_sample, n_samples=n_ens, save_dir=save_dir,
                       k=args.ensemble_k)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SemaPop-VAE+ (ITSC model, v2 protocol)")
    # data / conditioning (names must match prepare_data's expectations)
    p.add_argument("--data-dir", default=_PA.S("/data/pums_states/CA"))
    p.add_argument("--cols-config", default="cols_config.yaml")
    p.add_argument("--train-file", default="traindata_pers.csv")
    p.add_argument("--val-file", default="valdata_pers.csv")
    p.add_argument("--num-cols", nargs="*", default=None)
    p.add_argument("--cat-cols", nargs="*", default=None)
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--emb-file", default="embPers_traindata_modeA.pkl")
    p.add_argument("--emb-val-file", default="embPers_valdata_modeA.pkl")
    p.add_argument("--cond-source", default="persona",
                   choices=["persona", "rawattr_full", "rawattr_demo", "none", "gaussian"])
    p.add_argument("--cond-mode", default="mu", choices=["mr", "mu", "train"])
    p.add_argument("--cond-subspace", default="")
    p.add_argument("--prior-path", default=None)
    p.add_argument("--prior-family", default="gmm", choices=["gmm", "flow"])
    # consumed by prepare_data (shared with the GAN arm); not searched here
    p.add_argument("--demo-cols", nargs="*", default=None)
    p.add_argument("--gauss-dim", type=int, default=64)
    # model (published defaults)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--enc-hidden", type=int, default=512)
    p.add_argument("--dec-hidden", type=int, default=512)
    p.add_argument("--enc-depth", type=int, default=3)
    p.add_argument("--dec-depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--adapter-hidden", type=int, default=1024)
    p.add_argument("--adapter-dropout", type=float, default=0.1)
    # optimisation (published defaults)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta-warmup-epochs", type=int, default=0)
    p.add_argument("--lambda-m", type=float, default=2.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # marginal-loss construction (same knobs/defaults as the GAN arm)
    p.add_argument("--bw-scale", type=float, default=0.35)
    p.add_argument("--srmse-mode", default="eq11")
    # evaluation / protocol
    p.add_argument("--eval-config", default="")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-n", type=int, default=0)
    p.add_argument("--no-2way", action="store_true")
    p.add_argument("--f1-weight", type=float, default=0.0)
    p.add_argument("--select-on", default="val_srmse")
    p.add_argument("--tail-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--sample-batch", type=int, default=4096)
    p.add_argument("--generate-only", action="store_true",
                   help="skip training; load the checkpoint already in the run dir "
                        "and only re-run sampling/metrics")
    # 'sample' (multinomial) by default since 2026-08-29.  The published ITSC
    # configuration decodes categoricals by argmax; a five-seed PAIRED probe on this
    # tier (same checkpoints, same conditions -- results/catmode_probe) measured
    # multinomial better on NIP +0.071 (p=3e-4), NIR +0.077 (p=2e-3), TVD (p=.014)
    # and SRMSE-2 (p=.037) at a 1.1pp record-validity cost.  argmax remains one flag
    # away to reproduce the published behaviour; the paper reports the probe as the
    # justification for the switch.  NOTE: in-training checkpoint selection during the
    # search ran under argmax -- selection/reporting decode modes are disclosed apart.
    p.add_argument("--sample-cat-mode", default="sample", choices=["argmax", "sample"])
    p.add_argument("--n-samples", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-root", default=_PA.S("/modelSave"))
    p.add_argument("--tag", default="semapop_vae_v2")
    p.add_argument("--run-name", default="")
    p.add_argument("--no-deterministic", action="store_true")
    add_ensemble_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    set_torch_seed(args.seed, deterministic=not args.no_deterministic)
    data = prepare_data(args)
    run = args.run_name or f"{args.tag}_{datetime.now():%Y%m%d-%H%M%S}"
    save_dir = os.path.join(args.out_root, run)
    print(f"[run] {save_dir}", flush=True)
    return train(args, data, save_dir)


if __name__ == "__main__":
    main()
