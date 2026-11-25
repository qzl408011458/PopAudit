#!/usr/bin/env python
"""The BN baseline, run entirely inside pgmpy.

POP_AUDIT PORT (2026-11-23).  Copied from train_bn_pgmpy.py of the SemaPop repository at
commit e2b27f1 (2026-10-08).  The fitting, sampling and decoding code
is unchanged.  Added for pop_audit only: (1) the K=50 audit ensemble writer
(utils.ensemble_v2.write_ensemble; each realization is a pgmpy forward sample at the
ensemble seed, decoded with the same train-fit decoders as generated_samples.csv);
(2) --cols-config, accepted so the finalize argv matches the other trainers (the column
set is the eval config's order, as in the SemaPop repository); (3) pop_audit default paths.
Since 2026-11-23 this is the AUDITED BN of the paper (registry entry bn_pgmpy; it
replaced train_bn_v2.py, which is kept only to document the earlier runs).

WHY THIS REPLACES train_bn_v2.py's numpy backend.  A baseline should be standard library
code that readers can trust, so a hand-written structure learner is the
wrong thing to defend, however correct it is.  The rewrite was planned around pgmpy from the start;
the numpy backend existed on a speed assumption that turns out to be false -- pgmpy's
HillClimbSearch runs a full structure search in seconds.  Everything here is a library call,
so the paper's BN row is one sentence: pgmpy 1.1.2, HillClimbSearch under the score and
maximum in-degree its search selected (BDeu, 4; hpo/spaces.py), maximum-likelihood CPTs,
forward sampling and empirical decoding.

WHAT IS STILL OURS, and has to stay ours:
  discretisation   the BN's state space is the frozen EvalConfig, so the comparison happens
                   in the paper's matching space rather than a binning chosen per model
  numeric decoding a BN samples a BIN; a value has to be put back.  NumericDecoder in
                   empirical mode resamples a training value from that bin, so every emitted
                   number is in the training support.  Invisible to SRMSE and block P/R
                   (both re-discretise) but visible to record validity.
  splits           fit on train, select on val, reference untouched.

The BN is the only model that has to discretise: eight of the attributes are continuous,
and a discrete BN can only reach them through a grid.  It is given the grid the other models
are SELECTED under (configs/eval_national_X_hpo.yaml, quantile bins per continuous column), so
no model is scored on a partition of its own choosing.  numeric_columns_of reads the spec kinds rather
than a hard-coded list, so a config that declares a continuous column gets it decoded --
MODEL_NUMERIC survives only as the fallback for the older configs, which name their numeric
columns nowhere machine-readable.

The decoder: the validation criterion is computed on the BN's own grid, where 'empirical'
and 'midpoint' give the same value ('uniform', whose draws are rounded to integers, can land
in the next bin), so the search cannot choose between the first two.  The reported row
decodes with 'empirical', which resamples a training value from the sampled bin; it re-emits
training values, so the BN's continuous marginals stay close to the training marginals, and
the paper says so.

Against the literature this is still hill climbing rather than the greedy topological
ordering of Heckerman et al. (1995) that Jutras-Dube et al. (2024, s3.1) describe, and
forward rather than the rejection sampling of Sun & Erath (2015).  Both differences are
recorded in metrics.json so the paper can state them instead of glossing them.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[0])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)

import argparse, json, os, sys, time, warnings
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

# REPRODUCIBILITY.  pgmpy's HillClimbSearch.estimate picks the winning move with max() over
# a set of candidates, so ties are broken by set iteration order, which depends on Python's
# per-process hash seed.  Four runs of the identical command gave 74, 75, 77 and 78 edges;
# inside one process it is perfectly deterministic.  PYTHONHASHSEED has to be set before the
# interpreter starts, so the only way to fix it from inside the script is to re-exec.
# Verified: two separate processes then produce byte-identical edge sets.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.eval_v2 as ev                                     # noqa: E402
from model.bn_v2 import build_numeric_decoder                  # noqa: E402
from train_bn_v2 import decode_codes                           # noqa: E402
from utils.ensemble_v2 import add_ensemble_args, write_ensemble  # noqa: E402

import pgmpy                                                   # noqa: E402
from pgmpy.estimators import (HillClimbSearch, BIC, BDeu, BayesianEstimator,
                              MaximumLikelihoodEstimator)  # noqa: E402
from pgmpy.models import DiscreteBayesianNetwork               # noqa: E402
from pgmpy.sampling import BayesianModelSampling               # noqa: E402

#: Fallback for the 20- and 50-attribute configs, which name their numeric columns
#: nowhere machine-readable.  The mixed tier does not use it: numeric_columns_of reads the
#: spec kinds, so adding a continuous attribute to a config is enough to have the BN decode
#: it -- there is no second list to keep in step.
MODEL_NUMERIC = ["Age", "Household_Size"]


def numeric_columns_of(cfg) -> list:
    """Columns the BN must put a NUMBER back into after sampling a bin.

    Any spec that is not `categorical` came from a number, so a code alone is not an
    answer for it.  On the mixed tier that is the eight continuous attributes; on the
    older configs it degrades to MODEL_NUMERIC, which is what those runs used.
    """
    num = [c for c in cfg.order if cfg.specs[c].kind != "categorical"]
    return num or [c for c in MODEL_NUMERIC if c in cfg.order]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",
                   default=_PA.S("/data/pums_states/CA"))
    p.add_argument("--eval-config", default="configs/eval_v3_pums12.yaml")
    p.add_argument("--score", default="bic", choices=["bic", "bdeu"])
    p.add_argument("--max-indegree", type=int, default=3)
    p.add_argument("--ess", type=float, default=10.0, help="BDeu equivalent sample size")
    # Jutras-Dube et al. (2024) s3.1: "their parameters are estimated by maximum likelihood
    # estimation", and their repo (github.com/PascalJD/copulapopgen) leaves pomegranate's
    # pseudocount at 0.  Smoothing is not neutral here: every parent configuration unseen in
    # training gets a full prior row, and a draw landing there is close to a uniform draw
    # over the child -- manufactured impossible records, on exactly the axis (feasibility)
    # this table is about.  MLE is the literature's setting and the default here.
    p.add_argument("--cpt", default="mle", choices=["mle", "bdeu"])
    # A BN samples a BIN; something has to put a value back.  "uniform" draws between the
    # bin's edges, which is what FlowSyn's Discretizer.inverse_transform does and is the
    # assumption-free answer given only the bin.  "empirical" resamples a real training
    # value from that bin -- it makes the continuous marginal almost exact (normalised
    # Wasserstein 0.0024 against uniform's 0.3249 on 12-bin household income) by re-emitting
    # real values, which is record reuse at the value level; it also does not recover the
    # dependence the binning destroyed (mutual information retained is 93% either way).
    # It is the reported setting; the paper states that the BN's continuous values are
    # training values resampled within their bins.
    p.add_argument("--decode", default="uniform",
                   choices=["uniform", "empirical", "midpoint"])
    p.add_argument("--n-samples", type=int, default=0, help="0 = as many rows as val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", default=_PA.S("/modelSave"))
    p.add_argument("--tag", default="bn_pgmpy")
    # HPO plumbing.  This trainer was the fixed-config literature row and needed none of
    # it; the search space in hpo/spaces.py had been written for train_bn_v2.py and was
    # never exercised, because the spec carried no_hpo=True.  --device is accepted and
    # ignored: pgmpy is CPU-only, but hpo/spaces._base_argv passes it to every model.
    p.add_argument("--run-name", default="")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--select-on", default="val_srmse")
    p.add_argument("--f1-weight", type=float, default=0.0)
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--subsample-val", type=int, default=0)
    p.add_argument("--device", default="cpu", help="accepted and ignored (pgmpy is CPU)")
    p.add_argument("--cols-config", default="cols_config_v3.yaml",
                   help="pop_audit: echoed into metrics.json; the modelled columns are the "
                        "eval config's order, exactly as in llm_pop")
    add_ensemble_args(p)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    np.random.seed(a.seed)
    cfg = ev.EvalConfig.from_yaml(a.eval_config)
    COLS = list(cfg.order)

    train = pd.read_csv(f"{a.data_dir}/traindata_pers.csv", usecols=COLS, low_memory=False)[COLS]
    val = pd.read_csv(f"{a.data_dir}/valdata_pers.csv", usecols=COLS, low_memory=False)[COLS]
    enc = ev._as_encoded(train, cfg)
    codes = pd.DataFrame(enc.codes.astype(np.int16), columns=COLS)
    print(f"[data] train={codes.shape} cardinalities {min(enc.sizes)}..{max(enc.sizes)}",
          flush=True)

    # the bin -> value map, fitted on TRAIN only
    decoders: Dict[str, object] = {}
    num_cols = numeric_columns_of(cfg)
    for c in num_cols:
        j = COLS.index(c)
        decoders[c] = build_numeric_decoder(
            c, cfg.specs[c], pd.to_numeric(train[c], errors="coerce").to_numpy(float),
            enc.codes[:, j], a.decode)
    print(f"[decode] {a.decode} on {len(decoders)} numeric columns: "
          f"{', '.join(num_cols)}", flush=True)

    t0 = time.time()
    scorer = BIC(codes) if a.score == "bic" else BDeu(codes, equivalent_sample_size=a.ess)
    dag = HillClimbSearch(codes).estimate(
        scoring_method=scorer, max_indegree=a.max_indegree, max_iter=400, show_progress=False)
    t_struct = time.time() - t0
    edges = [(str(u), str(v)) for u, v in dag.edges()]
    print(f"[struct] {len(edges)} edges in {t_struct:.1f}s", flush=True)

    model = DiscreteBayesianNetwork(edges)
    # isolated nodes are dropped by DiscreteBayesianNetwork(edges); a BN over the full
    # attribute set has to keep them, or the sample comes back missing columns
    model.add_nodes_from([c for c in COLS if c not in model.nodes()])
    t1 = time.time()
    # pgmpy 1.1.2's DiscreteBayesianNetwork.fit takes no prior arguments; the prior lives
    # on the estimator, so the CPTs are built directly and added to the model
    if a.cpt == "mle":
        cpds = MaximumLikelihoodEstimator(model, codes).get_parameters()
    else:
        cpds = BayesianEstimator(model, codes).get_parameters(
            prior_type="BDeu", equivalent_sample_size=a.ess)
    model.add_cpds(*cpds)
    assert model.check_model()
    t_cpt = time.time() - t1
    n_params = sum(int(np.prod(cpd.cardinality)) for cpd in model.get_cpds())
    print(f"[cpt] {n_params:,} parameters in {t_cpt:.1f}s", flush=True)

    n_out = a.n_samples or len(val)
    t2 = time.time()
    sm = BayesianModelSampling(model).forward_sample(size=n_out, seed=a.seed,
                                                     show_progress=False)
    t_samp = time.time() - t2
    gen_codes = sm[COLS].to_numpy(dtype=np.int64)
    gen = decode_codes(gen_codes, cfg, decoders, rng)
    print(f"[sample] {len(gen):,} rows in {t_samp:.1f}s", flush=True)

    # val metrics, so hpo/objective.py can read this trainer.  It scored nothing before:
    # the row was produced once at the literature defaults and handed straight to the
    # reference-scale scorer.
    final_val_metrics = None
    if a.eval_config:
        import utils.eval_v2 as _ev
        _cfg = _ev.EvalConfig.from_yaml(a.eval_config)
        _g = gen if len(gen) == len(val) else gen.iloc[:len(val)]
        final_val_metrics = {
            "srmse_1": float(_ev.srmse_kway(val, _g, _cfg, k=1)["mean"]),
            "srmse_2": float(_ev.srmse_kway(val, _g, _cfg, k=2)["mean"]),
        }
        _pr = _ev.block_precision_recall(val, _g, _cfg)
        _f1 = [v["f1"] for b, v in _pr.items()
               if b != "all" and isinstance(v, dict) and "f1" in v]
        if _f1:
            final_val_metrics["block_f1"] = float(np.mean(_f1))
        print(f"[eval] val srmse_1={final_val_metrics['srmse_1']:.5f} "
              f"srmse_2={final_val_metrics['srmse_2']:.5f}", flush=True)

    run = os.path.join(a.out_root, a.run_name or f"{a.tag}_{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(run, exist_ok=True)

    # ---- pop_audit: K=50 audit ensemble (finalize only; utils/ensemble_v2 contract) ----
    # one pgmpy forward sample per ENSEMBLE_SEEDS seed, decoded with the SAME train-fit
    # decoders as generated_samples.csv; the decode rng is seeded by the same seed, as in
    # train_bn_v2.sample_df, so each realization is a deterministic function of its seed.
    ens_paths = []
    if a.ensemble:
        if a.ensemble_n <= 0:
            raise RuntimeError("--ensemble needs --ensemble-n > 0 (finalize passes |reference|)")
        sampler = BayesianModelSampling(model)

        def _ens_sample(seed: int, n: int) -> pd.DataFrame:
            s = sampler.forward_sample(size=n, seed=seed, show_progress=False)
            return decode_codes(s[COLS].to_numpy(dtype=np.int64), cfg, decoders,
                                np.random.default_rng(seed))

        ens_paths = write_ensemble(_ens_sample, n_samples=a.ensemble_n, save_dir=run,
                                   k=a.ensemble_k)
    gen.to_csv(f"{run}/generated_samples.csv", index=False)
    json.dump({"library": f"pgmpy {pgmpy.__version__}",
               "final_val_metrics": final_val_metrics,
               "tail_val_metrics": final_val_metrics,
               "select_on": a.select_on,
               "search": "HillClimbSearch (add/remove/reverse)",
               "score": a.score, "max_indegree": a.max_indegree,
               "numeric_decode": a.decode,
               "cpt_estimator": ("MaximumLikelihoodEstimator" if a.cpt == "mle"
                                 else f"BayesianEstimator BDeu ess={a.ess}"),
               "sampling": ("forward_sample (ancestral) -- NOT a deviation: pomegranate's "
                            "rejection sampler with evidence=None, which is what the "
                            "reference repo calls, reduces to ancestral sampling"),
               "deviation_from_literature":
                   "Jutras-Dube et al. 2024 s3.1 specify MDL scoring with the greedy "
                   "topological ordering of Heckerman et al. 1995 and rejection sampling "
                   "per Sun & Erath 2015; BIC is MDL up to a constant, the search and the "
                   "sampler are not the same algorithm",
               "numeric_columns": num_cols,
               "n_edges": len(edges), "n_params": n_params, "edges": edges,
               "seconds": {"structure": t_struct, "cpt": t_cpt, "sample": t_samp},
               "eval_config": a.eval_config, "data_dir": a.data_dir, "seed": a.seed,
               "cols_config": a.cols_config,
               "port": "pop_audit copy of llm_pop train_bn_pgmpy.py @ e2b27f1 + K=50 ensemble writer",
               "outputs": {"samples_csv": f"{run}/generated_samples.csv",
                           "ensemble_dir": (os.path.join(run, "generated_ensembles")
                                            if ens_paths else None),
                           "ensemble_k": len(ens_paths)}},
              open(f"{run}/metrics.json", "w"), indent=1)
    print(f"[done] {run}/generated_samples.csv", flush=True)


if __name__ == "__main__":
    main()
