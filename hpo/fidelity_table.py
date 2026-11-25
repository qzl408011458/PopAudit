#!/usr/bin/env python
"""Final base-year fidelity table from hpo_v3/final/*/aggregate.json (3-seed mean±sd
vs the FULL reference population). Rows = model (+ as-shipped / criterion variants),
cols = CA/TX/OH. Run any time; missing cells print as '—'."""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import glob, json, os, sys
F = _PA.S('/hpo_v3/final')
ROWS = [  # (label, dir-prefix)
    ('SemaPop-VAE+ (M-R main arm)', 'semapop_vae_{st}_train'),
    ('TTVAE (triangular)',          'ttvae_{st}'),
    ('BN (pgmpy)',                  'bn_pgmpy_{st}'),
    ('WGAN-GP-ZCR (if1)',           'wgan_zcr_{st}'),
    ('CTGAN — tuned',               'ctgan_{st}'),
    ('CTGAN — as-shipped',          'ctgan_defaults_{st}'),
    ('FlowSyn (official)',          'flowsyn_{st}'),
    ('Borysov VAE — tuned',         'borysov_{st}'),
    ('Borysov VAE — as-shipped',    'borysov_defaults_{st}'),
    ('TabDDPM — val_srmse resel.',  'tabddpm_{st}_valsrmse'),
    ('TabDDPM — ml_eff (authors)',  'tabddpm_{st}'),
    ('TVAE — tuned',                'tvae_ref_{st}'),
    ('TVAE — as-shipped',           'tvae_ref_defaults_{st}'),
]
metric = sys.argv[1] if len(sys.argv) > 1 else 'srmse_1'
print(f"{'model':30s} {'CA':>16s} {'TX':>16s} {'OH':>16s}    [{metric}, 3 seeds vs 2013 test split; oracle floor ≈0.022 (10k real sample, 0.015–0.030)]")
for label, pat in ROWS:
    cells = []
    for st in ('CA', 'TX', 'OH'):
        p = f'{F}/{pat.format(st=st)}/aggregate.json'
        try:
            r = json.load(open(p))['aggregate']['reference'][metric]
            cells.append(f"{r['mean']:.4f}±{r['sd']:.4f}")
        except Exception:
            cells.append('—')
    print(f"{label:30s} " + " ".join(f"{c:>16s}" for c in cells))
