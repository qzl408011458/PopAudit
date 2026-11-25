#!/usr/bin/env python
"""LaTeX version of hpo/fidelity_table.py: base-year fidelity (ref SRMSE-1, 3 seeds mean±sd)
for the 9 audited priors + as-shipped / criterion diagnostic rows -> main_experiments/tables/tab_fidelity_v3.tex"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import json, sys
F = _PA.S('/hpo_v3/final')
OUT = _PA.R('/main_experiments/tables/tab_fidelity_v3.tex')
ROWS = [
    (r'SemaPop-VAE+ (M-R arm, audited)', 'semapop_vae_{st}_train', True),
    (r'TTVAE (triangular sampler)',      'ttvae_{st}', True),
    (r'BN (pgmpy)',                       'bn_pgmpy_{st}', True),
    (r'WGAN-GP-ZCR',                      'wgan_zcr_{st}', True),
    (r'CTGAN, tuned',                     'ctgan_{st}', True),
    (r'\quad CTGAN, as shipped',          'ctgan_defaults_{st}', False),
    (r'FlowSyn (official protocol)',      'flowsyn_{st}', True),
    (r'Borysov-style VAE, tuned',         'borysov_{st}', True),
    (r'\quad Borysov-style VAE, as shipped', 'borysov_defaults_{st}', False),
    (r'TabDDPM, val-SRMSE selection',     'tabddpm_{st}_valsrmse', True),
    (r"\quad TabDDPM, authors' ML-efficacy selection", 'tabddpm_{st}', False),
    (r'TVAE, tuned',                      'tvae_ref_{st}', True),
    (r'\quad TVAE, as shipped',           'tvae_ref_defaults_{st}', False),
]
def cell(pat, st, m='srmse_1'):
    try:
        r = json.load(open(f'{F}/{pat.format(st=st)}/aggregate.json'))['aggregate']['reference'][m]
        return f"{r['mean']:.3f} $\\pm$ {r['sd']:.3f}"
    except Exception:
        return '--'
L = [r'\begin{table}[htbp]', r'\centering', r'\small',
     r'\caption{Base-year (2013) fidelity of the audited priors against the 2013 test split, an independent draw from the same weighted PUMS sample as the training split (Section~\ref{app:hpo}): SRMSE on one-way marginals (mean $\pm$ sd over three training seeds). Indented rows are diagnostic: the package/publication configuration of the same model (\emph{as shipped}) or the authors\textquotesingle{} own selection criterion. They are not included among the candidate priors. The oracle floor (a real 2013 sample of the same size) is $\approx 0.022$ ($0.015$ to $0.030$ across draws).}',
     r'\label{tab:fidelity-v3}', r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
     r'Prior & CA & TX & OH \\', r'\midrule']
for name, pat, audited in ROWS:
    L.append(f"{name} & " + " & ".join(cell(pat, st) for st in ('CA', 'TX', 'OH')) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open(OUT, 'w').write('\n'.join(L) + '\n'); print('wrote', OUT); print('\n'.join(L[8:22]))
