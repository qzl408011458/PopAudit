"""Run train_tabddpm_v2.main(argv) unchanged except that the CPU-only ml_eff multiseed
diagnostic (metrics.json only; never touches the ensemble) is stubbed out, and stdout is
line-buffered.  Usage: tabddpm_gen_wrapper.py <train_tabddpm_v2 argv...>"""
import sys as _sys, pathlib as _pl; _sys.path.append(str(_pl.Path(__file__).resolve().parents[4])); import popaudit_paths as _PA  # release path bootstrap (see popaudit_paths.py)
import sys, builtins, functools
sys.path.insert(0, _PA.R(""))
builtins.print = functools.partial(builtins.print, flush=True)
import utils.ml_efficacy as M
M.ml_efficiency_multiseed = lambda *a, **k: {"ml_eff_skipped": 1.0}
import train_tabddpm_v2 as T
T.main(sys.argv[1:])
