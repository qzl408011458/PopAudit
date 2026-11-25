"""g1fix (2026-11-24): Layer-3 re-solve (sigma_sens.do_l3, --save_projected_weights kept) for every
Layer-2-admitted case of every listed variant, in ONE shared worker pool (default 12 processes).
Skips cases whose layer3_summary.json already exists (restartable).
usage: l3_all_g1fix.py --variant v1 v2 ... [--jobs 12]"""
import sys, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE))
import decision_certificate_xstate_v3 as DC
from sigma_sens import cases, vdir, do_l3, VARIANTS

ap = argparse.ArgumentParser(); ap.add_argument("--variant", nargs="+", required=True)
ap.add_argument("--jobs", type=int, default=12); a = ap.parse_args()
todo = []
for v in a.variant:
    assert v in VARIANTS, v
    DC.BASE = vdir(v)
    for c in cases():
        if DC.layer2_test(DC.load_layer2(c[1], c[0], c[2]), 0.05)["accept"] and \
           not (vdir(v) / f"layer3_paperW_{c[0]}_{c[1]}_{c[2]}" / "layer3_summary.json").exists():
            todo.append((v, *c))
print(f"[{time.strftime('%T')}] {len(todo)} Layer-3 solves", flush=True)
t0 = time.time(); nfail = 0
with ThreadPoolExecutor(max_workers=a.jobs) as ex:
    futs = {ex.submit(do_l3, *t): t for t in todo}
    for i, fu in enumerate(as_completed(futs), 1):
        try:
            msg = fu.result()
        except Exception as e:
            msg = f"{futs[fu]} EXCEPTION {e!r}"
        if "ok" not in msg: nfail += 1
        print(f"[{time.strftime('%T')}] ({i}/{len(todo)}) {msg}", flush=True)
print(f"[{time.strftime('%T')}] done in {time.time()-t0:.0f}s, failures {nfail}", flush=True)
