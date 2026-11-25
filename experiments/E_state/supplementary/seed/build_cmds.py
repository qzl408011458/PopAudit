"""Build seed-2/3 ensemble-generation commands from the original finalize argv (log line 1).
Only changes: --out-root -> shadow final/, + --ensemble --ensemble-n N, and
--generate-only for SemaPop-VAE+/TabDDPM (checkpoints copied from hpo_v3/final, no retraining).
TTVAE, WGAN-GP-ZCR, BN are re-run with their exact finalize argv (+ensemble)."""
import sys, os, glob, shlex
HERE = os.path.dirname(os.path.abspath(__file__))
N = {"CA": 330109, "TX": 212756, "OH": 64112}
def build(orig_path, sample_batch=None, k=0):
    argv = shlex.split(open(orig_path).read().strip())
    name = os.path.basename(orig_path)[:-5]          # e.g. tabddpm_OH_valsrmse_s142
    study = name.rsplit("_s", 1)[0]
    st = [s for s in N if f"_{s}" in study][0]
    i = argv.index("--out-root"); argv[i + 1] = f"{HERE}/final/{study}"
    argv += ["--ensemble", "--ensemble-n", str(N[st])]
    if k: argv += ["--ensemble-k", str(k)]
    if study.startswith(("semapop_vae", "tabddpm")):
        argv += ["--generate-only"]
    if study.startswith("tabddpm"):
        argv[1] = f"{HERE}/tabddpm_gen_wrapper.py"
    if sample_batch and study.startswith("tabddpm") and st != "OH":   # OH keeps the default 4096
        argv += ["--sample-batch", str(sample_batch)]
    return name, study, st, argv
if __name__ == "__main__":
    sb = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for p in sorted(glob.glob(f"{HERE}/cmds/*.orig")):
        name, study, st, argv = build(p, sb)
        open(p[:-5] + ".sh", "w").write(" ".join(shlex.quote(a) for a in argv) + "\n")
        print(name)
