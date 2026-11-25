#!/usr/bin/env bash
# Regeneration after the group-1 fixes (2026-11-24): FIX 1 JWTRNS crosswalk for ACS>=2019 references,
# FIX 2 income classes in constant 2013 dollars (references 2018/21/24 + history 2009-2012), FIX 3 D on the
# category union (dA_union.py, used by make_figE_postproj / make_transfer_figures_v3).
# Identical step list to regen_after_sigmaH.sh; only the log directory differs (regen_logs_g1fix).
# Prerequisites, run first:
#   run_audit_v3.py --l23-only (and --l23-only --extras)   # Layer 2 + cells7 + Layer 3, Layer 1 reused
# Old outputs are backed up in a backup outside this release.  Order matters:
#   certificate -> stress/counterfactual -> tables -> sensitivity sweep (before figures!) -> figures
#   -> figE_postproj (after figures!) -> kappa recheck -> transfer figures -> ablation_l2 family.
# 2026-11-24 follow-up: 11_sensitivity_sweep moved before 08_figures (step names kept).  make_figures_v3.py figF(b)
# plots sensitivity/sensitivity_kappa2_curve.csv, so it must see the current sweep; the sweep reads only the
# audit dirs and xstate_valid_only.csv (step 01), none of 08/09/10/10b's outputs.
# --- release path setup (popaudit_paths equivalents; override via environment, see README) ---
: "${POPAUDIT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${POPAUDIT_PYTHON:=python}"
# --- end release path setup ---
set -u
PY=${POPAUDIT_PYTHON}
E=${POPAUDIT_ROOT}/main_experiments
V3=$E/audits_v3/PAPER_xstate_v3
L=$E/audits_v3/regen_logs_g1fix; mkdir -p $L
step () { local name=$1; shift; echo "[$(date +%T)] START $name"; ( "$@" ) > $L/$name.log 2>&1; local rc=$?; echo "[$(date +%T)] END   $name rc=$rc"; return 0; }

cd $E
step 01_certificate_primary  $PY decision_certificate_xstate_v3.py
cd $V3
step 01b_rebuild_lever_data $PY rebuild_lever_data.py
step 01c_build_budget_table $PY build_budget_table_v3.py
step 01d_baselines          $PY baselines/baseline_reproduction.py
cd $E
step 02_certificate_extras   $PY decision_certificate_xstate_v3.py --extras
cd $V3
# stale single-cell outputs of the old layout (backed up) are removed so nothing re-reads them
rm -rf $V3/l3_stress/tau_* $V3/l3_stress/l3_stress_curve.csv $V3/l3_counterfactual/w* $V3/l3_counterfactual/ref_w* $V3/l3_counterfactual/l3_counterfactual_curve.csv
step 03_stress_semapop       $PY l3_stress_case.py --model semapop_vae
step 04_stress_ttvae         $PY l3_stress_case.py --model ttvae
step 05_cf_semapop           $PY l3_counterfactual_telework.py --model semapop_vae
step 06_cf_ttvae             $PY l3_counterfactual_telework.py --model ttvae
cd ${POPAUDIT_ROOT}
step 07_paper_tables         $PY $V3/figures/make_paper_tables_v3.py
cd $V3/sensitivity
step 11_sensitivity_sweep    $PY sensitivity_sweep.py   # before 08: figF(b) reads sensitivity_kappa2_curve.csv
cd $V3/figures
step 08_figures              $PY make_figures_v3.py
step 09_figE_postproj        $PY make_figE_postproj.py
cd $V3/sensitivity
step 10_kappa_recheck        $PY kappa_recheck.py
step 10b_kappa_null_sim      $PY kappa_null_sim.py
cd $V3/figures
step 12_transfer_figures     $PY make_transfer_figures_v3.py
cd $V3/ablation_l2
step 13_run_ablation         $PY run_ablation.py
step 14_analyze              $PY analyze.py
step 15_analyze_byhorizon    $PY analyze_byhorizon.py
step 16_run_ensemble_table   $PY run_ensemble_table.py
step 17_semantics_figure     $PY make_L2_semantics_figure.py
step 18_no_L2_tables         $PY make_no_L2_tables.py
step 19_run_projeffect       $PY run_projeffect.py
step 20_commute_figure       $PY make_L2_commute_figure.py
step 21_wfh_age_tables       $PY make_wfh_age_tables.py
step 22_probe_commute        $PY probe_commute_structure.py
echo "[$(date +%T)] REGEN DONE"
