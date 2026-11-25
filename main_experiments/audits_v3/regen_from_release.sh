#!/usr/bin/env bash
# Re-run, from the files shipped in this repository alone, every step of regen_after_g1fix.sh that does not
# need the on-request inputs (projected weights, r* cells7 inputs, ensembles, processed PUMS), plus the
# supplementary table generators that work from released result CSVs.  Logs go to regen_logs_release/.
# Steps 03-06, 09, 12-16 and 18-22 of regen_after_g1fix.sh are skipped (they need the large inputs).
# make_figures_v3.py (step 08) also writes the PRE-projection tableE_decision_dA.csv and figE_decision.*,
# which step 09 normally overwrites with the post-projection versions; this driver therefore saves the
# shipped post-projection files before step 08 and puts them back afterwards.
#   bash main_experiments/audits_v3/regen_from_release.sh
set -u
AUD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$AUD/../.." && pwd)
E=$ROOT/main_experiments
V3=$AUD/PAPER_xstate_v3
S=$ROOT/supplementary_experiments
PY=${POPAUDIT_PYTHON:-python}
export POPAUDIT_PYTHON=$PY
L=$AUD/regen_logs_release; mkdir -p "$L"
step () { local name=$1; shift; echo "[$(date +%T)] START $name"; ( "$@" ) > "$L/$name.log" 2>&1; local rc=$?; echo "[$(date +%T)] END   $name rc=$rc"; return 0; }

cd "$E";        step 01_certificate_primary  $PY decision_certificate_xstate_v3.py
cd "$V3";       step 01b_rebuild_lever_data  $PY rebuild_lever_data.py
                step 01c_build_budget_table  $PY build_budget_table_v3.py
                step 01d_baselines           $PY baselines/baseline_reproduction.py
cd "$E";        step 02_certificate_extras   $PY decision_certificate_xstate_v3.py --extras
cd "$ROOT";     step 07_paper_tables         $PY "$V3/figures/make_paper_tables_v3.py"
cd "$V3/sensitivity"; step 11_sensitivity_sweep $PY sensitivity_sweep.py
KEEP=$(mktemp -d)
for f in figures/tables/tableE_decision_dA.csv figures/pdf/figE_decision.pdf figures/png/figE_decision.png; do
  mkdir -p "$KEEP/$(dirname $f)"; cp -p "$V3/$f" "$KEEP/$f"; done
cd "$V3/figures"; step 08_figures            $PY make_figures_v3.py
for f in figures/tables/tableE_decision_dA.csv figures/pdf/figE_decision.pdf figures/png/figE_decision.png; do
  cp -p "$KEEP/$f" "$V3/$f"; done; rm -rf "$KEEP"
cd "$V3/sensitivity"; step 10_kappa_recheck  $PY kappa_recheck.py
                step 10b_kappa_null_sim      $PY kappa_null_sim.py
cd "$V3/ablation_l2"; step 17_semantics_figure $PY make_L2_semantics_figure.py
cd "$V3";       step 24_headline_numbers     $PY headline_numbers.py
# supplementary tables from released result CSVs
cd "$S/srmse_threshold";  step S2_threshold_sweep      $PY threshold_sweep.py
cd "$S/tau_ladder";    step S3.6_tau_table          $PY make_table.py
cd "$S/sigma_sensitivity";  step S3.7_sigma_table        $PY make_table.py
cd "$S/seed_robustness";   step S3.8_seed_table         $PY make_seed_table.py
cd "$S/sigma_calibration"; step S4.2_sigma_calibration $PY sigma_lower_block.py
cd "$ROOT";     step S11_fidelity_table      $PY hpo/fidelity_table_tex.py
                step S11_fidelity_console    $PY hpo/fidelity_table.py
echo "[$(date +%T)] RELEASE REGEN DONE (logs in $L)"
