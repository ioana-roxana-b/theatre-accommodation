#!/usr/bin/env bash
# run_all.sh
# Full wordlists analysis of the 24 plays, in order: pairs + features, per-play convergence,
# reply-level mixed models, network analyses, LaTeX tables, descriptive stats, window check,
# and the linear mixed model on the DNM score.
#
# Usage (from anywhere, in WSL/Linux with the venv active):
#   bash scripts/plays/run_all.sh            # mixed models with nAGQ=0 (fast)
#   FULL=1 bash scripts/plays/run_all.sh     # mixed models with nAGQ=1 (final numbers, slower)
#   SKIP_PIPELINE=1 bash scripts/plays/run_all.sh   # reuse existing outputs/plays_*/wordlists
# Logs go to outputs/logs/.

set -euo pipefail
cd "$(dirname "$0")/../.."  # run from repo root

FULL_FLAG=""
[[ "${FULL:-0}" == "1" ]] && FULL_FLAG="--full"
LOGS=outputs/logs
mkdir -p "$LOGS"

step() {  # step <name> <command...>
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] $name"
    "$@" > "$LOGS/$name.log" 2>&1 || { echo "  FAILED, see $LOGS/$name.log"; exit 1; }
}

if [[ "${SKIP_PIPELINE:-0}" != "1" ]]; then
    step 01_preprocess       bash scripts/plays/preprocess_plays_wordlists.sh
    step 02_analysis         bash scripts/plays/analysis_plays_wordlists.sh
fi
step 03_alignment_table      python scripts/stats/build_alignment_table.py
step 04_glmm                 Rscript scripts/stats/alignment_glmm.R $FULL_FLAG
step 05_conditions           Rscript scripts/stats/alignment_conditions.R $FULL_FLAG
step 06_network_preservation python scripts/stats/network_preservation.py
step 06b_network_preservation_min10 python scripts/stats/network_preservation.py --min_pairs 10
step 07_network_similarity   python scripts/stats/network_similarity.py
step 08_tables_glmm          python scripts/paper/alignment_glmm_table.py
step 09_tables_network       python scripts/paper/network_preservation_table.py
step 10_compute_stats        python scripts/stats/compute_stats.py
step 11_sensitivity_window   python scripts/stats/sensitivity_window.py
step 11b_turn_length        python scripts/stats/turn_length.py
step 11c_turn_length_glmm   Rscript scripts/stats/turn_length_glmm.R $FULL_FLAG
step 11d_example_exchanges  python scripts/stats/example_exchanges.py
step 11e_robustness_checks  python scripts/stats/robustness_checks.py lexicon response addressee french seeds table

# linear mixed model on the DNM score (needs statsmodels: pip install -r requirements.txt)
if python -c "import statsmodels" 2>/dev/null; then
    step 12_mixed_effects_model  python scripts/stats/mixed_effects_model.py
    step 13_tables_mixed         python scripts/paper/mixed_effects_latex_table.py
    step 14_tables_mixed_std     python scripts/paper/mixed_effects_latex_table_enhanced.py
    step 15_convergent_validity  python scripts/stats/convergent_validity_check.py
else
    echo "  skipped linear mixed model: statsmodels not installed (pip install -r requirements.txt)"
fi

echo "[$(date +%H:%M:%S)] done. Descriptive stats: $LOGS/10_compute_stats.log"
