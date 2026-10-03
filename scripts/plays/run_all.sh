#!/usr/bin/env bash
# run_all.sh
# Every analysis in the paper, in order: pairs + features, per-play convergence and networks,
# reply-level mixed models, cross-language network comparisons, descriptive tests, linear mixed
# model, turn length, addressee window, and robustness checks.
#
# Usage (from anywhere, in WSL/Linux with the venv active):
#   bash scripts/plays/run_all.sh            # mixed models with nAGQ=0 (fast)
#   FULL=1 bash scripts/plays/run_all.sh     # mixed models with nAGQ=1 (as in the paper, slower)
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
for m in 10 20 30; do
    step 06_network_preservation_min$m python scripts/stats/network_preservation.py --min_pairs $m
done
step 07_network_similarity   python scripts/stats/network_similarity.py
step 08_compute_stats        python scripts/stats/compute_stats.py
step 09_mixed_effects_model  python scripts/stats/mixed_effects_model.py
step 10_sensitivity_window   python scripts/stats/sensitivity_window.py
step 11_turn_length          python scripts/stats/turn_length.py
step 12_turn_length_glmm     Rscript scripts/stats/turn_length_glmm.R $FULL_FLAG
step 13_robustness_checks    python scripts/stats/robustness_checks.py \
    table seeds response addressee french counts length reciprocity lexicon

echo "[$(date +%H:%M:%S)] done. Results in outputs/; descriptive tests in $LOGS/08_compute_stats.log"
