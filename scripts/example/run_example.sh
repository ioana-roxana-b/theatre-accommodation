#!/usr/bin/env bash
# run_example.sh
# Runs the word-list pipeline (the setup used in the paper) on the public-domain example pair:
# La Dame aux Camelias (FR) and Camille (EN). Outputs go to outputs/example/{en,fr}/<Play>/.
set -euo pipefail
cd "$(dirname "$0")/../.."  # run from repo root

CHAR_INFO="data/plays_dialogue/{play}/character_info.csv"
MIN_SUPPORT="${MIN_SUPPORT:-3}"

run() {  # run <lang> <play>
    local lang="$1" play="$2" out="outputs/example/$1/$2"
    echo "=== $play ($lang) ==="
    # with a single play, preprocessing writes straight into --out_dir
    python main.py --stage preprocess --mode wordlists --lang "$lang" \
        --play_dir "data/plays_dialogue/$play" --out_dir "$out" \
        --character_info "$CHAR_INFO" --non_adj_min_gap 3 --non_adj_max_gap 5
    python main.py --stage analysis --mode wordlists \
        --play_dir "$out" --out_dir "$out" --min_support "$MIN_SUPPORT" --title "$play"
}

run en Dumas_Camille_EN
run fr Dumas_DameAuxCamelias

echo
echo "Convergence scores: outputs/example/{en,fr}/<Play>/analysis_results/convergence_results/"
echo "Networks:           outputs/example/{en,fr}/<Play>/analysis_results/graph_results/"
