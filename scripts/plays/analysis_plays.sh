#!/usr/bin/env bash
# analysis_plays.sh
# Runs analysis for all preprocessed play corpora.
# Run preprocess_plays.sh (and export LIWC CSVs if MODE=liwc) before this.
#
# Usage:
#   bash analysis_plays.sh              # liwc mode (default)
#   MODE=spacy bash analysis_plays.sh

set -euo pipefail
cd "$(dirname "$0")/../.."  # run from repo root

MODE="${MODE:-liwc}"
MIN_SUPPORT="${MIN_SUPPORT:-3}"

run_analysis() {
    local corpus="$1"; shift
    local titles=("$@")

    echo "=== Analysis: ${corpus} (${MODE}) ==="
    for TITLE in "${titles[@]}"; do
        local prep_dir="outputs/plays_${corpus##*_}/${corpus}/${MODE}/${TITLE}"

        echo "  -> $TITLE"
        COMMON_ARGS=(
            --stage analysis
            --mode "$MODE"
            --play_dir "$prep_dir"
            --out_dir "$prep_dir"
            --min_support "$MIN_SUPPORT"
            --title "$TITLE"
        )

        if [[ "$MODE" == "liwc" ]]; then
            python main.py "${COMMON_ARGS[@]}" \
              --liwc_results          "$prep_dir/LIWC-22 Results - liwc_input - LIWC Analysis.csv" \
              --liwc_results_nonadj   "$prep_dir/LIWC-22 Results - liwc_input_nonadjacent - LIWC Analysis.csv" \
              --liwc_results_rand     "$prep_dir/LIWC-22 Results - liwc_input_randomized - LIWC Analysis.csv"
        else
            python main.py "${COMMON_ARGS[@]}"
        fi
    done
    echo "=== Done: ${corpus} ==="
    echo
}

# ── Beckett ──────────────────────────────────────────────────────────────────
run_analysis beckett_en \
    All_Plays Beckett_WaitingForGodot Beckett_Endgame

run_analysis beckett_fr \
    All_Plays Beckett_EnAttendantGodot Beckett_FinDePartie

# ── Becque ───────────────────────────────────────────────────────────────────
run_analysis becque_en \
    All_Plays Becque_TheVultures Becque_TheWomanOfParis

run_analysis becque_fr \
    All_Plays Becque_LesCorbeaux Becque_LaParisienne

# ── Brieux ───────────────────────────────────────────────────────────────────
run_analysis brieux_en \
    All_Plays Brieux_LaRobeRouge_EN Brieux_WomanOnHerOwn

run_analysis brieux_fr \
    All_Plays Brieux_LaFemmeSeule Brieux_LaRobeRouge

# ── Dumas ────────────────────────────────────────────────────────────────────
run_analysis dumas_en \
    All_Plays Dumas_Camille_EN Dumas_TheDemiMonde_EN

run_analysis dumas_fr \
    All_Plays Dumas_DameAuxCamelias Dumas_LeDemiMonde_FR

# ── Genet ────────────────────────────────────────────────────────────────────
run_analysis genet_en \
    All_Plays Genet_Balcony_EN Genet_TheMaids_EN

run_analysis genet_fr \
    All_Plays Genet_LeBalcon_FR Genet_LesBonnes_FR

# ── Ionesco ──────────────────────────────────────────────────────────────────
run_analysis ionesco_en \
    All_Plays Ionesco_BaldSoprano_EN Ionesco_Rhinoceros_EN

run_analysis ionesco_fr \
    All_Plays Ionesco_LaCantriceChauve_FR Ionesco_Rhinoceros_FR

echo "All play analyses complete."
