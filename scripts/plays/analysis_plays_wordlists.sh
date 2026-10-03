#!/usr/bin/env bash
# analysis_plays_wordlists.sh
# Runs word-list convergence analysis for EN and FR plays.

set -euo pipefail
cd "$(dirname "$0")/../.."  # run from repo root

MIN_SUPPORT="${MIN_SUPPORT:-3}"

run_analysis() {
    local base="$1"; shift
    local titles=("$@")

    echo "=== $base ==="
    for TITLE in "${titles[@]}"; do
        local dir="$base/$TITLE"
        echo "  -> $TITLE"
        python main.py \
          --stage analysis \
          --mode wordlists \
          --play_dir "$dir" \
          --out_dir "$dir" \
          --min_support "$MIN_SUPPORT" \
          --title "$TITLE"
    done
    echo
}

# ── English ──────────────────────────────────────────────────────────────────
run_analysis outputs/plays_en/wordlists \
    All_Plays \
    Beckett_WaitingForGodot \
    Beckett_Endgame \
    Becque_TheVultures \
    Becque_TheWomanOfParis \
    Brieux_LaRobeRouge_EN \
    Brieux_WomanOnHerOwn \
    Dumas_Camille_EN \
    Dumas_TheDemiMonde_EN \
    Genet_Balcony_EN \
    Genet_TheMaids_EN \
    Ionesco_BaldSoprano_EN \
    Ionesco_Rhinoceros_EN

# ── French ───────────────────────────────────────────────────────────────────
run_analysis outputs/plays_fr/wordlists \
    All_Plays \
    Beckett_EnAttendantGodot \
    Beckett_FinDePartie \
    Becque_LaParisienne \
    Becque_LesCorbeaux \
    Brieux_LaFemmeSeule \
    Brieux_LaRobeRouge \
    Dumas_DameAuxCamelias \
    Dumas_LeDemiMonde_FR \
    Genet_LeBalcon_FR \
    Genet_LesBonnes_FR \
    Ionesco_LaCantriceChauve_FR \
    Ionesco_Rhinoceros_FR

echo "All word-list analyses complete."
