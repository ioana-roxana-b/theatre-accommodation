#!/usr/bin/env bash
# preprocess_plays.sh
# Preprocesses all play corpora (by author × language) from data/plays_dialogue/ into outputs/plays_{en,fr}/<author>_<lang>/<MODE>/.
# Each corpus produces per-play subdirs AND an All_Plays aggregate.
#
# Usage:
#   bash preprocess_plays.sh              # liwc mode (default)
#   MODE=spacy bash preprocess_plays.sh

set -euo pipefail
cd "$(dirname "$0")/../.."  # run from repo root

MODE="${MODE:-liwc}"
LOWERCASE="${LOWERCASE:-1}"
CHAR_INFO="data/plays_dialogue/{play}/character_info.csv"
NON_ADJ_MIN="${NON_ADJ_MIN:-3}"
NON_ADJ_MAX="${NON_ADJ_MAX:-5}"

run_corpus() {
    local lang="$1"; shift
    local corpus="$1"; shift
    local plays=("$@")
    local out_dir="outputs/plays_${lang}/${corpus}/${MODE}"

    echo "=== Preprocessing: ${corpus} (${MODE}, lang=${lang}) ==="
    python main.py \
      --stage preprocess \
      --mode "$MODE" \
      --lang "$lang" \
      --play_dir "${plays[@]}" \
      --out_dir "$out_dir" \
      --character_info "$CHAR_INFO" \
      --non_adj_min_gap "$NON_ADJ_MIN" \
      --non_adj_max_gap "$NON_ADJ_MAX" \
      $( [ "$LOWERCASE" = "1" ] && echo --lowercase )
    echo "=== Done: ${corpus} ==="
    echo
}

# ── Beckett ──────────────────────────────────────────────────────────────────
run_corpus en beckett_en \
    data/plays_dialogue/Beckett_WaitingForGodot \
    data/plays_dialogue/Beckett_Endgame

run_corpus fr beckett_fr \
    data/plays_dialogue/Beckett_EnAttendantGodot \
    data/plays_dialogue/Beckett_FinDePartie

# ── Becque ───────────────────────────────────────────────────────────────────
run_corpus en becque_en \
    data/plays_dialogue/Becque_TheVultures \
    data/plays_dialogue/Becque_TheWomanOfParis

run_corpus fr becque_fr \
    data/plays_dialogue/Becque_LesCorbeaux \
    data/plays_dialogue/Becque_LaParisienne

# ── Brieux ───────────────────────────────────────────────────────────────────
run_corpus en brieux_en \
    data/plays_dialogue/Brieux_LaRobeRouge_EN \
    data/plays_dialogue/Brieux_WomanOnHerOwn

run_corpus fr brieux_fr \
    data/plays_dialogue/Brieux_LaFemmeSeule \
    data/plays_dialogue/Brieux_LaRobeRouge

# ── Dumas ────────────────────────────────────────────────────────────────────
run_corpus en dumas_en \
    data/plays_dialogue/Dumas_Camille_EN \
    data/plays_dialogue/Dumas_TheDemiMonde_EN

run_corpus fr dumas_fr \
    data/plays_dialogue/Dumas_DameAuxCamelias \
    data/plays_dialogue/Dumas_LeDemiMonde_FR

# ── Genet ────────────────────────────────────────────────────────────────────
run_corpus en genet_en \
    data/plays_dialogue/Genet_Balcony_EN \
    data/plays_dialogue/Genet_TheMaids_EN

run_corpus fr genet_fr \
    data/plays_dialogue/Genet_LeBalcon_FR \
    data/plays_dialogue/Genet_LesBonnes_FR

# ── Ionesco ──────────────────────────────────────────────────────────────────
run_corpus en ionesco_en \
    data/plays_dialogue/Ionesco_BaldSoprano_EN \
    data/plays_dialogue/Ionesco_Rhinoceros_EN

run_corpus fr ionesco_fr \
    data/plays_dialogue/Ionesco_LaCantriceChauve_FR \
    data/plays_dialogue/Ionesco_Rhinoceros_FR

echo "All play corpora preprocessed."
