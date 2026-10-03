#!/usr/bin/env bash
# preprocess_plays_spacy.sh
# Runs spaCy preprocessing for EN (en_core_web_sm) and FR (fr_core_news_sm) plays.
# Outputs go to outputs/plays_en/spacy/ and outputs/plays_fr/spacy/.

set -euo pipefail
cd "$(dirname "$0")/../.."  # run from repo root

CHAR_INFO="data/plays_dialogue/{play}/character_info.csv"
NON_ADJ_MIN="${NON_ADJ_MIN:-3}"
NON_ADJ_MAX="${NON_ADJ_MAX:-5}"

echo "=== English plays (spaCy, lang=en) ==="
python main.py \
  --stage preprocess \
  --mode spacy \
  --lang en \
  --play_dir \
    data/plays_dialogue/Beckett_WaitingForGodot \
    data/plays_dialogue/Beckett_Endgame \
    data/plays_dialogue/Becque_TheVultures \
    data/plays_dialogue/Becque_TheWomanOfParis \
    data/plays_dialogue/Brieux_LaRobeRouge_EN \
    data/plays_dialogue/Brieux_WomanOnHerOwn \
    data/plays_dialogue/Dumas_Camille_EN \
    data/plays_dialogue/Dumas_TheDemiMonde_EN \
    data/plays_dialogue/Genet_Balcony_EN \
    data/plays_dialogue/Genet_TheMaids_EN \
    data/plays_dialogue/Ionesco_BaldSoprano_EN \
    data/plays_dialogue/Ionesco_Rhinoceros_EN \
  --out_dir outputs/plays_en/spacy \
  --character_info "$CHAR_INFO" \
  --non_adj_min_gap "$NON_ADJ_MIN" \
  --non_adj_max_gap "$NON_ADJ_MAX"
echo

echo "=== French plays (spaCy, lang=fr) ==="
python main.py \
  --stage preprocess \
  --mode spacy \
  --lang fr \
  --play_dir \
    data/plays_dialogue/Beckett_EnAttendantGodot \
    data/plays_dialogue/Beckett_FinDePartie \
    data/plays_dialogue/Becque_LesCorbeaux \
    data/plays_dialogue/Becque_LaParisienne \
    data/plays_dialogue/Brieux_LaFemmeSeule \
    data/plays_dialogue/Brieux_LaRobeRouge \
    data/plays_dialogue/Dumas_DameAuxCamelias \
    data/plays_dialogue/Dumas_LeDemiMonde_FR \
    data/plays_dialogue/Genet_LeBalcon_FR \
    data/plays_dialogue/Genet_LesBonnes_FR \
    data/plays_dialogue/Ionesco_LaCantriceChauve_FR \
    data/plays_dialogue/Ionesco_Rhinoceros_FR \
  --out_dir outputs/plays_fr/spacy \
  --character_info "$CHAR_INFO" \
  --non_adj_min_gap "$NON_ADJ_MIN" \
  --non_adj_max_gap "$NON_ADJ_MAX"
echo

echo "spaCy preprocessing complete."
