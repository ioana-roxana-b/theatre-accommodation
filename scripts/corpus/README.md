# scripts/corpus/

How `data/plays_dialogue/` was built from the source editions. You only need this to rebuild
the corpus from scratch; the analysis starts from `data/plays_dialogue/`.
Run everything from the repo root.

```text
data/plays_pdf/  (scans) ──convert_pdf.sh──▶ data/ocr_raw/ ──clean_ocr.py──▶ data/ocr_clean/ ─┐
data/plays_txt/  (digital text) ──────────clean_plays.py──▶ data/plays_clean/ ─────────────────┤
                                                                                                ▼
                                    extract_dialogues.py ──▶ data/plays_dialogue/<Play>/
                                    fix_character_genders.py (patches character_info.csv)
                                    preprocessing/fix_*.py   (per-play annotation fixes)
                                    recompute_addressees.py  (one ±10 addressee rule for all plays)
```

| Script | What it does |
|--------|--------------|
| `convert_pdf.sh` | PDF → PNG (pdftoppm, 300 dpi) → Tesseract OCR, one `.txt` per PDF. Needs `sudo apt install poppler-utils tesseract-ocr tesseract-ocr-fra`. Uses French OCR only when the filename matches `Femme`/`Français`/`Brieux`; edit the regex for other French scans. |
| `clean_ocr.py` | Strips front matter, page headers and OCR noise from each OCR'd play. The cleaning rules are hardcoded per file. |
| `clean_plays.py` | Same for the plays that came as digital text. |
| `extract_dialogues.py` | Parses speaker labels and speeches with a parser per play (see the `PLAYS` registry at the bottom). Writes `quotation_info.csv`, `character_info.csv` and `play_text.txt` in the PDNC schema. Addressees are inferred from scene co-presence (±10 utterances). Importance category comes from the number of speeches. |
| `fix_character_genders.py` | Sets `Gender` in every `character_info.csv` from a name → gender lookup. |
| `recompute_addressees.py` | Rewrites the `addressees` column of every play with one rule: the other characters speaking within ±10 utterances (`preprocessing/addressees.py`). Needed because the annotation fixes renamed speakers without updating addressee lists. Keeps `quotation_info.csv.pre_addressees.bak`. |
| `txt_plays_to_csv.py` | An earlier, generic OCR-text → PDNC-CSV converter (wrote `data/plays_csv/`). Superseded by `extract_dialogues.py`; kept for reference. |

```bash
bash scripts/corpus/convert_pdf.sh
python scripts/corpus/clean_ocr.py
python scripts/corpus/clean_plays.py
python scripts/corpus/extract_dialogues.py
python scripts/corpus/fix_character_genders.py
# then the annotation fixes, see preprocessing/README.md
python scripts/corpus/recompute_addressees.py
```

`extract_dialogues.py` overwrites `data/plays_dialogue/`, so the annotation fixes and the
addressee recompute must be re-applied after it.
