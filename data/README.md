# data/

Most play texts and translations in the corpus are under copyright and are not distributed. One play
pair is included as a public-domain example:

| Folder | Play | Edition |
|--------|------|---------|
| `plays_dialogue/Dumas_DameAuxCamelias/` | Alexandre Dumas *fils*, *La Dame aux Camélias* (1852) | French original |
| `plays_dialogue/Dumas_Camille_EN/` | *Camille*, English translation by Matilda Heron | Cincinnati, 1856 |

Both are OCR'd from scans, so the text contains some OCR errors. The annotations (speakers, inferred
addressees, character gender and narrative prominence) are the ones used in the analysis.

```bash
bash scripts/example/run_example.sh     # preprocessing + analysis for the example pair
```

Expected layout:

```text
data/
  plays_dialogue/<Play>/          # pipeline input, one folder per play
    quotation_info.csv            #   dialogue turns (PDNC annotation schema)
    character_info.csv            #   speaker gender / narrative prominence
    play_text.txt

  # intermediate files from building the corpus (scripts/corpus/):
  plays_pdf/     scanned PDFs           -> convert_pdf.sh      -> ocr_raw/
  ocr_raw/       raw OCR text           -> clean_ocr.py        -> ocr_clean/
  plays_txt/     digital-source text    -> clean_plays.py      -> plays_clean/
  ocr_clean/, plays_clean/              -> extract_dialogues.py -> plays_dialogue/
```

`quotation_info.csv` needs the columns `quoteID, quoteText, speaker, addressees`; `character_info.csv`
needs one row per character with gender and narrative prominence (major / intermediate / minor).
