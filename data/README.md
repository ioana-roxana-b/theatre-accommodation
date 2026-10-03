# data/

Nothing in this folder is distributed: the play texts and translations are under copyright.

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
