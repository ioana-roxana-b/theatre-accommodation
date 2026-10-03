# scripts/corpus/

| Script | What it does |
| --- | --- |
| `recompute_addressees.py` | Rewrites the `addressees` column of every `data/plays_dialogue/<Play>/quotation_info.csv` with one rule: the other characters speaking within ±10 utterances (`preprocessing/addressees.py`). Keeps a `.pre_addressees.bak` copy. |

```bash
python scripts/corpus/recompute_addressees.py
```

The input files were built from the source editions by OCR and per-edition text cleaning: stage
directions in brackets or parentheses and lines that consist only of stage directions were removed,
and speaker labels were parsed into the PDNC CSV schema (`quoteID, quoteText, speaker, addressees`).
