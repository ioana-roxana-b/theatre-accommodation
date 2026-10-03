# preprocessing/

Stage 1 of the pipeline: turns a play's `quotation_info.csv` into A→B dialogue pairs with
function word features. Called through `main.py --stage preprocess`.

```text
quotation_info.csv
  → preprocess_data.py      clean quote text            → cleaned_quotes.csv
  → build_pairs.py          merge same-speaker turns    → merged_turns.csv
                            build A→B pairs             → pairs_A_to_B.csv
                              non-adjacent (gap 3–5)    → pairs_A_to_B_nonadjacent.csv
                              randomized (seed 123)     → pairs_A_to_B_randomized.csv
                            + gender / prominence from character_info.csv
  → features_wordlists.py   9 function word families    → pairs_features*.csv
```

| File | What it does |
| --- | --- |
| `preprocessing_pipeline.py` | Runs the steps above for one play, or for several plus an `All_Plays/` merge. |
| `preprocess_data.py` | Cleans quote text (whitespace, quotes, optional lowercasing). |
| `build_pairs.py` | Merges consecutive turns by one speaker; builds adjacent, non-adjacent and randomized pairs; attaches speaker metadata. |
| `features_wordlists.py` | Marks which of the 9 families each turn contains. Apostrophes are token boundaries, so French elided forms (*l'*, *d'*, *qu'*, …) are separate tokens. |
| `function_word_lists.py` | The English and French word lists for the 9 families. |
| `addressees.py` | Candidate addressees: the other characters speaking within ±10 utterances (used by `scripts/corpus/recompute_addressees.py`). |

The randomized pairs written here are a single shuffle; the paper averages over 1,000 shuffles
(`scripts/stats/robustness_checks.py table`).
