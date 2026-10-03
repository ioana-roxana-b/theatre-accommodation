# preprocessing/

Stage 1 of the pipeline: turns a play's `quotation_info.csv` into A→B dialogue
pairs with function-word features. Called through `main.py --stage preprocess`; you don't run
these modules directly (except the `fix_*.py` scripts).

## Flow

```text
quotation_info.csv
  → preprocess_data.py      clean quote text            → cleaned_quotes.csv
  → build_pairs.py          merge same-speaker turns    → merged_turns.csv
                            build A→B pairs             → pairs_A_to_B.csv
                              non-adjacent (gap 3–5)    → pairs_A_to_B_nonadjacent.csv
                              randomized (seed 123)     → pairs_A_to_B_randomized.csv
                            + gender/importance from character_info.csv
  → features (depends on --mode)                        → pairs_features*.csv  or  liwc_input*.csv
```

## Files

| File | What it does |
|------|--------------|
| `preprocessing_pipeline.py` | Orchestrates the steps above for one play (`run_one_play_prep`) or several plus an `All_Plays/` merge (`run_corpus_prep`). |
| `preprocess_data.py` | Cleans quote text (whitespace, quotes, optional lowercasing). |
| `build_pairs.py` | Merges consecutive turns by one speaker; builds adjacent / non-adjacent / randomized pairs; attaches speaker gender and category. |
| `features_wordlists.py` | **`--mode wordlists` (used in the paper).** Counts the 9 LIWC-aligned families using the lists in `function_word_lists.py`. |
| `function_word_lists.py` | The EN and FR word lists for the 9 families. Edit here to change the features. |
| `features.py` | `--mode spacy`: same 9 families via spaCy POS tags. |
| `features_stopwords.py` | `--mode stopwords`: every stopword (stopwords-iso) is its own family. |
| `prepare_liwc.py` | `--mode liwc`: writes `liwc_input*.csv` to score in LIWC-22. |

## Annotation fix scripts

One-off corrections for errors in specific plays' `quotation_info.csv` / `character_info.csv`.
They have **already been applied** to `data/plays_dialogue/` (the originals are kept as `.bak`).
Only re-run them if you rebuild the corpus with `scripts/corpus/extract_dialogues.py`.

| Script | Plays | Fixes |
|--------|-------|-------|
| `fix_speaker_conflation.py` | Ionesco_BaldSoprano_EN, Genet_Balcony_EN | characters merged as aliases (e.g. MR./MRS. SMITH) |
| `fix_maids_en.py` | Genet_TheMaids_EN | rebuilds the file; OCR-garbled SOLANGE labels had dropped turns |
| `fix_linewrapped.py` | Becque_LesCorbeaux, Dumas_Camille_EN | speeches truncated at the first line break |
| `fix_realists.py` | Brieux_LaRobeRouge_EN, Brieux_WomanOnHerOwn | duplicate / wrongly merged minor characters |

```bash
python preprocessing/fix_speaker_conflation.py data/plays_dialogue/Ionesco_BaldSoprano_EN
python preprocessing/fix_maids_en.py           data/plays_dialogue/Genet_TheMaids_EN
python preprocessing/fix_linewrapped.py        data/plays_dialogue/Becque_LesCorbeaux
python preprocessing/fix_realists.py           data/plays_dialogue/Brieux_WomanOnHerOwn
```

Genet_LesBonnes_FR also had 2 turns relabelled by hand (VOIX DE MADAME → MADAME); there is no script for that.

After fixing a play, re-run preprocessing and analysis for it (see `scripts/plays/`).
