# scripts/plays/

Run the pipeline (`main.py`) on all 24 plays. Each script works from any directory.
Every play gets its own output folder, plus `All_Plays/` for the whole language corpus.
Non-adjacent pairs use a 3–5 turn gap (`NON_ADJ_MIN` / `NON_ADJ_MAX`), and analysis uses
`MIN_SUPPORT=3`. Both can be overridden with env vars.

## Everything at once

```bash
bash scripts/plays/run_all.sh          # pipeline + mixed models + networks + tables + stats (~1 h)
FULL=1 SKIP_PIPELINE=1 bash scripts/plays/run_all.sh   # re-fit the mixed models with nAGQ=1
```

Logs for each step go to `outputs/logs/`.

## Word lists (the setup used in the paper)

```bash
bash scripts/plays/preprocess_plays_wordlists.sh   # → outputs/plays_{en,fr}/wordlists/<Play>/
bash scripts/plays/analysis_plays_wordlists.sh     # → .../<Play>/analysis_results/
```

## Other feature modes

| Mode | Preprocess | Analysis | Output |
|------|------------|----------|--------|
| spaCy POS | `preprocess_plays_spacy.sh` | `analysis_plays_spacy.sh` | `outputs/plays_{en,fr}/spacy/` |
| stopwords-iso | `preprocess_plays_stopwords.sh` | `analysis_plays_stopwords.sh` | `outputs/plays_{en,fr}/stopwords/` |
| LIWC-22 | `preprocess_plays.sh` | `MODE=liwc bash analysis_plays.sh` | `outputs/plays_{en,fr}/<author>_<lang>/liwc/` |

The spaCy modes need `en_core_web_sm` / `fr_core_news_sm`.

**LIWC** groups plays by author × language. After preprocessing, open each `liwc_input*.csv` in
LIWC-22 and save the export next to it with LIWC's default name
(`LIWC-22 Results - liwc_input - LIWC Analysis.csv`, and so on) before running the analysis.
