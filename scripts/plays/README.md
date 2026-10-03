# scripts/plays/

Run the pipeline (`main.py`) on all 24 plays. Each script works from any directory. Every play
gets its own output folder, plus `All_Plays/` for the whole language corpus. Non-adjacent pairs
use a 3–5 turn gap (`NON_ADJ_MIN` / `NON_ADJ_MAX`), and the analysis uses `MIN_SUPPORT=3`.

```bash
FULL=1 bash scripts/plays/run_all.sh                 # every analysis in the paper
bash scripts/plays/preprocess_plays_wordlists.sh     # → outputs/plays_{en,fr}/wordlists/<Play>/
bash scripts/plays/analysis_plays_wordlists.sh       # → .../<Play>/analysis_results/
```

`run_all.sh` writes one log per step to `outputs/logs/`. Without `FULL=1` the mixed models use a
faster approximation (nAGQ = 0); `SKIP_PIPELINE=1` reuses existing pipeline outputs.
