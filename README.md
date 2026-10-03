# Linguistic accommodation in dramatic dialogue

Code for measuring function word accommodation in a bilingual (English / French) corpus of 24 realist and
absurdist plays: the convergence score of Danescu-Niculescu-Mizil & Lee (2011), a reply level mixed effects
model (Doyle et al., 2016), accommodation networks, and their comparison across the two language versions
of each play.

The play texts are under copyright and are not included; `data/README.md` describes the expected input.

## Setup (Linux / WSL)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install r-base r-cran-lme4          # reply level models (scripts/stats/*.R)
```

## Run everything

```bash
python scripts/corpus/recompute_addressees.py   # once: one ±10 addressee rule for all plays
FULL=1 bash scripts/plays/run_all.sh            # pipeline, statistics, models, networks, tables
```

Step logs go to `outputs/logs/`. `SKIP_PIPELINE=1` reuses existing pipeline outputs; without `FULL=1` the
mixed models use a faster approximation (nAGQ = 0).

## Layout

| Folder | Contents |
|--------|----------|
| `main.py` | Pipeline entry point for one or more plays (`--stage preprocess` / `--stage analysis`). |
| `preprocessing/` | Turns, A→B pairs (adjacent, non-adjacent, randomized), function word features. |
| `analysis/` | Convergence scores, tests, plots, character networks. |
| `scripts/corpus/` | How the corpus was built from the source editions (OCR, cleaning, dialogue extraction). |
| `scripts/plays/` | Run the pipeline on all plays; `run_all.sh` runs every step. |
| `scripts/stats/` | Cross-play statistics, mixed effects models, network comparisons, robustness checks. |
| `scripts/paper/` | LaTeX tables. |
| `scripts/util/` | Re-draw plots and networks without recomputing. |

Each folder has its own README.
