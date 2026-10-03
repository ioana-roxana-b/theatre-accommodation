# Linguistic accommodation in dramatic dialogue

Code for measuring function word accommodation in a bilingual (English / French) corpus of 24 realist and
absurdist plays: the convergence score of Danescu-Niculescu-Mizil & Lee (2011), a reply level mixed effects
model (Doyle et al., 2016), accommodation networks, and their comparison across the two language versions
of each play.

Most play texts are under copyright and are not included; `data/README.md` describes the expected input.
One public-domain pair is included as an example: Dumas *fils*, *La Dame aux Camélias* (French original,
1852) and *Camille* (English translation by Matilda Heron, 1856).

```bash
bash scripts/example/run_example.sh     # → outputs/example/{en,fr}/<Play>/
```

## Setup (Linux / WSL)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install r-base r-cran-lme4          # reply level models (scripts/stats/*.R)
```

## Run everything

```bash
python scripts/corpus/recompute_addressees.py   # once: one ±10 addressee rule for all plays
FULL=1 bash scripts/plays/run_all.sh            # every analysis in the paper
```

Step logs go to `outputs/logs/`. `SKIP_PIPELINE=1` reuses existing pipeline outputs; without `FULL=1` the
mixed models use a faster approximation (nAGQ = 0).

## Layout

| Folder | Contents |
| --- | --- |
| `main.py` | Pipeline entry point for one or more plays (`--stage preprocess` / `--stage analysis`). |
| `preprocessing/` | Turns, A→B pairs (adjacent, non-adjacent, randomized), function word features. |
| `analysis/` | Convergence scores and character networks. |
| `scripts/corpus/` | The ±10 addressee rule applied to all plays. |
| `scripts/plays/` | Run the pipeline on all plays; `run_all.sh` runs every step. |
| `scripts/stats/` | Statistical tests, mixed effects models, network comparisons, robustness checks. |
| `scripts/example/` | The public-domain example. |

Each folder has its own README.
