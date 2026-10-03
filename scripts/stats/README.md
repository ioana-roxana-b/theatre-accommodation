# scripts/stats/

Cross-play statistics, run after `scripts/plays/analysis_plays_wordlists.sh`. They read
`outputs/plays_{en,fr}/wordlists/*/analysis_results/convergence_results/` and print results to
the terminal unless noted otherwise. Run them from the repo root.

| Script | What it does | Writes |
|--------|--------------|--------|
| `compute_stats.py` | All descriptive tests: adjacent vs randomized / non-adjacent, genre, language, per-author, per-play, importance, gender (t-tests, Cohen's d, bootstrap CIs, Bonferroni). | stdout |
| `author_stats.py` | Per-author adjacent vs randomized and vs non-adjacent. | stdout |
| `genre_nonadj.py` | Genre-level conditions + family-level absurdist vs realist. | stdout |
| `mixed_effects_model.py` | Linear mixed model on DNM scores: convergence ~ genre + language + gender dyad + directed importance + family, with a random intercept per play and a dyad variance component; likelihood-ratio test per factor; robustness model with a random intercept per author (play and dyad nested). Needs `statsmodels`; takes a few minutes. | `outputs/mixed_effects_{fixed_effects,lrt_importance,nested_robustness,model_info}.csv` |
| `build_alignment_table.py` | Builds the reply-level table for the GLMMs: one row per pair × family for adjacent, non-adjacent and randomized pairs, with A/B usage, turn lengths, and a `work` ID linking EN/FR versions. | `outputs/alignment/reply_level.csv.gz` |
| `alignment_glmm.R` | **Reply-level logistic mixed model (recommended).** Models whether B uses a family given that A just did. Accommodation = the `a_uses` effect; each factor is tested as an `a_uses:factor` interaction. Random effects: work (with accommodation slope), language version within work, and speaker × family baseline. Needs R + lme4. | `outputs/alignment/glmm_*.csv` |
| `alignment_conditions.R` | Same model on adjacent vs non-adjacent vs randomized pairs: is the `a_uses` effect larger for adjacent replies (overall and per family)? Model-based version of the control comparison in Danescu-Niculescu-Mizil & Lee (2011). | `outputs/alignment/conditions_*.csv` |
| `turn_length.py` | Play-level turn length check: mean / median reply length and share of long replies (≥ 50 tokens) per play vs DNM convergence; Spearman correlations and OLS of convergence on movement and log mean turn length. | `outputs/turn_length/*.csv` |
| `turn_length_glmm.R` | Reply-level version: the main GLMM plus the play's mean log turn length as a moderator (`a_uses:play_len`); LRTs for genre given turn length and turn length given genre. `--full` for nAGQ = 1. | `outputs/alignment/turnlen_*.csv` |
| `robustness_checks.py` | Robustness checks on the DNM score, recomputed from the pair tables with the pipeline's rules: randomized baseline over many seeds (`seeds`), without the B-response filter (`response`), single-addressee turns only (`addressee`), French scores without *le/la/les/l'* (`french`), word list sizes (`lexicon`). | `outputs/robustness/*.csv` |
| `example_exchanges.py` | Candidate examples of accommodation for the paper: runs of short consecutive adjacent turns ranked by rarer families picked up from the previous turn, printed with the original text and the shared families. Options `--turns`, `--max-tokens`, `--top`, `--plays`. | `outputs/examples/example_exchanges.{md,csv}` |
| `network_preservation.py` | **Cross-language network preservation.** Aligns characters across EN/FR versions (`character_alignment.csv`) and asks whether the same A→B edges keep their accommodation weight, sign, and each character's out/in-strength across languages. Permutation tests within each work; pooled test over all 12. Run with `--min_pairs 10` to keep only well-sampled edges. | `outputs/network_preservation/*.csv` |
| `network_similarity.py` | Cross-play version: 12 × 12 EN × FR distance matrix from alignment-free descriptors of how accommodation is organised (reciprocity, divergent share, source/target spread, weight vs centrality). Tests whether each play's own translation is nearest, against a matching null and a **topology null** (weights shuffled within each network), because matched versions share their cast. ~3 min. | `outputs/network_similarity/*.csv` |
| `character_alignment.csv` | Hand-made EN↔FR character mapping per work (e.g. Camille ↔ MARGUERITE). Characters that exist in only one version are left out. Rows with a `CHECK` note are uncertain. | (input) |
| `mixed_effects_summary.py` | Readable terminal summary of the model CSVs. | stdout |
| `convergent_validity_check.py` | Genre / language / gender: univariate t-test and Cohen's d next to the model's likelihood-ratio test (run the model first). | stdout |
| `sensitivity_window.py` | Re-runs the pipeline with addressee windows of 5/10/15 utterances to check that the results don't depend on the window. | `outputs/sensitivity/` |

```bash
# reply-level GLMM (R: sudo apt install r-base r-cran-lme4)
python scripts/stats/build_alignment_table.py
Rscript scripts/stats/alignment_glmm.R          # fast (nAGQ=0)
Rscript scripts/stats/alignment_glmm.R --full   # final numbers (nAGQ=1, slower)
Rscript scripts/stats/alignment_conditions.R     # adjacent vs controls (--full / --test too)
Rscript scripts/stats/turn_length_glmm.R --full  # genre vs play mean turn length
python scripts/stats/turn_length.py              # play-level turn length vs convergence
python scripts/stats/example_exchanges.py        # candidate examples, outputs/examples/
python scripts/paper/alignment_glmm_table.py

# cross-language network preservation (needs reply_level.csv.gz from the first step)
python scripts/stats/network_preservation.py
python scripts/stats/network_similarity.py
python scripts/paper/network_preservation_table.py

python scripts/stats/compute_stats.py
python scripts/stats/mixed_effects_model.py
python scripts/stats/mixed_effects_summary.py
python scripts/paper/mixed_effects_latex_table.py   # LaTeX tables from the model
```

**Why the reply-level model:** the dyad-level score `conv = p1 − p0` gives a dyad with 3 triggers the
same weight as one with 300, and it is capped by the family's base rate (it can't exceed 1 − p0), so
common families look like they accommodate less. Exactly: p1 − p0 = P(A doesn't use t) × [P(B|A used t) − P(B|A didn't)],
so the trigger effect is shrunk by how often the family occurs (see Doyle, Yurovsky & Frank 2016). Modelling each reply in log-odds fixes both.
It also compares languages within the same work instead of treating EN and FR as unrelated plays.
