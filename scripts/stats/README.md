# scripts/stats/

Cross-play statistics, run after `scripts/plays/preprocess_plays_wordlists.sh` and
`analysis_plays_wordlists.sh`. Run from the repo root; `scripts/plays/run_all.sh` runs all of them in order.

| Script | Paper section | What it does | Writes |
| --- | --- | --- | --- |
| `compute_stats.py` | Results (t-tests) | Adjacent vs non-adjacent, movement, language, gender and prominence comparisons: independent t-tests, Cohen's d, bootstrap CIs (5,000 resamples), Bonferroni within families of tests. | stdout |
| `robustness_checks.py` | Corpus level, Appendix | `table`: adjacent vs randomized with the randomized control averaged over 1,000 shuffles (permutation p). Also `seeds`, `response`, `addressee`, `french`, `counts`, `length`, `reciprocity`, `lexicon`. | `outputs/robustness/` |
| `mixed_effects_model.py` | Statistical testing, Joint models | Linear mixed model on the DNM scores (play intercept, dyad variance component), likelihood ratio test per factor, author-intercept robustness model. | `outputs/mixed_effects_*.csv` |
| `build_alignment_table.py` | Reply level model | One row per A→B pair × family for the three conditions. | `outputs/alignment/reply_level.csv.gz` |
| `alignment_glmm.R` | Reply level model, Joint models | Logistic mixed model of B's use of a family given A's (lme4); LRT per factor through its interaction with A_f. `--full` for nAGQ = 1. | `outputs/alignment/glmm_*.csv` |
| `alignment_conditions.R` | Joint models | Same model on adjacent, non-adjacent and randomized pairs. | `outputs/alignment/conditions_*.csv` |
| `turn_length.py` | Appendix (turn length) | Play mean turn length vs convergence. | `outputs/turn_length/` |
| `turn_length_glmm.R` | Joint models, Appendix | Reply level model with play mean turn length as a moderator. | `outputs/alignment/turnlen_*.csv` |
| `sensitivity_window.py` | Corpus level | Reruns the pipeline with addressee windows of ±5, ±10 and ±15 utterances. | `outputs/sensitivity/` |
| `network_preservation.py` | Cross-language network comparison | Matched EN/FR edges: Spearman ρ, sign agreement vs chance, out/in-strength ρ; permutation tests within each play pair and pooled. `--min_pairs N` keeps edges with ≥ N pairs. | `outputs/network_preservation/` |
| `network_similarity.py` | Appendix (whole networks) | 12 × 12 EN × FR distance between network descriptors, with a matching null and a topology null. | `outputs/network_similarity/` |
| `character_alignment.csv` | Cross-language network comparison | Hand-made EN↔FR character mapping per play pair. | (input) |

R scripts need `r-base` and `r-cran-lme4`.
