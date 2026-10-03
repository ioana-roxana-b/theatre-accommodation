# scripts/paper/

Generate the LaTeX tables for the paper. Run from the repo root.

| Script | Input | Output |
|--------|-------|--------|
| `mixed_effects_latex_table.py` | `outputs/mixed_effects_*.csv` (run `scripts/stats/mixed_effects_model.py` first) | `paper/tables/mixed_effects_latex_table.tex` (fixed + random effects), `mixed_effects_lrt.tex` (likelihood-ratio test per factor), `mixed_effects_genre_robustness.tex` (primary vs play-within-author) |
| `mixed_effects_latex_table_enhanced.py` | same | `paper/tables/mixed_effects_latex_table_enhanced.tex` (raw + standardized β), `mixed_effects_relative_importance.tex` (levels ranked by \|β_std\|; for whole-factor importance use the likelihood-ratio table instead) |
| `alignment_glmm_table.py` | `outputs/alignment/glmm_*.csv` (run `scripts/stats/alignment_glmm.R` first) | `paper/tables/alignment_glmm_lrt.tex` (does accommodation differ by each factor), `alignment_glmm_levels.tex` (odds ratio per level), `alignment_glmm_genre_family.tex`, `alignment_glmm_conditions.tex` (adjacent vs controls, if `alignment_conditions.R` was run) |
| `network_preservation_table.py` | `outputs/network_preservation/*.csv` (run `scripts/stats/network_preservation.py` first) | `paper/tables/network_preservation.tex`: per work, EN/FR edge-weight ρ, sign agreement vs chance, character role ρ; pooled permutation tests. Also `network_similarity.tex` if `network_similarity.py` was run |
| `gen_table.py` | `data/plays_dialogue/` + `outputs/plays_*/wordlists/` | Corpus statistics table (characters, turns, tokens, dyads), **computed from the data**. Printed to stdout. |
| `gen_table2.py` | none | Table 1 of the paper (by genre, with translation footnotes). **Numbers are hardcoded**; update them by hand, e.g. from `gen_table.py`. Printed to stdout. |

```bash
python scripts/stats/mixed_effects_model.py
python scripts/paper/mixed_effects_latex_table.py
python scripts/paper/mixed_effects_latex_table_enhanced.py
python scripts/paper/gen_table.py  > paper/tables/corpus_stats.tex
python scripts/paper/gen_table2.py > paper/tables/corpus_table1.tex
```
