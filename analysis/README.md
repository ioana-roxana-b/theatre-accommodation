# analysis/

Stage 2 of the pipeline: convergence scores, significance tests, plots and character networks
for one preprocessed play. Called through `main.py --stage analysis`; results go to
`<out_dir>/analysis_results/`.

## Flow

```text
pairs_features*.csv (or LIWC-22 exports + pairs)
  → convergence_score_*.py   conv = P(B uses family | A used it) − P(B uses family)
                             per dyad × family       → convergence_results/conv_by_pair_feature_<mode>*.csv
                             per dyad (edge)         → convergence_results/conv_edge_summary_<mode>*.csv
  → stats_tests.py           adjacent vs non-adjacent vs randomized  → stats_results/
  → visualisation.py         heatmaps, bar charts, gender/importance → plots/  (+ plots/tables/*.csv)
  → graph_analysis.py        directed character network, reciprocity, communities → graph_results/
  → proximity.py             speaker proximity per family            → proximity_results/
```

## Files

| File | What it does |
|------|--------------|
| `analysis_pipeline.py` | Runs everything above for one play; entry point used by `main.py`. |
| `convergence_score_spacy.py` | Convergence from `pairs_features*.csv`. Used by the `wordlists`, `spacy` and `stopwords` modes. |
| `convergence_score_liwc.py` | Same measure for LIWC mode, on the merged LIWC flags. |
| `merge_liwc.py` | LIWC mode only: joins the LIWC-22 export back onto the pairs as a_*/b_* presence flags. |
| `stats_tests.py` | Paired t-tests and bootstrap CIs between conditions; breakdowns by gender and importance dyad. |
| `visualisation.py` | All plots in `plots/`. |
| `graph_analysis.py` | Network metrics and plots; `graph_summary.txt` has reciprocity and reciprocity_correlation. |
| `proximity.py` | Proximity between speakers' family usage rates. |
