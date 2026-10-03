# analysis/

Stage 2 of the pipeline: convergence scores and the character network for one preprocessed play.
Called through `main.py --stage analysis`; results go to `<out_dir>/analysis_results/`.

```text
pairs_features{,_nonadjacent,_randomized}.csv
  → convergence_score.py   conv = P(B uses f | A used f) − P(B uses f), per dyad × family, kept if A uses f
                           in ≥ 3 turns and B uses f in ≥ 2 of the replies to them
                             → convergence_results/conv_by_pair_feature_wordlists*.csv
                           edge weight per dyad (mean over families, plain and trigger-weighted)
                             → convergence_results/conv_edge_summary_wordlists*.csv
  → graph_analysis.py      directed character network: reciprocity, density, communities,
                           assortativity, Gini, PageRank entropy → graph_results/
```

| File | What it does |
| --- | --- |
| `analysis_pipeline.py` | Runs the two steps for one play; entry point used by `main.py`. |
| `convergence_score.py` | Convergence score per dyad and family, and per-dyad edge summaries. |
| `graph_analysis.py` | Network descriptors (`graph_summary.txt`) and the network figure. |
