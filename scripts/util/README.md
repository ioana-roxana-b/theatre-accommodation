# scripts/util/

Re-draw figures without recomputing convergence, e.g. after changing plot styles in
`analysis/visualisation.py` or `analysis/graph_analysis.py`. Both scripts find every
`convergence_results/` folder under `outputs/` and redo that step in place.

```bash
python scripts/util/regen_plots.py    # → .../analysis_results/plots/
python scripts/util/regen_graphs.py   # → .../analysis_results/graph_results/
```
