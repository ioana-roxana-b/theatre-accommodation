"""Re-run only the graph analysis step for all plays (no convergence recompute)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import graph_analysis

root = Path(__file__).resolve().parents[2]

MODES = ["liwc", "spacy", "wordlists", "stopwords"]

seen = set()
jobs = []

for mode in MODES:
    for csv_path in sorted(root.glob(f"outputs/**/**/convergence_results/conv_edge_summary_{mode}.csv")):
        conv_dir = csv_path.parent
        key = str(conv_dir)
        if key not in seen:
            seen.add(key)
            jobs.append((conv_dir, mode))

for conv_dir, mode in jobs:
    graph_dir = conv_dir.parent / "graph_results"
    graph_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{mode}] {conv_dir.parent.parent.name}/{conv_dir.parent.parent.parent.name} ...")
    graph_analysis.run_graph_analysis(
        conv_dir, graph_dir, mode, positive_only=False, play_title=None
    )

print(f"\nDone. Regenerated graph plots for {len(jobs)} directories.")
