"""Re-run only the visualization step for all plays (no convergence recompute)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import visualisation

root = Path(__file__).resolve().parents[2]

# Collect every directory that has a conv_by_pair_feature_*.csv
conv_dirs = []
for csv_path in sorted(root.glob("outputs/**/**/convergence_results/conv_by_pair_feature_*.csv")):
    mode = csv_path.stem.split("_")[-1]          # liwc or spacy
    conv_dir = csv_path.parent
    if (conv_dir, mode) not in conv_dirs:
        conv_dirs.append((conv_dir, mode))

# Deduplicate: keep the conv_dir once per mode (multiple CSVs may share a dir)
seen = set()
unique = []
for conv_dir, mode in conv_dirs:
    key = (str(conv_dir), mode)
    if key not in seen:
        seen.add(key)
        unique.append((conv_dir, mode))

for conv_dir, mode in unique:
    plot_dir = conv_dir.parent / "plots"
    visualisation.run_visuals_for_play(
        in_dir=conv_dir,
        out_root=plot_dir,
        mode=mode,
        play_title=None,
    )

print(f"Done. Regenerated plots for {len(unique)} directories.")
