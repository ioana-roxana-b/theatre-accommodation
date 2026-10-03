"""
Recompute the addressees column of every data/plays_dialogue/<Play>/quotation_info.csv with
one rule for all plays: the other characters speaking within +/- 10 utterances.

A pair A->B only counts if B is among A's addressees, so the same rule must hold for every play.
The first run keeps the previous file as quotation_info.csv.pre_addressees.bak.

Usage:  python scripts/corpus/recompute_addressees.py [--window 10]
"""
import argparse
import ast
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocessing.addressees import recompute_addressees  # noqa: E402

CORPUS = Path("data/plays_dialogue")


def main():
    ap = argparse.ArgumentParser(description="Recompute addressees with a +/- window rule.")
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    for play_dir in sorted(p for p in CORPUS.iterdir() if (p / "quotation_info.csv").exists()):
        qi = play_dir / "quotation_info.csv"
        backup = play_dir / "quotation_info.csv.pre_addressees.bak"
        if not backup.exists():
            shutil.copy2(qi, backup)
        df = pd.read_csv(qi, encoding="utf-8-sig")
        before = df["addressees"].apply(lambda s: len(ast.literal_eval(s)) if isinstance(s, str) else 0).mean()
        new = recompute_addressees(df, args.window)
        after = new["addressees"].apply(lambda s: len(ast.literal_eval(s))).mean()
        new.to_csv(qi, index=False, encoding="utf-8-sig")
        print(f"{play_dir.name:32s} {len(df):5d} quotes  addressees/quote {before:4.2f} -> {after:4.2f}")


if __name__ == "__main__":
    main()
