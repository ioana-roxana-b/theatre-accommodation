from __future__ import annotations
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from preprocessing import preprocessing_pipeline
from analysis import analysis_pipeline


def main():
    """
    Parse CLI arguments and run the requested pipeline stage.
    Params:
        None
    Returns:
        None
    """
    ap = argparse.ArgumentParser(description="Run one stage of the accommodation pipeline.")
    ap.add_argument(
        "--stage",
        choices=["preprocess", "analysis"],
        required=True,
        help="Select which pipeline stage to execute.",
    )
    ap.add_argument(
        "--mode",
        choices=["wordlists"],
        default="wordlists",
        help="Feature set: explicit word lists for the 9 LIWC-aligned function word families.",
    )
    ap.add_argument("--play_dir", nargs="+", required=True,
                    help="Either a single play folder or a list of them for corpus-level preprocessing.")
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Destination folder for per-play outputs or the merged corpus.",
    )
    ap.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text while preprocessing (ignored for analysis stage).",
    )
    ap.add_argument("--character_info", type=str, default=None,
                    help="Path to character_info.csv for preprocess.")
    ap.add_argument(
        "--min_support",
        type=int,
        default=10,
        help="Minimum number of A turns containing a family for a dyad-family score.",
    )
    ap.add_argument("--title", dest="title", default=None)
    ap.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language of the word lists ('en' or 'fr').",
    )
    ap.add_argument(
        "--non_adj_min_gap",
        type=int,
        default=2,
        help="Minimum turn gap for non-adjacent pairs (default 2, i.e. one intervening turn).",
    )
    ap.add_argument(
        "--non_adj_max_gap",
        type=int,
        default=2,
        help="Maximum turn gap for non-adjacent pairs (default 2, i.e. one intervening turn).",
    )

    args = ap.parse_args()

    stage = args.stage.lower()
    mode = args.mode.lower()
    out_root = Path(args.out_dir)

    if stage == "preprocess":
        if len(args.play_dir) > 1:
            # multi-play case
            print("[stage] Running corpus-level preprocessing...")
            preprocessing_pipeline.run_corpus_prep(
                play_dirs=args.play_dir,
                out_dir=out_root,
                mode=mode,
                lowercase=args.lowercase,
                character_info_pattern=args.character_info,
                lang=args.lang,
                non_adj_min_gap=args.non_adj_min_gap,
                non_adj_max_gap=args.non_adj_max_gap,
            )
        else:
            # single play case
            play_dir = Path(args.play_dir[0])
            preprocessing_pipeline.run_one_play_prep(
                play_dir=play_dir,
                out_dir=out_root,
                mode=mode,
                lowercase=args.lowercase,
                character_info_arg=args.character_info,
                lang=args.lang,
                non_adj_min_gap=args.non_adj_min_gap,
                non_adj_max_gap=args.non_adj_max_gap,
            )
        print(f"[preprocess] Done -> {out_root}")

    else:
        # ------------ analysis ------------
        in_dir = Path(args.play_dir[0])
        out_root.mkdir(parents=True, exist_ok=True)

        analysis_pipeline.run_full_analysis(
            in_dir=in_dir,
            out_dir=out_root,
            mode=mode,
            min_support=args.min_support,
            play_title=args.title,
        )

        print(f"[analysis] Done -> {out_root}")


if __name__ == "__main__":
    main()
