from __future__ import annotations
from pathlib import Path

import pandas as pd

from preprocessing import preprocess_data          # cleans the quotation csv
from preprocessing import build_pairs              # merges contiguous turns and builds A→B pairs
from preprocessing import features_wordlists       # 9 LIWC-aligned function word families


def _check_mode(mode: str) -> None:
    if mode.lower() != "wordlists":
        raise ValueError("mode must be 'wordlists'")


def run_one_play_prep(
    play_dir: Path,
    out_dir: Path,
    mode: str,
    lowercase: bool,
    character_info_arg: str | None,
    lang: str = "en",
    non_adj_min_gap: int = 2,
    non_adj_max_gap: int = 2,
) -> None:
    """
    Generate per-play preprocessing artifacts for downstream analysis.
    Params:
        play_dir: Path to the play folder containing source CSVs.
        out_dir: Output directory where derived artifacts are written.
        mode: Feature set; only "wordlists".
        lowercase: Whether to lowercase quotes during cleaning.
        character_info_arg: Optional override path or {play} pattern for metadata.
    Returns:
        None
    """
    _check_mode(mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    quotes_csv = play_dir/"quotation_info.csv"
    if not quotes_csv.exists():
        raise FileNotFoundError(f"Missing {quotes_csv}")

    # character metadata
    character_info_csv: Path | None = None
    if character_info_arg:
        character_info_csv = (
            Path(character_info_arg.format(play=play_dir.name))
            if "{play}" in character_info_arg
            else Path(character_info_arg)
        )
    else:
        auto_candidate = play_dir / "character_info.csv"
        if auto_candidate.exists():
            character_info_csv = auto_candidate

    # 1 clean quotes
    cleaned_csv = out_dir/"cleaned_quotes.csv"
    preprocess_data.preprocess_quotes(
        path=str(quotes_csv),
        out_path=str(cleaned_csv),
        lowercase=lowercase,
    )

    # 2 merge and pairs
    quotes_df = build_pairs.load_quotes(Path(cleaned_csv))
    merged_df = build_pairs.merge_contiguous_turns(quotes_df)
    merged_path = out_dir/"merged_turns.csv"
    merged_df.to_csv(merged_path, index=False)

    # adjacent
    pairs_df_adj = build_pairs.build_pairs_from_merged(merged_df)
    if character_info_csv and Path(character_info_csv).exists():
        pairs_df_adj = build_pairs.attach_metadata_to_pairs(pairs_df_adj, character_info_csv)
    pairs_path_adj = out_dir/"pairs_A_to_B.csv"
    pairs_df_adj.to_csv(pairs_path_adj, index=False)

    # nonadjacent
    pairs_df_nonadj = build_pairs.build_pairs_from_merged_nonadjacent(
        merged_df, min_gap=non_adj_min_gap, max_gap=non_adj_max_gap
    )
    if character_info_csv and Path(character_info_csv).exists():
        pairs_df_nonadj = build_pairs.attach_metadata_to_pairs(pairs_df_nonadj, character_info_csv)
    pairs_path_nonadj = out_dir/"pairs_A_to_B_nonadjacent.csv"
    pairs_df_nonadj.to_csv(pairs_path_nonadj, index=False)

    # randomized
    pairs_df_rand = build_pairs.build_pairs_randomized_from_adjacent(pairs_df_adj, seed=123)
    pairs_path_rand = out_dir/"pairs_A_to_B_randomized.csv"
    pairs_df_rand.to_csv(pairs_path_rand, index=False)

    # 3 word-list features for all three conditions
    features_wordlists.process_pairs(pairs_path_adj, out_dir/"pairs_features.csv", lang=lang)
    features_wordlists.process_pairs(pairs_path_nonadj, out_dir/"pairs_features_nonadjacent.csv", lang=lang)
    features_wordlists.process_pairs(pairs_path_rand, out_dir/"pairs_features_randomized.csv", lang=lang)
    print(f"[prep] word-list features written in {out_dir}")


def run_corpus_prep(
    play_dirs,
    out_dir,
    mode,
    lowercase=False,
    character_info_pattern=None,
    lang: str = "en",
    non_adj_min_gap: int = 2,
    non_adj_max_gap: int = 2,
):
    """
    Build merged corpus artifacts by running preprocessing for each play.
    Params:
        play_dirs: Iterable of play directories to include in the corpus.
        out_dir: Root output directory where corpus artifacts are written.
        mode: Feature set; only "wordlists".
        lowercase: Whether to lowercase quotes during cleaning.
        character_info_pattern: Optional {play} pattern to locate metadata files.
    Returns:
        None
    """
    _check_mode(mode)
    out_root = Path(out_dir)
    out_all = out_root/"All_Plays"
    out_root.mkdir(parents=True, exist_ok=True)
    out_all.mkdir(parents=True, exist_ok=True)

    all_adj, all_non, all_rand = [], [], []

    for nd in play_dirs:
        play_dir = Path(nd)
        title = play_dir.name
        print(f"[corpus] processing {title}...")

        # Run normal preprocessing
        out_play = out_root/title
        run_one_play_prep(
            play_dir=play_dir,
            out_dir=out_play,
            mode=mode,
            lowercase=lowercase,
            character_info_arg=character_info_pattern.format(play=title) if character_info_pattern else None,
            lang=lang,
            non_adj_min_gap=non_adj_min_gap,
            non_adj_max_gap=non_adj_max_gap,
        )

        # Load per-play outputs
        adj = pd.read_csv(out_play/"pairs_A_to_B.csv")
        non = pd.read_csv(out_play/"pairs_A_to_B_nonadjacent.csv")
        rnd = pd.read_csv(out_play/"pairs_A_to_B_randomized.csv")
        for df in [adj, non, rnd]:
            df["play"] = title

        all_adj.append(adj)
        all_non.append(non)
        all_rand.append(rnd)

    # Concatenate everything
    adj_all = pd.concat(all_adj, ignore_index=True)
    non_all = pd.concat(all_non, ignore_index=True)
    rnd_all = pd.concat(all_rand, ignore_index=True)

    # Write unified pairs
    adj_path = out_all/"pairs_A_to_B.csv"
    non_path = out_all/"pairs_A_to_B_nonadjacent.csv"
    rnd_path = out_all/"pairs_A_to_B_randomized.csv"
    adj_all.to_csv(adj_path, index=False)
    non_all.to_csv(non_path, index=False)
    rnd_all.to_csv(rnd_path, index=False)

    print(f"[corpus] merged pairs written to {out_all}")

    features_wordlists.process_pairs(adj_path, out_all/"pairs_features.csv", lang=lang)
    features_wordlists.process_pairs(non_path, out_all/"pairs_features_nonadjacent.csv", lang=lang)
    features_wordlists.process_pairs(rnd_path, out_all/"pairs_features_randomized.csv", lang=lang)
    print(f"[corpus] word-list feature CSVs saved in {out_all}")

    print(f"[corpus] done. Combined corpus ready at {out_all}")
