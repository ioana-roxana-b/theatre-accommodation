from __future__ import annotations
from pathlib import Path
from typing import Optional

from analysis import convergence_score, graph_analysis

# raw family names in pairs_features*.csv -> names used in the outputs
FAMILY_RENAME = {
    "article": "articles",
    "auxverb": "auxiliary_verbs",
    "conj": "conjunctions",
    "adverb": "adverbs",
    "ipron": "impersonal_pronouns",
    "negate": "negations",
    "ppron": "personal_pronouns",
    "prep": "prepositions",
    "quant": "quantifiers",
}


def run_one_play_analysis(
    in_dir: Path,
    out_conv_dir: Path,
    mode: str,
    min_support: int,
) -> None:
    """
    Compute convergence data for one play.

    Params:
        in_dir: Preprocess outputs folder that contains pairs files.
        out_conv_dir: Destination folder for conv_by_pair_feature_* and conv_edge_summary_* CSVs.
        mode: Feature set; only "wordlists".
        min_support: Minimum number of A triggers per family within a dyad to include that family.

    Returns:
        None. Writes convergence CSVs to out_conv_dir.
    """
    if mode.lower() != "wordlists":
        raise ValueError("mode must be 'wordlists'")
    out_conv_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("pairs_features.csv",
         "conv_by_pair_feature_wordlists.csv",
         "conv_edge_summary_wordlists.csv"),
        ("pairs_features_nonadjacent.csv",
         "conv_by_pair_feature_wordlists_nonadjacent.csv",
         "conv_edge_summary_wordlists_nonadjacent.csv"),
        ("pairs_features_randomized.csv",
         "conv_by_pair_feature_wordlists_randomized.csv",
         "conv_edge_summary_wordlists_randomized.csv"),
    ]
    for in_name, out_pf, out_edge in configs:
        in_path = in_dir / in_name
        if not in_path.exists():
            print(f"[wordlists] Missing {in_path}, skipping.")
            continue
        convergence_score.compute_convergence(
            pairs_features_path=in_path,
            out_conv_pair_feature=out_conv_dir / out_pf,
            out_conv_edge=out_conv_dir / out_edge,
            min_support=min_support,
            family_rename=FAMILY_RENAME,
            allowed_families=set(FAMILY_RENAME.values()),
            min_resp_support=2,
        )


def run_full_analysis(
    in_dir: Path,
    out_dir: Path,
    mode: str,
    min_support: int,
    play_title: Optional[str] = None,
) -> None:
    """
    Run the complete analysis workflow for one play.

    Params:
        in_dir: Preprocess outputs folder for this play.
        out_dir: Root folder where analysis_results will be created.
        mode: Feature set; only "wordlists".
        min_support: Minimum number of A triggers per family within a dyad to include that family.

    Returns:
        None. Creates analysis_results/ with convergence_results/ and graph_results/.
    """
    base = out_dir / "analysis_results"
    conv_dir = base / "convergence_results"
    graph_dir = base / "graph_results"

    for p in (conv_dir, graph_dir):
        p.mkdir(parents=True, exist_ok=True)

    run_one_play_analysis(in_dir=in_dir, out_conv_dir=conv_dir, mode=mode, min_support=min_support)

    graph_analysis.run_graph_analysis(conv_dir, graph_dir, mode, positive_only=False, play_title=play_title)

    print(f"[analysis] Complete. Results written to {base}")
