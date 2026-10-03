"""
Addressee-window sensitivity check.

Re-runs compute_addressees() on data/plays_dialogue/{play}/quotation_info.csv
with window=5/10/15, replaces the addressees column, then replays:
  preprocess_quotes -> merge_contiguous_turns -> build_pairs
  -> features_wordlists -> convergence_score_spacy

Usage:  python scripts/stats/sensitivity_window.py   (from repo root)
"""
from __future__ import annotations
import ast, sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing import preprocess_data, build_pairs, features_wordlists
from analysis   import convergence_score_spacy

PLAY_DIR    = Path("data/plays_dialogue")
OUT_ROOT    = Path("outputs") / "sensitivity"
WINDOWS     = [5, 10, 15]
MIN_SUPPORT = 2

PLAYS_EN = [
    "Beckett_WaitingForGodot", "Beckett_Endgame",
    "Genet_Balcony_EN",        "Genet_TheMaids_EN",
    "Ionesco_BaldSoprano_EN",  "Ionesco_Rhinoceros_EN",
    "Becque_TheVultures",      "Becque_TheWomanOfParis",
    "Brieux_LaRobeRouge_EN",   "Brieux_WomanOnHerOwn",
    "Dumas_Camille_EN",        "Dumas_TheDemiMonde_EN",
]
PLAYS_FR = [
    "Beckett_EnAttendantGodot", "Beckett_FinDePartie",
    "Genet_LeBalcon_FR",        "Genet_LesBonnes_FR",
    "Ionesco_LaCantriceChauve_FR", "Ionesco_Rhinoceros_FR",
    "Becque_LesCorbeaux",       "Becque_LaParisienne",
    "Brieux_LaRobeRouge",       "Brieux_LaFemmeSeule",
    "Dumas_DameAuxCamelias",    "Dumas_LeDemiMonde_FR",
]

FAMILY_RENAME = {
    "article":  "articles",      "auxverb": "auxiliary_verbs",
    "conj":     "conjunctions",  "adverb":  "adverbs",
    "ipron":    "impersonal_pronouns", "negate": "negations",
    "ppron":    "personal_pronouns",   "prep":   "prepositions",
    "quant":    "quantifiers",
}
ALLOWED_FAMILIES = set(FAMILY_RENAME.values())


def recompute_addressees_csv(qi_path: Path, window: int) -> pd.DataFrame:
    """
    Load quotation_info.csv, recompute addressees using ±window utterance window,
    return modified DataFrame ready for preprocess_quotes.
    """
    from preprocessing.addressees import recompute_addressees
    return recompute_addressees(pd.read_csv(qi_path, encoding="utf-8-sig"), window)


def cohend(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = np.sqrt(((n1-1)*np.std(a, ddof=1)**2 + (n2-1)*np.std(b, ddof=1)**2) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def run_one_play(play: str, lang: str, window: int) -> pd.Series | None:
    qi_path   = PLAY_DIR / play / "quotation_info.csv"
    char_info = PLAY_DIR / play / "character_info.csv"
    if not qi_path.exists():
        print(f"  [skip] {play}: no quotation_info.csv")
        return None

    out_dir = OUT_ROOT / f"window_{window}" / play
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Recompute addressees and save modified quotation_info
    qi_mod = out_dir / "quotation_info_mod.csv"
    df_mod = recompute_addressees_csv(qi_path, window)
    df_mod.to_csv(qi_mod, index=False)

    # 2. Clean quotes
    cleaned = out_dir / "cleaned_quotes.csv"
    preprocess_data.preprocess_quotes(str(qi_mod), str(cleaned), lowercase=False)

    # 3. Merge + build pairs
    quotes_df = build_pairs.load_quotes(cleaned)
    merged_df = build_pairs.merge_contiguous_turns(quotes_df)
    pairs_adj  = build_pairs.build_pairs_from_merged(merged_df)
    pairs_non  = build_pairs.build_pairs_from_merged_nonadjacent(merged_df, min_gap=3, max_gap=5)  # as in the plays scripts
    pairs_rand = build_pairs.build_pairs_randomized_from_adjacent(pairs_adj)

    if char_info.exists():
        pairs_adj  = build_pairs.attach_metadata_to_pairs(pairs_adj,  char_info)
        pairs_non  = build_pairs.attach_metadata_to_pairs(pairs_non,  char_info)
        pairs_rand = build_pairs.attach_metadata_to_pairs(pairs_rand, char_info)

    adj_path  = out_dir / "pairs_A_to_B.csv"
    non_path  = out_dir / "pairs_A_to_B_nonadjacent.csv"
    rand_path = out_dir / "pairs_A_to_B_randomized.csv"
    pairs_adj.to_csv(adj_path,   index=False)
    pairs_non.to_csv(non_path,   index=False)
    pairs_rand.to_csv(rand_path, index=False)

    # 4. Features
    lang_tag = "en" if lang == "en" else "fr"
    feat_adj  = out_dir / "pairs_features.csv"
    feat_non  = out_dir / "pairs_features_nonadjacent.csv"
    feat_rand = out_dir / "pairs_features_randomized.csv"
    features_wordlists.process_pairs(adj_path,  feat_adj,  lang=lang_tag)
    features_wordlists.process_pairs(non_path,  feat_non,  lang=lang_tag)
    features_wordlists.process_pairs(rand_path, feat_rand, lang=lang_tag)

    # 5. Convergence
    conv_adj  = out_dir / "conv_by_pair_feature.csv"
    conv_rand = out_dir / "conv_by_pair_feature_randomized.csv"
    conv_non  = out_dir / "conv_by_pair_feature_nonadjacent.csv"
    for feat, conv, lbl in [
        (feat_adj,  conv_adj,  "conv_edge.csv"),
        (feat_rand, conv_rand, "conv_edge_rand.csv"),
        (feat_non,  conv_non,  "conv_edge_non.csv"),
    ]:
        convergence_score_spacy.compute_convergence_spacy(
            pairs_features_path=feat,
            out_conv_pair_feature=conv,
            out_conv_edge=out_dir / lbl,
            min_support=MIN_SUPPORT,
            family_rename=FAMILY_RENAME,
            allowed_families=ALLOWED_FAMILIES,
            min_resp_support=2,
        )

    return pairs_adj  # return for n_pairs count


def collect(window: int, condition: str) -> pd.Series:
    suffix = "" if condition == "adj" else f"_{condition}"
    base   = OUT_ROOT / f"window_{window}"
    vals   = []
    for csv in base.glob(f"*/conv_by_pair_feature{suffix}.csv"):
        try:
            df = pd.read_csv(csv, encoding="utf-8-sig")
            if "conv" in df.columns:
                vals.append(df["conv"].dropna())
        except Exception:
            pass
    return pd.concat(vals).reset_index(drop=True) if vals else pd.Series(dtype=float)


def fmt_p(p):
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


# ── Main ─────────────────────────────────────────────────────────────────────
print("Addressee-window sensitivity analysis")
print("=" * 70)

summary_rows = []

for window in WINDOWS:
    print(f"\n[window={window}] processing {len(PLAYS_EN)+len(PLAYS_FR)} plays...")
    n_pairs_total = 0
    for play in PLAYS_EN:
        result = run_one_play(play, "en", window)
        if result is not None:
            n_pairs_total += len(result)
    for play in PLAYS_FR:
        result = run_one_play(play, "fr", window)
        if result is not None:
            n_pairs_total += len(result)

    adj  = collect(window, "adj")
    rand = collect(window, "randomized")
    non  = collect(window, "nonadjacent")

    _, p_rand = stats.ttest_ind(adj.values, rand.values)
    _, p_non  = stats.ttest_ind(adj.values, non.values) if len(non) > 1 else (None, float("nan"))
    d_rand = cohend(adj.values, rand.values)
    d_non  = cohend(adj.values, non.values) if len(non) > 1 else float("nan")

    row = dict(
        window=window,
        adj=round(adj.mean(), 3),
        non=round(non.mean(), 3),
        rand=round(rand.mean(), 3),
        d_adj_rand=round(d_rand, 3),
        p_adj_rand=fmt_p(p_rand),
        d_adj_non=round(d_non, 3),
        p_adj_non=fmt_p(p_non),
        n_conv_rows=len(adj),
        n_adj_pairs=n_pairs_total,
    )
    summary_rows.append(row)
    print(f"  adj={row['adj']:.3f}  non={row['non']:.3f}  rand={row['rand']:.3f} | "
          f"d(a-r)={row['d_adj_rand']:+.3f} p={row['p_adj_rand']} | "
          f"d(a-n)={row['d_adj_non']:+.3f} p={row['p_adj_non']} | "
          f"n_conv={row['n_conv_rows']}  n_pairs={row['n_adj_pairs']}")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print(f"{'Win':>4}  {'adj':>6}  {'non':>6}  {'rand':>6}  "
      f"{'d(a-r)':>7}  {'p(a-r)':>9}  {'d(a-n)':>7}  {'p(a-n)':>9}  {'n_pairs':>8}")
for r in summary_rows:
    print(f"  {r['window']:2d}   {r['adj']:.3f}   {r['non']:.3f}   {r['rand']:.3f}   "
          f"{r['d_adj_rand']:+.3f}   {r['p_adj_rand']:>9}  {r['d_adj_non']:+.3f}   "
          f"{r['p_adj_non']:>9}  {r['n_adj_pairs']:>8}")

out_csv = OUT_ROOT / "window_sensitivity_summary.csv"
pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
print(f"\nSummary saved to {out_csv}")
