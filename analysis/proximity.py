import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_proximity_from_conv_csv(conv_csv_path, family_col="family"):
    """
    Expects columns:
      n_pairs, n_triggers, p0, family
    Computes per-dyad proximity and returns a tidy DataFrame
    with columns: family, proximity, pA, pB, n_pairs
    """
    df = pd.read_csv(conv_csv_path)

    required = {"n_pairs", "n_triggers", "p0", family_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # base rates
    df["pA"] = df["n_triggers"] / df["n_pairs"].replace(0, np.nan)
    df["pB"] = df["p0"]

    # drop rows without valid denominators
    df = df.dropna(subset=["pA", "pB"]).copy()

    # clamp to [0, 1] for safety
    df["pA"] = df["pA"].clip(0, 1)
    df["pB"] = df["pB"].clip(0, 1)

    # proximity per dyad per family
    df["proximity"] = 1.0 - (df["pA"] - df["pB"]).abs()
    df["proximity"] = df["proximity"].clip(0, 1)

    return df[[family_col, "proximity", "pA", "pB", "n_pairs"]].copy()

def _sem(x):
    x = np.asarray(x)
    n = np.count_nonzero(~np.isnan(x))
    if n <= 1:
        return 0.0
    return np.nanstd(x, ddof=1) / math.sqrt(n)

def plot_family_proximity(df_prox, out_png, title="Mean proximity by family"):
    fam_order = (
        df_prox.groupby("family")["proximity"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    means = df_prox.groupby("family")["proximity"].mean().reindex(fam_order)
    ses = df_prox.groupby("family")["proximity"].apply(_sem).reindex(fam_order)

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(fam_order)), means.values, yerr=ses.values, capsize=3)
    plt.xticks(range(len(fam_order)), fam_order, rotation=30, ha="right")
    plt.ylabel("Proximity")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def write_family_proximity_tables(df_prox, out_csv_unweighted, out_csv_weighted):
    # unweighted by dyad
    unweighted = (
        df_prox.groupby("family")
        .agg(
            mean_proximity=("proximity", "mean"),
            sd=("proximity", "std"),
            n_dyads=("proximity", "size"),
            mean_pA=("pA", "mean"),
            mean_pB=("pB", "mean"),
        )
        .reset_index()
        .sort_values("mean_proximity", ascending=False)
    )
    unweighted.to_csv(out_csv_unweighted, index=False)

    # weighted by number of pairs in the dyad
    w = df_prox.copy()
    w["w"] = w["n_pairs"].clip(lower=1)
    def wmean(x, w):
        x, w = np.asarray(x), np.asarray(w)
        return np.average(x, weights=w)
    weighted = (
        w.groupby("family")
        .apply(lambda g: pd.Series({
            "weighted_mean_proximity": wmean(g["proximity"], g["w"]),
            "weighted_mean_pA": wmean(g["pA"], g["w"]),
            "weighted_mean_pB": wmean(g["pB"], g["w"]),
            "sum_pairs": int(g["w"].sum()),
            "n_dyads": int(len(g)),
        }))
        .reset_index()
        .sort_values("weighted_mean_proximity", ascending=False)
    )
    weighted.to_csv(out_csv_weighted, index=False)

def compute_and_visualise_proximity_from_conv(conv_csv_path, out_dir, label="adjacent"):
    os.makedirs(out_dir, exist_ok=True)
    df_prox = compute_proximity_from_conv_csv(conv_csv_path, family_col="family")

    out_csv_unw = os.path.join(out_dir, f"proximity_by_family_{label}_unweighted.csv")
    out_csv_w   = os.path.join(out_dir, f"proximity_by_family_{label}_weighted.csv")
    write_family_proximity_tables(df_prox, out_csv_unw, out_csv_w)

    out_png = os.path.join(out_dir, f"proximity_by_family_{label}.png")
    plot_family_proximity(df_prox, out_png, title=f"Mean proximity by family [{label}]")

    # optional quick diagnostic scatter of family means versus mean Conv if conv is present
    # you can join on family from the same conv csv if desired
