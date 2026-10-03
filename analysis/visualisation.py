from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.stats import ttest_1samp

# =========================
# Color palette
# =========================
OI_BLUE   = "#3B4992"
OI_ORANGE = "#EE0011"
OI_GREEN  = "#008B45"
OI_RED    = "#EE0011"
OI_PURPLE = "#631879"
OI_BLACK  = "#000000"
OI_GREY   = "#7F7F7F"

# =========================
# Paper style
# =========================
DPI_EXPORT = 100
FONT_PT    = 20
PAPER_RC = {
    "font.size": FONT_PT,
    "font.weight": "bold",           # <-- make all text bold
    "axes.titlesize": FONT_PT + 6,
    "axes.titleweight": "bold",
    "axes.labelsize": FONT_PT + 4,
    "axes.labelweight": "bold",
    "xtick.labelsize": FONT_PT + 2,
    "ytick.labelsize": FONT_PT + 2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "legend.fontsize": FONT_PT,
    "legend.title_fontsize": FONT_PT,
    "figure.titlesize": FONT_PT + 6,
    "savefig.dpi": DPI_EXPORT,
    "axes.linewidth": 1.1,
}
plt.rcParams.update(PAPER_RC)
plt.rcParams["figure.constrained_layout.use"] = True

# --- crisp export settings ---
PNG_DPI = 600                  # for quick previews
EXPORT_PNG = True
EXPORT_PDF = True              # use the PDF in the paper for perfect text and lines

plt.rcParams.update({
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,        # embed TrueType so text looks right in editors
    "ps.fonttype": 42,
    "figure.autolayout": False,
    "patch.antialiased": True,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.title_fontsize": FONT_PT,   # keep legend title normal sized
})

from pathlib import Path

def savefig_all(out_png: Path, fig=None):
    """Save both PNG and PDF for publication quality."""
    if fig is None:
        fig = plt.gcf()
    if EXPORT_PNG:
        fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.03)


# =========================
# Family order and abbreviations
# =========================
FAMILY_ORDER: List[str] = [
    "negations",
    "impersonal_pronouns",
    "quantifiers",
    "auxiliary_verbs",
    "adverbs",
    "personal_pronouns",
    "conjunctions",
    "articles",
    "prepositions",
]

ABBREV_MAP = {
    "negations": "NEG",
    "impersonal_pronouns": "IP",
    "quantifiers": "QUANT",
    "auxiliary_verbs": "AUX",
    "adverbs": "ADV",
    "personal_pronouns": "PP",
    "conjunctions": "CONJ",
    "articles": "ART",
    "prepositions": "PREP",
}

def abbrev_label(name: str) -> str:
    key = str(name).strip().lower()
    if key in ABBREV_MAP:
        return ABBREV_MAP[key]
    # generic fallback to upper initials
    tokens = re.split(r"[\s_]+", key)
    tokens = [t for t in tokens if t]
    return "".join(t[0] for t in tokens).upper() if tokens else str(name).upper()

def ordered_index(seq) -> List[str]:
    # keep only known families in the given order
    items = [s for s in FAMILY_ORDER if s in set(map(str, seq))]
    # append any unknowns at the end in original order
    seen = set(items)
    items.extend([s for s in seq if s not in seen])
    return items

# =========================
# Scalable sizing helpers
# =========================
BASE_H = 4.8
PER_EDGE_ROW = 0.55

CELL_W = 0.36
CELL_H = 0.44
LEFT_MARGIN  = 1.2
RIGHT_MARGIN = 0.6
TOP_MARGIN   = 0.8
BOTTOM_MARGIN = 0.6
CBAR_W = 0.6

def h_rows(n: int, per: float = PER_EDGE_ROW) -> float:
    return min(6.2, max(BASE_H, per * float(n)))

def w_cats(n: int, base: float = 7.2, per: float = 0.32) -> float:
    w = max(base, per * float(n) + 3.6)
    return min(8.0, w)

def heat_figsize(n_rows: int, n_cols: int) -> Tuple[float, float]:
    w = LEFT_MARGIN + n_cols * CELL_W + CBAR_W + RIGHT_MARGIN
    h = TOP_MARGIN   + n_rows * CELL_H + BOTTOM_MARGIN
    return (min(8.0, max(6.4, w)), min(6.4, max(4.8, h)))

# =========================
# CSV helpers
# =========================
def csv_target(out_png: Path, csv_dir: Optional[Path]) -> Path:
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)
        return (csv_dir / out_png.name).with_suffix(".csv")
    return out_png.with_suffix(".csv")

def csv_named(out_png: Path, csv_dir: Optional[Path], new_name: str) -> Path:
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)
        return csv_dir / new_name
    return out_png.with_name(new_name)

def export_family_key(names, out_csv: Path):
    uniq = pd.Series(sorted(set(map(str, names))))
    df_map = pd.DataFrame({
        "family_full": uniq,
        "family_abbrev": [abbrev_label(x) for x in uniq]
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_map.to_csv(out_csv, index=False, encoding="utf-8-sig")

# =========================
# Gender levels
# =========================
def plot_gender_levels(
    pf: pd.DataFrame,
    out_png: Path,
    who: str = "responder",
    n_perm: int = 2000,
    csv_dir: Optional[Path] = None,
    play_title: Optional[str] = None,
) -> None:
    if pf is None or pf.empty:
        return
    side_col = "b_gender" if who == "responder" else "a_gender"
    if side_col not in pf.columns:
        return

    df = pf.dropna(subset=[side_col, "family", "conv"]).copy()
    df[side_col] = df[side_col].astype(str).str.upper().str[0]
    df = df[df[side_col].isin(["F", "M"])]
    if df.empty:
        return

    # enforce family order
    df["family"] = df["family"].astype(str)
    fam_ord = ordered_index(df["family"].unique())

    g = df.groupby(["family", side_col])["conv"].mean().unstack(side_col)
    if g is None or g.empty:
        return
    g = g.reindex(index=fam_ord)
    cols = [c for c in ["F", "M"] if c in g.columns]
    if not cols:
        return
    g_plot = g[cols].fillna(0.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(w_cats(len(g_plot.index)), h_rows(len(g_plot))))
    x = np.arange(len(g_plot.index))
    w = 0.42 if len(cols) == 2 else 0.6

    gender_labels = {"F": "FEMALE", "M": "MALE"}
    gender_colors = {"F": OI_PURPLE, "M": OI_ORANGE}

    for i, c in enumerate(cols):
        plt.bar(
            x + i * w,
            g_plot[c].to_numpy(),
            width=w,
            label=gender_labels.get(c, c),
            color=gender_colors.get(c, plt.get_cmap("cividis")(0.6)),
            edgecolor=OI_BLACK,
            linewidth=0.7,
            alpha=0.95,
        )

    xlabels = [abbrev_label(xx) for xx in g_plot.index]
    plt.xticks(x + w * (len(cols) - 1) / 2, xlabels, rotation=30, ha="right")
    plt.tick_params(axis="x", pad=4)
    plt.subplots_adjust(bottom=0.22)
    plt.ylabel("Average convergence  Conv(t)")
    base = "Responder gender effect" if who == "responder" else "Initiator gender effect"
    plt.title(base + (f"  ({play_title})" if play_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.legend(title="GENDER", frameon=False, title_fontsize=FONT_PT-8)
    plt.tight_layout()
    savefig_all(out_png)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    g_out = g_plot.reset_index().rename(columns={"index": "family"})
    g_out = g_out.assign(play=play_title if play_title else "")
    g_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # differences and permutation p
    fam_order = list(g_plot.index)
    diffs_FM = []
    pvals = []
    rng = np.random.default_rng(42)

    for fam in fam_order:
        fam_rows = df[df["family"] == fam]
        f_vals = fam_rows[fam_rows[side_col] == "F"]["conv"].to_numpy()
        m_vals = fam_rows[fam_rows[side_col] == "M"]["conv"].to_numpy()
        if f_vals.size == 0 and m_vals.size == 0:
            diffs_FM.append(np.nan)
            pvals.append(np.nan)
            continue
        d_obs = float(np.nanmean(f_vals) - np.nanmean(m_vals))
        diffs_FM.append(d_obs)

        labels = fam_rows[side_col].to_numpy()
        vals = fam_rows["conv"].to_numpy()
        mask = np.isin(labels, ["F", "M"]) & np.isfinite(vals)
        labels = labels[mask]
        vals = vals[mask]
        if labels.size == 0 or (labels == "F").sum() == 0 or (labels == "M").sum() == 0:
            pvals.append(np.nan)
        else:
            nF = int((labels == "F").sum())
            count = 0
            for _ in range(n_perm):
                perm = rng.permutation(labels.size)
                f_idx = perm[:nF]
                m_idx = perm[nF:]
                diff_perm = vals[f_idx].mean() - vals[m_idx].mean()
                if abs(diff_perm) >= abs(d_obs):
                    count += 1
            pvals.append((count + 1) / (n_perm + 1))

    diff_df = pd.DataFrame({
        "family": fam_order,
        "diff_F_minus_M": diffs_FM,
        "diff_M_minus_F": [-d if np.isfinite(d) else np.nan for d in diffs_FM],
        "perm_p_two_sided": pvals,
        "play": play_title if play_title else "",
    })

    diff_out_png_FM = out_png.with_name(out_png.stem + "_diff_F_minus_M.png")
    plt.figure(figsize=(11.0, h_rows(len(diff_df))))
    plt.barh([abbrev_label(x) for x in diff_df["family"]][::-1],
             diff_df["diff_F_minus_M"].to_numpy()[::-1],
             alpha=0.95, color=OI_GREEN, edgecolor=OI_BLACK, linewidth=0.7)
    plt.axvline(0, color=OI_BLACK, linewidth=1.2, linestyle="--", alpha=0.85)
    plt.xlabel("Difference in convergence  F minus M")
    plt.title("Gender difference in convergence  positive means women adapt more"
              + (f"  ({play_title})" if play_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(diff_out_png_FM, bbox_inches="tight")
    plt.close()

    diff_out_png_MF = out_png.with_name(out_png.stem + "_diff_M_minus_F.png")
    plt.figure(figsize=(11.0, h_rows(len(diff_df))))
    plt.barh([abbrev_label(x) for x in diff_df["family"]][::-1],
             diff_df["diff_M_minus_F"].to_numpy()[::-1],
             alpha=0.95, color=OI_BLUE, edgecolor=OI_BLACK, linewidth=0.7)
    plt.axvline(0, color=OI_BLACK, linewidth=1.2, linestyle="--", alpha=0.85)
    plt.xlabel("Difference in convergence  M minus F")
    plt.title("Gender difference in convergence  positive means men adapt more"
              + (f"  ({play_title})" if play_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(diff_out_png_MF, bbox_inches="tight")
    plt.close()

    diff_csv = csv_named(out_png, csv_dir, out_png.with_suffix(".diffs.csv").name)
    diff_df.to_csv(diff_csv, index=False, encoding="utf-8-sig")

# =========================
# Importance levels
# =========================
def plot_importance_levels(
    pf: pd.DataFrame,
    out_png: Path,
    who: str = "initiator",
    csv_dir: Optional[Path] = None,
    play_title: Optional[str] = None,
) -> None:
    if pf is None or pf.empty:
        return
    cat_col = "a_category" if who == "initiator" else "b_category"
    if cat_col not in pf.columns:
        return

    df = pf.dropna(subset=["family", "conv"]).copy()
    df[cat_col] = (df[cat_col].astype(str).str.strip().str.lower()
                   .replace({"maj": "major", "main": "major", "primary": "major",
                             "int": "intermediate", "mid": "intermediate",
                             "min": "minor"}))
    levels = ["major", "intermediate", "minor"]
    df = df[df[cat_col].isin(levels)]
    if df.empty:
        return

    df["family"] = df["family"].astype(str)
    fam_ord = ordered_index(df["family"].unique())

    g = df.groupby(["family", cat_col])["conv"].mean().unstack(cat_col)
    if g is None or g.empty:
        return
    present = [c for c in levels if c in g.columns]
    if not present:
        return
    g = g[present].fillna(0.0).reindex(index=fam_ord)

    colors = {"major": OI_BLUE, "intermediate": OI_ORANGE, "minor": OI_GREEN}

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(w_cats(len(g.index)), h_rows(len(g))))
    x = np.arange(len(g.index))
    w = 0.30
    for i, c in enumerate(present):
        plt.bar(x + i*w, g[c].to_numpy(), width=w, label=c.upper(),
                color=colors.get(c, plt.get_cmap("cividis")(0.6)),
                edgecolor=OI_BLACK, linewidth=0.7, alpha=0.95)

    xlabels = [abbrev_label(xx) for xx in g.index]
    plt.xticks(x + w*(len(present)-1)/2, xlabels, rotation=30, ha="right")
    plt.tick_params(axis="x", pad=4)
    plt.subplots_adjust(bottom=0.22)
    plt.ylabel("Average convergence  Conv(t)")
    base = "Initiator importance effect" if who == "initiator" else "Responder importance effect"
    plt.title(base + (f"  ({play_title})" if play_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.legend(title="IMPORTANCE", frameon=False, title_fontsize=FONT_PT-8)
    plt.tight_layout()
    savefig_all(out_png)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    g_out = g.reset_index().assign(play=play_title if play_title else "")
    g_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

# =========================
# Two mode edge heatmaps
# =========================
def plot_edge_category_heatmap(
    edges: pd.DataFrame,
    out_png: Path,
    title: str,
    csv_dir: Optional[Path] = None,
    play_title: Optional[str] = None,
) -> None:
    if edges is None or edges.empty:
        return
    if not {"a_category", "b_category"}.issubset(edges.columns):
        return

    weight_col = next((c for c in ["mean_conv_w", "weight_work", "mean_conv"] if c in edges.columns), None)
    if weight_col is None:
        return

    df = edges.copy()
    norm_names = {"maj": "major", "main": "major", "primary": "major",
                  "int": "intermediate", "mid": "intermediate",
                  "min": "minor"}
    for c in ["a_category", "b_category"]:
        df[c] = df[c].astype(str).str.strip().str.lower().replace(norm_names)
    levels = ["major", "intermediate", "minor"]
    df = df[df["a_category"].isin(levels) & df["b_category"].isin(levels)]

    mean_p = pd.DataFrame(index=levels, columns=levels, dtype=float)
    n_p    = pd.DataFrame(index=levels, columns=levels, dtype=int)
    for a in levels:
        for b in levels:
            vals = pd.to_numeric(df.loc[(df["a_category"] == a) & (df["b_category"] == b), weight_col],
                                 errors="coerce").dropna().to_numpy()
            n_p.loc[a, b] = int(vals.size)
            mean_p.loc[a, b] = float(vals.mean()) if vals.size else np.nan

    disp = [s.capitalize() for s in levels]
    mean_p.index = disp; mean_p.columns = disp
    n_p.index  = disp; n_p.columns  = disp

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=heat_figsize(3, 3), constrained_layout=True)
    M = mean_p.to_numpy()
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.0
    vlim = vmax if vmax > 0 else 0.1
    cmap = plt.get_cmap("PiYG").copy()
    cmap.set_bad(color="#D9D9D9")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")

    ax.set_xticks(range(len(disp)), disp)
    ax.set_yticks(range(len(disp)), disp)
    ax.set_xticks(np.arange(-0.5, len(disp), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(disp), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Removed text annotation from cells
    for i, ia in enumerate(disp):
        for j, jb in enumerate(disp):
            mu = mean_p.loc[ia, jb]
            if not np.isfinite(mu):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, hatch="///", edgecolor=OI_BLACK, linewidth=0.7, alpha=0.6))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.3f")
    cbar.set_label("Mean edge Conv  weighted when available")
    cbar.ax.tick_params(labelsize=FONT_PT - 1)

    ax.set_title(f"{title}" + (f"  ({play_title})" if play_title else ""))
    ax.set_xlabel("Responder importance  B")
    ax.set_ylabel("Initiator importance  A")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close()

    tall = []
    for a in disp:
        for b in disp:
            tall.append({
                "Initiator importance  A": a,
                "Responder importance  B": b,
                "mean": mean_p.loc[a, b],
                "n": n_p.loc[a, b],
                "play": play_title if play_title else "",
            })
    pd.DataFrame(tall).to_csv(csv_target(out_png, csv_dir), index=False, encoding="utf-8-sig")


def plot_edge_gender_heatmap(
    edges: pd.DataFrame,
    out_png: Path,
    title: str,
    csv_dir: Optional[Path] = None,
    play_title: Optional[str] = None,
) -> None:
    if edges is None or edges.empty:
        return
    if not {"a_gender", "b_gender"}.issubset(edges.columns):
        return

    weight_col = next((c for c in ["mean_conv_w", "weight_work", "mean_conv"] if c in edges.columns), None)
    if weight_col is None:
        return

    df = edges.copy()
    df["a_gender"] = df["a_gender"].astype(str).str.upper().str[0]
    df["b_gender"] = df["b_gender"].astype(str).str.upper().str[0]
    df = df[df["a_gender"].isin(["F", "M"]) & df["b_gender"].isin(["F", "M"])]

    levels = ["F", "M"]
    mean_p = pd.DataFrame(index=levels, columns=levels, dtype=float)
    n_p    = pd.DataFrame(index=levels, columns=levels, dtype=int)
    for a in levels:
        for b in levels:
            vals = pd.to_numeric(df.loc[(df["a_gender"] == a) & (df["b_gender"] == b), weight_col],
                                 errors="coerce").dropna().to_numpy()
            n_p.loc[a, b] = int(vals.size)
            mean_p.loc[a, b] = float(vals.mean()) if vals.size else np.nan

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=heat_figsize(2, 2), constrained_layout=True)
    M = mean_p.to_numpy()
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.0
    vlim = vmax if vmax > 0 else 0.1
    cmap = plt.get_cmap("PiYG").copy()
    cmap.set_bad(color="#D9D9D9")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")

    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(levels)), levels)
    ax.set_xticks(np.arange(-0.5, len(levels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(levels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Removed the text annotation block inside the cells
    for i, ia in enumerate(levels):
        for j, jb in enumerate(levels):
            mu = mean_p.loc[ia, jb]
            if not np.isfinite(mu):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, hatch="///", edgecolor=OI_BLACK, linewidth=0.7, alpha=0.6))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.3f")
    cbar.set_label("Mean edge Conv  weighted when available")
    cbar.ax.tick_params(labelsize=FONT_PT - 1)

    ax.set_title(f"{title}" + (f"  ({play_title})" if play_title else ""))
    ax.set_xlabel("Responder gender  B")
    ax.set_ylabel("Initiator gender  A")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close()

    out_rows = []
    for a in levels:
        for b in levels:
            out_rows.append({"Initiator gender  A": a,
                             "Responder gender  B": b,
                             "mean": mean_p.loc[a, b],
                             "n": n_p.loc[a, b],
                             "play": play_title if play_title else ""})
    pd.DataFrame(out_rows).to_csv(csv_target(out_png, csv_dir), index=False, encoding="utf-8-sig")


# =========================
# Per family p0 vs p1
# =========================
def plot_family_p1_p0(
    pf: pd.DataFrame,
    out_png: Path,
    title: str,
    alpha: float = 0.05,
    csv_dir: Optional[Path] = None,
    play_title: Optional[str] = None,
) -> None:
    if pf is None or pf.empty or not {"family", "p0", "p1"}.issubset(pf.columns):
        return

    rows = []
    for fam, g in pf.groupby("family", dropna=False):
        per_pair = pd.DataFrame({
            "p0": pd.to_numeric(g["p0"], errors="coerce"),
            "p1": pd.to_numeric(g["p1"], errors="coerce"),
        }).dropna()
        if per_pair.empty:
            continue

        p0 = per_pair["p0"].to_numpy()
        p1 = per_pair["p1"].to_numpy()
        diffs = p1 - p0

        mu0 = float(p0.mean())
        mu1 = float(p1.mean())

        try:
            t_stat, p_one = ttest_1samp(diffs, 0.0, alternative="greater")
        except TypeError:
            t_stat, p_two = ttest_1samp(diffs, 0.0)
            p_one = p_two / 2 if np.nanmean(diffs) > 0 else 1.0

        rows.append({
            "family": fam,
            "mean_p0": mu0,
            "mean_p1": mu1,
            "diff": mu1 - mu0,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_val_ttest_one_sided": float(p_one) if np.isfinite(p_one) else np.nan,
            "significant_ttest": bool(np.isfinite(p_one) and p_one < alpha),
            "n_pairs": int(diffs.size),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    # enforce family order
    df["family"] = df["family"].astype(str)
    order_df = ordered_index(df["family"].unique())
    # keep only those present in FAMILY_ORDER first
    df["__ord__"] = df["family"].apply(lambda x: order_df.index(x) if x in order_df else 10_000)
    df = df.sort_values("__ord__").drop(columns="__ord__").reset_index(drop=True)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(w_cats(len(df)), h_rows(len(df))))
    x = np.arange(len(df))
    w = 0.36

    c0 = OI_BLUE
    c1 = "#8A91C7"

    plt.bar(x - w/2, df["mean_p0"], width=w, color=c0, edgecolor=OI_BLACK, linewidth=0.7, label="P(B)")
    plt.bar(x + w/2, df["mean_p1"], width=w, color=c1, edgecolor=OI_BLACK, linewidth=0.7, label="P(B|A)")

    y_top = np.maximum(df["mean_p0"], df["mean_p1"]).to_numpy()
    for i, sig in enumerate(df["significant_ttest"].tolist()):
        if sig:
            plt.text(i, y_top[i] * 1.02, "*", ha="center", va="bottom", fontsize=FONT_PT)

    labels = [abbrev_label(x) for x in df["family"]]
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.tick_params(axis="x", pad=4)
    plt.subplots_adjust(bottom=0.22)
    plt.ylabel("Probability")
    plt.ylim(0, max(0.85, float(np.nanmax(np.c_[df["mean_p0"], df["mean_p1"]])) * 1.12))
    plt.title(f"{title}" + (f"  ({play_title})" if play_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.legend(frameon=False, loc="upper left", title_fontsize=FONT_PT-8)
    plt.tight_layout()
    savefig_all(out_png)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    df.assign(play=play_title if play_title else "").to_csv(out_csv, index=False, encoding="utf-8-sig")

# =========================
# Condition comparisons
# =========================
def plot_condition_comparison(
    pf_adj: Optional[pd.DataFrame],
    pf_non: Optional[pd.DataFrame],
    pf_rand: Optional[pd.DataFrame],
    out_png: Path,
    title: str,
    csv_dir: Optional[Path] = None,
    play_title: Optional[str] = None,
) -> None:
    frames = []
    if pf_adj is not None and not pf_adj.empty:
        frames.append(pf_adj.groupby("family")["conv"].mean().rename("Adjacent").to_frame())
    if pf_non is not None and not pf_non.empty:
        frames.append(pf_non.groupby("family")["conv"].mean().rename("Nonadjacent").to_frame())
    if pf_rand is not None and not pf_rand.empty:
        frames.append(pf_rand.groupby("family")["conv"].mean().rename("Randomized").to_frame())
    if not frames:
        return

    M = pd.concat(frames, axis=1).dropna(how="all")
    if M.empty:
        return

    # enforce family order
    fam_ord = ordered_index(M.index.tolist())
    M = M.reindex(index=fam_ord)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(w_cats(len(M.index)), h_rows(len(M))))
    xs = np.arange(len(M.index))
    w = 0.28
    palette = {"Adjacent": OI_BLUE, "Nonadjacent": OI_ORANGE, "Randomized": OI_GREEN}
    cols = [c for c in ["Adjacent", "Nonadjacent", "Randomized"] if c in M.columns]

    for i, c in enumerate(cols):
        plt.bar(xs + i*w, M[c].to_numpy(), width=w, label=c.upper(), alpha=0.95,
                color=palette.get(c, plt.get_cmap("cividis")(0.6)),
                edgecolor=OI_BLACK, linewidth=0.7)

    labels = [abbrev_label(x) for x in M.index]
    plt.xticks(xs + w, labels, rotation=30, ha="right")
    plt.tick_params(axis="x", pad=4)
    plt.subplots_adjust(bottom=0.22)
    plt.ylabel("Mean Conv(t)")
    plt.title(f"{title}" + (f"  ({play_title})" if play_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.legend(frameon=False, title_fontsize=FONT_PT-8)
    plt.tight_layout()
    savefig_all(out_png)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    M.assign(play=play_title if play_title else "").to_csv(out_csv, encoding="utf-8-sig")

# =========================
# Runner
# =========================
def run_visuals_for_play(
    in_dir: Path,
    out_root: Path,
    mode: str,
    min_support: int = 10,   # kept for compatibility
    max_edges: int = 50,     # kept for compatibility
    top_k: int = 20,         # kept for compatibility
    csv_subdir: str = "tables",
    play_title: Optional[str] = None,
) -> None:
    print(f"[viz] Running visualisation for {in_dir}  {mode}")

    pf_path = in_dir / f"conv_by_pair_feature_{mode}.csv"
    edge_path = in_dir / f"conv_edge_summary_{mode}.csv"
    pf_non_path = in_dir / f"conv_by_pair_feature_{mode}_nonadjacent.csv"
    pf_rand_path = in_dir / f"conv_by_pair_feature_{mode}_randomized.csv"

    if not pf_path.exists() or not edge_path.exists():
        print("[viz] Missing inputs, skipping.")
        return

    def _safe_read(p):
        try:
            return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None
        except pd.errors.EmptyDataError:
            return None

    pf = _safe_read(pf_path)
    edges = _safe_read(edge_path)
    if pf is None or edges is None:
        print("[viz] Empty or missing core inputs, skipping.")
        return
    pf_non = _safe_read(pf_non_path)
    pf_rand = _safe_read(pf_rand_path)

    out_root.mkdir(parents=True, exist_ok=True)
    csv_dir = out_root / csv_subdir

    # export family abbreviation key once
    try:
        export_family_key(pf["family"].dropna().unique(), csv_dir / "family_abbrev_key.csv")
    except Exception:
        pass

    # gender effects
    plot_gender_levels(
        pf, out_root / "gender_levels_responder.png",
        who="responder", csv_dir=csv_dir, play_title=None
    )
    plot_gender_levels(
        pf, out_root / "gender_levels_initiator.png",
        who="initiator", csv_dir=csv_dir, play_title=None
    )

    # importance effects
    plot_importance_levels(
        pf, out_root / "importance_levels_initiator.png",
        who="initiator", csv_dir=csv_dir, play_title=None
    )
    plot_importance_levels(
        pf, out_root / "importance_levels_responder.png",
        who="responder", csv_dir=csv_dir, play_title=None
    )

    # condition comparison
    plot_condition_comparison(
        pf, pf_non, pf_rand, out_root / "adj_non_rand.png",
        "Family means across conditions", csv_dir=csv_dir, play_title=None
    )

    # two mode edge heatmaps
    plot_edge_gender_heatmap(
        edges, out_root / "edge_heatmap_gender.png",
        "Mean edge convergence by gender  A to B", csv_dir=csv_dir, play_title=None
    )
    plot_edge_category_heatmap(
        edges, out_root / "edge_heatmap_importance.png",
        "Mean edge convergence by importance  A to B", csv_dir=csv_dir, play_title=None
    )

    # per family p0 vs p1
    plot_family_p1_p0(
        pf, out_root / "family_p0_p1.png",
        "P (B_f)  vs  P (B_f | A_f)  by family", csv_dir=csv_dir, play_title=None
    )

    print(f"[viz] Visuals written to {out_root}  CSVs in {csv_dir}")
