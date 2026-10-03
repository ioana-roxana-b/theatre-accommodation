from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, FancyBboxPatch

import matplotlib as mpl
import matplotlib.pyplot as plt

def use_big_style(font_pt: int = 28, save_dpi: int = 600, show_dpi: int = 180):
    mpl.rcParams.update({
        "font.size": font_pt,
        "font.weight": "bold",
        "axes.titlesize": font_pt + 8,
        "axes.titleweight": "bold",
        "axes.labelsize": font_pt + 6,
        "axes.labelweight": "bold",
        "xtick.labelsize": font_pt + 2,
        "ytick.labelsize": font_pt + 2,
        "legend.fontsize": font_pt,
        "legend.title_fontsize": font_pt + 2,
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "figure.dpi": show_dpi,     # on screen preview size
        "savefig.dpi": save_dpi,    # export sharpness
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    plt.rcParams["figure.constrained_layout.use"] = True

use_big_style(font_pt=30, save_dpi=800, show_dpi=220)


def load_edges(edges_csv: Path, prefer_weight_col: str = "mean_conv_w") -> pd.DataFrame:
    """
    Load edge-level convergence results and pick the working weight column.

    Params:
        edges_csv: Path to conv_edge_summary_*.csv produced by the convergence step.
        prefer_weight_col: Name of the weight column to prefer when available.

    Returns:
        DataFrame with columns a_speaker b_speaker weight_work and any available a_* / b_* metadata.
    """
    df = pd.read_csv(edges_csv, encoding="utf-8-sig")

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    ren = {}
    if "A" in df.columns and "a_speaker" not in df.columns:
        ren["A"] = "a_speaker"
    if "B" in df.columns and "b_speaker" not in df.columns:
        ren["B"] = "b_speaker"
    if ren:
        df = df.rename(columns=ren)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()

    # Choose the working weight column
    weight_col = None
    if prefer_weight_col in df.columns:
        weight_col = prefer_weight_col
    elif "mean_conv" in df.columns:
        weight_col = "mean_conv"
    else:
        for cand in ["conv_mean_unw", "conv_mean_w", "conv_mean"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "mean_conv"})
                weight_col = "mean_conv"
                break
    if weight_col is None:
        raise ValueError("No edge weight column found. Need mean_conv_w or mean_conv")

    required = {"a_speaker", "b_speaker", weight_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in edges CSV: {sorted(missing)}")

    keep = ["a_speaker", "b_speaker", weight_col, "n_pairs", "n_features", "mean_conv", "mean_conv_w", "n_triggers_total"]
    keep += [c for c in df.columns if c.startswith("a_") or c.startswith("b_")]
    keep = pd.Index(keep).drop_duplicates().tolist()
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df.dropna(subset=[weight_col, "a_speaker", "b_speaker"])

    if "n_pairs" in df.columns:
        df = df[df["n_pairs"].fillna(0) > 0].copy()

    for col in ("a_speaker", "b_speaker"):
        df[col] = df[col].astype(str).map(lambda s: " ".join(s.split()))

    df = df.rename(columns={weight_col: "weight_work"})
    return df


def build_graph(
    edges: pd.DataFrame,
    positive_only: bool = False,
    weight_col: str = "weight_work",
) -> nx.DiGraph:
    """
    Build a directed graph where edges A->B are weighted by convergence.

    Params:
        edges: Edge table from load_edges with a_speaker b_speaker and weight_work.
        positive_only: Keep only edges with positive weight.
        weight_col: Column name to use as weight.

    Returns:
        A directed NetworkX DiGraph with edge attribute weight.
    """
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        a = str(r["a_speaker"]).strip()
        b = str(r["b_speaker"]).strip()
        w = float(r.get(weight_col, 0.0))
        if positive_only and w <= 0:
            continue
        G.add_edge(a, b, weight=w)
    return G


def compute_metrics(G: nx.DiGraph) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute node metrics and global graph statistics.

    Params:
        G: Directed graph with edge attribute weight.

    Returns:
        A tuple of
        First is a DataFrame of node metrics including strength and betweenness
        Second is a dict with n_nodes n_edges density and reciprocity.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty after filtering")

    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())

    out_strength = {n: 0.0 for n in G.nodes()}
    in_strength = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        aw = abs(w)
        out_strength[u] += aw
        in_strength[v] += aw

    # Weighted betweenness: transform weights to lengths inline
    if G.number_of_edges() > 0:
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 0.0))
            length = 1.0 / (abs(w) + 1e-9)
            H.add_edge(u, v, length=length)
        btw_w = nx.betweenness_centrality(H, normalized=True, weight="length")
    else:
        btw_w = {n: 0.0 for n in G.nodes()}

    btw = nx.betweenness_centrality(G, normalized=True) if G.number_of_edges() > 0 else {n: 0.0 for n in G.nodes()}

    df_nodes = pd.DataFrame({
        "character": list(G.nodes()),
        "out_degree": [out_deg.get(n, 0) for n in G.nodes()],
        "in_degree": [in_deg.get(n, 0) for n in G.nodes()],
        "adaptability_out_strength": [out_strength.get(n, 0.0) for n in G.nodes()],
        "adaptability_in_strength": [in_strength.get(n, 0.0) for n in G.nodes()],
        "betweenness": [btw.get(n, 0.0) for n in G.nodes()],
        "betweenness_weighted": [btw_w.get(n, 0.0) for n in G.nodes()],
    })

    genders = nx.get_node_attributes(G, "gender")
    if genders:
        df_nodes["gender"] = df_nodes["character"].map(genders)
    cats = nx.get_node_attributes(G, "category")
    if cats:
        df_nodes["category"] = df_nodes["character"].map(cats)

    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "reciprocity": nx.reciprocity(G) if G.number_of_edges() > 1 else np.nan,
    }
    return df_nodes.sort_values("adaptability_out_strength", ascending=False), stats


def detect_communities(G: nx.DiGraph) -> pd.DataFrame:
    """
    Detect undirected communities via greedy modularity.

    Params:
        G: Directed graph to project to undirected for community detection.

    Returns:
        DataFrame with columns character and community_id.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["character", "community_id"])
    UG = G.to_undirected(reciprocal=False)
    from networkx.algorithms import community
    comms = list(community.greedy_modularity_communities(UG, weight=None))
    mapping = {n: cid for cid, nodes in enumerate(comms) for n in nodes}
    return pd.DataFrame({"character": list(mapping.keys()),
                         "community_id": list(mapping.values())})


def reciprocity_pair_table(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Build a paired table of A->B and B->A weights for mutual dyads.

    Params:
        edges: Edge DataFrame with a_speaker b_speaker and weight_work.

    Returns:
        DataFrame with columns a_speaker b_speaker w_ab w_ba for mutual edges only.
    """
    df = edges[["a_speaker", "b_speaker", "weight_work"]].copy()
    df = df.rename(columns={"weight_work": "w_ab"})
    df_rev = edges[["a_speaker", "b_speaker", "weight_work"]].copy()
    df_rev = df_rev.rename(columns={"a_speaker": "b_speaker", "b_speaker": "a_speaker", "weight_work": "w_ba"})
    return df.merge(df_rev, on=["a_speaker", "b_speaker"], how="inner")


def plot_reciprocity(paired: pd.DataFrame, out_png: Path, out_csv: Path, play_title: Optional[str] = None) -> Tuple[float, float]:
    if paired.empty:
        paired.to_csv(out_csv, index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(8.0, 6.2))
        ax.text(0.5, 0.5, "No mutual dyads", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_png, bbox_inches="tight", dpi=260)
        plt.close(fig)
        return np.nan, np.nan

    r, _ = pearsonr(paired["w_ab"], paired["w_ba"])
    sign_match = float(np.mean(np.sign(paired["w_ab"]) == np.sign(paired["w_ba"])))

    point_color = "#3B4992"
    diagonal_color = "#4D4D4D"
    grid_color = "#BFBFBF"

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.scatter(
        paired["w_ab"], paired["w_ba"],
        s=26, alpha=0.8,
        facecolor=point_color,
        edgecolor="white", linewidth=0.5
    )

    lim = max(abs(paired["w_ab"]).max(), abs(paired["w_ba"]).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", linewidth=1.3, color=diagonal_color)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("A -> B weight")
    ax.set_ylabel("B -> A weight")
    base = f"Reciprocity r={r:.3f}  sign match={sign_match:.3f}"
    ax.set_title(f"{play_title} — {base}" if play_title else base)

    ax.grid(True, linestyle=":", linewidth=0.9, color=grid_color, alpha=0.85)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=260)
    plt.close(fig)

    paired.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return float(r), float(sign_match)

def plot_graph_simple_bidirectional(
    G: nx.DiGraph,
    out_png: Path,
    color_attr: str = "gender",
    show_labels: bool = True,
    figsize: Tuple[float, float] = (36.0, 24.0),   # larger canvas for LaTeX
    font_size: int = 42,                           # readable at print scale
    title: Optional[str] = None,
) -> None:
    """
    Plot a bidirectional convergence network with:
    - uniform node size
    - shortened names (max two words)
    - large layout and external bottom-left legend
    - arrows pointing near nodes, not overlapping
    """
    if G.number_of_nodes() == 0:
        return

    # --- simplify names (keep only first two words) ---
    def clean_name(n: str) -> str:
        parts = str(n).strip().split()
        return " ".join(parts[:2]) if len(parts) > 2 else " ".join(parts)

    G = nx.relabel_nodes(G, clean_name)

    # --- node colors ---
    attr_values = nx.get_node_attributes(G, color_attr)
    palette_nodes = {"F": "#EE0011", "M": "#3B4992"}
    node_colors = [palette_nodes.get(attr_values.get(n, "U"), "#999999") for n in G.nodes()]

    # --- uniform node size ---
    node_size = 2000
    node_radius_pt = np.sqrt(node_size / np.pi)

    # --- layout ---
    pos = nx.spring_layout(G, seed=7, k=1.15, iterations=150)

    edge_color = "#4D4D4D"

    def edge_width(w: float) -> float:
        return 2.0 + 6.0 * min(abs(w), 0.15) / 0.15

    fig, ax = plt.subplots(figsize=figsize)

    # --- nodes ---
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="#4D4D4D",
        linewidths=1.5,
        ax=ax,
        # zorder=2.2,
    )

    # --- edges with arrows pointing near nodes ---
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        a = FancyArrowPatch(
            pos[u],
            pos[v],
            arrowstyle="-|>",
            mutation_scale=36,            # big arrowheads
            shrinkA=node_radius_pt + 6,   # gap before node
            shrinkB=node_radius_pt + 8,
            connectionstyle="arc3,rad=0.0",
            lw=edge_width(w),
            color=edge_color,
            alpha=0.9,
            zorder=1.8,
        )
        if w < 0:
            a.set_linestyle((0, (3, 3)))
        import matplotlib.patheffects as pe
        a.set_path_effects([
            pe.Stroke(linewidth=a.get_linewidth() + 1.2, foreground="white"),
            pe.Normal(),
        ])
        ax.add_patch(a)

    # --- labels ---
    if show_labels:
        label_pos = {n: (x, y + 0.025) for n, (x, y) in pos.items()}
        texts = nx.draw_networkx_labels(
            G,
            label_pos,
            labels={n: str(n) for n in G.nodes()},
            font_size=font_size + 8,
            font_weight="bold",
            font_color="black",
            ax=ax,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#BFBFBF", alpha=0.9),
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=False,
        )
        for t in texts.values():
            t.set_zorder(3.5)

    # --- legend (outside bottom-left) ---
    handles_nodes = []
    present_attrs = sorted(set(v for v in attr_values.values() if v in palette_nodes))
    for k in present_attrs:
        handles_nodes.append(
            Line2D([0], [0], marker="o", linestyle="", markersize=14,
                   markerfacecolor=palette_nodes[k], markeredgecolor="#4D4D4D", label=k)
        )
    edge_handles = [
        Line2D([0], [0], color=edge_color, lw=4.5, linestyle="-", label="Convergence"),
        Line2D([0], [0], color=edge_color, lw=4.5, linestyle="--", label="Divergence"),
    ]
    all_handles = edge_handles + handles_nodes
    if all_handles:
        leg = ax.legend(
            handles=all_handles,
            title=None,
            loc="lower left",
            bbox_to_anchor=(-0.05, -0.05),  # push outside figure area
            frameon=False,
            handlelength=2.8,
            handletextpad=0.8,
            borderaxespad=0.8,
            fontsize=font_size + 10,
        )

    # --- title ---
    if title:
        ax.set_title(title, fontweight="bold", fontsize=font_size + 15, pad=35)

    ax.set_axis_off()

    # reduce margins so graph is centered
    fig.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.08)

    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def subgraph_direction_by_pair(G: nx.DiGraph, which: str) -> nx.DiGraph:
    """
    Split each unordered pair {u,v} into exactly one plotted direction per figure.
    which='AtoB' keeps only min(u,v) -> max(u,v)
    which='BtoA' keeps only max(u,v) -> min(u,v)
    """
    assert which in ("AtoB", "BtoA")
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))

    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        u0, v0 = (u, v) if u < v else (v, u)
        if which == "AtoB":
            keep = (u == u0) and (v == v0)
        else:
            keep = (u == v0) and (v == u0)
        if keep:
            H.add_edge(u, v, **d)
    return H

def plot_graph_direction_only(
    G: nx.DiGraph,
    out_png: Path,
    direction: str = "AtoB",  # "AtoB" or "BtoA"
    color_attr: str = "gender",
    show_labels: bool = True,
    figsize: Tuple[float, float] = (14.5, 10.5),
    font_size: int = 11,
    play_title: Optional[str] = None,
) -> None:
    H = subgraph_direction_by_pair(G, "AtoB" if direction == "AtoB" else "BtoA")
    base = "Convergence network — A->B only (per pair forward)" if direction == "AtoB" \
           else "Convergence network — B->A only (per pair reverse)"
    title = f"{play_title} — {base}" if play_title else base
    plot_graph_simple_bidirectional(
        H,
        out_png=out_png,
        color_attr=color_attr,
        show_labels=show_labels,
        figsize=figsize,
        font_size=font_size,
        title=title,
    )

def plot_graph_communities_simple(
    G: nx.DiGraph,
    out_png: Path,
    community_attr: str = "community_id",
    show_labels: bool = True,
    figsize: Tuple[float, float] = (12.5, 8.5),
    font_size: int = 10,
    title: Optional[str] = None,
) -> None:
    if G.number_of_nodes() == 0:
        return

    G = nx.relabel_nodes(G, lambda n: " ".join(str(n).split()))
    pos = nx.spring_layout(G, seed=7, k=1.15, iterations=100)

    # outgoing strength for node size
    out_strength = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        out_strength[u] += abs(float(d.get("weight", 0.0)))
    node_sizes = {n: 220.0 + 300.0 * np.sqrt(max(out_strength.get(n, 0.0), 0.0)) for n in G.nodes()}

    # stronger community hull colors
    hull_fills = [
        "#A8BAD5", "#F5A9A9", "#A8D5B5", "#D5A8CF", "#FAD7A0",
        "#AED6F1", "#F9C6C6", "#B2D8C8", "#D7C0E2", "#FDEBD0"
    ]
    stroke = "#4D4D4D"
    edge_color = "#7F7F7F"
    node_face = "#B3B3B3"
    node_edge = stroke

    attr = nx.get_node_attributes(G, community_attr)
    if not attr:
        attr = {n: 0 for n in G.nodes()}
    groups: Dict[int, List[str]] = {}
    for n, cid in attr.items():
        groups.setdefault(int(cid), []).append(n)

    # gender based node colors
    palette_nodes = {"F": "#EE0011", "M": "#3B4992"}
    genders = nx.get_node_attributes(G, "gender")
    node_colors = [palette_nodes.get(genders.get(n, "U"), "#999999") for n in G.nodes()]

    fig, ax = plt.subplots(figsize=figsize)

    # draw hulls with stronger presence
    for i, (cid, members) in enumerate(sorted(groups.items(), key=lambda x: x[0])):
        pts = np.array([pos[n] for n in members])
        color = hull_fills[i % len(hull_fills)]
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            poly = Polygon(
                pts[hull.vertices],
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.5,
                zorder=0.5,
            )
            ax.add_patch(poly)
        else:
            xs, ys = pts[:, 0], pts[:, 1]
            pad = 0.05
            x0, x1 = xs.min() - pad, xs.max() + pad
            y0, y1 = ys.min() - pad, ys.max() + pad
            box = FancyBboxPatch(
                (x0, y0), x1 - x0, y1 - y0,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=color,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.5,
                zorder=0.5,
            )
            ax.add_patch(box)

    # edges
    def edge_width(w: float) -> float:
        return 0.6 + 2.8 * min(abs(w), 0.15) / 0.15

    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        a = FancyArrowPatch(
            pos[u], pos[v],
            arrowstyle="-|>",
            mutation_scale=10,
            shrinkA=0.55 * np.sqrt(node_sizes[u]),
            shrinkB=0.55 * np.sqrt(node_sizes[v]),
            connectionstyle="arc3,rad=0.0",
            lw=edge_width(w),
            color=edge_color,
            alpha=0.85,
            zorder=1.2,
        )
        if w < 0:
            a.set_linestyle((0, (2, 2)))
        import matplotlib.patheffects as pe
        a.set_path_effects([
            pe.Stroke(linewidth=a.get_linewidth() + 0.8, foreground="white"),
            pe.Normal()
        ])
        ax.add_patch(a)

    # nodes
    nodes_coll = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,            
        node_size=[node_sizes[n] for n in G.nodes()],
        edgecolors=node_edge,
        linewidths=0.9,
        ax=ax,
    )

    try:
        nodes_coll.set_zorder(2.0)
    except Exception:
        pass

    # labels near nodes
    if show_labels:
        label_pos = {n: (x, y + 0.02) for n, (x, y) in pos.items()}
        nx.draw_networkx_labels(
            G,
            label_pos,
            labels={n: str(n) for n in G.nodes()},
            font_size=font_size,
            font_color="black",
            ax=ax,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#BFBFBF", alpha=0.8),
            verticalalignment="center",
            horizontalalignment="center",
            clip_on=False,
        )


    # legend: edges, communities, gender
    edge_handles = [
        Line2D([0], [0], color=edge_color, lw=2.2, linestyle="-", label="Positive convergence"),
        Line2D([0], [0], color=edge_color, lw=2.2, linestyle="--", label="Negative convergence"),
    ]

    comm_handles = []
    for i, cid in enumerate(sorted(groups.keys())):
        comm_handles.append(
            Line2D([0], [0], marker="s", linestyle="", markersize=11,
                markerfacecolor=hull_fills[i % len(hull_fills)],
                markeredgecolor=hull_fills[i % len(hull_fills)],
                label=f"Community {cid}")
        )

    gender_handles = []
    if genders:
        present = sorted(set(genders.values()))
        name_map = {"F": "Female", "M": "Male"}
        for g in present:
            gender_handles.append(
                Line2D([0], [0], marker="o", linestyle="", markersize=9,
                    markerfacecolor=palette_nodes.get(g, "#999999"),
                    markeredgecolor=node_edge,
                    label=name_map.get(g, "Unknown"))
            )

    handles = edge_handles + comm_handles + gender_handles
    if handles:
        ax.legend(handles=handles, title="Legend", loc="upper left", frameon=False)

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=260)
    plt.close(fig)


def run_graph_analysis(
    in_dir: Path,
    out_dir: Path,
    mode: str,
    positive_only: bool = False,
    prefer_weight_col: str = "mean_conv_w",
    play_title: Optional[str] = None,
) -> None:
    """
    Build and analyze the convergence graph and write all artifacts.

    Params:
        in_dir: Folder with convergence results, typically {root}/analysis_results/convergence_results.
        out_dir: Destination folder for node_metrics.csv communities.csv reciprocity files graph image and summary.
        mode: feature set ("wordlists"), used to pick the right edges filename.
        positive_only: Keep only edges with positive weight.
        prefer_weight_col: Which edge weight column to prefer if present.

    Returns:
        None. Writes graph data, plots, tables and a summary text file to out_dir.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_name = f"conv_edge_summary_{mode.lower()}.csv"
    edges_csv = in_dir / edges_name
    if not edges_csv.exists():
        print(f"[graph] Skip: edges file not found: {edges_csv}")
        return

    # Load edges and build graph
    edges = load_edges(edges_csv, prefer_weight_col=prefer_weight_col)
    G = build_graph(edges, positive_only=positive_only, weight_col="weight_work")

    # Attach node attributes from modal values found in the edges table
    # Gender
    gmap: Dict[str, str] = {}
    for side in ["a", "b"]:
        s_col = f"{side}_speaker"
        g_col = f"{side}_gender"
        if g_col in edges.columns:
            tmp = edges[[s_col, g_col]].dropna()
            if not tmp.empty:
                mode_val = tmp.groupby(s_col)[g_col].agg(lambda x: x.value_counts().idxmax())
                gmap.update(mode_val.to_dict())
    if gmap:
        nx.set_node_attributes(G, gmap, "gender")

    # Category / importance
    cmap: Dict[str, str] = {}
    for side in ["a", "b"]:
        s_col = f"{side}_speaker"
        c_col = f"{side}_category"
        if c_col in edges.columns:
            tmp = edges[[s_col, c_col]].dropna()
            if not tmp.empty:
                mode_val = tmp.groupby(s_col)[c_col].agg(lambda x: x.value_counts().idxmax())
                cmap.update(mode_val.to_dict())
    if cmap:
        nx.set_node_attributes(G, cmap, "category")

    # Node and global metrics
    df_nodes, stats = compute_metrics(G)
    df_nodes.to_csv(out_dir / "node_metrics.csv", index=False, encoding="utf-8-sig")

    # Communities
    comm = detect_communities(G)
    comm.to_csv(out_dir / "communities.csv", index=False, encoding="utf-8-sig")

    # Attach community IDs so plotting can access them
    if not comm.empty:
        nx.set_node_attributes(G, dict(zip(comm["character"], comm["community_id"])), "community_id")


    # Assortativity and two-mode A->B tables (done inline here)
    from networkx.algorithms.assortativity import attribute_assortativity_coefficient

    assort_gender = None
    genders_attr = nx.get_node_attributes(G, "gender")
    if genders_attr:
        UG = G.to_undirected()
        nx.set_node_attributes(UG, genders_attr, "gender")
        try:
            assort_gender = float(attribute_assortativity_coefficient(UG, "gender"))
        except Exception:
            assort_gender = None

    assort_category = None
    cats_attr = nx.get_node_attributes(G, "category")
    if cats_attr:
        UG2 = G.to_undirected()
        nx.set_node_attributes(UG2, cats_attr, "category")
        try:
            assort_category = float(attribute_assortativity_coefficient(UG2, "category"))
        except Exception:
            assort_category = None

    # Two-mode edge averages
    if {"a_gender", "b_gender"}.issubset(edges.columns):
        gtab = (edges.groupby(["a_gender", "b_gender"], dropna=False)
                      .agg(mean_weight=("weight_work", "mean"), count=("weight_work", "size"))
                      .reset_index())
        gtab.to_csv(out_dir / "edge_table_gender_A_to_B.csv", index=False, encoding="utf-8-sig")

    if {"a_category", "b_category"}.issubset(edges.columns):
        ctab = (edges.groupby(["a_category", "b_category"], dropna=False)
                      .agg(mean_weight=("weight_work", "mean"), count=("weight_work", "size"))
                      .reset_index())
        ctab.to_csv(out_dir / "edge_table_category_A_to_B.csv", index=False, encoding="utf-8-sig")

     # Reciprocity
    paired = reciprocity_pair_table(edges)
    rec_r, rec_sign = plot_reciprocity(
        paired,
        out_png=out_dir / "reciprocity_scatter.png",
        out_csv=out_dir / "reciprocity_pairs.csv",
        play_title=play_title,
    )

    # Graph plots (simple + directions)
    base_title = f"{play_title} — Convergence network" if play_title else "Convergence network"
    plot_graph_simple_bidirectional(
        G,
        out_png=out_dir / "graph_network_simple.png",
        color_attr="gender",
        show_labels=True,
        figsize=(14.5, 10.5),
        font_size=11,
        title=base_title,
    )
    plot_graph_direction_only(
        G,
        out_png=out_dir / "graph_A_to_B.png",
        direction="AtoB",
        color_attr="gender",
        show_labels=True,
        figsize=(14.5, 10.5),
        font_size=11,
        play_title=play_title,
    )
    plot_graph_direction_only(
        G,
        out_png=out_dir / "graph_B_to_A.png",
        direction="BtoA",
        color_attr="gender",
        show_labels=True,
        figsize=(14.5, 10.5),
        font_size=11,
        play_title=play_title,
    )
    plot_graph_communities_simple(
        G,
        out_png=out_dir / "graph_communities_simple.png",
        community_attr="community_id",
        show_labels=True,            # names inside nodes
        figsize=(12.5, 8.5),
        font_size=10,
        title=(f"{play_title} — Communities" if play_title else "Communities"),
    )

    # Summary file written inline
    lines: List[str] = []
    lines.append(f"nodes {stats.get('n_nodes')}")
    lines.append(f"edges {stats.get('n_edges')}")
    lines.append(f"density {stats.get('density'):.4f}")
    rec = stats.get("reciprocity")
    lines.append(f"reciprocity {rec:.4f}" if rec is not None and not np.isnan(rec) else "reciprocity NA")
    if assort_gender is not None:
        lines.append(f"gender_assortativity {assort_gender:.4f}")
    if assort_category is not None:
        lines.append(f"category_assortativity {assort_category:.4f}")
    if rec_r is not None and not np.isnan(rec_r):
        lines.append(f"reciprocity_correlation {rec_r:.4f}")
    if rec_sign is not None and not np.isnan(rec_sign):
        lines.append(f"reciprocity_sign_match {rec_sign:.4f}")

    if not edges.empty:
        E = {(r.a_speaker, r.b_speaker) for r in edges.itertuples(index=False)}
        mutual = sum(1 for a, b in E if (b, a) in E)
        lines.append(f"mutual_edge_pairs {mutual // 2}")

    if not comm.empty:
        k = len(comm["community_id"].unique())
        counts = comm["community_id"].value_counts().sort_index()
        lines.append(f"communities_detected {k}")
        lines.append("community_sizes " + ", ".join(f"{cid}:{cnt}" for cid, cnt in counts.items()))

    ew = edges["weight_work"].to_numpy()
    mean_w = float(np.nanmean(ew)) if ew.size else np.nan
    pos_share = float(np.mean(ew > 0)) if ew.size else np.nan
    neg_share = float(np.mean(ew < 0)) if ew.size else np.nan

    w_abs = np.abs(ew[np.isfinite(ew)])
    if w_abs.size == 0:
        edge_weight_gini = np.nan
    else:
        x = np.sort(w_abs)
        cum = np.cumsum(x)
        edge_weight_gini = 1.0 - 2.0 * np.sum(cum) / (x.sum() * (x.size)) + 1.0 / x.size

    share_both_pos = share_both_neg = asymm_mean = np.nan
    if not paired.empty:
        wa = paired["w_ab"].to_numpy()
        wb = paired["w_ba"].to_numpy()
        both_pos = (wa > 0) & (wb > 0)
        both_neg = (wa < 0) & (wb < 0)
        share_both_pos = float(np.mean(both_pos))
        share_both_neg = float(np.mean(both_neg))
        denom = np.abs(wa) + np.abs(wb)
        asymm = np.abs(wa - wb) / np.where(denom > 0, denom, np.nan)
        asymm_mean = float(np.nanmean(asymm))

    out_strength = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        out_strength[u] += abs(float(d.get("weight", 0.0)))
    svals = np.asarray(list(out_strength.values()), float)
    if svals.size == 0:
        out_strength_gini = np.nan
        top10_share = np.nan
    else:
        xs = np.sort(svals)
        cum = np.cumsum(xs)
        out_strength_gini = 1.0 - 2.0 * np.sum(cum) / (xs.sum() * xs.size) + 1.0 / xs.size
        k = max(1, int(np.ceil(0.10 * xs.size)))
        top10_share = float(np.sum(xs[-k:]) / xs.sum()) if xs.sum() > 0 else np.nan

    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            w = abs(float(d.get("weight", 0.0)))
            if w > 0:
                H.add_edge(u, v, weight=w)
        pr = nx.pagerank(H, weight="weight") if H.number_of_edges() > 0 else {n: 1.0/len(H) for n in H}
        p = np.asarray(list(pr.values()), float)
        p = p / p.sum() if p.sum() > 0 else p
        ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0))
        pagerank_entropy = float(ent / np.log(len(p))) if len(p) > 1 else np.nan  # normalize to [0,1]
    else:
        pagerank_entropy = np.nan

    try:
        UGw = nx.Graph()
        for u, v, d in G.edges(data=True):
            w = abs(float(d.get("weight", 0.0)))
            if w > 0:
                if UGw.has_edge(u, v):
                    UGw[u][v]["weight"] += w
                else:
                    UGw.add_edge(u, v, weight=w)
        if UGw.number_of_edges() > 0:
            from networkx.algorithms import community
            parts = community.greedy_modularity_communities(UGw, weight="weight")
            partition = {n: i for i, S in enumerate(parts) for n in S}
            modularity_Q = community.modularity(UGw, parts, weight="weight")
            n_comms_Q = len(parts)
        else:
            modularity_Q = np.nan
            n_comms_Q = 0
    except Exception:
        modularity_Q = np.nan
        n_comms_Q = 0

    lines.append(f"mean_edge_weight {mean_w:.4f}" if np.isfinite(mean_w) else "mean_edge_weight NA")
    lines.append(f"pos_share {pos_share:.4f}" if np.isfinite(pos_share) else "pos_share NA")
    lines.append(f"neg_share {neg_share:.4f}" if np.isfinite(neg_share) else "neg_share NA")
    lines.append(f"edge_weight_gini {edge_weight_gini:.4f}" if np.isfinite(edge_weight_gini) else "edge_weight_gini NA")
    lines.append(f"reciprocity_both_positive {share_both_pos:.4f}" if np.isfinite(share_both_pos) else "reciprocity_both_positive NA")
    lines.append(f"reciprocity_both_negative {share_both_neg:.4f}" if np.isfinite(share_both_neg) else "reciprocity_both_negative NA")
    lines.append(f"asymmetry_index_mean {asymm_mean:.4f}" if np.isfinite(asymm_mean) else "asymmetry_index_mean NA")
    lines.append(f"out_strength_gini {out_strength_gini:.4f}" if np.isfinite(out_strength_gini) else "out_strength_gini NA")
    lines.append(f"top10pct_strength_share {top10_share:.4f}" if np.isfinite(top10_share) else "top10pct_strength_share NA")
    lines.append(f"pagerank_entropy {pagerank_entropy:.4f}" if np.isfinite(pagerank_entropy) else "pagerank_entropy NA")
    lines.append(f"modularity_Q {modularity_Q:.4f}" if np.isfinite(modularity_Q) else "modularity_Q NA")
    lines.append(f"modularity_communities {n_comms_Q}")


    (out_dir / "graph_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[graph] Results saved -> {out_dir}")
