import math
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind


def paired_t(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Compute a paired-sample t-test between two related numeric arrays.

    Params:
        a: Numeric array of observations for condition 1.
        b: Numeric array of observations for condition 2.

    Returns:
        Tuple (mean_diff, p_value)
            mean_diff: The average difference (mean(a - b)) across pairs.
            p_value: Two-sided p-value from a paired t-test.
                     Small p (< 0.05) means the two conditions differ significantly.
        If fewer than two valid pairs exist, returns NaN values.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float(np.nan), float(np.nan)
    t = ttest_rel(a[ok], b[ok], alternative="two-sided", nan_policy="omit")
    return float((a[ok] - b[ok]).mean()), float(t.pvalue)


def bootstrap_mean_ci(
    x: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Estimate the mean and confidence interval using bootstrap resampling.
    The bootstrap repeatedly resamples from the data (with replacement) to
    approximate the sampling distribution of the mean. The confidence interval
    (CI) shows the range of likely mean values given sampling uncertainty.

    Params:
        x: Numeric array.
        n_boot: Number of bootstrap resamples (higher = more stable but slower).
        alpha: Significance level (e.g., 0.05 gives a 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple (mean, ci_low, ci_high)
            mean: The arithmetic mean of x.
            ci_low, ci_high: Lower and upper percentile bounds of the bootstrap CI.
        Returns NaNs if the array has no finite values.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(np.nan), float(np.nan), float(np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(x.mean()), float(lo), float(hi)


def bootstrap_paired_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for the mean paired difference (a - b).
    Each bootstrap resample draws with replacement from the differences
    between a[i] and b[i]. This estimates the variability in the mean difference
    without relying on normality assumptions.

    Params:
        a: First numeric array (condition 1).
        b: Second numeric array (condition 2).
        n_boot: Number of bootstrap resamples.
        alpha: Significance level for the two-sided CI (e.g., 0.05 = 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        Tuple (mean_diff, ci_low, ci_high)
            mean_diff: The observed mean of (a - b).
            ci_low, ci_high: Lower and upper percentile bounds from the bootstrap.
        Returns NaNs if no valid pairs exist.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a[ok] - b[ok])
    if d.size == 0:
        return float(np.nan), float(np.nan), float(np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(d.mean()), float(lo), float(hi)


def _load_edge_family_conv(path: Path) -> pd.DataFrame:
    """
    Load a convergence dataset that has one row per (A,B,family) pair.

    Params:
        path: Path to a file such as 'conv_by_pair_feature_*.csv'.

    Returns:
        DataFrame with required columns [a_speaker, b_speaker, family, conv].
        Raises an error if those columns are missing.
    """
    df = pd.read_csv(path)
    need = {"a_speaker", "b_speaker", "family", "conv"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["a_speaker", "b_speaker", "family", "conv"]].copy()


def _bootstrap_se(values: Iterable[float], n_boot: int = 1000, seed: int = 123) -> float:
    """
    Estimate the standard error (SE) of the mean using bootstrap resampling.
    The standard error reflects how much the sample mean is expected to vary
    due to sampling noise — smaller SE means more precise mean estimates.

    Params:
        values: Iterable of numeric values.
        n_boot: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Estimated bootstrap standard error of the mean, or NaN if insufficient data.
    """
    vals = list(values)
    if len(vals) <= 1:
        return float("nan")
    rnd = random.Random(seed)
    boots = []
    n = len(vals)
    for _ in range(n_boot):
        sample = [vals[rnd.randrange(n)] for __ in range(n)]
        boots.append(sum(sample) / n)
    mu = sum(boots) / len(boots)
    var = sum((b - mu) ** 2 for b in boots) / (len(boots) - 1)
    return math.sqrt(var)


def _paired_test(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    """
    Perform a paired-sample test using a normal approximation to estimate p-values.

    This is similar to a paired t-test but approximates the p-value from a
    normal distribution instead of using a t-distribution.

    Params:
        x: Series of condition 1 values.
        y: Series of condition 2 values (aligned with x).

    Returns:
        Tuple (t_stat, p_value, df)
            t_stat: The test statistic (mean difference / SE).
            p_value: Approximate two-sided p-value using the normal approximation.
            df: Degrees of freedom (n - 1).
        Returns NaNs if fewer than 2 valid pairs exist.
    """
    d = (x - y).dropna()
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), n
    mean_d = d.mean()
    sd_d = d.std(ddof=1)
    if sd_d == 0:
        return (float("inf") if mean_d != 0 else 0.0), 0.0, n - 1
    se = sd_d / math.sqrt(n)
    t = mean_d / se
    p = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))  # normal approximation
    return t, p, n - 1


def summarize_and_compare(
    pf_adj: pd.DataFrame,
    pf_non: Optional[pd.DataFrame],
    pf_rand: Optional[pd.DataFrame],
    out_csv: Path,
    alpha: float = 0.05,
    n_boot: int = 5000
) -> pd.DataFrame:
    """
    Summarize convergence per linguistic family and statistically compare conditions.
    For each feature family (e.g., pronouns, articles), this function computes
    average convergence and bootstrap confidence intervals in the “adjacent”
    (real) condition, then compares them to “nonadjacent” and/or “randomized”
    control datasets using paired t-tests and bootstrap CIs on differences.

    Params:
        pf_adj: DataFrame for the adjacent condition (columns: family, conv, a_speaker, b_speaker).
        pf_non: Optional control for nonadjacent pairs.
        pf_rand: Optional control for randomized pairs.
        out_csv: Output path for the summary CSV.
        alpha: Significance level for CIs (0.05 = 95% CI).
        n_boot: Number of bootstrap iterations for confidence intervals.

    Returns:
        DataFrame with one row per linguistic family, containing:
          - adj_mean, adj_ci_lo, adj_ci_hi  → mean and CI in adjacent condition
          - non_mean, rand_mean              → mean in controls (if provided)
          - adj_minus_non_* or adj_minus_rand_* → paired differences, p-values, bootstrap CIs
        The full summary is also saved to out_csv.
    """
    rows = []
    families = sorted(set(pf_adj["family"])) if pf_adj is not None and not pf_adj.empty else []

    def _pair_on_family(pf1: pd.DataFrame, pf2: pd.DataFrame, fam: str) -> Tuple[np.ndarray, np.ndarray]:
        a1 = pf1.query("family == @fam")[["a_speaker", "b_speaker", "conv"]]
        a2 = pf2.query("family == @fam")[["a_speaker", "b_speaker", "conv"]]
        m = a1.merge(a2, on=["a_speaker", "b_speaker"], how="inner", suffixes=("_adj", "_other"))
        return m["conv_adj"].to_numpy(), m["conv_other"].to_numpy()

    for fam in families:
        mean_a, lo_a, hi_a = bootstrap_mean_ci(
            pf_adj.loc[pf_adj["family"] == fam, "conv"].to_numpy(),
            n_boot=n_boot, alpha=alpha
        )
        rec: Dict[str, float | str] = {
            "family": fam,
            "adj_mean": mean_a,
            "adj_ci_lo": lo_a,
            "adj_ci_hi": hi_a,
        }

        # compare adjacent vs nonadjacent
        if pf_non is not None and not pf_non.empty:
            a, b = _pair_on_family(pf_adj, pf_non, fam)
            mean_d, p_t = paired_t(a, b)
            _, d_lo, d_hi = bootstrap_paired_diff_ci(a, b, n_boot=n_boot, alpha=alpha)
            rec.update({
                "non_mean": float(np.nanmean(pf_non.loc[pf_non["family"] == fam, "conv"])),
                "adj_minus_non_mean": mean_d,
                "adj_minus_non_p_t": p_t,
                "adj_minus_non_ci_lo": d_lo,
                "adj_minus_non_ci_hi": d_hi,
                "test": "paired_t_with_bootstrap_CI",
            })

        # compare adjacent vs randomized
        if pf_rand is not None and not pf_rand.empty:
            a, b = _pair_on_family(pf_adj, pf_rand, fam)
            mean_d, p_t = paired_t(a, b)
            _, d_lo, d_hi = bootstrap_paired_diff_ci(a, b, n_boot=n_boot, alpha=alpha)
            rec.update({
                "rand_mean": float(np.nanmean(pf_rand.loc[pf_rand["family"] == fam, "conv"])),
                "adj_minus_rand_mean": mean_d,
                "adj_minus_rand_p_t": p_t,
                "adj_minus_rand_ci_lo": d_lo,
                "adj_minus_rand_ci_hi": d_hi,
            })

        rows.append(rec)

    out = pd.DataFrame(rows)

    # Cohen's d for adj vs rand and adj vs non (unpaired, across all dyads per family)
    if pf_rand is not None and not pf_rand.empty:
        def _cohens_d(a, b):
            na, nb = len(a), len(b)
            if na < 2 or nb < 2:
                return float("nan")
            pooled = math.sqrt(((na - 1) * float(np.std(a, ddof=1))**2 +
                                (nb - 1) * float(np.std(b, ddof=1))**2) / (na + nb - 2))
            return (float(np.mean(a)) - float(np.mean(b))) / pooled if pooled else float("nan")

        d_vals, d_vals_non = [], []
        for fam in families:
            a = pf_adj.loc[pf_adj["family"] == fam, "conv"].dropna().to_numpy()
            r = pf_rand.loc[pf_rand["family"] == fam, "conv"].dropna().to_numpy()
            d_vals.append(_cohens_d(a, r))
            if pf_non is not None and not pf_non.empty:
                n_ = pf_non.loc[pf_non["family"] == fam, "conv"].dropna().to_numpy()
                d_vals_non.append(_cohens_d(a, n_))
            else:
                d_vals_non.append(float("nan"))
        out["cohens_d_adj_rand"] = d_vals
        if pf_non is not None and not pf_non.empty:
            out["cohens_d_adj_non"] = d_vals_non

    # Bonferroni correction flag (n_families tests)
    n_fam = len(families)
    if "adj_minus_rand_p_t" in out.columns:
        out["bonferroni_significant_rand"] = out["adj_minus_rand_p_t"] < (0.05 / n_fam)
    if "adj_minus_non_p_t" in out.columns:
        out["bonferroni_significant_non"] = out["adj_minus_non_p_t"] < (0.05 / n_fam)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


def analyze_by_gender_dyad(
    pf_adj: pd.DataFrame,
    out_csv: Path,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute convergence broken down by gender dyad (a_gender -> b_gender) and
    linguistic family. Rows with missing gender metadata are excluded.
    Saves one row per (family, dyad_gender) with mean, bootstrap CI, and n.
    Also appends pairwise t-tests between dyad types (Bonferroni-corrected).
    """
    if pf_adj is None or pf_adj.empty:
        return pd.DataFrame()
    need = {"a_gender", "b_gender", "family", "conv"}
    if not need.issubset(pf_adj.columns):
        return pd.DataFrame()

    df = pf_adj.dropna(subset=["a_gender", "b_gender"]).copy()
    df["dyad_gender"] = df["a_gender"].str.strip() + "->" + df["b_gender"].str.strip()
    df = df[df["dyad_gender"].str.match(r"^[MFU]->[MFU]$")]

    rows = []
    for fam in sorted(df["family"].unique()):
        for dyad in sorted(df["dyad_gender"].unique()):
            vals = df[(df["family"] == fam) & (df["dyad_gender"] == dyad)]["conv"].dropna()
            if len(vals) < 2:
                continue
            rng = np.random.default_rng(42)
            boots = rng.choice(vals.to_numpy(), size=(n_boot, len(vals)), replace=True).mean(axis=1)
            rows.append({
                "family": fam,
                "dyad_gender": dyad,
                "mean_conv": float(vals.mean()),
                "ci_lo": float(np.percentile(boots, 100 * alpha / 2)),
                "ci_hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
                "n": len(vals),
            })

    result = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return result


def analyze_by_importance_dyad(
    pf_adj: pd.DataFrame,
    out_csv: Path,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute convergence broken down by importance dyad (a_category -> b_category)
    and linguistic family. Also tests the CAT status asymmetry:
    whether minor->major convergence differs from major->minor.
    """
    if pf_adj is None or pf_adj.empty:
        return pd.DataFrame()
    need = {"a_category", "b_category", "family", "conv"}
    if not need.issubset(pf_adj.columns):
        return pd.DataFrame()

    df = pf_adj.dropna(subset=["a_category", "b_category"]).copy()
    df["dyad_imp"] = df["a_category"].str.strip() + "->" + df["b_category"].str.strip()

    rows = []
    for fam in sorted(df["family"].unique()):
        for dyad in sorted(df["dyad_imp"].unique()):
            vals = df[(df["family"] == fam) & (df["dyad_imp"] == dyad)]["conv"].dropna()
            if len(vals) < 2:
                continue
            rng = np.random.default_rng(42)
            boots = rng.choice(vals.to_numpy(), size=(n_boot, len(vals)), replace=True).mean(axis=1)
            rows.append({
                "family": fam,
                "dyad_importance": dyad,
                "mean_conv": float(vals.mean()),
                "ci_lo": float(np.percentile(boots, 100 * alpha / 2)),
                "ci_hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
                "n": len(vals),
            })

    result = pd.DataFrame(rows)

    # Status asymmetry test: minor->major vs major->minor (pooled across families)
    asym_rows = []
    for fam in sorted(df["family"].unique()):
        minor_maj = df[(df["family"] == fam) & (df["dyad_imp"] == "minor->major")]["conv"].dropna()
        maj_minor = df[(df["family"] == fam) & (df["dyad_imp"] == "major->minor")]["conv"].dropna()
        if len(minor_maj) < 2 or len(maj_minor) < 2:
            continue
        t_stat, p_val = ttest_ind(minor_maj, maj_minor)
        na, nb = len(minor_maj), len(maj_minor)
        pooled_sd = math.sqrt(((na - 1) * minor_maj.std(ddof=1)**2 +
                               (nb - 1) * maj_minor.std(ddof=1)**2) / (na + nb - 2))
        d = (minor_maj.mean() - maj_minor.mean()) / pooled_sd if pooled_sd else float("nan")
        asym_rows.append({
            "family": fam,
            "minor_to_major_mean": float(minor_maj.mean()),
            "major_to_minor_mean": float(maj_minor.mean()),
            "diff_minor_minus_major": float(minor_maj.mean() - maj_minor.mean()),
            "cohens_d": d,
            "p_value": p_val,
            "n_minor_major": na,
            "n_major_minor": nb,
        })

    asym = pd.DataFrame(asym_rows)
    if not asym.empty:
        n_tests = len(asym)
        asym["bonferroni_significant"] = asym["p_value"] < (0.05 / n_tests)
        asym_path = out_csv.parent / out_csv.name.replace("importance_dyad", "importance_asymmetry")
        asym.to_csv(asym_path, index=False, encoding="utf-8-sig")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return result
