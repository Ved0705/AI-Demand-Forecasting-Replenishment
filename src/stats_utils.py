"""Statistical primitives for Phase 3.

Separated from the analysis orchestration so each function can be unit-tested
against known values. Nothing here touches the database.

Two themes run through this module:

1.  **Daily retail sales are not independent observations.** Autocorrelation,
    weekly seasonality and trend all violate the i.i.d. assumption behind the
    ordinary bootstrap and behind every standard test's standard error. Where a
    statistic depends on temporal structure we use a moving-block bootstrap;
    where it does not (cross-sectional series statistics) the ordinary
    bootstrap is correct.

2.  **Significance is not effect size.** With thousands of observations almost
    any difference reaches p < 0.05. Every comparison here reports an effect
    size and a confidence interval, and the interpretation leans on those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------

def describe_distribution(x: Sequence[float], name: str = "") -> dict:
    """Descriptive statistics chosen to survive zero-inflation.

    For intermittent retail demand the mean is dominated by structural zeros
    and the standard deviation is inflated by them, so both are reported
    alongside quantiles and a separate summary of the NON-ZERO values. The
    non-zero summary is what "demand size" means when demand occurs at all.
    """
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"name": name, "n": 0}

    nz = a[a > 0]
    out = {
        "name": name,
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "sd": float(a.std(ddof=1)) if a.size > 1 else np.nan,
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "zero_share": float((a == 0).mean()),
        "skewness": float(stats.skew(a)) if a.size > 2 else np.nan,
    }
    # CV is only meaningful for a positive-mean, ratio-scale quantity.
    out["cv"] = float(a.std(ddof=1) / a.mean()) if a.mean() > 0 and a.size > 1 else np.nan
    out["nonzero_n"] = int(nz.size)
    out["nonzero_mean"] = float(nz.mean()) if nz.size else np.nan
    out["nonzero_median"] = float(np.median(nz)) if nz.size else np.nan
    out["nonzero_cv"] = (
        float(nz.std(ddof=1) / nz.mean()) if nz.size > 1 and nz.mean() > 0 else np.nan
    )
    return out


def gini(x: Sequence[float]) -> float:
    """Concentration of demand across series. 0 = perfectly even, 1 = all in one."""
    a = np.sort(np.asarray(x, dtype=float))
    a = a[~np.isnan(a)]
    if a.size == 0 or a.sum() == 0:
        return np.nan
    if (a < 0).any():
        raise ValueError("gini requires non-negative values")
    n = a.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

@dataclass
class CIResult:
    statistic: float
    lower: float
    upper: float
    method: str
    n_boot: int
    resample_unit: str

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "ci_method": self.method,
            "n_boot": self.n_boot,
            "resample_unit": self.resample_unit,
        }


def bootstrap_ci(
    x: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    resample_unit: str = "observation",
) -> CIResult:
    """Ordinary (i.i.d.) percentile bootstrap.

    Valid only when observations are exchangeable. Use this for
    CROSS-SECTIONAL statistics - e.g. the distribution of per-series zero
    share, where each series contributes one value and ordering is
    meaningless. Do NOT use it on a daily time series.
    """
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return CIResult(np.nan, np.nan, np.nan, "percentile_iid", n_boot, resample_unit)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = np.array([statistic(a[i]) for i in idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CIResult(float(statistic(a)), float(lo), float(hi),
                    "percentile_iid", n_boot, resample_unit)


def optimal_block_length(n: int) -> int:
    """Rule-of-thumb block length ~ n^(1/3).

    Blocks must be long enough to carry the dependence structure and short
    enough to give many distinct blocks. For a daily retail series n^(1/3)
    comfortably exceeds the 7-day weekly cycle at n >= ~350.
    """
    return max(2, int(round(n ** (1 / 3))))


def moving_block_bootstrap_ci(
    x: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    block_length: int | None = None,
    resample_unit: str = "day (moving block)",
) -> CIResult:
    """Moving-block bootstrap for statistics of a dependent time series.

    Resampling individual days would destroy autocorrelation and weekly
    seasonality, producing confidence intervals far too narrow - the classic
    way a time-series analysis reports false precision. Resampling contiguous
    blocks preserves short-range dependence within each block.
    """
    a = np.asarray(x, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if n < 4:
        return CIResult(
            float(statistic(a)) if n else np.nan, np.nan, np.nan,
            "moving_block", n_boot, resample_unit,
        )

    L = block_length or optimal_block_length(n)
    L = min(L, n)
    n_blocks = int(np.ceil(n / L))
    max_start = n - L

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))
    offsets = np.arange(L)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = a[(starts[b][:, None] + offsets).ravel()][:n]
        boot[b] = statistic(sample)

    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CIResult(float(statistic(a)), float(lo), float(hi),
                    f"moving_block(L={L})", n_boot, resample_unit)


def block_bootstrap_diff_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    block_length: int | None = None,
) -> CIResult:
    """CI for the difference in means of two dependent series.

    Each group is block-resampled independently. Both groups are day-indexed
    subsets of the same store's history, so each retains its own dependence.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 4 or b.size < 4:
        return CIResult(np.nan, np.nan, np.nan, "moving_block_diff", n_boot, "day")

    La = block_length or optimal_block_length(a.size)
    Lb = block_length or optimal_block_length(b.size)
    rng = np.random.default_rng(seed)

    def resample(arr: np.ndarray, L: int, r: np.random.Generator) -> np.ndarray:
        n = arr.size
        L = min(L, n)
        nb = int(np.ceil(n / L))
        starts = r.integers(0, n - L + 1, size=nb)
        return arr[(starts[:, None] + np.arange(L)).ravel()][:n]

    boot = np.array([
        resample(a, La, rng).mean() - resample(b, Lb, rng).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CIResult(float(a.mean() - b.mean()), float(lo), float(hi),
                    f"moving_block_diff(La={La},Lb={Lb})", n_boot, "day")


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta: P(X > Y) - P(X < Y), in [-1, 1].

    Non-parametric and distribution-free, so it is valid for zero-inflated
    demand where Cohen's d - which presumes roughly normal, comparable-variance
    groups - is not.

    Computed from the Mann-Whitney U statistic rather than by pairwise
    comparison, which would be O(n*m) and infeasible at these sizes.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return np.nan
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (n * m) - 1)


def interpret_cliffs_delta(d: float) -> str:
    """Romano et al. thresholds. Labels, not verdicts."""
    if np.isnan(d):
        return "undefined"
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def relative_difference(x: Sequence[float], y: Sequence[float]) -> float:
    """(mean(x) - mean(y)) / mean(y), as a percentage.

    Reported alongside Cliff's delta because a rank-based effect size is hard
    to act on commercially, while a percentage difference is immediately
    interpretable.
    """
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if b.size == 0 or a.size == 0 or b.mean() == 0:
        return np.nan
    return float(100 * (a.mean() - b.mean()) / b.mean())


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    question: str
    null_hypothesis: str
    alt_hypothesis: str
    test: str
    n_group1: int
    n_group2: int
    statistic: float
    p_value: float
    effect_size: float
    effect_metric: str
    effect_label: str
    rel_diff_pct: float
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    ci_method: str = ""
    notes: str = ""
    p_adjusted: float = np.nan
    significant_fdr: bool = False

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


def mann_whitney_test(
    x: Sequence[float],
    y: Sequence[float],
    question: str,
    null_hypothesis: str,
    alt_hypothesis: str,
    notes: str = "",
    ci: CIResult | None = None,
) -> TestResult:
    """Mann-Whitney U with a rank-based effect size.

    Chosen over the t-test because daily unit sales are heavily right-skewed
    and zero-inflated; the t-test's normality assumption is not remotely met at
    series level and only approximately met even for store-day aggregates.

    The independence caveat still applies: consecutive days are correlated, so
    the p-value is anti-conservative. That is why every call passes a
    block-bootstrap CI and why interpretation rests on the effect size.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]

    if a.size < 2 or b.size < 2:
        return TestResult(question, null_hypothesis, alt_hypothesis,
                          "mann_whitney_u", a.size, b.size, np.nan, np.nan,
                          np.nan, "cliffs_delta", "undefined", np.nan,
                          notes=notes + " insufficient data")

    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cliffs_delta(a, b)
    out = TestResult(
        question=question,
        null_hypothesis=null_hypothesis,
        alt_hypothesis=alt_hypothesis,
        test="mann_whitney_u",
        n_group1=int(a.size),
        n_group2=int(b.size),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=d,
        effect_metric="cliffs_delta",
        effect_label=interpret_cliffs_delta(d),
        rel_diff_pct=relative_difference(a, b),
        notes=notes,
    )
    if ci is not None:
        out.ci_lower, out.ci_upper, out.ci_method = ci.lower, ci.upper, ci.method
    return out


def kruskal_test(
    groups: dict[str, Sequence[float]],
    question: str,
    null_hypothesis: str,
    alt_hypothesis: str,
    notes: str = "",
) -> TestResult:
    """Kruskal-Wallis for 3+ groups, with epsilon-squared as the effect size.

    Used instead of running every pairwise comparison: one omnibus test asks
    whether ANY group differs, which is the right first question and avoids
    inflating the comparison count.
    """
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[~np.isnan(a)] for a in arrays]
    arrays = [a for a in arrays if a.size >= 2]
    if len(arrays) < 2:
        return TestResult(question, null_hypothesis, alt_hypothesis,
                          "kruskal_wallis", 0, 0, np.nan, np.nan, np.nan,
                          "epsilon_squared", "undefined", np.nan,
                          notes=notes + " insufficient groups")

    res = stats.kruskal(*arrays)
    n = sum(a.size for a in arrays)
    k = len(arrays)
    # epsilon^2 = (H - k + 1) / (n - k), the standard KW effect size.
    eps2 = (res.statistic - k + 1) / (n - k) if n > k else np.nan
    eps2 = float(max(0.0, eps2)) if not np.isnan(eps2) else np.nan

    label = "negligible" if eps2 < 0.01 else (
        "small" if eps2 < 0.06 else ("medium" if eps2 < 0.14 else "large")
    )
    return TestResult(
        question=question,
        null_hypothesis=null_hypothesis,
        alt_hypothesis=alt_hypothesis,
        test="kruskal_wallis",
        n_group1=n,
        n_group2=k,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=eps2,
        effect_metric="epsilon_squared",
        effect_label=label,
        rel_diff_pct=np.nan,
        notes=notes,
    )


def permutation_test_diff_means(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    """Two-sided permutation p-value for a difference in means.

    Makes no distributional assumption, but DOES assume exchangeability under
    the null - which daily autocorrelation violates. Reported as a
    cross-check on Mann-Whitney, not as a fix for dependence.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n = a.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:n].mean() - pool[n:].mean()) >= obs:
            count += 1
    return float((count + 1) / (n_perm + 1))


# ---------------------------------------------------------------------------
# Multiple testing
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05):
    """Benjamini-Hochberg FDR control.

    Returns (rejected, adjusted_p). NaN p-values pass through as NaN and are
    excluded from the correction rather than silently counted.

    BH rather than Bonferroni: with a family of related retail hypotheses,
    controlling the expected proportion of false discoveries is the sensible
    target. Bonferroni controls the probability of ANY false positive, which is
    needlessly severe here and would hide real effects.
    """
    p = np.asarray(p_values, dtype=float)
    out_adj = np.full(p.shape, np.nan)
    out_rej = np.zeros(p.shape, dtype=bool)

    valid = ~np.isnan(p)
    pv = p[valid]
    n = pv.size
    if n == 0:
        return out_rej, out_adj

    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward.
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    adj_orig = np.empty(n)
    adj_orig[order] = adj
    out_adj[valid] = adj_orig
    out_rej[valid] = adj_orig <= alpha
    return out_rej, out_adj


def apply_fdr(results: list[TestResult], alpha: float = 0.05) -> list[TestResult]:
    """Correct a family of tests in place and return it."""
    rejected, adjusted = benjamini_hochberg([r.p_value for r in results], alpha)
    for r, rej, adj in zip(results, rejected, adjusted):
        r.p_adjusted = float(adj)
        r.significant_fdr = bool(rej)
    return results


def results_to_frame(results: list[TestResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in results])
