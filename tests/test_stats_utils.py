"""Tests for the Phase 3 statistical primitives.

Each function is pinned against a hand-computable or analytically known value.
Statistical code is exactly where a silent error is most dangerous: a wrong
effect size or a broken FDR correction produces plausible numbers that no
downstream check catches.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from src.stats_utils import (
    benjamini_hochberg,
    block_bootstrap_diff_ci,
    bootstrap_ci,
    cliffs_delta,
    describe_distribution,
    gini,
    interpret_cliffs_delta,
    kruskal_test,
    mann_whitney_test,
    moving_block_bootstrap_ci,
    optimal_block_length,
    permutation_test_diff_means,
    relative_difference,
)


# --- descriptive -----------------------------------------------------------

def test_describe_known_values():
    d = describe_distribution([0, 0, 0, 1, 2, 3, 4])
    assert d["n"] == 7
    assert d["zero_share"] == pytest.approx(3 / 7)
    assert d["median"] == 1
    assert d["nonzero_n"] == 4
    assert d["nonzero_mean"] == pytest.approx(2.5)


def test_describe_separates_zero_inflation_from_demand_size():
    """The whole point: overall mean and non-zero mean must differ."""
    d = describe_distribution([0] * 90 + [10] * 10)
    assert d["mean"] == pytest.approx(1.0)
    assert d["nonzero_mean"] == pytest.approx(10.0)
    assert d["zero_share"] == pytest.approx(0.9)


def test_describe_handles_empty_and_all_zero():
    assert describe_distribution([])["n"] == 0
    d = describe_distribution([0, 0, 0])
    assert d["zero_share"] == 1.0
    assert np.isnan(d["cv"])          # mean is 0 -> CV undefined, not inf
    assert np.isnan(d["nonzero_mean"])


def test_gini_bounds():
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-9)     # perfectly even
    assert gini([0, 0, 0, 100]) > 0.7                              # concentrated
    assert np.isnan(gini([0, 0, 0]))


def test_gini_rejects_negatives():
    with pytest.raises(ValueError):
        gini([-1, 2, 3])


# --- effect sizes ----------------------------------------------------------

def test_cliffs_delta_complete_separation():
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)


def test_cliffs_delta_identical_groups_is_zero():
    assert cliffs_delta([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.0)


def test_cliffs_delta_matches_manual_pairwise():
    """Pin the U-statistic shortcut against the O(n*m) definition."""
    rng = np.random.default_rng(0)
    x, y = rng.normal(1, 1, 40), rng.normal(0, 1, 35)
    manual = np.mean([np.sign(a - b) for a in x for b in y])
    assert cliffs_delta(x, y) == pytest.approx(manual, abs=1e-9)


def test_cliffs_delta_labels():
    assert interpret_cliffs_delta(0.05) == "negligible"
    assert interpret_cliffs_delta(0.25) == "small"
    assert interpret_cliffs_delta(0.40) == "medium"
    assert interpret_cliffs_delta(0.90) == "large"
    assert interpret_cliffs_delta(np.nan) == "undefined"


def test_relative_difference():
    assert relative_difference([110] * 10, [100] * 10) == pytest.approx(10.0)
    assert np.isnan(relative_difference([1, 2], [0, 0]))


# --- multiple testing ------------------------------------------------------

def test_bh_matches_scipy_reference():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    _, adj = benjamini_hochberg(p, 0.05)
    expected = sps.false_discovery_control(p, method="bh")
    np.testing.assert_allclose(adj, expected, rtol=1e-9)


def test_bh_is_monotonic_and_bounded():
    rng = np.random.default_rng(1)
    p = np.sort(rng.uniform(0, 1, 50))
    _, adj = benjamini_hochberg(p)
    assert np.all(np.diff(adj) >= -1e-12), "adjusted p must be non-decreasing"
    assert adj.max() <= 1.0 and adj.min() >= 0.0


def test_bh_is_less_conservative_than_bonferroni():
    p = [0.01, 0.02, 0.03, 0.04]
    _, adj = benjamini_hochberg(p)
    assert np.all(adj <= np.array(p) * len(p) + 1e-12)


def test_bh_passes_nan_through_and_excludes_it():
    """A NaN p-value must not be counted in the correction denominator."""
    rej, adj = benjamini_hochberg([0.01, np.nan, 0.02])
    assert np.isnan(adj[1])
    assert not rej[1]
    # n=2 valid tests, not 3.
    assert adj[0] == pytest.approx(0.02)


def test_bh_all_nan():
    rej, adj = benjamini_hochberg([np.nan, np.nan])
    assert np.all(np.isnan(adj))
    assert not rej.any()


# --- bootstrap -------------------------------------------------------------

def test_bootstrap_is_deterministic_under_seed():
    x = np.random.default_rng(3).normal(10, 2, 200)
    a = bootstrap_ci(x, seed=42, n_boot=500)
    b = bootstrap_ci(x, seed=42, n_boot=500)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_bootstrap_ci_brackets_the_statistic():
    x = np.random.default_rng(4).normal(10, 2, 500)
    ci = bootstrap_ci(x, np.mean, 1000, seed=42)
    assert ci.lower < ci.statistic < ci.upper


def test_bootstrap_ci_covers_true_mean_for_iid_data():
    x = np.random.default_rng(5).normal(10, 2, 800)
    ci = bootstrap_ci(x, np.mean, 1500, seed=42)
    assert ci.lower <= 10.0 <= ci.upper


def test_block_length_rule():
    assert optimal_block_length(1000) == 10
    assert optimal_block_length(8) == 2
    assert optimal_block_length(2) >= 2


def test_block_bootstrap_is_deterministic():
    x = np.random.default_rng(6).normal(0, 1, 300)
    a = moving_block_bootstrap_ci(x, seed=7, n_boot=300)
    b = moving_block_bootstrap_ci(x, seed=7, n_boot=300)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_block_bootstrap_wider_than_iid_on_autocorrelated_data():
    """The core justification for using it.

    On a strongly autocorrelated series the i.i.d. bootstrap reports false
    precision. The block bootstrap should give a materially wider interval.
    """
    rng = np.random.default_rng(8)
    n = 1000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + rng.normal()      # AR(1), phi = 0.9

    iid = bootstrap_ci(x, np.mean, 800, seed=1)
    blk = moving_block_bootstrap_ci(x, np.mean, 800, seed=1)
    assert (blk.upper - blk.lower) > 1.5 * (iid.upper - iid.lower)


def test_block_bootstrap_similar_to_iid_on_independent_data():
    """Sanity check the other direction: no autocorrelation, similar widths."""
    x = np.random.default_rng(9).normal(0, 1, 1000)
    iid = bootstrap_ci(x, np.mean, 800, seed=1)
    blk = moving_block_bootstrap_ci(x, np.mean, 800, seed=1)
    ratio = (blk.upper - blk.lower) / (iid.upper - iid.lower)
    assert 0.6 < ratio < 1.7


def test_block_bootstrap_diff_brackets_observed_difference():
    rng = np.random.default_rng(10)
    a, b = rng.normal(12, 2, 400), rng.normal(10, 2, 400)
    ci = block_bootstrap_diff_ci(a, b, 600, seed=42)
    assert ci.lower < ci.statistic < ci.upper
    assert ci.statistic == pytest.approx(a.mean() - b.mean())


def test_bootstrap_handles_tiny_samples_without_crashing():
    ci = moving_block_bootstrap_ci([1.0, 2.0], n_boot=50)
    assert np.isnan(ci.lower)


# --- tests -----------------------------------------------------------------

def test_mann_whitney_detects_a_real_shift():
    rng = np.random.default_rng(11)
    a, b = rng.normal(12, 2, 300), rng.normal(10, 2, 300)
    r = mann_whitney_test(a, b, "q", "h0", "h1")
    assert r.p_value < 0.01
    assert r.effect_size > 0
    assert r.rel_diff_pct > 0


def test_mann_whitney_null_case_is_not_significant():
    rng = np.random.default_rng(12)
    a, b = rng.normal(10, 2, 300), rng.normal(10, 2, 300)
    assert mann_whitney_test(a, b, "q", "h0", "h1").p_value > 0.05


def test_mann_whitney_insufficient_data_is_nan_not_crash():
    r = mann_whitney_test([1.0], [2.0], "q", "h0", "h1")
    assert np.isnan(r.p_value)
    assert "insufficient" in r.notes


def test_kruskal_effect_size_is_bounded():
    rng = np.random.default_rng(13)
    groups = {"a": rng.normal(10, 2, 200), "b": rng.normal(12, 2, 200),
              "c": rng.normal(14, 2, 200)}
    r = kruskal_test(groups, "q", "h0", "h1")
    assert r.p_value < 0.01
    assert 0.0 <= r.effect_size <= 1.0


def test_kruskal_identical_groups_gives_negligible_effect():
    rng = np.random.default_rng(14)
    groups = {k: rng.normal(10, 2, 200) for k in "abc"}
    r = kruskal_test(groups, "q", "h0", "h1")
    assert r.effect_size < 0.05


def test_permutation_test_is_deterministic_and_bounded():
    rng = np.random.default_rng(15)
    a, b = rng.normal(11, 2, 80), rng.normal(10, 2, 80)
    p1 = permutation_test_diff_means(a, b, 500, seed=3)
    p2 = permutation_test_diff_means(a, b, 500, seed=3)
    assert p1 == p2
    assert 0 < p1 <= 1


# --- zero-inflated realism -------------------------------------------------

def test_rank_test_works_on_zero_inflated_data():
    """The situation the whole module exists for."""
    rng = np.random.default_rng(16)
    a = rng.poisson(0.5, 500) * (rng.random(500) < 0.4)
    b = rng.poisson(0.5, 500) * (rng.random(500) < 0.2)
    r = mann_whitney_test(a, b, "q", "h0", "h1")
    assert not np.isnan(r.p_value)
    assert not np.isnan(r.effect_size)
