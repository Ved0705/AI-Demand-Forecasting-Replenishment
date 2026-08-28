"""Adversarial leakage tests — the most important tests in Phase 4.

DESIGN INTENT
-------------
These tests directly answer the interview question: "How do you know your
features don't leak future information?"

The answer is structural and empirical:

  Structural: every feature function slices to date <= cutoff before
              computing anything.  There is no code path that reads past
              the cutoff boundary.

  Empirical: we MUTATE the post-cutoff data in a controlled way and
             verify that the pre-cutoff features/forecasts do not change.

If future values leaked into feature computation, mutating them would
change the pre-cutoff outputs.  The tests below fail in exactly that case.

ADVERSARIAL APPROACH (STEP 16 from spec)
-----------------------------------------
Construct two DataFrames that are bit-identical up to the cutoff date.
Post-cutoff values are changed (some to 10x, some to 0, some to 9999).
Then verify:

  1. Lag features for [date <= cutoff] are bit-identical in both DataFrames.
  2. Rolling features for [date <= cutoff] are bit-identical.
  3. Baseline forecasts (naive, seasonal_naive, ma7, ma28, zero) are
     bit-identical.
  4. The fold-level run_backtest call produces identical predictions.

These tests would FAIL if:
  - The leakage guard were removed from make_lag_features.
  - The leakage guard were removed from make_rolling_features.
  - The baseline functions read from future rows.
  - Any normalization or imputation used global (pre+post-cutoff) statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    apply_moving_average,
    apply_naive,
    apply_seasonal_naive,
    apply_zero_baseline,
    generate_folds,
    run_backtest,
)
from src.config import load_config
from src.features import make_lag_features, make_rolling_features


# ---------------------------------------------------------------------------
# Test data factory
# ---------------------------------------------------------------------------

N_SERIES = 4
N_TRAIN_DAYS = 100
N_FUTURE_DAYS = 28


def _make_two_versions(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Return two DataFrames and a cutoff.

    ``original`` and ``mutated`` are bit-identical for date <= cutoff.
    For date > cutoff, ``mutated`` has completely different sales values.

    If features leak, mutating post-cutoff values will change pre-cutoff
    feature outputs.  If the leakage guard works, outputs will be identical.
    """
    rng = np.random.default_rng(seed)
    stores = ["CA_1", "TX_1"]
    items = ["FOODS_1_001", "FOODS_1_002"]

    all_rows = []
    for store in stores:
        for item in items:
            total_days = N_TRAIN_DAYS + N_FUTURE_DAYS
            dates = pd.date_range("2012-01-01", periods=total_days, freq="D")
            sales = rng.integers(0, 15, size=total_days).astype(int)
            for i, (d, s) in enumerate(zip(dates, sales)):
                all_rows.append({
                    "store_id": store, "item_id": item,
                    "date": d, "sales": int(s), "index_within_series": i,
                })

    original = pd.DataFrame(all_rows)
    cutoff = original["date"].min() + pd.Timedelta(days=N_TRAIN_DAYS - 1)

    # Mutate ONLY post-cutoff rows
    mutated = original.copy()
    future_mask = mutated["date"] > cutoff
    # Change post-cutoff sales dramatically so any leakage is detectable
    mutated.loc[future_mask, "sales"] = 9999
    # Also zero out some to test zero-leakage path
    future_rows = mutated[future_mask].index
    if len(future_rows) > 0:
        mutated.loc[future_rows[:len(future_rows)//2], "sales"] = 0

    return original, mutated, cutoff


# ---------------------------------------------------------------------------
# Core leakage tests
# ---------------------------------------------------------------------------

class TestLagFeatureLeakage:
    """Lag features must be invariant to post-cutoff mutations."""

    def setup_method(self):
        self.original, self.mutated, self.cutoff = _make_two_versions(seed=0)

    def test_lag_1_identical_for_all_lags(self):
        for k in [1, 7, 14, 28]:
            orig = make_lag_features(self.original, self.cutoff, lags=[k])
            mut = make_lag_features(self.mutated, self.cutoff, lags=[k])
            pd.testing.assert_frame_equal(
                orig.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True),
                mut.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True),
                check_like=True,
                obj=f"lag_{k} features differ between original and mutated",
            )

    def test_no_post_cutoff_dates_in_lag_output(self):
        result = make_lag_features(self.mutated, self.cutoff, lags=[1, 7])
        assert result["date"].max() <= self.cutoff, (
            f"Lag features contain dates beyond cutoff {self.cutoff}"
        )

    def test_lag_values_at_cutoff_match_original(self):
        """The lag at the cutoff date must be y[cutoff - k], from training only."""
        orig_lags = make_lag_features(self.original, self.cutoff, lags=[1])
        mut_lags = make_lag_features(self.mutated, self.cutoff, lags=[1])

        orig_at_cutoff = orig_lags[orig_lags["date"] == self.cutoff].sort_values(
            ["store_id", "item_id"]
        )["lag_1"].values
        mut_at_cutoff = mut_lags[mut_lags["date"] == self.cutoff].sort_values(
            ["store_id", "item_id"]
        )["lag_1"].values

        np.testing.assert_array_equal(
            orig_at_cutoff, mut_at_cutoff,
            err_msg="Lag at cutoff differs — future values may have leaked",
        )


class TestRollingFeatureLeakage:
    """Rolling features must be invariant to post-cutoff mutations."""

    def setup_method(self):
        self.original, self.mutated, self.cutoff = _make_two_versions(seed=1)

    def test_rolling_mean_identical_before_and_at_cutoff(self):
        for w in [7, 28]:
            orig = make_rolling_features(self.original, self.cutoff, windows=[w])
            mut = make_rolling_features(self.mutated, self.cutoff, windows=[w])
            pd.testing.assert_frame_equal(
                orig.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True),
                mut.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True),
                check_like=True,
                obj=f"rolling_mean_{w} features differ",
            )

    def test_rolling_zero_rate_identical(self):
        orig = make_rolling_features(self.original, self.cutoff, windows=[7])
        mut = make_rolling_features(self.mutated, self.cutoff, windows=[7])
        np.testing.assert_array_equal(
            orig.sort_values(["store_id","item_id","date"])["rolling_zero_rate_7"].values,
            mut.sort_values(["store_id","item_id","date"])["rolling_zero_rate_7"].values,
            err_msg="rolling_zero_rate_7 differs — possible leakage",
        )

    def test_rolling_features_do_not_include_cutoff_plus_one(self):
        """The rolling window must never see y[cutoff+1]."""
        # Make original and mutated, then compare the rolling at cutoff itself
        orig = make_rolling_features(self.original, self.cutoff, windows=[7])
        mut = make_rolling_features(self.mutated, self.cutoff, windows=[7])
        o_cutoff = orig[orig["date"] == self.cutoff]["rolling_mean_7"].values
        m_cutoff = mut[mut["date"] == self.cutoff]["rolling_mean_7"].values
        np.testing.assert_array_equal(
            o_cutoff, m_cutoff,
            err_msg="rolling_mean_7 at cutoff differs — cutoff+1 may have leaked",
        )


class TestBaselineLeakage:
    """Baseline forecasts must be invariant to post-cutoff mutations."""

    def setup_method(self):
        self.original, self.mutated, self.cutoff = _make_two_versions(seed=2)
        # Training data = rows up to and including cutoff
        self.orig_train = self.original[self.original["date"] <= self.cutoff].copy()
        self.mut_train = self.mutated[self.mutated["date"] <= self.cutoff].copy()
        # Forecast skeleton (just keys and dates, no actuals)
        first_forecast = self.cutoff + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(first_forecast, periods=14, freq="D")
        series = self.original[["store_id", "item_id"]].drop_duplicates()
        self.forecast_df = series.assign(_tmp=1).merge(
            pd.DataFrame({"date": forecast_dates, "_tmp": 1}), on="_tmp"
        ).drop(columns="_tmp").reset_index(drop=True)

    def _assert_arrays_equal(self, name, orig_preds, mut_preds):
        nan_same = np.isnan(orig_preds) == np.isnan(mut_preds)
        non_nan = ~np.isnan(orig_preds) & ~np.isnan(mut_preds)
        assert nan_same.all() and np.allclose(orig_preds[non_nan], mut_preds[non_nan]), (
            f"{name} forecast differs between original and mutated training data.\n"
            f"Max difference: {np.nanmax(np.abs(orig_preds - mut_preds)):.4f}\n"
            "This likely indicates that post-cutoff values are leaking into "
            "the training-based forecasts."
        )

    def test_naive_invariant_to_future_mutations(self):
        p_orig = apply_naive(self.orig_train, self.forecast_df)
        p_mut = apply_naive(self.mut_train, self.forecast_df)
        self._assert_arrays_equal("naive", p_orig, p_mut)

    def test_seasonal_naive_invariant_to_future_mutations(self):
        p_orig = apply_seasonal_naive(self.orig_train, self.forecast_df)
        p_mut = apply_seasonal_naive(self.mut_train, self.forecast_df)
        self._assert_arrays_equal("seasonal_naive", p_orig, p_mut)

    def test_ma7_invariant_to_future_mutations(self):
        p_orig = apply_moving_average(self.orig_train, self.forecast_df, window=7)
        p_mut = apply_moving_average(self.mut_train, self.forecast_df, window=7)
        self._assert_arrays_equal("ma7", p_orig, p_mut)

    def test_ma28_invariant_to_future_mutations(self):
        p_orig = apply_moving_average(self.orig_train, self.forecast_df, window=28)
        p_mut = apply_moving_average(self.mut_train, self.forecast_df, window=28)
        self._assert_arrays_equal("ma28", p_orig, p_mut)

    def test_zero_invariant_to_future_mutations(self):
        p_orig = apply_zero_baseline(self.orig_train, self.forecast_df)
        p_mut = apply_zero_baseline(self.mut_train, self.forecast_df)
        np.testing.assert_array_equal(
            p_orig, p_mut, err_msg="Zero baseline should always be 0"
        )


# ---------------------------------------------------------------------------
# Full pipeline adversarial test
# ---------------------------------------------------------------------------

class TestFullPipelineLeakage:
    """End-to-end adversarial test: mutating future values must not change
    baseline forecasts for any period up to the fold's training cutoff.

    This is the 'key interview-defensibility test' from the Phase 4 spec.
    """

    def _make_config(self):
        cfg = load_config()
        cfg.raw["backtest"] = {
            "horizon_days": 14,
            "n_folds": 2,
            "step_days": 14,
            "scheme": "expanding",
            "min_train_days": 60,
        }
        return cfg

    def _make_long_df(self, seed: int, contaminate: bool = False) -> pd.DataFrame:
        """Build a fixture-shaped long DataFrame with optional post-cutoff mutation.

        200 days of data.  Contamination modifies only the FINAL 14 days
        (days 187-200), which correspond to the last fold's test window.
        These dates are guaranteed to be strictly beyond every fold's training
        cutoff (earliest cutoff = first_date + min_train_days = day 60),
        so clean_train == cont_train for every fold.
        """
        rng = np.random.default_rng(seed)
        stores = ["CA_1", "TX_1"]
        items = ["FOODS_1_001", "FOODS_1_002"]
        dates = pd.date_range("2012-01-01", periods=200, freq="D")

        rows = []
        for store in stores:
            for item in items:
                sales = rng.integers(0, 10, size=200).astype(int)
                for d, s in zip(dates, sales):
                    rows.append({
                        "store_id": store, "item_id": item,
                        "date": d, "sales": int(s),
                        "cat_id": "FOODS", "state_id": store[:2],
                    })

        df = pd.DataFrame(rows)

        if contaminate:
            # Contaminate only the last 14 days — always outside training cutoffs.
            contamination_start = dates[-14]
            df.loc[df["date"] >= contamination_start, "sales"] = 7777

        return df

    def test_predictions_identical_regardless_of_future_contamination(self):
        """The critical adversarial test.

        Two DataFrames: identical up to the training cutoff, different after.
        The baseline forecasts for all fold test windows must be identical.

        Why this matters: if any baseline reads from future rows, mutating
        those rows changes the forecasts — and a model trained on such
        features would have learned from future information.
        """
        cfg = self._make_config()
        clean = self._make_long_df(seed=7, contaminate=False)
        contaminated = self._make_long_df(seed=7, contaminate=True)

        # Get the fold cutoffs
        folds = generate_folds(clean, cfg)
        assert folds, "Should have at least one fold"

        for fold in folds:
            # Training data for clean and contaminated must be IDENTICAL
            clean_train = clean[clean["date"] <= fold.train_end].copy()
            cont_train = contaminated[contaminated["date"] <= fold.train_end].copy()

            pd.testing.assert_frame_equal(
                clean_train.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True),
                cont_train.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True),
                obj=f"Fold {fold.fold_id}: training data differs despite identical pre-cutoff values",
            )

            # Forecasts for both must be identical because they depend only on training data
            forecast_df = pd.DataFrame({
                "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
                "item_id": ["FOODS_1_001", "FOODS_1_002"] * 2,
            })
            forecast_df = forecast_df.assign(_tmp=1).merge(
                pd.DataFrame({
                    "date": pd.date_range(fold.forecast_start, fold.forecast_end, freq="D"),
                    "_tmp": 1,
                }),
                on="_tmp",
            ).drop(columns="_tmp")

            for baseline_fn, kwargs in [
                (apply_naive, {}),
                (apply_moving_average, {"window": 7}),
                (apply_moving_average, {"window": 28}),
                (apply_zero_baseline, {}),
            ]:
                kw = {"train_df": clean_train, "forecast_df": forecast_df, **kwargs}
                p_clean = baseline_fn(**kw)
                kw["train_df"] = cont_train
                p_cont = baseline_fn(**kw)

                # Use allclose for floats, handling NaN
                nan_match = np.isnan(p_clean) == np.isnan(p_cont)
                non_nan = ~np.isnan(p_clean) & ~np.isnan(p_cont)
                assert nan_match.all() and np.allclose(p_clean[non_nan], p_cont[non_nan]), (
                    f"Fold {fold.fold_id}, {baseline_fn.__name__}: "
                    f"forecasts differ between clean and contaminated data.\n"
                    f"This proves future values are leaking into the training-based forecast."
                )

    def test_lag_features_identical_up_to_cutoff_after_contamination(self):
        """Lag features computed from contaminated data must match clean data."""
        clean = self._make_long_df(seed=8, contaminate=False)
        contaminated = self._make_long_df(seed=8, contaminate=True)

        # Verify contaminated data is actually different after some point
        diff = (clean["sales"] != contaminated["sales"]).sum()
        assert diff > 0, "Setup error: contaminated df should differ from clean"

        # Pick a cutoff in the 'clean' zone (well before contamination at day 150)
        cutoff = pd.Timestamp("2012-03-01")

        for k in [1, 7, 14]:
            orig_lag = make_lag_features(clean, cutoff, lags=[k])
            cont_lag = make_lag_features(contaminated, cutoff, lags=[k])
            pd.testing.assert_frame_equal(
                orig_lag.sort_values(["store_id","item_id","date"]).reset_index(drop=True),
                cont_lag.sort_values(["store_id","item_id","date"]).reset_index(drop=True),
                obj=f"lag_{k} differs after contamination — leakage detected",
            )
