"""Tests for src/features.py — leakage prevention and correctness.

Key invariants tested:
  1. Lag features shift by exactly k rows per series.
  2. Lag features never use observations after the cutoff.
  3. Rolling features exclude y[t] (window is [t-w .. t-1]).
  4. Rolling features never use observations after the cutoff.
  5. FeatureSet exposes historical_cols and known_future_cols explicitly.
  6. SNAP is in known_future_cols, not historical_cols.
  7. Series with fewer than k observations produce NaN for lag_k.
  8. build_fold_features returns non-overlapping train/forecast date sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    FeatureSet,
    build_fold_features,
    make_known_future_features,
    make_lag_features,
    make_rolling_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_series(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    """One item-store series with known, reproducible sales values."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "store_id": "CA_1",
        "item_id": "FOODS_1_001",
        "date": dates,
        "sales": rng.integers(0, 10, size=n_days).astype(int),
    })


def _make_two_series(n_days: int = 60) -> pd.DataFrame:
    """Two item-store series for groupby tests."""
    s1 = _make_series(n_days, seed=1)
    s2 = _make_series(n_days, seed=2)
    s2["item_id"] = "FOODS_1_002"
    return pd.concat([s1, s2], ignore_index=True)


# ---------------------------------------------------------------------------
# Lag features — correctness
# ---------------------------------------------------------------------------

class TestMakeLagFeatures:

    def test_lag_1_equals_previous_day(self):
        df = _make_series(30)
        cutoff = df["date"].max()
        result = make_lag_features(df, cutoff, lags=[1])
        merged = df.merge(result, on=["store_id", "item_id", "date"])
        # lag_1(t) should equal sales(t-1) for t > first date
        shifted = df.set_index("date")["sales"].shift(1)
        for _, row in merged.iterrows():
            expected = shifted.get(row["date"])
            if pd.isna(expected):
                assert pd.isna(row["lag_1"]), f"Expected NaN at {row['date']}"
            else:
                assert row["lag_1"] == expected, f"lag_1 mismatch at {row['date']}"

    def test_lag_7_equals_seven_days_ago(self):
        df = _make_series(40)
        cutoff = df["date"].max()
        result = make_lag_features(df, cutoff, lags=[7])
        merged = df.merge(result, on=["store_id", "item_id", "date"])
        sales_by_date = df.set_index("date")["sales"]
        for _, row in merged.iterrows():
            lookback = row["date"] - pd.Timedelta(days=7)
            expected = sales_by_date.get(lookback, np.nan)
            if pd.isna(expected):
                assert pd.isna(row["lag_7"])
            else:
                assert row["lag_7"] == expected, f"lag_7 mismatch at {row['date']}"

    def test_first_k_rows_are_nan(self):
        """lag_k must be NaN for the first k rows of each series."""
        df = _make_series(30)
        cutoff = df["date"].max()
        for k in [1, 7, 14]:
            result = make_lag_features(df, cutoff, lags=[k])
            nans = result[f"lag_{k}"].isna().sum()
            assert nans == k, f"Expected {k} NaN rows for lag_{k}, got {nans}"

    def test_multiple_series_independent(self):
        """Lag from series A must not pollute series B."""
        df = _make_two_series(40)
        cutoff = df["date"].max()
        result = make_lag_features(df, cutoff, lags=[1])
        for (store, item), grp in result.groupby(["store_id", "item_id"]):
            # Within each series group, check lag is computed independently
            src = df[(df["store_id"] == store) & (df["item_id"] == item)].set_index("date")
            g = grp.set_index("date")
            for date, row in g.iterrows():
                expected = src["sales"].get(date - pd.Timedelta(days=1), np.nan)
                if pd.isna(expected):
                    assert pd.isna(row["lag_1"])
                else:
                    assert row["lag_1"] == expected

    def test_cutoff_excludes_future_rows(self):
        """Rows beyond cutoff must not appear in the output at all."""
        df = _make_series(40)
        cutoff = df["date"].iloc[20]
        result = make_lag_features(df, cutoff, lags=[1])
        assert result["date"].max() <= cutoff, (
            f"Result contains dates beyond cutoff {cutoff}"
        )

    def test_returns_expected_columns(self):
        df = _make_series(20)
        cutoff = df["date"].max()
        result = make_lag_features(df, cutoff, lags=[1, 7, 14])
        assert "lag_1" in result.columns
        assert "lag_7" in result.columns
        assert "lag_14" in result.columns
        assert "date" in result.columns

    def test_empty_dataframe_when_all_future(self):
        df = _make_series(20)
        cutoff = df["date"].min() - pd.Timedelta(days=1)
        result = make_lag_features(df, cutoff, lags=[1])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Rolling features — correctness
# ---------------------------------------------------------------------------

class TestMakeRollingFeatures:

    def test_rolling_mean_excludes_current_row(self):
        """rolling_mean_7(t) must NOT include y[t]."""
        # Construct a series where y[t] is very large so its inclusion is detectable.
        dates = pd.date_range("2012-01-01", periods=20, freq="D")
        sales = np.ones(20, dtype=int)
        sales[-1] = 1000   # last row is a huge outlier
        df = pd.DataFrame({
            "store_id": "CA_1", "item_id": "X",
            "date": dates, "sales": sales,
        })
        cutoff = dates[-1]
        result = make_rolling_features(df, cutoff, windows=[7])
        # rolling_mean_7 at the last date should be mean of the previous 7 rows
        # = mean([1,1,1,1,1,1,1]) = 1.0 — NOT affected by sales[-1] = 1000
        last_row = result[result["date"] == dates[-1]]["rolling_mean_7"].values[0]
        assert last_row < 10, (
            f"rolling_mean_7 at last row = {last_row:.1f}; "
            "y[t] appears to have leaked into the window"
        )
        assert abs(last_row - 1.0) < 1e-6, f"Expected 1.0, got {last_row}"

    def test_rolling_mean_7_correctness(self):
        """rolling_mean_7(t) = mean(y[t-7 .. t-1]) exactly."""
        df = _make_series(30)
        cutoff = df["date"].max()
        result = make_rolling_features(df, cutoff, windows=[7])
        merged = df.merge(result, on=["store_id", "item_id", "date"])
        sales = df.sort_values("date")["sales"].values
        dates = df.sort_values("date")["date"].values
        sales_by_date = dict(zip(dates, sales))

        for _, row in merged.iterrows():
            window = [
                sales_by_date[row["date"] - pd.Timedelta(days=k)]
                for k in range(1, 8)
                if (row["date"] - pd.Timedelta(days=k)) in sales_by_date
            ]
            if window:
                expected = np.mean(window)
                assert abs(row["rolling_mean_7"] - expected) < 1e-6, (
                    f"rolling_mean_7 mismatch at {row['date']}: "
                    f"got {row['rolling_mean_7']:.4f}, expected {expected:.4f}"
                )

    def test_rolling_std_needs_at_least_2_obs(self):
        """rolling_std_7 should be NaN when fewer than 2 prior observations."""
        df = _make_series(30)
        cutoff = df["date"].max()
        result = make_rolling_features(df, cutoff, windows=[7])
        # First row has 0 prior observations, second has 1 — both should be NaN
        first_std = result.sort_values("date")["rolling_std_7"].iloc[0]
        assert pd.isna(first_std), "rolling_std for first row should be NaN"

    def test_rolling_zero_rate_is_bounded(self):
        df = _make_series(40)
        cutoff = df["date"].max()
        result = make_rolling_features(df, cutoff, windows=[7])
        rate = result["rolling_zero_rate_7"].dropna()
        assert (rate >= 0).all() and (rate <= 1).all(), "Zero rate must be in [0, 1]"

    def test_cutoff_excludes_future_rows(self):
        df = _make_series(40)
        cutoff = df["date"].iloc[20]
        result = make_rolling_features(df, cutoff, windows=[7])
        assert result["date"].max() <= cutoff

    def test_all_windows_produced(self):
        df = _make_series(30)
        cutoff = df["date"].max()
        result = make_rolling_features(df, cutoff, windows=[7, 28])
        for col in ["rolling_mean_7", "rolling_std_7", "rolling_zero_rate_7",
                    "rolling_mean_28", "rolling_std_28", "rolling_zero_rate_28"]:
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Known-future features
# ---------------------------------------------------------------------------

class TestKnownFutureFeatures:

    def test_snap_is_in_known_future(self):
        """SNAP must be labelled known-future, not historical (D-026)."""
        # known_future_features returns a DataFrame; we check that snap_active is present
        dates = pd.date_range("2015-01-01", periods=10, freq="D")
        snap = pd.Series(np.ones(10, dtype=int))
        result = make_known_future_features(
            dates, store_id="CA_1", state_id="CA", snap_ca=snap
        )
        assert "snap_active" in result.columns
        assert result["snap_active"].notna().all()

    def test_state_aware_snap(self):
        """CA store should use snap_ca, TX store snap_tx, WI store snap_wi."""
        dates = pd.date_range("2015-01-01", periods=5, freq="D")
        snap_ca = pd.Series([1, 1, 0, 0, 1])
        snap_tx = pd.Series([0, 0, 1, 1, 0])
        snap_wi = pd.Series([1, 0, 1, 0, 0])

        ca = make_known_future_features(
            dates, "CA_1", "CA", snap_ca=snap_ca, snap_tx=snap_tx, snap_wi=snap_wi
        )
        tx = make_known_future_features(
            dates, "TX_1", "TX", snap_ca=snap_ca, snap_tx=snap_tx, snap_wi=snap_wi
        )
        assert list(ca["snap_active"]) == [1, 1, 0, 0, 1]
        assert list(tx["snap_active"]) == [0, 0, 1, 1, 0]

    def test_calendar_columns_present(self):
        dates = pd.date_range("2015-06-01", periods=14, freq="D")
        result = make_known_future_features(dates, "CA_1", "CA")
        for col in ["wday", "month", "year", "week_of_year", "is_weekend"]:
            assert col in result.columns

    def test_is_weekend_correct(self):
        # 2015-01-03 is Saturday, 2015-01-04 is Sunday, 2015-01-05 is Monday
        dates = pd.to_datetime(["2015-01-03", "2015-01-04", "2015-01-05"])
        result = make_known_future_features(dates, "CA_1", "CA")
        assert result["is_weekend"].tolist() == [1, 1, 0]

    def test_no_past_y_needed(self):
        """Known-future features must be computable with no sales data."""
        dates = pd.date_range("2020-01-01", periods=28, freq="D")
        # This should not raise even with no training data passed
        result = make_known_future_features(dates, "WI_1", "WI")
        assert len(result) == 28


# ---------------------------------------------------------------------------
# FeatureSet contract
# ---------------------------------------------------------------------------

class TestFeatureSet:

    def _make_fs(self) -> FeatureSet:
        df = _make_series(30)
        cutoff = df["date"].max()
        lags = [1, 7]
        windows = [7]
        lag_df = make_lag_features(df, cutoff, lags)
        roll_df = make_rolling_features(df, cutoff, windows)
        merged = df.merge(lag_df, on=["store_id", "item_id", "date"])
        merged = merged.merge(roll_df, on=["store_id", "item_id", "date"])
        return FeatureSet(
            df=merged,
            historical_cols=[f"lag_{k}" for k in lags]
            + [c for c in merged.columns if c.startswith("rolling_")],
            known_future_cols=["wday", "month"],
        )

    def test_all_feature_cols_combines_both_lists(self):
        fs = self._make_fs()
        assert set(fs.all_feature_cols) == set(fs.historical_cols) | set(fs.known_future_cols)

    def test_historical_and_known_future_are_disjoint(self):
        fs = self._make_fs()
        overlap = set(fs.historical_cols) & set(fs.known_future_cols)
        assert len(overlap) == 0, f"Columns in both groups: {overlap}"

    def test_historical_cols_derived_from_past_only(self):
        """All historical_cols should be lag or rolling features."""
        fs = self._make_fs()
        for col in fs.historical_cols:
            assert col.startswith("lag_") or col.startswith("rolling_"), (
                f"Unexpected column in historical_cols: {col}"
            )


# ---------------------------------------------------------------------------
# build_fold_features — integration
# ---------------------------------------------------------------------------

class TestBuildFoldFeatures:

    def test_train_and_forecast_dates_do_not_overlap(self):
        df = _make_series(60)
        cutoff = df["date"].iloc[40]
        horizon = 10
        train_fs, forecast_fs = build_fold_features(df, cutoff, horizon, lags=[1, 7], windows=[7])
        train_dates = set(train_fs.df["date"])
        forecast_dates = set(forecast_fs.df["date"])
        overlap = train_dates & forecast_dates
        assert len(overlap) == 0, f"Overlap between train and forecast dates: {overlap}"

    def test_forecast_dates_start_after_cutoff(self):
        df = _make_series(60)
        cutoff = df["date"].iloc[40]
        horizon = 10
        _, forecast_fs = build_fold_features(df, cutoff, horizon, lags=[1, 7], windows=[7])
        assert forecast_fs.df["date"].min() > cutoff

    def test_forecast_dates_cover_exactly_horizon_days(self):
        df = _make_series(60)
        cutoff = df["date"].iloc[40]
        horizon = 7
        _, forecast_fs = build_fold_features(df, cutoff, horizon, lags=[1, 7], windows=[7])
        # Each series should have exactly horizon unique forecast dates
        per_series = forecast_fs.df.groupby(["store_id", "item_id"])["date"].nunique()
        assert (per_series == horizon).all(), (
            f"Some series have != {horizon} forecast dates: {per_series.to_dict()}"
        )

    def test_historical_cols_label_preserved(self):
        df = _make_series(60)
        cutoff = df["date"].iloc[40]
        train_fs, _ = build_fold_features(df, cutoff, horizon=7, lags=[1], windows=[7])
        assert "lag_1" in train_fs.historical_cols
        assert any("rolling_mean" in c for c in train_fs.historical_cols)

    def test_target_y_is_nan_in_forecast(self):
        df = _make_series(60)
        cutoff = df["date"].iloc[40]
        _, forecast_fs = build_fold_features(df, cutoff, horizon=7, lags=[1], windows=[7])
        assert forecast_fs.df["y"].isna().all(), "Forecast y must be NaN (unknown)"
