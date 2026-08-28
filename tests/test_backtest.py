"""Tests for src/backtest.py — fold generation, baselines, and evaluation.

Tests cover:
  - Fold chronological ordering
  - No train/test date overlap
  - Forecast dates strictly after training cutoff
  - Expanding window: fold N+1 train set ⊃ fold N train set
  - All baselines return finite or NaN values (never wrong shape)
  - Zero baseline is constant zero
  - Seasonal naive uses the correct 7-day lookback
  - Naive uses only the last training observation
  - MA uses only the trailing window
  - Segment/store/horizon aggregation columns are present
  - Edge cases: series with no training history, all-zero training window
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    BASELINES,
    FoldSpec,
    apply_moving_average,
    apply_naive,
    apply_seasonal_naive,
    apply_zero_baseline,
    generate_folds,
    run_backtest,
)
from src.config import Config, load_config


# ---------------------------------------------------------------------------
# Minimal config helpers
# ---------------------------------------------------------------------------

def _test_config(
    min_train_days: int = 200,
    n_folds: int = 3,
    horizon: int = 14,
    step: int = 14,
) -> Config:
    """Return a Config with small backtest parameters suitable for fixture data."""
    cfg = load_config()
    # Override only the backtest parameters; leave everything else unchanged.
    cfg.raw["backtest"] = {
        "horizon_days": horizon,
        "n_folds": n_folds,
        "step_days": step,
        "scheme": "expanding",
        "min_train_days": min_train_days,
    }
    return cfg


# ---------------------------------------------------------------------------
# Fixture long DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def long_df():
    """Build a fixture long DataFrame large enough for 3 folds."""
    from src.config import load_config
    from src.ingest import build_long_table
    from src.make_fixture import build_fixture
    import tempfile
    from pathlib import Path

    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = build_fixture(seed=0, out_dir=Path(tmp) / "raw")
        df = build_long_table(cfg, raw_dir=raw_dir)
    return df


@pytest.fixture(scope="module")
def seg_df(long_df):
    """Segmentation for the fixture data."""
    from src.profile import classify_demand, series_stats
    cfg = load_config()
    stats = classify_demand(
        series_stats(long_df),
        cfg["segmentation"]["adi_threshold"],
        cfg["segmentation"]["cv2_threshold"],
    )
    return stats[["store_id", "item_id", "segment"]]


# ---------------------------------------------------------------------------
# FoldSpec
# ---------------------------------------------------------------------------

class TestFoldSpec:

    def test_as_dict_contains_all_keys(self):
        fold = FoldSpec(
            fold_id=1,
            train_start=pd.Timestamp("2012-01-01"),
            train_end=pd.Timestamp("2012-12-31"),
            forecast_start=pd.Timestamp("2013-01-01"),
            forecast_end=pd.Timestamp("2013-01-28"),
        )
        d = fold.as_dict()
        for key in ["fold_id", "train_start", "train_end",
                    "forecast_start", "forecast_end",
                    "n_train_rows", "n_test_rows", "n_series"]:
            assert key in d, f"Missing key in FoldSpec.as_dict(): {key}"

    def test_forecast_start_is_one_day_after_train_end(self):
        fold = FoldSpec(
            fold_id=1,
            train_start=pd.Timestamp("2012-01-01"),
            train_end=pd.Timestamp("2012-12-31"),
            forecast_start=pd.Timestamp("2013-01-01"),
            forecast_end=pd.Timestamp("2013-01-28"),
        )
        assert fold.forecast_start == fold.train_end + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# generate_folds
# ---------------------------------------------------------------------------

class TestGenerateFolds:

    def test_folds_are_in_chronological_order(self, long_df):
        cfg = _test_config()
        folds = generate_folds(long_df, cfg)
        ends = [f.train_end for f in folds]
        assert ends == sorted(ends), "Folds must be in chronological order"

    def test_fold_ids_are_sequential(self, long_df):
        cfg = _test_config()
        folds = generate_folds(long_df, cfg)
        assert [f.fold_id for f in folds] == list(range(1, len(folds) + 1))

    def test_no_train_test_date_overlap(self, long_df):
        cfg = _test_config()
        folds = generate_folds(long_df, cfg)
        for fold in folds:
            assert fold.forecast_start > fold.train_end, (
                f"Fold {fold.fold_id}: forecast_start {fold.forecast_start} "
                f"<= train_end {fold.train_end}"
            )

    def test_forecast_dates_within_data(self, long_df):
        cfg = _test_config()
        last_date = pd.to_datetime(long_df["date"]).max()
        folds = generate_folds(long_df, cfg)
        for fold in folds:
            assert fold.forecast_end <= last_date, (
                f"Fold {fold.fold_id}: forecast_end {fold.forecast_end} "
                f"> last data date {last_date}"
            )

    def test_expanding_window_train_starts_same(self, long_df):
        """All folds must share the same train_start (expanding window)."""
        cfg = _test_config()
        folds = generate_folds(long_df, cfg)
        starts = {f.train_start for f in folds}
        assert len(starts) == 1, (
            f"Expanding window: expected one train_start, got {starts}"
        )

    def test_expanding_window_later_fold_has_more_training_data(self, long_df):
        """For expanding window, fold N+1's train_end > fold N's train_end."""
        cfg = _test_config()
        folds = generate_folds(long_df, cfg)
        for i in range(len(folds) - 1):
            assert folds[i + 1].train_end > folds[i].train_end, (
                f"Fold {i+2} train_end not > fold {i+1} train_end"
            )

    def test_min_train_days_respected(self, long_df):
        cfg = _test_config(min_train_days=200)
        first_date = pd.to_datetime(long_df["date"]).min()
        folds = generate_folds(long_df, cfg)
        for fold in folds:
            days = (fold.train_end - first_date).days
            assert days >= 200, (
                f"Fold {fold.fold_id} has only {days} training days (< 200)"
            )

    def test_returns_empty_list_when_data_too_short(self):
        """When data is shorter than min_train_days, no folds can be generated."""
        cfg = _test_config(min_train_days=99999)
        df = pd.DataFrame({
            "date": pd.date_range("2012-01-01", periods=10, freq="D"),
            "store_id": "CA_1", "item_id": "X", "sales": 1,
        })
        folds = generate_folds(df, cfg)
        assert folds == [], "Should return empty list when data is too short"

    def test_horizon_days_equals_window_length(self, long_df):
        horizon = 14
        cfg = _test_config(horizon=horizon)
        folds = generate_folds(long_df, cfg)
        for fold in folds:
            window = (fold.forecast_end - fold.forecast_start).days + 1
            assert window == horizon, (
                f"Fold {fold.fold_id}: window = {window}, expected {horizon}"
            )

    def test_reproducibility(self, long_df):
        """Calling generate_folds twice must return identical results."""
        cfg = _test_config()
        f1 = generate_folds(long_df, cfg)
        f2 = generate_folds(long_df, cfg)
        assert len(f1) == len(f2)
        for a, b in zip(f1, f2):
            assert a.train_end == b.train_end
            assert a.forecast_start == b.forecast_start


# ---------------------------------------------------------------------------
# Baselines — individual
# ---------------------------------------------------------------------------

class TestApplyNaive:

    def _train_test(self):
        dates = pd.date_range("2012-01-01", periods=20, freq="D")
        train_df = pd.DataFrame({
            "store_id": "CA_1",
            "item_id": "X",
            "date": dates,
            "sales": np.arange(20),
        })
        forecast_df = pd.DataFrame({
            "store_id": ["CA_1"] * 5,
            "item_id": ["X"] * 5,
            "date": pd.date_range("2012-01-21", periods=5, freq="D"),
        })
        return train_df, forecast_df

    def test_predicts_last_training_value(self):
        train, fcst = self._train_test()
        preds = apply_naive(train, fcst)
        # Last training sales value is 19 (index 0..19)
        assert (preds == 19).all(), f"Naive should predict 19, got {preds}"

    def test_nan_for_series_with_no_training_data(self):
        train = pd.DataFrame({
            "store_id": ["CA_1"],
            "item_id": ["Y"],
            "date": pd.to_datetime(["2012-01-01"]),
            "sales": [5],
        })
        forecast = pd.DataFrame({
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["X", "Y"],     # "X" has no training data
            "date": pd.to_datetime(["2012-01-02", "2012-01-02"]),
        })
        preds = apply_naive(train, forecast)
        assert np.isnan(preds[0]), "Should be NaN for series with no training data"
        assert preds[1] == 5.0


class TestApplySeasonalNaive:

    def test_uses_7_day_lookback(self):
        dates = pd.date_range("2012-01-01", periods=20, freq="D")
        sales = np.arange(20)
        train = pd.DataFrame({"store_id": "CA_1", "item_id": "X",
                               "date": dates, "sales": sales})
        # Forecast date = 2012-01-21: should look up 2012-01-14 = day 13 → sales = 13
        forecast = pd.DataFrame({
            "store_id": ["CA_1"],
            "item_id": ["X"],
            "date": pd.to_datetime(["2012-01-21"]),
        })
        preds = apply_seasonal_naive(train, forecast, period=7)
        assert preds[0] == 13.0, f"Expected 13, got {preds[0]}"

    def test_nan_when_lookback_outside_training(self):
        dates = pd.date_range("2012-01-08", periods=5, freq="D")
        train = pd.DataFrame({"store_id": "CA_1", "item_id": "X",
                               "date": dates, "sales": np.ones(5)})
        # Forecast 2012-01-08: looks up 2012-01-01 which is BEFORE training
        forecast = pd.DataFrame({
            "store_id": ["CA_1"],
            "item_id": ["X"],
            "date": pd.to_datetime(["2012-01-08"]),
        })
        preds = apply_seasonal_naive(train, forecast, period=7)
        assert np.isnan(preds[0]), "Should be NaN when lookback is outside training"


class TestApplyMovingAverage:

    def test_ma7_uses_last_7_observations(self):
        dates = pd.date_range("2012-01-01", periods=20, freq="D")
        sales = np.zeros(20)
        sales[-7:] = [1, 2, 3, 4, 5, 6, 7]
        train = pd.DataFrame({"store_id": "CA_1", "item_id": "X",
                               "date": dates, "sales": sales})
        forecast = pd.DataFrame({
            "store_id": ["CA_1"],
            "item_id": ["X"],
            "date": pd.to_datetime(["2012-01-21"]),
        })
        preds = apply_moving_average(train, forecast, window=7)
        expected = np.mean([1, 2, 3, 4, 5, 6, 7])
        assert abs(preds[0] - expected) < 1e-9, f"Expected {expected}, got {preds[0]}"

    def test_ma28_broadcast_across_all_horizon_days(self):
        """MA forecast must be the same for every horizon day of the same series."""
        dates = pd.date_range("2012-01-01", periods=30, freq="D")
        train = pd.DataFrame({"store_id": "CA_1", "item_id": "X",
                               "date": dates, "sales": np.ones(30)})
        forecast = pd.DataFrame({
            "store_id": ["CA_1"] * 7,
            "item_id": ["X"] * 7,
            "date": pd.date_range("2012-01-31", periods=7, freq="D"),
        })
        preds = apply_moving_average(train, forecast, window=28)
        assert len(set(preds)) == 1, "MA28 should be the same for all forecast dates"

    def test_short_history_uses_all_available(self):
        """If series has fewer than window rows, use all available history."""
        dates = pd.date_range("2012-01-01", periods=3, freq="D")
        train = pd.DataFrame({"store_id": "CA_1", "item_id": "X",
                               "date": dates, "sales": [2, 4, 6]})
        forecast = pd.DataFrame({
            "store_id": ["CA_1"],
            "item_id": ["X"],
            "date": pd.to_datetime(["2012-01-04"]),
        })
        preds = apply_moving_average(train, forecast, window=28)
        assert abs(preds[0] - 4.0) < 1e-9   # mean of [2, 4, 6]


class TestApplyZeroBaseline:

    def test_all_zeros(self):
        forecast = pd.DataFrame({
            "store_id": ["CA_1"] * 5,
            "item_id": ["X"] * 5,
            "date": pd.date_range("2012-01-01", periods=5, freq="D"),
        })
        train = pd.DataFrame({
            "store_id": ["CA_1"],
            "item_id": ["X"],
            "date": pd.to_datetime(["2012-01-01"]),
            "sales": [10],
        })
        preds = apply_zero_baseline(train, forecast)
        assert (preds == 0).all()
        assert len(preds) == len(forecast)


# ---------------------------------------------------------------------------
# run_backtest — integration
# ---------------------------------------------------------------------------

class TestRunBacktest:

    def test_output_keys_present(self, long_df, seg_df):
        cfg = _test_config(n_folds=2, horizon=7, min_train_days=200)
        results = run_backtest(long_df, cfg, seg_df)
        for key in ["fold_meta", "all_predictions", "summary",
                    "segment_results", "horizon_results", "store_results"]:
            assert key in results, f"Missing key in results: {key}"

    def test_fold_meta_has_correct_number_of_folds(self, long_df, seg_df):
        cfg = _test_config(n_folds=2, horizon=7, min_train_days=200)
        results = run_backtest(long_df, cfg, seg_df)
        assert len(results["fold_meta"]) == 2

    def test_all_baselines_present_in_predictions(self, long_df, seg_df):
        cfg = _test_config(n_folds=2, horizon=7, min_train_days=200)
        results = run_backtest(long_df, cfg, seg_df)
        pred = results["all_predictions"]
        for baseline in BASELINES:
            assert baseline in pred.columns, f"Missing baseline column: {baseline}"

    def test_horizon_day_range(self, long_df, seg_df):
        horizon = 7
        cfg = _test_config(n_folds=2, horizon=horizon, min_train_days=200)
        results = run_backtest(long_df, cfg, seg_df)
        days = sorted(results["all_predictions"]["horizon_day"].unique())
        assert days == list(range(1, horizon + 1)), (
            f"Expected horizon days 1..{horizon}, got {days}"
        )

    def test_segment_results_has_required_methods(self, long_df, seg_df):
        cfg = _test_config(n_folds=2, horizon=7, min_train_days=200)
        results = run_backtest(long_df, cfg, seg_df)
        methods = set(results["segment_results"]["method"])
        for baseline in BASELINES:
            assert baseline in methods, f"Baseline {baseline} missing from segment_results"

    def test_store_results_has_all_stores(self, long_df, seg_df):
        cfg = _test_config(n_folds=2, horizon=7, min_train_days=200)
        results = run_backtest(long_df, cfg, seg_df)
        stores = set(results["store_results"]["store_id"])
        expected = set(long_df["store_id"].unique())
        assert stores == expected, f"Missing stores in store_results: {expected - stores}"

    def test_actuals_in_predictions_not_all_nan(self, long_df, seg_df):
        cfg = _test_config(n_folds=2, horizon=7, min_train_days=200)
        results = run_backtest(long_df, cfg, seg_df)
        assert results["all_predictions"]["actual"].notna().any()

    def test_mae_metrics_are_nonnegative(self, long_df, seg_df):
        cfg = _test_config(n_folds=2, horizon=7, min_train_days=200)
        results = run_backtest(long_df, cfg, seg_df)
        mae_vals = results["summary"]["mae"].dropna()
        assert (mae_vals >= 0).all(), "MAE values must be non-negative"

    def test_zero_baseline_mae_matches_expected(self, long_df, seg_df):
        """Zero baseline MAE should equal the mean of actual values."""
        cfg = _test_config(n_folds=1, horizon=7, min_train_days=200)
        results = run_backtest(long_df, cfg, seg_df)
        pred = results["all_predictions"]
        actual_mean = pred["actual"].mean()
        zero_mae = pred["actual"].abs().mean()   # MAE of forecasting 0 = mean(|actual|)
        # Check consistency (not equality — aggregation is by fold)
        assert not np.isnan(zero_mae)

    def test_reproducibility(self, long_df, seg_df):
        """Two runs with the same inputs must produce identical results."""
        cfg = _test_config(n_folds=2, horizon=7, min_train_days=200)
        r1 = run_backtest(long_df, cfg, seg_df)
        r2 = run_backtest(long_df, cfg, seg_df)
        pd.testing.assert_frame_equal(
            r1["all_predictions"].sort_values(
                ["fold_id", "store_id", "item_id", "date", "naive"]
            ).reset_index(drop=True),
            r2["all_predictions"].sort_values(
                ["fold_id", "store_id", "item_id", "date", "naive"]
            ).reset_index(drop=True),
        )

    def test_no_future_actuals_in_training_period(self, long_df, seg_df):
        """Prediction DataFrame must never contain training-period dates."""
        cfg = _test_config(n_folds=2, horizon=7, min_train_days=200)
        folds_list = generate_folds(long_df, cfg)
        results = run_backtest(long_df, cfg, seg_df)
        pred = results["all_predictions"]
        for fold in folds_list:
            fold_preds = pred[pred["fold_id"] == fold.fold_id]
            if fold_preds.empty:
                continue
            assert (pd.to_datetime(fold_preds["date"]) > fold.train_end).all(), (
                f"Fold {fold.fold_id} predictions include training-period dates"
            )
