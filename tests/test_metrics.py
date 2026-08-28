"""Tests for src/metrics.py — correctness and edge-case behaviour.

Every metric is verified against a hand-calculated example so the numbers
come from first principles, not from the code being tested.

Edge cases tested:
  - Empty arrays → NaN
  - All-zero actuals → WAPE NaN (denominator = 0)
  - All-zero actuals → MAPE NaN (no valid rows)
  - NaN pairs are dropped before computation
  - Perfect forecast → error metrics = 0
  - sMAPE when both actual and forecast are 0 → term = 0 (not undefined)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import (
    bias,
    evaluate_predictions,
    mae,
    mape,
    rmse,
    smape,
    summarise_metrics,
    wape,
)


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

class TestMAE:

    def test_hand_calculated(self):
        # |3-1| + |3-5| + |3-4| = 2 + 2 + 1 = 5; mean = 5/3
        actual = [1, 5, 4]
        forecast = [3, 3, 3]
        assert abs(mae(actual, forecast) - 5 / 3) < 1e-9

    def test_perfect_forecast_is_zero(self):
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_empty_returns_nan(self):
        assert np.isnan(mae([], []))

    def test_nan_pairs_dropped(self):
        # Only the non-NaN pair (1, 2) is used: |1-2| = 1
        assert mae([1, np.nan], [2, 5]) == 1.0

    def test_symmetry(self):
        # MAE(a, f) == MAE(f, a)
        a, f = [2, 4, 6], [1, 5, 3]
        assert abs(mae(a, f) - mae(f, a)) < 1e-9

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mae([1, 2], [1])


# ---------------------------------------------------------------------------
# WAPE
# ---------------------------------------------------------------------------

class TestWAPE:

    def test_hand_calculated(self):
        # actual = [2, 4]; forecast = [1, 5]
        # sum(|actual-forecast|) = |2-1| + |4-5| = 1 + 1 = 2
        # sum(|actual|) = 6
        # WAPE = 2/6 = 1/3
        assert abs(wape([2, 4], [1, 5]) - 1 / 3) < 1e-9

    def test_zero_denominator_returns_nan(self):
        """When all actuals are zero WAPE is undefined."""
        result = wape([0, 0, 0], [1, 2, 3])
        assert np.isnan(result), f"Expected NaN, got {result}"

    def test_perfect_forecast_is_zero(self):
        assert wape([3, 5, 7], [3, 5, 7]) == 0.0

    def test_empty_returns_nan(self):
        assert np.isnan(wape([], []))

    def test_all_nan_actuals_returns_nan(self):
        assert np.isnan(wape([np.nan, np.nan], [1.0, 2.0]))

    def test_partial_nan_dropped(self):
        # Only pair (4, 2) is valid: |4-2|/4 → WAPE = 2/4 = 0.5
        result = wape([4, np.nan], [2, 100])
        assert abs(result - 0.5) < 1e-9

    def test_wape_is_not_symmetric(self):
        """WAPE(a, f) != WAPE(f, a) in general (denominator depends on a)."""
        # actual=[1, 9], forecast=[4, 4]
        # WAPE(a,f) = (|1-4|+|9-4|)/(1+9) = (3+5)/10 = 0.8
        # WAPE(f,a) = (|4-1|+|4-9|)/(4+4) = (3+5)/8 = 1.0
        a, f = [1, 9], [4, 4]
        assert abs(wape(a, f) - wape(f, a)) > 1e-9


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

class TestRMSE:

    def test_hand_calculated(self):
        # errors = [1, -1, 2]; MSE = (1+1+4)/3 = 2; RMSE = sqrt(2)
        actual = [1, 5, 4]
        forecast = [2, 4, 6]
        expected = math.sqrt((1 + 1 + 4) / 3)
        assert abs(rmse(actual, forecast) - expected) < 1e-9

    def test_perfect_forecast_is_zero(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_empty_returns_nan(self):
        assert np.isnan(rmse([], []))

    def test_rmse_geq_mae(self):
        """RMSE >= MAE for any input (by Jensen's inequality)."""
        a = [1, 5, 0, 3, 2]
        f = [2, 3, 1, 0, 4]
        assert rmse(a, f) >= mae(a, f) - 1e-9


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------

class TestBias:

    def test_over_forecast_is_positive(self):
        # forecast is always 1 above actual → bias = +1
        actual = [1, 2, 3]
        forecast = [2, 3, 4]
        assert abs(bias(actual, forecast) - 1.0) < 1e-9

    def test_under_forecast_is_negative(self):
        actual = [2, 3, 4]
        forecast = [1, 2, 3]
        assert abs(bias(actual, forecast) - (-1.0)) < 1e-9

    def test_unbiased_is_zero(self):
        # errors cancel out: +1, -1, 0 → mean = 0
        assert abs(bias([1, 3, 4], [2, 2, 4])) < 1e-9

    def test_empty_returns_nan(self):
        assert np.isnan(bias([], []))


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------

class TestSMAPE:

    def test_hand_calculated(self):
        # actual=1, forecast=3: 2*|1-3|/(|1|+|3|) = 4/4 = 1.0
        # actual=5, forecast=5: 0
        # mean = 0.5
        assert abs(smape([1, 5], [3, 5]) - 0.5) < 1e-9

    def test_both_zero_contributes_zero(self):
        """When actual=0 and forecast=0, the term should be 0 (not NaN)."""
        result = smape([0, 2], [0, 2])
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_forecast_zero_actual_nonzero(self):
        # actual=4, forecast=0: 2*4/(4+0) = 2.0
        assert abs(smape([4], [0]) - 2.0) < 1e-9

    def test_perfect_forecast_is_zero(self):
        assert smape([1, 2, 3], [1, 2, 3]) == 0.0

    def test_bounded_between_0_and_2(self):
        import numpy as np
        rng = np.random.default_rng(42)
        a = rng.integers(0, 20, 100).astype(float)
        f = rng.integers(0, 20, 100).astype(float)
        result = smape(a, f)
        assert 0 <= result <= 2.0, f"sMAPE = {result} is out of [0, 2]"


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------

class TestMAPE:

    def test_only_nonzero_actuals_used(self):
        """MAPE must silently exclude zero-actual rows."""
        # actual = [0, 4]; only the row (4, 2) is valid
        # |4-2|/4 = 0.5
        result = mape([0, 4], [5, 2])
        assert abs(result - 0.5) < 1e-9, f"Expected 0.5, got {result}"

    def test_all_zero_actuals_returns_nan(self):
        result = mape([0, 0, 0], [1, 2, 3])
        assert np.isnan(result), "MAPE must return NaN when all actuals are zero"

    def test_perfect_forecast_is_zero(self):
        assert mape([1, 2, 3], [1, 2, 3]) == 0.0

    def test_hand_calculated(self):
        # actual=[2, 4], forecast=[1, 6]
        # |2-1|/2 + |4-6|/4 = 0.5 + 0.5 = 1.0; mean = 0.5
        assert abs(mape([2, 4], [1, 6]) - 0.5) < 1e-9

    def test_empty_returns_nan(self):
        assert np.isnan(mape([], []))


# ---------------------------------------------------------------------------
# summarise_metrics
# ---------------------------------------------------------------------------

class TestSummariseMetrics:

    def test_all_keys_present(self):
        result = summarise_metrics([1, 2, 3], [1, 2, 3])
        for key in ["mae", "wape", "rmse", "bias", "smape", "mape", "n_obs"]:
            assert key in result, f"Missing key: {key}"

    def test_perfect_forecast_all_zeros(self):
        result = summarise_metrics([2, 4, 6], [2, 4, 6])
        assert result["mae"] == 0.0
        assert result["wape"] == 0.0
        assert result["rmse"] == 0.0
        assert result["bias"] == 0.0

    def test_prefix_applied(self):
        result = summarise_metrics([1, 2], [1, 2], prefix="naive_")
        assert all(k.startswith("naive_") for k in result)

    def test_n_obs_correct(self):
        result = summarise_metrics([1, np.nan, 3], [1, 2, 3])
        assert result["n_obs"] == 2   # one NaN pair dropped


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------

class TestEvaluatePredictions:

    def _make_df(self):
        import pandas as pd
        return pd.DataFrame({
            "actual": [1, 2, 3, 4, 5, 6],
            "method_a": [1, 2, 3, 4, 5, 6],   # perfect
            "method_b": [2, 3, 4, 5, 6, 7],   # always +1
            "store_id": ["A", "A", "A", "B", "B", "B"],
        })

    def test_evaluates_all_methods(self):
        import pandas as pd
        df = self._make_df()
        result = evaluate_predictions(df, method_cols=["method_a", "method_b"])
        assert set(result["method"]) == {"method_a", "method_b"}

    def test_grouped_evaluation(self):
        import pandas as pd
        df = self._make_df()
        result = evaluate_predictions(
            df, method_cols=["method_a"], group_cols=["store_id"]
        )
        assert "store_id" in result.columns
        assert len(result) == 2   # one row per store per method
