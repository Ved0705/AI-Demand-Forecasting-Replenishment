"""Adversarial leakage tests for Phase 5 forecasting models."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.forecasting_models import (
    apply_croston, apply_sba, apply_tsb, apply_ses, GlobalXGBoostForecaster
)
from src.features import build_fold_features
from src.model_runner import run_phase5_backtest

@pytest.fixture
def dummy_config():
    from src.config import load_config
    cfg = load_config()
    cfg.raw["backtest"] = {
        "scheme": "expanding",
        "horizon_days": 7,
        "n_folds": 1,
        "step_days": 7,
        "min_train_days": 30
    }
    return cfg

@pytest.fixture
def dummy_data():
    dates = pd.date_range("2015-01-01", "2015-02-15", freq="D") # 46 days
    n_series = 2
    
    rows = []
    for s_id in range(n_series):
        for d in dates:
            rows.append({
                "store_id": "CA_1",
                "item_id": f"ITEM_{s_id}",
                "cat_id": "HOBBIES",
                "dept_id": "HOBBIES_1",
                "state_id": "CA",
                "date": d,
                "sales": np.random.randint(0, 5),
                "wday": d.dayofweek + 1,
                "month": d.month,
                "year": d.year,
                "week_of_year": d.isocalendar().week,
                "is_weekend": int(d.dayofweek in [5, 6]),
                "snap_active": np.random.randint(0, 2),
                "is_event": 0
            })
    return pd.DataFrame(rows)

def test_intermittent_models_leakage(dummy_data):
    """Ensure per-series models only see data up to the cutoff."""
    cutoff = pd.Timestamp("2015-01-31")
    
    # Base dataset
    df_base = dummy_data.copy()
    
    # Contaminated dataset (mutate post-cutoff sales to 9999)
    df_contam = dummy_data.copy()
    df_contam.loc[df_contam["date"] > cutoff, "sales"] = 9999
    
    # They should produce the exact same forecast for a given series
    # because the runner slices the data before passing to models.
    # We test the model functions themselves here directly by simulating the slice.
    
    y_base = df_base.loc[(df_base["date"] <= cutoff) & (df_base["item_id"] == "ITEM_0"), "sales"].values
    y_contam = df_contam.loc[(df_contam["date"] <= cutoff) & (df_contam["item_id"] == "ITEM_0"), "sales"].values
    
    # Trivial check that the slice worked identically
    np.testing.assert_array_equal(y_base, y_contam)
    
    # Model check
    assert apply_croston(y_base) == apply_croston(y_contam)
    assert apply_sba(y_base) == apply_sba(y_contam)
    assert apply_tsb(y_base) == apply_tsb(y_contam)
    assert apply_ses(y_base) == apply_ses(y_contam)

def test_full_pipeline_leakage(dummy_data, dummy_config):
    """End-to-end leakage test for Phase 5 runner."""
    cutoff = pd.Timestamp("2015-02-08") # Last 7 days are forecast
    
    df_base = dummy_data.copy()
    
    df_contam = dummy_data.copy()
    df_contam.loc[df_contam["date"] > cutoff, "sales"] = 9999
    
    # Run backtest on both
    res_base = run_phase5_backtest(df_base, dummy_config)
    res_contam = run_phase5_backtest(df_contam, dummy_config)
    
    pred_base = res_base["all_predictions"]
    pred_contam = res_contam["all_predictions"]
    
    # All predictions should be identical between the two runs
    # even though actuals (which are merged in for metrics) differ.
    
    cols_to_check = ["croston", "sba", "tsb", "ses", "xgboost"]
    
    for col in cols_to_check:
        np.testing.assert_allclose(
            pred_base[col].values,
            pred_contam[col].values,
            err_msg=f"Leakage detected in {col}"
        )
