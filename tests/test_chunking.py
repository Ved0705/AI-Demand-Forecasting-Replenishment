import numpy as np
import pandas as pd
import pytest

from src.features import build_fold_features, FeatureSet

@pytest.fixture
def dummy_data():
    dates = pd.date_range("2015-01-01", "2015-02-15", freq="D")
    n_series = 5
    
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
                "sales": float(np.random.randint(0, 5)),
                "wday": d.dayofweek + 1,
                "month": d.month,
                "year": d.year,
                "week_of_year": d.isocalendar().week,
                "is_weekend": int(d.dayofweek in [5, 6]),
                "snap_active": np.random.randint(0, 2),
                "is_event": 0
            })
    return pd.DataFrame(rows)

def test_chunked_feature_generation_equivalence(dummy_data):
    """Verify that chunked feature generation exactly matches global generation."""
    cutoff = pd.Timestamp("2015-01-31")
    horizon = 14
    key_cols = ["store_id", "item_id"]
    
    # 1. Global (all at once)
    train_global, forecast_global = build_fold_features(
        dummy_data, cutoff, horizon, key_cols=key_cols
    )
    
    # 2. Chunked
    keys_df = dummy_data[key_cols].drop_duplicates().reset_index(drop=True)
    chunk_size = 2
    chunks = [keys_df.iloc[i:i + chunk_size] for i in range(0, len(keys_df), chunk_size)]
    
    train_fs_list = []
    forecast_fs_list = []
    hist_cols, cal_cols = [], []
    
    for chunk in chunks:
        chunk_df = dummy_data.merge(chunk, on=key_cols, how="inner")
        t_fs, f_fs = build_fold_features(
            chunk_df, cutoff, horizon, key_cols=key_cols
        )
        train_fs_list.append(t_fs.df)
        forecast_fs_list.append(f_fs.df)
        if not hist_cols and t_fs.historical_cols:
            hist_cols = t_fs.historical_cols
            cal_cols = t_fs.known_future_cols
            
    train_chunked = pd.concat(train_fs_list, ignore_index=True)
    forecast_chunked = pd.concat(forecast_fs_list, ignore_index=True)
    
    # Sort both to ensure identical ordering
    sort_cols = key_cols + ["date"]
    train_global.df = train_global.df.sort_values(sort_cols).reset_index(drop=True)
    train_chunked = train_chunked.sort_values(sort_cols).reset_index(drop=True)
    
    forecast_global.df = forecast_global.df.sort_values(sort_cols).reset_index(drop=True)
    forecast_chunked = forecast_chunked.sort_values(sort_cols).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(train_global.df, train_chunked)
    pd.testing.assert_frame_equal(forecast_global.df, forecast_chunked)
    
    assert train_global.historical_cols == hist_cols
    assert train_global.known_future_cols == cal_cols
