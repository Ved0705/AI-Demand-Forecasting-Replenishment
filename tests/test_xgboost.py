import numpy as np
import pandas as pd
import pytest
from src.forecasting_models import GlobalXGBoostForecaster

def test_xgboost_wrapper():
    # Dummy data
    train_df = pd.DataFrame({
        "lag_1": [0, 1, 0, 5, 2],
        "rolling_mean_7": [0.0, 0.1, 0.1, 1.0, 1.5],
        "wday": [1, 2, 3, 4, 5],
        "y": [0, 0, 5, 2, 1]
    })
    
    forecast_df = pd.DataFrame({
        "lag_1": [1, 5],
        "rolling_mean_7": [1.5, 1.2],
        "wday": [6, 7]
    })
    
    features = ["lag_1", "rolling_mean_7", "wday"]
    
    model = GlobalXGBoostForecaster(n_estimators=5, max_depth=2)
    model.fit(train_df, features)
    
    preds = model.predict(forecast_df, features)
    
    assert len(preds) == 2
    assert all(p >= 0.0 for p in preds)

def test_xgboost_requires_fit():
    model = GlobalXGBoostForecaster()
    forecast_df = pd.DataFrame({"f1": [1]})
    with pytest.raises(RuntimeError):
        model.predict(forecast_df, ["f1"])
