import numpy as np
import pytest
from src.forecasting_models import apply_ses
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def test_ses_all_zeros():
    y = np.zeros(10)
    assert apply_ses(y) == 0.0

def test_ses_matches_statsmodels():
    np.random.seed(42)
    y = np.random.poisson(2, 100).astype(float)
    
    # Use statsmodels to get its initialization and alpha
    sm_model = SimpleExpSmoothing(y, initialization_method="heuristic").fit()
    alpha = sm_model.params['smoothing_level']
    l0 = sm_model.params['initial_level']
    
    # Get statsmodels' forecast for the next step (which is the last smoothed level)
    sm_forecast = sm_model.forecast(1)[0]
    
    # Compare with our vectorized SES using the same alpha and initial level
    fast_forecast = apply_ses(y, alpha=alpha, initial_level=l0)
    
    assert np.isclose(sm_forecast, fast_forecast)

def test_ses_no_negative():
    y = np.array([0, 0, -5, -10])
    res = apply_ses(y, alpha=0.5)
    assert res >= 0.0
