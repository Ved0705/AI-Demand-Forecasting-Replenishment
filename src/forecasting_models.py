"""Phase 5 — Forecasting Model Implementations.

This module contains implementations of intermittent demand models,
a statistical model, and a global Machine Learning model.

All models adhere to the strict fold architecture.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
except ImportError:
    logger.warning("xgboost is not installed. GlobalXGBoostForecaster will not work.")

# ---------------------------------------------------------------------------
# Fast Vectorized SES (Matches statsmodels.tsa.holtwinters.SimpleExpSmoothing)
# ---------------------------------------------------------------------------

def apply_ses(y: np.ndarray, alpha: float = 0.1, initial_level: float | None = None) -> float:
    """Vectorized Simple Exponential Smoothing (SES).
    
    y_hat[t+1] = alpha * y[t] + (1 - alpha) * y_hat[t]
    
    If initial_level is None, it defaults to the heuristic used by statsmodels 
    (mean of first 10 observations, or all if len < 10).
    
    Returns the forecast for the next period (which is flat for all future horizons).
    """
    n = len(y)
    if n == 0:
        return 0.0
    
    if initial_level is None:
        initial_level = float(np.mean(y[:min(10, n)]))
        
    curr_l = initial_level
    for i in range(n):
        curr_l = alpha * y[i] + (1 - alpha) * curr_l
        
    return max(0.0, float(curr_l))

# ---------------------------------------------------------------------------
# Intermittent Demand Models
# ---------------------------------------------------------------------------

def apply_croston(y: np.ndarray, alpha: float = 0.1) -> float:
    """Croston's Method for intermittent demand.
    
    Smooths demand sizes (z) and intervals between demands (p) separately.
    Returns z_hat / p_hat for the forecast.
    """
    n = len(y)
    if n == 0:
        return 0.0
        
    # Find non-zero indices
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) == 0:
        return 0.0
        
    # Initialize
    z_hat = y[nz_idx[0]]
    p_hat = nz_idx[0] + 1.0 # Interval from start to first demand
    
    last_idx = nz_idx[0]
    
    for i in range(1, len(nz_idx)):
        idx = nz_idx[i]
        q = idx - last_idx
        z = y[idx]
        
        z_hat = alpha * z + (1 - alpha) * z_hat
        p_hat = alpha * q + (1 - alpha) * p_hat
        
        last_idx = idx
        
    if p_hat == 0.0:
        return 0.0
        
    # Forecast is the ratio
    return max(0.0, float(z_hat / p_hat))

def apply_sba(y: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> float:
    """Syntetos-Boylan Approximation (SBA).
    
    Applies a bias-correction factor (1 - beta/2) to Croston's estimate.
    We use beta to smooth intervals instead of alpha in the original SBA formulation,
    but commonly alpha is used for both if not specified. We allow separate alpha/beta.
    """
    n = len(y)
    if n == 0:
        return 0.0
        
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) == 0:
        return 0.0
        
    z_hat = y[nz_idx[0]]
    p_hat = nz_idx[0] + 1.0
    
    last_idx = nz_idx[0]
    
    for i in range(1, len(nz_idx)):
        idx = nz_idx[i]
        q = idx - last_idx
        z = y[idx]
        
        z_hat = alpha * z + (1 - alpha) * z_hat
        p_hat = beta * q + (1 - beta) * p_hat
        
        last_idx = idx
        
    if p_hat == 0.0:
        return 0.0
        
    bias_correction = 1.0 - (beta / 2.0)
    forecast = (z_hat / p_hat) * bias_correction
    return max(0.0, float(forecast))

def apply_tsb(y: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> float:
    """Teunter-Syntetos-Babai (TSB) method.
    
    Smooths demand size (z) for non-zero periods, and demand probability (p) 
    for all periods (updated to 1 if demand occurs, 0 otherwise).
    Forecast is z_hat * p_hat.
    """
    n = len(y)
    if n == 0:
        return 0.0
        
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) == 0:
        return 0.0
        
    # Initialize
    # The paper suggests initializing p_hat as mean probability, z_hat as mean size
    p_hat = len(nz_idx) / n
    z_hat = np.mean(y[nz_idx])
    
    for i in range(n):
        if y[i] > 0:
            z_hat = alpha * y[i] + (1 - alpha) * z_hat
            p_hat = beta * 1.0 + (1 - beta) * p_hat
        else:
            p_hat = beta * 0.0 + (1 - beta) * p_hat
            
    return max(0.0, float(z_hat * p_hat))

# ---------------------------------------------------------------------------
# Global ML Model wrapper
# ---------------------------------------------------------------------------

class GlobalXGBoostForecaster:
    """Wrapper for a global XGBoost model that forecasts all series together.
    
    Uses reg:tweedie to handle zero-inflated, positive count data.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        self.model: xgb.XGBRegressor | None = None
        self.params = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": 1.5, # 1.0=Poisson, 2.0=Gamma, 1.5 is common for compound Poisson-Gamma
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "random_state": 42
        }
        self.params.update(kwargs)
        
    def fit(self, train_df: pd.DataFrame, features: list[str], target: str = "y") -> None:
        """Fit the global XGBoost model on the training fold."""
        X = train_df[features]
        y = train_df[target]
        
        # XGBoost handles NaNs naturally in tree splits, so we don't strictly need to impute,
        # but we must ensure categorical encoding if needed. We assume features passed are numeric.
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        
    def predict(self, forecast_df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Predict for the forecast fold."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict.")
            
        X = forecast_df[features]
        preds = self.model.predict(X)
        
        # Hard non-negativity guard even with Tweedie just to be safe
        preds = np.maximum(preds, 0.0)
        return preds
