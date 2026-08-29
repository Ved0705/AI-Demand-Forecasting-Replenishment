import numpy as np
import pytest
from src.forecasting_models import apply_croston, apply_sba, apply_tsb

def test_croston_all_zeros():
    y = np.zeros(10)
    assert apply_croston(y) == 0.0

def test_croston_single_demand():
    y = np.zeros(10)
    y[5] = 10
    # Demand size will be smoothed toward 10.
    # Interval will be smoothed toward 6 (index 5 + 1).
    res = apply_croston(y, alpha=0.1)
    assert res > 0
    assert res <= 10.0

def test_croston_leading_zeros():
    y1 = np.array([0, 0, 0, 10, 10, 10])
    y2 = np.array([10, 10, 10, 10, 10, 10])
    res1 = apply_croston(y1, alpha=0.1)
    res2 = apply_croston(y2, alpha=0.1)
    # The intervals for y1 will initially be larger than y2, 
    # so the forecast for y1 should be somewhat smaller initially, but strictly positive.
    assert res1 > 0
    assert res1 < res2

def test_sba_bias_correction():
    y = np.array([0, 5, 0, 5, 0, 5, 0, 5])
    cros = apply_croston(y, alpha=0.1)
    sba = apply_sba(y, alpha=0.1, beta=0.1)
    # SBA applies (1 - beta/2) correction
    assert np.isclose(sba, cros * (1 - 0.1/2))

def test_sba_all_zeros():
    y = np.zeros(10)
    assert apply_sba(y) == 0.0

def test_tsb_all_zeros():
    y = np.zeros(10)
    assert apply_tsb(y) == 0.0

def test_tsb_changing_occurrence():
    y1 = np.array([0, 0, 0, 0, 0, 5, 5, 5, 5, 5])
    y2 = np.array([5, 5, 5, 5, 5, 0, 0, 0, 0, 0])
    res1 = apply_tsb(y1, alpha=0.1, beta=0.1)
    res2 = apply_tsb(y2, alpha=0.1, beta=0.1)
    # y1 has demand at the end, probability is increasing
    # y2 has no demand at the end, probability is decaying to 0
    assert res1 > res2

def test_no_negative_predictions():
    # Even with weird data, models should return >= 0
    y = np.array([0, 1, 0, -5, 0, 10]) # -5 shouldn't happen in our data, but test robustness
    assert apply_croston(y) >= 0.0
    assert apply_sba(y) >= 0.0
    assert apply_tsb(y) >= 0.0
