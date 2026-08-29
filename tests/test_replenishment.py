"""Tests for src/replenishment.py — the DECISION layer.

Covers: safety-stock/reorder-point math, risk-proxy loading, and the full
edge-case matrix for decide_one_series (missing forecast, missing inventory,
zero demand, negative/zero inventory position, high-uncertainty segments,
outlier forecasts, deterministic reproducibility).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.config import Config, load_config
from src.replenishment import (
    ReplenishmentPolicy,
    RISK_HIGH_OUTLIER,
    RISK_HIGH_UNCERTAINTY,
    RISK_LOW,
    RISK_NO_DATA,
    RISK_STOCKOUT,
    REASON_MISSING_FORECAST,
    REASON_MISSING_INVENTORY,
    REASON_NO_ORDER_NEEDED,
    REASON_ORDER_BELOW_ROP,
    REASON_ZERO_DEMAND_SERIES,
    build_forecast_risk_summary,
    build_replenishment_recommendations,
    compute_order_up_to_level,
    compute_reorder_point,
    compute_safety_stock,
    decide_one_series,
    derive_safety_stock_z,
    load_segment_risk_proxy,
    resolve_sigma,
    simulate_inventory_position,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _base_config(**phase6_overrides) -> Config:
    cfg = load_config()
    cfg.raw = dict(cfg.raw)
    cfg.raw["replenishment"] = {
        "lead_time_days": 7,
        "service_level": 0.95,
        "review_period_days": 7,
        "initial_inventory_days_of_cover": 14,
    }
    cfg.raw["phase6"] = {
        "safety_stock_z_override": None,
        "fallback_sigma": 1.0,
        "outlier_forecast_multiplier": 10.0,
        "risk_source": "reports/does_not_exist.csv",
        **phase6_overrides,
    }
    return cfg


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------

class TestSafetyStockMath:

    def test_z_matches_scipy_norm_ppf(self):
        z = derive_safety_stock_z(0.95)
        assert abs(z - stats.norm.ppf(0.95)) < 1e-9

    def test_z_override_bypasses_derivation(self):
        assert derive_safety_stock_z(0.95, override=2.5) == 2.5

    def test_z_rejects_invalid_service_level(self):
        with pytest.raises(ValueError):
            derive_safety_stock_z(1.5)
        with pytest.raises(ValueError):
            derive_safety_stock_z(0.0)

    def test_safety_stock_formula(self):
        ss = compute_safety_stock(sigma_daily=2.0, lead_time_days=9.0, z=1.5)
        assert abs(ss - (1.5 * 2.0 * math.sqrt(9.0))) < 1e-9

    def test_safety_stock_rejects_negative_inputs(self):
        with pytest.raises(ValueError):
            compute_safety_stock(-1.0, 7, 1.5)
        with pytest.raises(ValueError):
            compute_safety_stock(1.0, -7, 1.5)

    def test_reorder_point_adds_lead_time_demand_and_safety_stock(self):
        rop = compute_reorder_point(forecast_daily_mean=3.0, lead_time_days=7.0, safety_stock=5.0)
        assert rop == 3.0 * 7.0 + 5.0

    def test_reorder_point_clips_negative_forecast(self):
        rop = compute_reorder_point(forecast_daily_mean=-3.0, lead_time_days=7.0, safety_stock=5.0)
        assert rop == 5.0

    def test_order_up_to_covers_lead_time_plus_review_period(self):
        oul = compute_order_up_to_level(2.0, lead_time_days=7.0, review_period_days=7.0, safety_stock=4.0)
        assert oul == 2.0 * 14.0 + 4.0

    def test_simulate_inventory_position(self):
        pos = simulate_inventory_position(forecast_daily_mean=4.0, initial_days_of_cover=14.0)
        assert pos == 56.0

    def test_simulate_inventory_position_never_negative(self):
        assert simulate_inventory_position(-5.0, 14.0) == 0.0
        assert simulate_inventory_position(4.0, -14.0) == 0.0


# ---------------------------------------------------------------------------
# Risk proxy
# ---------------------------------------------------------------------------

class TestRiskProxy:

    def test_missing_report_returns_empty_map(self, tmp_path):
        cfg = _base_config(risk_source=str(tmp_path / "nope.csv"))
        assert load_segment_risk_proxy(cfg) == {}

    def test_loads_xgboost_rmse_by_segment(self, tmp_path):
        report = tmp_path / "segment_model_comparison.csv"
        pd.DataFrame({
            "fold_id": [1, 1, 2, 2],
            "segment": ["smooth", "smooth", "smooth", "smooth"],
            "method": ["xgboost", "tsb", "xgboost", "tsb"],
            "rmse": [2.0, 9.0, 4.0, 9.0],
        }).to_csv(report, index=False)
        cfg = _base_config(risk_source=str(report))
        risk_map = load_segment_risk_proxy(cfg)
        assert risk_map["smooth"] == pytest.approx(3.0)  # mean of 2.0, 4.0

    def test_resolve_sigma_uses_fallback_when_segment_missing(self):
        assert resolve_sigma("lumpy", {"smooth": 2.0}, fallback_sigma=1.5) == 1.5

    def test_resolve_sigma_uses_fallback_when_segment_none(self):
        assert resolve_sigma(None, {"smooth": 2.0}, fallback_sigma=1.5) == 1.5

    def test_resolve_sigma_uses_map_value_when_present(self):
        assert resolve_sigma("smooth", {"smooth": 2.0}, fallback_sigma=1.5) == 2.0


# ---------------------------------------------------------------------------
# decide_one_series — edge case matrix
# ---------------------------------------------------------------------------

class TestDecideOneSeries:

    def _policy(self, **overrides) -> ReplenishmentPolicy:
        cfg = _base_config(**overrides.pop("phase6", {}))
        return ReplenishmentPolicy.from_config(cfg)

    def test_missing_forecast_is_explicit_edge_case(self):
        result = decide_one_series(None, "smooth", self._policy(), {}, inventory_position=10)
        assert result["decision_reason"] == REASON_MISSING_FORECAST
        assert result["risk_flag"] == RISK_NO_DATA
        assert result["recommended_order_qty"] is None

    def test_nan_forecast_treated_as_missing(self):
        result = decide_one_series(float("nan"), "smooth", self._policy(), {}, inventory_position=10)
        assert result["decision_reason"] == REASON_MISSING_FORECAST

    def test_missing_inventory_position(self):
        result = decide_one_series(3.0, "smooth", self._policy(), {}, inventory_position=None)
        assert result["decision_reason"] == REASON_MISSING_INVENTORY
        assert result["risk_flag"] == RISK_NO_DATA
        assert result["recommended_order_qty"] is None
        # Safety stock / reorder point are still computed (forecast-derivable)
        assert result["safety_stock"] is not None
        assert result["reorder_point"] is not None

    def test_zero_demand_series_no_order_when_position_ge_safety_stock(self):
        policy = self._policy()
        result = decide_one_series(0.0, "smooth", policy, {}, inventory_position=1000.0)
        assert result["decision_reason"] == REASON_ZERO_DEMAND_SERIES
        assert result["recommended_order_qty"] == 0.0

    def test_zero_demand_series_orders_up_to_safety_stock_if_below(self):
        policy = self._policy()
        result = decide_one_series(0.0, "smooth", policy, {}, inventory_position=0.0)
        assert result["decision_reason"] == REASON_ORDER_BELOW_ROP
        assert result["recommended_order_qty"] == pytest.approx(result["safety_stock"])

    def test_negative_forecast_is_clipped_to_zero(self):
        policy = self._policy()
        result = decide_one_series(-5.0, "smooth", policy, {}, inventory_position=1000.0)
        assert result["forecast_daily_mean"] == 0.0

    def test_order_recommended_when_position_below_reorder_point(self):
        policy = self._policy()
        result = decide_one_series(5.0, "smooth", policy, {"smooth": 1.0}, inventory_position=0.0)
        assert result["decision_reason"] == REASON_ORDER_BELOW_ROP
        assert result["recommended_order_qty"] > 0

    def test_no_order_when_position_above_target(self):
        policy = self._policy()
        result = decide_one_series(1.0, "smooth", policy, {"smooth": 0.5}, inventory_position=10_000.0)
        assert result["decision_reason"] == REASON_NO_ORDER_NEEDED
        assert result["recommended_order_qty"] == 0.0

    def test_negative_inventory_position_treated_literally_as_backorder(self):
        policy = self._policy()
        result = decide_one_series(2.0, "smooth", policy, {"smooth": 1.0}, inventory_position=-20.0)
        assert result["decision_reason"] == REASON_ORDER_BELOW_ROP
        assert result["risk_flag"] == RISK_STOCKOUT
        # Order quantity must be larger than it would be from a zero position.
        zero_case = decide_one_series(2.0, "smooth", policy, {"smooth": 1.0}, inventory_position=0.0)
        assert result["recommended_order_qty"] > zero_case["recommended_order_qty"]

    def test_high_uncertainty_segment_flagged(self):
        policy = self._policy()
        result = decide_one_series(2.0, "intermittent", policy, {"intermittent": 1.0}, inventory_position=1000.0)
        assert result["risk_flag"] == RISK_HIGH_UNCERTAINTY

    def test_stockout_risk_overrides_high_uncertainty(self):
        policy = self._policy()
        result = decide_one_series(2.0, "intermittent", policy, {"intermittent": 1.0}, inventory_position=0.0)
        assert result["risk_flag"] == RISK_STOCKOUT

    def test_high_forecast_outlier_flag(self):
        policy = self._policy()
        result = decide_one_series(
            100.0, "smooth", policy, {"smooth": 1.0},
            inventory_position=10_000.0, historical_mean_sales=1.0,
        )
        assert result["risk_flag"] == RISK_HIGH_OUTLIER

    def test_low_risk_default(self):
        policy = self._policy()
        result = decide_one_series(
            2.0, "smooth", policy, {"smooth": 1.0},
            inventory_position=10_000.0, historical_mean_sales=2.0,
        )
        assert result["risk_flag"] == RISK_LOW

    def test_deterministic_repeat_calls(self):
        policy = self._policy()
        a = decide_one_series(3.0, "smooth", policy, {"smooth": 1.0}, inventory_position=5.0)
        b = decide_one_series(3.0, "smooth", policy, {"smooth": 1.0}, inventory_position=5.0)
        assert a == b


# ---------------------------------------------------------------------------
# ReplenishmentPolicy config validation
# ---------------------------------------------------------------------------

class TestReplenishmentPolicy:

    def test_missing_required_assumption_raises(self):
        cfg = _base_config()
        cfg.raw["replenishment"] = {"lead_time_days": 7}  # missing the rest
        with pytest.raises(ValueError):
            ReplenishmentPolicy.from_config(cfg)

    def test_all_business_values_come_from_config(self):
        cfg = _base_config()
        cfg.raw["replenishment"]["lead_time_days"] = 21
        policy = ReplenishmentPolicy.from_config(cfg)
        assert policy.lead_time_days == 21


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def _forecast_df(n_series=3, horizon=28, daily=2.0, segment="smooth"):
    rows = []
    for i in range(n_series):
        for d in range(1, horizon + 1):
            rows.append({
                "store_id": "CA_1",
                "item_id": f"ITEM_{i}",
                "forecast_date": pd.Timestamp("2016-05-23") + pd.Timedelta(days=d - 1),
                "forecast_units": daily,
                "horizon_day": d,
                "segment": segment,
            })
    return pd.DataFrame(rows)


class TestBuildReplenishmentRecommendations:

    def test_simulated_inventory_when_none_supplied(self):
        cfg = _base_config()
        df = _forecast_df()
        rec = build_replenishment_recommendations(df, cfg, inventory_df=None, risk_map={"smooth": 1.0})
        assert (rec["inventory_position"] > 0).all()
        assert set(rec["store_id"]) == {"CA_1"}
        assert len(rec) == 3

    def test_user_supplied_inventory_used_when_given(self):
        cfg = _base_config()
        df = _forecast_df()
        inv = pd.DataFrame({
            "store_id": ["CA_1"] * 3,
            "item_id": ["ITEM_0", "ITEM_1", "ITEM_2"],
            "inventory_position": [0.0, 500.0, -10.0],
        })
        rec = build_replenishment_recommendations(df, cfg, inventory_df=inv, risk_map={"smooth": 1.0})
        rec = rec.set_index("item_id")
        assert rec.loc["ITEM_0", "inventory_position"] == 0.0
        assert rec.loc["ITEM_2", "inventory_position"] == -10.0
        assert rec.loc["ITEM_2", "risk_flag"] == RISK_STOCKOUT

    def test_empty_forecast_returns_empty_frame_with_schema(self):
        cfg = _base_config()
        empty = pd.DataFrame(columns=["store_id", "item_id", "forecast_units", "horizon_day", "segment"])
        rec = build_replenishment_recommendations(empty, cfg)
        assert rec.empty
        assert "recommended_order_qty" in rec.columns

    def test_output_schema(self):
        cfg = _base_config()
        df = _forecast_df()
        rec = build_replenishment_recommendations(df, cfg, risk_map={"smooth": 1.0})
        expected = {
            "store_id", "item_id", "forecast_horizon", "forecast_daily_mean",
            "inventory_position", "lead_time_days", "safety_stock", "reorder_point",
            "recommended_order_qty", "decision_reason", "risk_flag",
        }
        assert expected.issubset(set(rec.columns))


class TestBuildForecastRiskSummary:

    def test_forecast_only_has_no_inventory_or_order_fields(self):
        cfg = _base_config()
        df = _forecast_df()
        risk = build_forecast_risk_summary(df, cfg, risk_map={"smooth": 1.0})
        assert "inventory_position" not in risk.columns
        assert "recommended_order_qty" not in risk.columns
        assert "safety_stock" in risk.columns
        assert len(risk) == 3

    def test_intermittent_segment_flagged_high_uncertainty(self):
        cfg = _base_config()
        df = _forecast_df(segment="intermittent")
        risk = build_forecast_risk_summary(df, cfg, risk_map={"intermittent": 1.0})
        assert (risk["risk_flag"] == RISK_HIGH_UNCERTAINTY).all()
