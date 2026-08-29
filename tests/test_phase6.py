"""Tests for src/phase6_run.py — production forecast generation.

Covers: forecast schema/horizon correctness, non-negativity, deterministic
reproducibility, model persistence (train-once/reuse), the missing-future-
calendar fallback, and adversarial leakage (post-cutoff sales mutation must
not change the forecast or downstream replenishment decisions) — proving
forecast generation and the decision layer stay separated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.phase6_run import generate_production_forecast, KEY_COLS
from src.replenishment import build_replenishment_recommendations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_config(tmp_path):
    cfg = load_config()
    cfg.raw = dict(cfg.raw)
    cfg.raw["backtest"] = {
        "scheme": "expanding",
        "horizon_days": 7,
        "n_folds": 1,
        "step_days": 7,
        "min_train_days": 30,
    }
    cfg.raw["phase6"] = {
        "model_dir": str(tmp_path / "models"),
        "model_name": "test_phase6_xgb",
        "safety_stock_z_override": None,
        "fallback_sigma": 1.0,
        "outlier_forecast_multiplier": 10.0,
        "risk_source": "reports/does_not_exist.csv",
    }
    cfg.raw["replenishment"] = {
        "lead_time_days": 7,
        "service_level": 0.95,
        "review_period_days": 7,
        "initial_inventory_days_of_cover": 14,
    }
    return cfg


@pytest.fixture
def dummy_data():
    dates = pd.date_range("2015-01-01", "2015-02-15", freq="D")  # 46 days
    rows = []
    for s_id in range(2):
        for d in dates:
            rows.append({
                "store_id": "CA_1",
                "item_id": f"ITEM_{s_id}",
                "cat_id": "HOBBIES",
                "dept_id": "HOBBIES_1",
                "state_id": "CA",
                "date": d,
                "sales": (s_id * 3 + d.dayofyear) % 6,
                "wday": d.dayofweek + 1,
                "month": d.month,
                "year": d.year,
                "week_of_year": d.isocalendar().week,
                "is_weekend": int(d.dayofweek in [5, 6]),
                "snap_active": d.day % 2,
                "is_event": 0,
            })
    return pd.DataFrame(rows)


NONEXISTENT_CALENDAR = Path("does/not/exist/calendar.csv")


# ---------------------------------------------------------------------------
# Forecast generation: schema, horizon, non-negativity
# ---------------------------------------------------------------------------

class TestGenerateProductionForecast:

    def test_output_schema(self, dummy_data, dummy_config):
        out, meta, cutoff = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        assert list(out.columns) == ["store_id", "item_id", "forecast_date", "forecast_units", "horizon_day"]

    def test_horizon_correctness(self, dummy_data, dummy_config):
        out, meta, cutoff = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        per_series = out.groupby(KEY_COLS)["horizon_day"].apply(lambda s: sorted(s.tolist()))
        for horizons in per_series:
            assert horizons == list(range(1, 8))  # horizon_days=7

    def test_forecast_dates_start_after_cutoff(self, dummy_data, dummy_config):
        out, meta, cutoff = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        assert out["forecast_date"].min() > cutoff

    def test_training_cutoff_defaults_to_last_observed_date(self, dummy_data, dummy_config):
        out, meta, cutoff = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        assert cutoff == dummy_data["date"].max()
        assert meta["training_cutoff"] == str(cutoff.date())

    def test_non_negative_forecasts(self, dummy_data, dummy_config):
        out, meta, cutoff = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        assert (out["forecast_units"] >= 0).all()

    def test_no_future_actual_sales_used(self, dummy_data, dummy_config):
        """Forecast dates and training dates must never overlap."""
        out, meta, cutoff = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        assert (out["forecast_date"] > cutoff).all()
        train_dates = set(dummy_data.loc[dummy_data["date"] <= cutoff, "date"])
        assert not (set(out["forecast_date"]) & train_dates)

    def test_missing_future_calendar_does_not_crash(self, dummy_data, dummy_config):
        """calendar.csv absent -> explicit documented fallback (NaN), not a crash."""
        out, meta, cutoff = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        assert len(out) > 0
        assert out["forecast_units"].notna().all()


# ---------------------------------------------------------------------------
# Determinism & model persistence
# ---------------------------------------------------------------------------

class TestModelPersistence:

    def test_second_run_reuses_persisted_model(self, dummy_data, dummy_config):
        out1, meta1, _ = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        assert meta1["reused_existing_model"] is False

        out2, meta2, _ = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        assert meta2["reused_existing_model"] is True
        pd.testing.assert_series_equal(
            out1["forecast_units"].reset_index(drop=True),
            out2["forecast_units"].reset_index(drop=True),
        )

    def test_force_retrain_ignores_persisted_model(self, dummy_data, dummy_config):
        _, meta1, _ = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
        )
        assert meta1["reused_existing_model"] is False
        _, meta2, _ = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR, force_retrain=True,
        )
        assert meta2["reused_existing_model"] is False

    def test_deterministic_forecast_output(self, dummy_data, dummy_config):
        out1, _, _ = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR, force_retrain=True,
        )
        out2, _, _ = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR, force_retrain=True,
        )
        np.testing.assert_allclose(
            out1.sort_values(KEY_COLS + ["forecast_date"])["forecast_units"].values,
            out2.sort_values(KEY_COLS + ["forecast_date"])["forecast_units"].values,
        )


# ---------------------------------------------------------------------------
# Adversarial leakage
# ---------------------------------------------------------------------------

class TestPhase6Leakage:

    def test_post_cutoff_mutation_does_not_change_forecast(self, dummy_data, dummy_config):
        cutoff = dummy_data["date"].max()
        contaminated = dummy_data.copy()
        # Mutate the LAST training day (still <= cutoff) is not a valid leakage
        # probe (that's real permitted data); instead we prove there is no
        # data beyond cutoff for this fixture, then prove that shifting the
        # cutoff back and contaminating the now-future tail leaves the
        # earlier-cutoff forecast untouched.
        earlier_cutoff = dummy_data["date"].sort_values().unique()[-8]  # 7 days before max
        contaminated.loc[contaminated["date"] > earlier_cutoff, "sales"] = 999999

        out_base, _, _ = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
            training_cutoff=pd.Timestamp(earlier_cutoff), force_retrain=True,
        )
        out_contam, _, _ = generate_production_forecast(
            contaminated, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
            training_cutoff=pd.Timestamp(earlier_cutoff), force_retrain=True,
        )

        np.testing.assert_allclose(
            out_base.sort_values(KEY_COLS + ["forecast_date"])["forecast_units"].values,
            out_contam.sort_values(KEY_COLS + ["forecast_date"])["forecast_units"].values,
            err_msg="Post-cutoff sales mutation changed the production forecast",
        )

    def test_replenishment_decision_unaffected_by_future_sales(self, dummy_data, dummy_config):
        """End-to-end: forecast -> decision must be identical whether or not
        post-cutoff sales are mutated, and the decision layer never sees the
        contamination because it only consumes forecast_units.
        """
        earlier_cutoff = dummy_data["date"].sort_values().unique()[-8]
        contaminated = dummy_data.copy()
        contaminated.loc[contaminated["date"] > earlier_cutoff, "sales"] = 999999

        out_base, _, _ = generate_production_forecast(
            dummy_data, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
            training_cutoff=pd.Timestamp(earlier_cutoff), force_retrain=True,
        )
        out_contam, _, _ = generate_production_forecast(
            contaminated, dummy_config, calendar_path=NONEXISTENT_CALENDAR,
            training_cutoff=pd.Timestamp(earlier_cutoff), force_retrain=True,
        )

        rec_base = build_replenishment_recommendations(out_base, dummy_config, risk_map={})
        rec_contam = build_replenishment_recommendations(out_contam, dummy_config, risk_map={})

        pd.testing.assert_frame_equal(
            rec_base.sort_values(KEY_COLS).reset_index(drop=True),
            rec_contam.sort_values(KEY_COLS).reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Forecast/decision separation
# ---------------------------------------------------------------------------

class TestForecastDecisionSeparation:

    def test_replenishment_module_does_not_import_xgboost(self):
        """src/replenishment.py must stay a pure decision layer: no model
        training/inference imports, so forecasting and decision logic cannot
        silently entangle.
        """
        src = Path("src/replenishment.py").read_text(encoding="utf-8")
        import_lines = [l for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        assert not any("xgboost" in l.lower() for l in import_lines)
        assert not any("GlobalXGBoostForecaster" in l for l in import_lines)

    def test_decision_layer_only_needs_forecast_units_and_segment(self, dummy_data, dummy_config):
        """The decision engine must be callable from a hand-built forecast
        frame with no dependency on phase6_run's model training path.
        """
        forecast_df = pd.DataFrame({
            "store_id": ["CA_1"] * 3,
            "item_id": ["A", "B", "C"],
            "forecast_units": [1.0, 0.0, 5.0],
            "horizon_day": [1, 1, 1],
            "segment": ["smooth", "intermittent", "lumpy"],
        })
        rec = build_replenishment_recommendations(forecast_df, dummy_config, risk_map={})
        assert len(rec) == 3
