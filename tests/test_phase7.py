"""Tests for src/phase7_interface.py — the read-only Phase 7 presentation layer.

All tests construct a small synthetic set of precomputed "Phase 6 outputs"
under tmp_path rather than touching the real reports/phase6/ (production
scale) artifacts, so this suite runs in milliseconds and never re-triggers
forecasting or model training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.phase7_interface import (
    DataUnavailableError,
    NotFoundError,
    Phase7DataStore,
    ValidationError,
    explain_decision,
    validate_identifier,
)
from src.replenishment import (
    REASON_MISSING_INVENTORY,
    REASON_NO_ORDER_NEEDED,
    REASON_ORDER_BELOW_ROP,
    RISK_HIGH_UNCERTAINTY,
    RISK_LOW,
    RISK_STOCKOUT,
)


# ---------------------------------------------------------------------------
# Fixture: a small synthetic Phase 6 output set
# ---------------------------------------------------------------------------

def _write_store(
    tmp_path: Path,
    with_recommendations: bool = True,
    run_mode: str = "replenishment",
    inventory_source: str = "simulated",
    with_run_metadata: bool = True,
) -> Phase7DataStore:
    reports_dir = tmp_path / "phase6"
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2016-05-23", periods=3, freq="D")
    forecast_rows = []
    for store, item, seg, units in [
        ("CA_1", "FOODS_1_001", "smooth", [5.0, 6.0, 5.5]),
        ("CA_1", "FOODS_1_002", "intermittent", [0.0, 0.0, 1.0]),
        ("TX_1", "HOBBIES_1_001", "lumpy", [2.0, 0.0, 3.0]),
    ]:
        for d, u in zip(dates, units):
            forecast_rows.append({
                "store_id": store, "item_id": item, "forecast_date": d.date().isoformat(),
                "forecast_units": u, "horizon_day": (d - dates[0]).days + 1, "segment": seg,
            })
    pd.DataFrame(forecast_rows).to_csv(reports_dir / "forecast_summary.csv", index=False)

    risk_rows = [
        {"store_id": "CA_1", "item_id": "FOODS_1_001", "segment": "smooth", "risk_flag": RISK_LOW},
        {"store_id": "CA_1", "item_id": "FOODS_1_002", "segment": "intermittent", "risk_flag": RISK_HIGH_UNCERTAINTY},
        {"store_id": "TX_1", "item_id": "HOBBIES_1_001", "segment": "lumpy", "risk_flag": RISK_STOCKOUT},
    ]
    pd.DataFrame(risk_rows).to_csv(reports_dir / "risk_summary.csv", index=False)

    if with_recommendations:
        rec_rows = [
            {"store_id": "CA_1", "item_id": "FOODS_1_001", "forecast_horizon": 3,
             "forecast_daily_mean": 5.5, "inventory_position": 100.0, "lead_time_days": 7.0,
             "safety_stock": 3.0, "reorder_point": 41.5, "recommended_order_qty": 0.0,
             "decision_reason": REASON_NO_ORDER_NEEDED, "risk_flag": RISK_LOW},
            {"store_id": "CA_1", "item_id": "FOODS_1_002", "forecast_horizon": 3,
             "forecast_daily_mean": 0.33, "inventory_position": 1.0, "lead_time_days": 7.0,
             "safety_stock": 2.0, "reorder_point": 4.3, "recommended_order_qty": 5.0,
             "decision_reason": REASON_ORDER_BELOW_ROP, "risk_flag": RISK_HIGH_UNCERTAINTY},
            {"store_id": "TX_1", "item_id": "HOBBIES_1_001", "forecast_horizon": 3,
             "forecast_daily_mean": 1.67, "inventory_position": 0.0, "lead_time_days": 7.0,
             "safety_stock": 5.0, "reorder_point": 16.7, "recommended_order_qty": 20.0,
             "decision_reason": REASON_ORDER_BELOW_ROP, "risk_flag": RISK_STOCKOUT},
        ]
        pd.DataFrame(rec_rows).to_csv(reports_dir / "replenishment_recommendations.csv", index=False)

    if with_run_metadata:
        run_metadata = {
            "mode": run_mode,
            "inventory_source": inventory_source if run_mode == "replenishment" else "not_applicable",
            "training_cutoff": "2016-05-22",
            "reused_existing_model": False,
            "horizon_days": 3,
            "forecast_window": {"start": "2016-05-23", "end": "2016-05-25"},
            "n_series": 3,
            "generated_at": "2026-01-01T00:00:00+00:00",
        }
        (reports_dir / "run_metadata.json").write_text(json.dumps(run_metadata), encoding="utf-8")

    model_meta = {
        "xgboost_params": {"objective": "reg:tweedie", "n_estimators": 100, "max_depth": 6,
                            "learning_rate": 0.1, "random_state": 42, "enable_categorical": True},
        "feature_cols": ["lag_1", "lag_7", "wday", "store_id"],
        "training_cutoff": "2016-05-22",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "n_train_rows": 12345,
        "n_series": 3,
    }
    model_meta_path = model_dir / "meta.json"
    model_meta_path.write_text(json.dumps(model_meta), encoding="utf-8")

    phase5_path = tmp_path / "phase5_model_comparison.csv"
    pd.DataFrame({
        "fold_id": [1, 2, 3, 4, 5],
        "method": ["xgboost"] * 5,
        "wape": [0.77, 0.78, 0.79, 0.76, 0.78],
        "mae": [0.98, 0.99, 1.0, 0.97, 0.98],
    }).to_csv(phase5_path, index=False)

    return Phase7DataStore(
        reports_dir=reports_dir, model_meta_path=model_meta_path, phase5_comparison_path=phase5_path,
    )


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_ok_when_all_present(self, tmp_path):
        store = _write_store(tmp_path)
        h = store.health()
        assert h["status"] == "ok"
        assert all(h["checks"].values())

    def test_health_degraded_when_missing(self, tmp_path):
        store = _write_store(tmp_path)
        (store.reports_dir / "risk_summary.csv").unlink()
        h = store.health()
        assert h["status"] == "degraded"
        assert h["checks"]["risk_summary"] is False


# ---------------------------------------------------------------------------
# 2. Metadata / traceability
# ---------------------------------------------------------------------------

class TestMetadataTraceability:

    def test_metadata_reports_training_cutoff_and_params(self, tmp_path):
        store = _write_store(tmp_path)
        meta = store.metadata()
        assert meta["training_cutoff"] == "2016-05-22"
        assert meta["xgboost_params"]["objective"] == "reg:tweedie"
        assert meta["n_feature_cols"] == 4
        assert meta["run_mode"] == "replenishment"
        assert meta["inventory_source"] == "simulated"

    def test_metadata_includes_phase5_selection_provenance(self, tmp_path):
        store = _write_store(tmp_path)
        meta = store.metadata()
        assert meta["model_selected_metrics"]["n_folds"] == 5
        assert meta["model_selected_metrics"]["wape"] == pytest.approx(0.776, abs=0.01)


# ---------------------------------------------------------------------------
# 3-6. Forecast query
# ---------------------------------------------------------------------------

class TestForecastQuery:

    def test_valid_forecast_request(self, tmp_path):
        store = _write_store(tmp_path)
        result = store.get_forecast("CA_1", "FOODS_1_001")
        assert result["store_id"] == "CA_1"
        assert result["item_id"] == "FOODS_1_001"
        assert result["forecast_cutoff_date"] == "2016-05-22"
        assert result["model_name"] == "GlobalXGBoostForecaster"

    def test_invalid_item_raises_not_found(self, tmp_path):
        store = _write_store(tmp_path)
        with pytest.raises(NotFoundError):
            store.get_forecast("CA_1", "DOES_NOT_EXIST")

    def test_forecast_horizon_correctness(self, tmp_path):
        store = _write_store(tmp_path)
        result = store.get_forecast("CA_1", "FOODS_1_001")
        assert result["forecast_horizon"] == 3
        assert len(result["forecast_dates"]) == 3
        assert len(result["predicted_units"]) == 3
        assert result["forecast_dates"] == ["2016-05-23", "2016-05-24", "2016-05-25"]

    def test_deterministic_response(self, tmp_path):
        store = _write_store(tmp_path)
        a = store.get_forecast("CA_1", "FOODS_1_001")
        b = store.get_forecast("CA_1", "FOODS_1_001")
        assert a == b

    def test_no_negative_forecasts(self, tmp_path):
        store = _write_store(tmp_path)
        for store_id, item_id in [("CA_1", "FOODS_1_001"), ("CA_1", "FOODS_1_002"), ("TX_1", "HOBBIES_1_001")]:
            result = store.get_forecast(store_id, item_id)
            assert all(v >= 0 for v in result["predicted_units"])

    def test_forecast_cutoff_prefers_run_metadata_over_stale_model_meta(self, tmp_path):
        """If a persisted model (trained on different data) is reused for
        this run, forecast_cutoff_date must reflect the CURRENT run's cutoff
        (run_metadata), not the model artifact's own training cutoff —
        otherwise auditability is silently wrong. Both are still surfaced.
        """
        store = _write_store(tmp_path)
        # model_meta's cutoff (2016-05-22) intentionally differs from the
        # run_metadata cutoff written in _write_store (also 2016-05-22 by
        # default) — force a mismatch to exercise the divergence path.
        run_meta_path = store.reports_dir / "run_metadata.json"
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        run_meta["training_cutoff"] = "2013-04-07"
        run_meta_path.write_text(json.dumps(run_meta), encoding="utf-8")

        result = store.get_forecast("CA_1", "FOODS_1_001")
        assert result["forecast_cutoff_date"] == "2013-04-07"
        assert result["model_trained_cutoff"] == "2016-05-22"
        assert result["model_reused_from_different_cutoff"] is True

    def test_forecast_does_not_leak_raw_sales_field(self, tmp_path):
        """The interface only ever exposes precomputed forecast_units, never
        a raw 'sales' actuals column (which it never loads in the first place).
        """
        store = _write_store(tmp_path)
        result = store.get_forecast("CA_1", "FOODS_1_001")
        assert "sales" not in result
        assert "actual" not in result


# ---------------------------------------------------------------------------
# 7-9. Replenishment query
# ---------------------------------------------------------------------------

class TestReplenishmentQuery:

    def test_output_schema(self, tmp_path):
        store = _write_store(tmp_path)
        r = store.get_replenishment("TX_1", "HOBBIES_1_001")
        expected = {
            "store_id", "item_id", "mode", "inventory_source", "forecast_daily_mean",
            "inventory_position", "lead_time_days", "safety_stock", "reorder_point",
            "recommended_order_qty", "decision_reason", "risk_flag", "explanation",
            "simulation_notice",
        }
        assert expected.issubset(set(r.keys()))

    def test_simulation_mode_labelled_and_explained(self, tmp_path):
        store = _write_store(tmp_path, inventory_source="simulated")
        r = store.get_replenishment("TX_1", "HOBBIES_1_001")
        assert r["mode"] == "REPLENISHMENT-SIMULATION"
        assert r["inventory_source"] == "simulated"
        assert "SIMULATED" in r["simulation_notice"]

    def test_user_supplied_inventory_labelled_differently(self, tmp_path):
        store = _write_store(tmp_path, inventory_source="user_supplied")
        r = store.get_replenishment("TX_1", "HOBBIES_1_001")
        assert r["inventory_source"] == "user_supplied"
        assert "supplied" in r["simulation_notice"]

    def test_forecast_only_mode_has_no_recommendation(self, tmp_path):
        store = _write_store(tmp_path, with_recommendations=False, run_mode="forecast-only")
        r = store.get_replenishment("CA_1", "FOODS_1_001")
        assert r["mode"] == "FORECAST-ONLY"
        assert r["decision_reason"] == REASON_MISSING_INVENTORY
        assert r["recommended_order_qty"] is None

    def test_missing_inventory_still_resolves_via_forecast_lookup(self, tmp_path):
        """Even with no recommendations file, an item WITH a forecast should
        not silently error — it should return the FORECAST-ONLY shape.
        """
        store = _write_store(tmp_path, with_recommendations=False)
        r = store.get_replenishment("CA_1", "FOODS_1_002")
        assert r["mode"] == "FORECAST-ONLY"

    def test_missing_inventory_and_missing_forecast_raises_not_found(self, tmp_path):
        store = _write_store(tmp_path, with_recommendations=False)
        with pytest.raises(NotFoundError):
            store.get_replenishment("CA_1", "NOPE")

    def test_unknown_inventory_provenance_when_run_metadata_absent(self, tmp_path):
        """Older Phase 6 runs (pre-run_metadata.json) must not silently claim
        certainty about inventory provenance.
        """
        store = _write_store(tmp_path, with_run_metadata=False)
        r = store.get_replenishment("TX_1", "HOBBIES_1_001")
        assert r["inventory_source"] == "unknown"
        assert "unknown" in r["simulation_notice"].lower()

    def test_no_negative_recommended_order_qty(self, tmp_path):
        store = _write_store(tmp_path)
        for store_id, item_id in [("CA_1", "FOODS_1_001"), ("CA_1", "FOODS_1_002"), ("TX_1", "HOBBIES_1_001")]:
            r = store.get_replenishment(store_id, item_id)
            assert r["recommended_order_qty"] is None or r["recommended_order_qty"] >= 0


# ---------------------------------------------------------------------------
# 10. Risk ranking
# ---------------------------------------------------------------------------

class TestRiskRanking:

    def test_risk_ranking_orders_stockout_first(self, tmp_path):
        store = _write_store(tmp_path)
        ranked = store.get_risk(top=10)
        assert ranked[0]["risk_flag"] == RISK_STOCKOUT

    def test_risk_top_limits_results(self, tmp_path):
        store = _write_store(tmp_path)
        ranked = store.get_risk(top=1)
        assert len(ranked) == 1

    def test_risk_store_filter(self, tmp_path):
        store = _write_store(tmp_path)
        ranked = store.get_risk(top=10, store_id="CA_1")
        assert all(r["store_id"] == "CA_1" for r in ranked)

    def test_risk_includes_explanation(self, tmp_path):
        store = _write_store(tmp_path)
        ranked = store.get_risk(top=10)
        assert all("explanation" in r for r in ranked)

    def test_risk_rejects_non_positive_top(self, tmp_path):
        store = _write_store(tmp_path)
        with pytest.raises(ValidationError):
            store.get_risk(top=0)


# ---------------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------------

class TestSummary:

    def test_summary_counts(self, tmp_path):
        store = _write_store(tmp_path)
        s = store.get_summary()
        assert s["n_series"] == 3
        assert s["n_recommendations"] == 3
        assert s["n_orders_recommended"] == 2  # two rows have qty > 0

    def test_summary_forecast_only_has_no_recommendations(self, tmp_path):
        store = _write_store(tmp_path, with_recommendations=False)
        s = store.get_summary()
        assert s["n_recommendations"] == 0
        assert "note" in s


# ---------------------------------------------------------------------------
# 12-14. Validation / error handling
# ---------------------------------------------------------------------------

class TestValidationAndErrors:

    def test_validate_identifier_accepts_normal_ids(self):
        assert validate_identifier("store_id", "CA_1") == "CA_1"

    @pytest.mark.parametrize("bad", ["", "../etc/passwd", "CA 1", "CA/1", "a" * 100, None])
    def test_validate_identifier_rejects_malformed(self, bad):
        with pytest.raises(ValidationError):
            validate_identifier("store_id", bad)

    def test_forecast_rejects_malformed_store_id(self, tmp_path):
        store = _write_store(tmp_path)
        with pytest.raises(ValidationError):
            store.get_forecast("../../etc", "FOODS_1_001")

    def test_missing_reports_dir_raises_data_unavailable(self, tmp_path):
        store = Phase7DataStore(
            reports_dir=tmp_path / "does_not_exist",
            model_meta_path=tmp_path / "models" / "meta.json",
            phase5_comparison_path=tmp_path / "phase5.csv",
        )
        with pytest.raises(DataUnavailableError):
            _ = store.forecast_df

    def test_missing_model_meta_raises_data_unavailable(self, tmp_path):
        store = _write_store(tmp_path)
        store.model_meta_path = tmp_path / "models" / "does_not_exist.json"
        with pytest.raises(DataUnavailableError):
            store.metadata()


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

class TestExplainability:

    def test_explain_known_codes(self):
        text = explain_decision(REASON_ORDER_BELOW_ROP, RISK_STOCKOUT)
        assert "reorder point" in text
        assert "stockout" in text.lower() or "safety" in text.lower()

    def test_explain_handles_missing_codes_gracefully(self):
        text = explain_decision(None, None)
        assert text == "No explanation available."

    def test_explain_never_overclaims_model_certainty(self):
        """The explanation vocabulary must not phrase business-rule outcomes
        as model predictions (e.g. must not say the model 'predicts a
        stockout')."""
        for text in list(pytest_module_reason_texts()):
            assert "predicts" not in text.lower()


def pytest_module_reason_texts():
    from src.phase7_interface import _REASON_EXPLANATIONS, _RISK_EXPLANATIONS
    return list(_REASON_EXPLANATIONS.values()) + list(_RISK_EXPLANATIONS.values())


# ---------------------------------------------------------------------------
# Interface does not duplicate forecasting/decision logic
# ---------------------------------------------------------------------------

class TestNoLogicDuplication:

    def test_interface_does_not_import_xgboost_or_forecasting(self):
        src = Path("src/phase7_interface.py").read_text(encoding="utf-8")
        import_lines = [l for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        assert not any("xgboost" in l.lower() for l in import_lines)
        assert not any("forecasting_models" in l for l in import_lines)
        assert not any("phase6_run" in l for l in import_lines)

    def test_interface_never_reads_raw_parquet(self):
        src = Path("src/phase7_interface.py").read_text(encoding="utf-8")
        assert "read_parquet" not in src
        assert "sales_long" not in src
