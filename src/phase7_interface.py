"""Phase 7 — Production Retail Intelligence Interface.

A thin, read-only presentation layer over the Phase 6 outputs. It does NOT
forecast, does NOT train or load XGBoost, and does NOT read raw sales data —
it only reads the precomputed CSV/JSON artifacts that `src/phase6_run.py`
already wrote to `reports/phase6/` and `models/`.

    DATA -> SQL/DATA FOUNDATION -> FEATURES -> FORECAST -> RISK/UNCERTAINTY
         -> REPLENISHMENT DECISION -> [PHASE 7 INTERFACE]  (this module)

Concerns are kept separate:
  - Data access:       Phase7DataStore (loads the precomputed Phase 6 artifacts)
  - Forecast/decision:  already computed by Phase 5/6 — never reimplemented here
  - Business rules:     the explanation mappings below only describe Phase 6's
                        existing decision_reason/risk_flag codes in prose; they
                        introduce no new business logic or thresholds
  - Presentation/API:   the CLI functions and `main()` at the bottom

WHY A CLI, NOT A WEB API
-------------------------
FastAPI/uvicorn/pydantic are not installed in this project's environment
(requirements.txt has none of them). Adding a web framework purely to satisfy
an interface requirement would be exactly the kind of unneeded infrastructure
the Phase 7 brief warns against. Every prior phase in this repository already
ships as a `python -m src.<module>` CLI; Phase 7 follows that convention. See
DECISION_LOG D-031.

PERFORMANCE
-----------
Never loads the 14M-row processed dataset and never re-trains or re-invokes
XGBoost. All queries are served from the small precomputed Phase 6 CSVs
(<= a few hundred thousand rows), read once per process and cached on the
Phase7DataStore instance for the life of that instance.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.replenishment import (
    REASON_MISSING_FORECAST,
    REASON_MISSING_INVENTORY,
    REASON_NO_ORDER_NEEDED,
    REASON_ORDER_BELOW_ROP,
    REASON_ZERO_DEMAND_SERIES,
    RISK_HIGH_OUTLIER,
    RISK_HIGH_UNCERTAINTY,
    RISK_LOW,
    RISK_NO_DATA,
    RISK_STOCKOUT,
)

KEY_COLS = ["store_id", "item_id"]
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

MODEL_NAME = "GlobalXGBoostForecaster"
MODEL_SELECTION_SOURCE = "reports/phase5/model_selection.md (Phase 5 benchmark, 9,147 series / 5 folds)"


# ---------------------------------------------------------------------------
# Controlled errors (never leak a raw traceback to a CLI/API caller)
# ---------------------------------------------------------------------------

class Phase7Error(Exception):
    """Base class for controlled Phase 7 interface errors."""


class ValidationError(Phase7Error):
    """A user-supplied identifier or parameter failed validation."""


class NotFoundError(Phase7Error):
    """The requested store/item has no precomputed Phase 6 output."""


class DataUnavailableError(Phase7Error):
    """A required Phase 6/5 artifact is missing on disk."""


def validate_identifier(name: str, value: str) -> str:
    """Reject anything that isn't a plain M5-style identifier.

    This is the interface's only user-supplied input, so it is validated
    strictly (alnum/underscore/hyphen, bounded length) before ever being used
    to filter a DataFrame. Nothing here is used to build a file path, so
    there is no path-traversal surface, but malformed input is still
    rejected explicitly rather than silently coerced.
    """
    if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
        raise ValidationError(
            f"Invalid {name}: {value!r}. Expected alphanumeric/underscore/hyphen, 1-64 chars."
        )
    return value


# ---------------------------------------------------------------------------
# Explanation vocabulary — prose ONLY for codes Phase 6 already computed.
# No new thresholds or business logic are introduced here.
# ---------------------------------------------------------------------------

_REASON_EXPLANATIONS = {
    REASON_ORDER_BELOW_ROP: (
        "High replenishment priority: inventory position is below the reorder "
        "point (expected demand over the lead time plus safety stock), so an "
        "order is recommended to reach the order-up-to level."
    ),
    REASON_NO_ORDER_NEEDED: (
        "No order recommended: inventory position already meets or exceeds "
        "the target level for this review cycle."
    ),
    REASON_MISSING_FORECAST: (
        "No recommendation possible: no Phase 6 forecast exists for this "
        "store/item combination."
    ),
    REASON_MISSING_INVENTORY: (
        "Recommendation deferred: no inventory position was supplied or "
        "simulated for this series (forecast-only query)."
    ),
    REASON_ZERO_DEMAND_SERIES: (
        "No order needed: forecast demand is zero and current inventory "
        "covers the configured safety stock."
    ),
}

_RISK_EXPLANATIONS = {
    RISK_STOCKOUT: (
        "Inventory position is at or below the calculated safety-stock "
        "threshold — elevated stockout exposure under the current assumptions."
    ),
    RISK_HIGH_UNCERTAINTY: (
        "This item is in an intermittent/lumpy demand segment, where the "
        "Phase 5 backtest measured higher forecast error historically."
    ),
    RISK_HIGH_OUTLIER: (
        "The forecast is well above this series' own historical mean daily "
        "sales — flagged for manual review, not auto-suppressed."
    ),
    RISK_LOW: "No unusual risk indicators from the available data.",
    RISK_NO_DATA: "Insufficient forecast or inventory data to assess risk.",
}


def explain_decision(decision_reason: str | None, risk_flag: str | None) -> str:
    """Business-language explanation for a Phase 6 decision_reason/risk_flag
    pair. Deliberately does NOT say "XGBoost predicts a stockout" — the model
    only produces a point forecast; the stockout/priority language is a
    business RULE applied to that forecast plus the inventory assumption.
    """
    parts = []
    if decision_reason in _REASON_EXPLANATIONS:
        parts.append(_REASON_EXPLANATIONS[decision_reason])
    elif decision_reason:
        parts.append(f"Decision reason: {decision_reason}.")
    if risk_flag in _RISK_EXPLANATIONS:
        parts.append(_RISK_EXPLANATIONS[risk_flag])
    elif risk_flag:
        parts.append(f"Risk flag: {risk_flag}.")
    return " ".join(parts) if parts else "No explanation available."


_RISK_SEVERITY_ORDER = {
    RISK_STOCKOUT: 0,
    RISK_HIGH_OUTLIER: 1,
    RISK_HIGH_UNCERTAINTY: 2,
    RISK_NO_DATA: 3,
    RISK_LOW: 4,
    None: 5,
}


# ---------------------------------------------------------------------------
# Data access layer
# ---------------------------------------------------------------------------

@dataclass
class Phase7DataStore:
    """Read-only access to precomputed Phase 6 outputs. Loads each artifact
    at most once (cached on the instance); never touches raw sales data.
    """

    reports_dir: Path
    model_meta_path: Path
    phase5_comparison_path: Path

    _forecast_df: pd.DataFrame | None = None
    _risk_df: pd.DataFrame | None = None
    _recommendations_df: pd.DataFrame | None = None
    _run_metadata: dict | None = None
    _model_meta: dict | None = None
    _phase5_provenance: dict | None = None

    @classmethod
    def from_config(cls, cfg: Config, fixture: bool = False) -> "Phase7DataStore":
        reports_dir = cfg.path("reports") / ("phase6_fixture" if fixture else "phase6")
        p6 = cfg.get("phase6", {})
        model_dir = cfg.root / p6.get("model_dir", "models")
        model_name = p6.get("model_name", "phase6_xgboost_production")
        return cls(
            reports_dir=reports_dir,
            model_meta_path=model_dir / f"{model_name}_meta.json",
            phase5_comparison_path=cfg.path("reports") / "phase5" / "model_comparison.csv",
        )

    # -- lazy loaders --------------------------------------------------

    def _read_csv(self, name: str) -> pd.DataFrame:
        path = self.reports_dir / name
        if not path.exists():
            raise DataUnavailableError(
                f"{path} not found. Run `python -m src.phase6_run` first to "
                "generate Phase 6 outputs; Phase 7 never regenerates them."
            )
        return pd.read_csv(path)

    @property
    def forecast_df(self) -> pd.DataFrame:
        if self._forecast_df is None:
            df = self._read_csv("forecast_summary.csv")
            df["forecast_date"] = pd.to_datetime(df["forecast_date"])
            self._forecast_df = df
        return self._forecast_df

    @property
    def risk_df(self) -> pd.DataFrame:
        if self._risk_df is None:
            self._risk_df = self._read_csv("risk_summary.csv")
        return self._risk_df

    @property
    def recommendations_df(self) -> pd.DataFrame | None:
        if self._recommendations_df is None:
            path = self.reports_dir / "replenishment_recommendations.csv"
            self._recommendations_df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        return self._recommendations_df if not self._recommendations_df.empty else None

    @property
    def run_metadata(self) -> dict | None:
        if self._run_metadata is None:
            path = self.reports_dir / "run_metadata.json"
            self._run_metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return self._run_metadata or None

    @property
    def model_meta(self) -> dict:
        if self._model_meta is None:
            if not self.model_meta_path.exists():
                raise DataUnavailableError(
                    f"{self.model_meta_path} not found. No production model has been trained yet."
                )
            self._model_meta = json.loads(self.model_meta_path.read_text(encoding="utf-8"))
        return self._model_meta

    @property
    def phase5_provenance(self) -> dict:
        """XGBoost's own Phase 5 out-of-sample benchmark numbers, read from
        the structured CSV Phase 5 already produced — never recomputed here.
        """
        if self._phase5_provenance is None:
            if not self.phase5_comparison_path.exists():
                self._phase5_provenance = {}
            else:
                df = pd.read_csv(self.phase5_comparison_path)
                sub = df.loc[df["method"] == "xgboost"]
                self._phase5_provenance = {
                    "wape": float(sub["wape"].mean()) if not sub.empty else None,
                    "mae": float(sub["mae"].mean()) if not sub.empty else None,
                    "n_folds": int(sub["fold_id"].nunique()) if not sub.empty else 0,
                    "source": MODEL_SELECTION_SOURCE,
                }
        return self._phase5_provenance

    # -- health ----------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Cheap existence/readability checks only — no data is loaded or
        computed beyond file-existence and (for JSON) a parse.
        """
        checks = {}
        for label, path in [
            ("forecast_summary", self.reports_dir / "forecast_summary.csv"),
            ("risk_summary", self.reports_dir / "risk_summary.csv"),
            ("model_meta", self.model_meta_path),
        ]:
            checks[label] = path.exists()
        status = "ok" if all(checks.values()) else "degraded"
        return {"status": status, "checks": checks, "reports_dir": str(self.reports_dir)}

    # -- metadata ----------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        model_meta = self.model_meta
        run_meta = self.run_metadata or {}
        return {
            "model_name": MODEL_NAME,
            "model_selected_by": "Phase 5 benchmark (see model_selected_metrics)",
            "model_selected_metrics": self.phase5_provenance,
            "xgboost_params": model_meta.get("xgboost_params"),
            "feature_cols": model_meta.get("feature_cols"),
            "n_feature_cols": len(model_meta.get("feature_cols", []) or []),
            "training_cutoff": model_meta.get("training_cutoff"),
            "model_trained_at": model_meta.get("trained_at"),
            "n_train_rows": model_meta.get("n_train_rows"),
            "n_series_trained": model_meta.get("n_series"),
            "run_mode": run_meta.get("mode"),
            "inventory_source": run_meta.get("inventory_source"),
            "forecast_window": run_meta.get("forecast_window"),
            "horizon_days": run_meta.get("horizon_days"),
        }

    # -- forecast query ----------------------------------------------------

    def get_forecast(self, store_id: str, item_id: str) -> dict[str, Any]:
        store_id = validate_identifier("store_id", store_id)
        item_id = validate_identifier("item_id", item_id)

        sub = self.forecast_df.loc[
            (self.forecast_df["store_id"] == store_id) & (self.forecast_df["item_id"] == item_id)
        ].sort_values("forecast_date")
        if sub.empty:
            raise NotFoundError(
                f"No Phase 6 forecast found for store_id={store_id!r}, item_id={item_id!r}. "
                "It may be outside the subset that was forecast, or misspelled."
            )

        model_meta = self.model_meta
        run_meta = self.run_metadata or {}
        risk_row = self.risk_df.loc[
            (self.risk_df["store_id"] == store_id) & (self.risk_df["item_id"] == item_id)
        ]
        risk_flag = risk_row["risk_flag"].iloc[0] if not risk_row.empty else None
        segment = sub["segment"].iloc[0] if "segment" in sub.columns else None

        # The persisted production model can be REUSED across runs on
        # different data (e.g. a fixture smoke test) whenever its
        # hyperparameters/features match (see train_or_load_production_model
        # in src/phase6_run.py). When that happens, the model artifact's own
        # training_cutoff (model_meta) no longer matches the cutoff that was
        # actually used to slice features for THIS run's forecast dates
        # (run_metadata, written fresh by every phase6_run.py invocation).
        # Prefer the per-run cutoff for forecast_cutoff_date — it is what
        # governs this specific forecast's leakage guarantee — and surface
        # both explicitly so a caller can see if the model itself was fit on
        # different (e.g. production-scale) data than the run being queried.
        run_cutoff = run_meta.get("training_cutoff")
        model_cutoff = model_meta.get("training_cutoff")
        cutoff_mismatch = bool(run_cutoff and model_cutoff and run_cutoff != model_cutoff)

        return {
            "store_id": store_id,
            "item_id": item_id,
            "forecast_cutoff_date": run_cutoff or model_cutoff,
            "forecast_horizon": int(sub["horizon_day"].nunique()),
            "forecast_dates": [d.date().isoformat() for d in sub["forecast_date"]],
            "predicted_units": [round(float(v), 4) for v in sub["forecast_units"]],
            "segment": segment,
            "risk_flag": risk_flag,
            "model_name": MODEL_NAME,
            "model_trained_at": model_meta.get("trained_at"),
            "model_trained_cutoff": model_cutoff,
            "model_reused_from_different_cutoff": cutoff_mismatch,
            "n_feature_cols": len(model_meta.get("feature_cols", []) or []),
        }

    # -- replenishment query -------------------------------------------------

    def get_replenishment(self, store_id: str, item_id: str) -> dict[str, Any]:
        store_id = validate_identifier("store_id", store_id)
        item_id = validate_identifier("item_id", item_id)

        rec_df = self.recommendations_df
        run_meta = self.run_metadata or {}
        mode_label = "FORECAST-ONLY" if run_meta.get("mode") == "forecast-only" or rec_df is None else "REPLENISHMENT-SIMULATION"
        inventory_source = run_meta.get("inventory_source", "unknown")

        if rec_df is None:
            # Confirm the item at least has a forecast, so the error is precise.
            _ = self.get_forecast(store_id, item_id)
            return {
                "store_id": store_id,
                "item_id": item_id,
                "mode": "FORECAST-ONLY",
                "inventory_source": "not_applicable",
                "recommended_order_qty": None,
                "decision_reason": REASON_MISSING_INVENTORY,
                "risk_flag": None,
                "explanation": explain_decision(REASON_MISSING_INVENTORY, None),
                "note": "This Phase 6 run was forecast-only; no replenishment recommendations were computed.",
            }

        sub = rec_df.loc[(rec_df["store_id"] == store_id) & (rec_df["item_id"] == item_id)]
        if sub.empty:
            raise NotFoundError(
                f"No Phase 6 replenishment recommendation found for store_id={store_id!r}, item_id={item_id!r}."
            )
        row = sub.iloc[0].to_dict()

        if inventory_source == "simulated":
            simulation_notice = (
                "Inventory position for this series was SIMULATED from "
                "replenishment.initial_inventory_days_of_cover (config), not "
                "observed. This is a scenario/simulation output, not a "
                "measured business outcome — see DECISION_LOG D-004/D-029."
            )
        elif inventory_source == "user_supplied":
            simulation_notice = "Inventory position was supplied via --inventory-csv (real data)."
        else:
            simulation_notice = (
                "Inventory provenance for this run is unknown (run predates "
                "run_metadata.json). Treat inventory_position as SIMULATED "
                "unless independently verified — see DECISION_LOG D-004."
            )

        return {
            "store_id": store_id,
            "item_id": item_id,
            "mode": mode_label,
            "inventory_source": inventory_source,
            "forecast_daily_mean": row.get("forecast_daily_mean"),
            "inventory_position": row.get("inventory_position"),
            "lead_time_days": row.get("lead_time_days"),
            "safety_stock": row.get("safety_stock"),
            "reorder_point": row.get("reorder_point"),
            "recommended_order_qty": row.get("recommended_order_qty"),
            "decision_reason": row.get("decision_reason"),
            "risk_flag": row.get("risk_flag"),
            "explanation": explain_decision(row.get("decision_reason"), row.get("risk_flag")),
            "simulation_notice": simulation_notice,
        }

    # -- risk / prioritization -------------------------------------------

    def get_risk(self, top: int = 20, store_id: str | None = None) -> list[dict[str, Any]]:
        if top <= 0:
            raise ValidationError(f"top must be a positive integer; got {top}")
        if store_id is not None:
            store_id = validate_identifier("store_id", store_id)

        rec_df = self.recommendations_df
        base = rec_df if rec_df is not None else self.risk_df
        if base is None or base.empty:
            return []

        df = base.copy()
        if store_id is not None:
            df = df.loc[df["store_id"] == store_id]

        df["_severity"] = df["risk_flag"].map(_RISK_SEVERITY_ORDER).fillna(5)
        sort_cols = ["_severity"]
        ascending = [True]
        if "recommended_order_qty" in df.columns:
            sort_cols.append("recommended_order_qty")
            ascending.append(False)
        df = df.sort_values(sort_cols, ascending=ascending).head(top)

        out_cols = [c for c in [
            "store_id", "item_id", "forecast_daily_mean", "inventory_position",
            "safety_stock", "reorder_point", "recommended_order_qty",
            "decision_reason", "risk_flag",
        ] if c in df.columns]
        records = df[out_cols].to_dict("records")
        for r in records:
            r["explanation"] = explain_decision(r.get("decision_reason"), r.get("risk_flag"))
        return records

    # -- summary -------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        forecast_df = self.forecast_df
        n_series = int(forecast_df[KEY_COLS].drop_duplicates().shape[0])
        by_store = (
            forecast_df[KEY_COLS].drop_duplicates()
            .groupby("store_id").size().to_dict()
        )
        by_segment = (
            forecast_df[KEY_COLS + ["segment"]].drop_duplicates(subset=KEY_COLS)
            .groupby("segment").size().to_dict()
            if "segment" in forecast_df.columns else {}
        )

        risk_counts = (
            self.risk_df["risk_flag"].value_counts().to_dict()
            if "risk_flag" in self.risk_df.columns else {}
        )

        rec_df = self.recommendations_df
        summary: dict[str, Any] = {
            "n_series": n_series,
            "by_store": by_store,
            "by_segment": by_segment,
            "risk_flag_counts": risk_counts,
            "run_metadata": self.run_metadata,
        }
        if rec_df is not None:
            summary["n_recommendations"] = int(len(rec_df))
            summary["n_orders_recommended"] = int((rec_df["recommended_order_qty"].fillna(0) > 0).sum())
            summary["decision_reason_counts"] = rec_df["decision_reason"].value_counts().to_dict()
        else:
            summary["n_recommendations"] = 0
            summary["note"] = "forecast-only run: no replenishment recommendations available"
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _handle(fn, *args, **kwargs) -> int:
    try:
        _print_json(fn(*args, **kwargs))
        return 0
    except Phase7Error as e:
        print(json.dumps({"error": type(e).__name__, "message": str(e)}, indent=2), file=sys.stderr)
        return 1
    except Exception as e:  # pragma: no cover - last-resort guard, never a raw traceback
        print(json.dumps({"error": "InternalError", "message": str(e)}, indent=2), file=sys.stderr)
        return 1


def build_store(fixture: bool = False) -> Phase7DataStore:
    cfg = load_config()
    return Phase7DataStore.from_config(cfg, fixture=fixture)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase7-interface",
        description="Phase 7 — read-only retail intelligence interface over Phase 6 outputs.",
    )
    parser.add_argument("--fixture", action="store_true", help="Query the fixture Phase 6 run instead of production.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Check that Phase 6 outputs are available.")
    sub.add_parser("metadata", help="Model/configuration provenance.")

    p_forecast = sub.add_parser("forecast", help="Forecast for one store/item.")
    p_forecast.add_argument("store_id")
    p_forecast.add_argument("item_id")

    p_repl = sub.add_parser("replenishment", help="Replenishment recommendation for one store/item.")
    p_repl.add_argument("store_id")
    p_repl.add_argument("item_id")

    p_risk = sub.add_parser("risk", help="Top-N series ranked by replenishment risk/priority.")
    p_risk.add_argument("--top", type=int, default=20)
    p_risk.add_argument("--store", dest="store_id", default=None)

    sub.add_parser("summary", help="Aggregate summary of the current Phase 6 run.")

    args = parser.parse_args(argv)
    store = build_store(fixture=args.fixture)

    if args.command == "health":
        return _handle(store.health)
    if args.command == "metadata":
        return _handle(store.metadata)
    if args.command == "forecast":
        return _handle(store.get_forecast, args.store_id, args.item_id)
    if args.command == "replenishment":
        return _handle(store.get_replenishment, args.store_id, args.item_id)
    if args.command == "risk":
        return _handle(store.get_risk, args.top, args.store_id)
    if args.command == "summary":
        return _handle(store.get_summary)

    parser.error(f"Unknown command: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    sys.exit(main())
