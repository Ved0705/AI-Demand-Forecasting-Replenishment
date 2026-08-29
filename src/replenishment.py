"""Phase 6 — Replenishment decision engine.

This module is the DECISION layer only. It never fits or calls a forecasting
model; it consumes forecasts (store_id, item_id, forecast_date, forecast_units,
horizon_day) produced elsewhere (src/phase6_run.py) plus explicit business
inputs, and turns them into auditable replenishment recommendations.

ARCHITECTURE
------------
    FORECAST  ->  UNCERTAINTY/RISK  ->  INVENTORY STATE  ->  DECISION

Every number that is not derived from the forecast is a declared business
assumption (config.yaml `replenishment:` / `phase6:` sections), never an M5
observation. See DECISION_LOG.md D-004 and D-027..D-029.

UNCERTAINTY METHODOLOGY (documented, not calibrated)
-----------------------------------------------------
XGBoost point predictions carry no native uncertainty estimate. Rather than
fabricate one, this module derives a per-segment demand-error sigma from the
Phase 5 backtest's out-of-sample RMSE (reports/phase5/segment_model_comparison.csv,
method == "xgboost"), averaged across the 5 folds. RMSE approximates the
error standard deviation under the (unverified but small, see phase5 report)
assumption that bias is close to zero.

This is a RISK PROXY, not a calibrated prediction interval:
  - It is a segment-level average, not per-series or per-date.
  - It was measured historically across all 5 backtest folds; it is NEVER
    computed from the actuals of the specific decision being evaluated.
  - It says nothing about how well-calibrated a resulting 95% coverage claim
    would be — no calibration study has been run.

Safety stock and reorder point use it purely as an ordering signal, not as a
scientifically validated service-level guarantee.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decision reason / risk flag vocabularies (kept as plain strings so output
# CSVs stay human-readable and grep-able).
# ---------------------------------------------------------------------------

REASON_ORDER_BELOW_ROP = "inventory_position_below_reorder_point"
REASON_NO_ORDER_NEEDED = "inventory_position_at_or_above_target"
REASON_MISSING_FORECAST = "missing_forecast"
REASON_MISSING_INVENTORY = "missing_inventory_data"
REASON_ZERO_DEMAND_SERIES = "zero_forecast_no_order"

RISK_STOCKOUT = "STOCKOUT_RISK"
RISK_HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY_SEGMENT"
RISK_HIGH_OUTLIER = "HIGH_FORECAST_OUTLIER"
RISK_LOW = "LOW_RISK"
RISK_NO_DATA = "INSUFFICIENT_DATA"

_HIGH_UNCERTAINTY_SEGMENTS = {"intermittent", "lumpy", "unknown"}


# ---------------------------------------------------------------------------
# Safety-stock / reorder-point math (pure functions — unit tested directly)
# ---------------------------------------------------------------------------

def derive_safety_stock_z(service_level: float, override: float | None = None) -> float:
    """Convert a target service level into a safety-stock z-multiplier.

    Uses the inverse standard normal CDF (scipy.stats.norm.ppf), the standard
    textbook mapping from a one-sided cycle-service-level target to a
    z-score for the safety-stock formula z * sigma * sqrt(lead_time).

    ``override`` lets config.yaml bypass the derivation entirely with an
    explicit number (phase6.safety_stock_z_override); null (None) means
    derive it from service_level, which is the default and recommended path.
    """
    if override is not None:
        return float(override)
    if not (0.0 < service_level < 1.0):
        raise ValueError(f"service_level must be in (0, 1); got {service_level}")
    return float(stats.norm.ppf(service_level))


def compute_safety_stock(sigma_daily: float, lead_time_days: float, z: float) -> float:
    """Safety stock = z * sigma_daily * sqrt(lead_time_days).

    Standard periodic-review safety-stock formula assuming i.i.d. daily
    demand error across the lead time. sigma_daily is a risk PROXY (see
    module docstring), so treat the result as an ordering signal, not a
    guaranteed service level.
    """
    if sigma_daily < 0 or lead_time_days < 0:
        raise ValueError("sigma_daily and lead_time_days must be non-negative")
    return float(z * sigma_daily * math.sqrt(lead_time_days))


def compute_reorder_point(forecast_daily_mean: float, lead_time_days: float, safety_stock: float) -> float:
    """Reorder point = expected demand over lead time + safety stock."""
    forecast_daily_mean = max(0.0, forecast_daily_mean)
    return float(forecast_daily_mean * lead_time_days + safety_stock)


def compute_order_up_to_level(
    forecast_daily_mean: float,
    lead_time_days: float,
    review_period_days: float,
    safety_stock: float,
) -> float:
    """Order-up-to level for periodic (R, S) review: covers lead time + one
    review cycle, since the next chance to order is one review period away.
    """
    forecast_daily_mean = max(0.0, forecast_daily_mean)
    return float(forecast_daily_mean * (lead_time_days + review_period_days) + safety_stock)


def simulate_inventory_position(forecast_daily_mean: float, initial_days_of_cover: float) -> float:
    """Simulated starting inventory position when no real on-hand data exists.

    Purely a scenario default (config phase6/replenishment), documented as
    such wherever it is used — never presented as an observed M5 quantity.
    """
    return float(max(0.0, forecast_daily_mean) * max(0.0, initial_days_of_cover))


# ---------------------------------------------------------------------------
# Risk proxy loading
# ---------------------------------------------------------------------------

def load_segment_risk_proxy(cfg: Config, method: str = "xgboost") -> dict[str, float]:
    """Load per-segment demand-error sigma from the Phase 5 backtest report.

    Uses RMSE of the selected model (``method``, default xgboost) averaged
    across all 5 backtest folds, from reports/phase5/segment_model_comparison.csv.
    Out-of-sample by construction (it is the Phase 5 backtest output) and
    never touches the actuals of the decision currently being scored.

    Returns {} if the report is unavailable — callers must fall back to
    ``phase6.fallback_sigma`` (see ``resolve_sigma``).
    """
    path = Path(cfg.get("phase6", {}).get("risk_source", "reports/phase5/segment_model_comparison.csv"))
    if not path.is_absolute():
        path = cfg.root / path
    if not path.exists():
        logger.warning("Risk source %s not found; risk proxy unavailable.", path)
        return {}

    df = pd.read_csv(path)
    sub = df.loc[df["method"] == method]
    if sub.empty:
        logger.warning("No rows for method=%r in %s; risk proxy unavailable.", method, path)
        return {}
    return sub.groupby("segment")["rmse"].mean().to_dict()


def resolve_sigma(segment: str | None, risk_map: dict[str, float], fallback_sigma: float) -> float:
    """Segment sigma, falling back to a documented config placeholder.

    Missing/None/'unknown' segment, or a segment absent from the risk map,
    all resolve to fallback_sigma rather than raising — this is an explicit,
    deterministic edge-case handling path (see class docstring / tests).
    """
    if segment is not None and segment in risk_map:
        val = risk_map[segment]
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return float(val)
    return float(fallback_sigma)


# ---------------------------------------------------------------------------
# Per-row decision engine
# ---------------------------------------------------------------------------

@dataclass
class ReplenishmentPolicy:
    """Bundle of config-driven business assumptions. Never hardcoded values —
    always constructed from config.yaml (see ``ReplenishmentPolicy.from_config``).
    """

    lead_time_days: float
    service_level: float
    review_period_days: float
    initial_inventory_days_of_cover: float
    safety_stock_z_override: float | None
    fallback_sigma: float
    outlier_forecast_multiplier: float

    @classmethod
    def from_config(cls, cfg: Config) -> "ReplenishmentPolicy":
        rep = cfg["replenishment"]
        p6 = cfg.get("phase6", {})
        required = ("lead_time_days", "service_level", "review_period_days",
                    "initial_inventory_days_of_cover")
        missing = [k for k in required if k not in rep]
        if missing:
            raise ValueError(
                f"config.yaml replenishment: section is missing required "
                f"assumption(s) {missing}. Add them explicitly rather than "
                "letting the decision engine fabricate a default."
            )
        return cls(
            lead_time_days=float(rep["lead_time_days"]),
            service_level=float(rep["service_level"]),
            review_period_days=float(rep["review_period_days"]),
            initial_inventory_days_of_cover=float(rep["initial_inventory_days_of_cover"]),
            safety_stock_z_override=p6.get("safety_stock_z_override"),
            fallback_sigma=float(p6.get("fallback_sigma", 1.0)),
            outlier_forecast_multiplier=float(p6.get("outlier_forecast_multiplier", 10.0)),
        )


def decide_one_series(
    forecast_daily_mean: float | None,
    segment: str | None,
    policy: ReplenishmentPolicy,
    risk_map: dict[str, float],
    inventory_position: float | None = None,
    historical_mean_sales: float | None = None,
) -> dict:
    """Apply the full decision policy to a single series and return an
    auditable record. Every branch below is an explicit, deterministic edge
    case (see tests/test_replenishment.py for the full matrix).
    """
    # --- Edge case: no forecast available ---------------------------------
    if forecast_daily_mean is None or (isinstance(forecast_daily_mean, float) and np.isnan(forecast_daily_mean)):
        return {
            "forecast_daily_mean": None,
            "lead_time_days": policy.lead_time_days,
            "safety_stock": None,
            "reorder_point": None,
            "recommended_order_qty": None,
            "decision_reason": REASON_MISSING_FORECAST,
            "risk_flag": RISK_NO_DATA,
        }

    # Negative predictions cannot occur from GlobalXGBoostForecaster (it hard
    # -clips at 0), but the decision layer clips defensively too so it never
    # trusts an upstream guarantee silently.
    forecast_daily_mean = max(0.0, float(forecast_daily_mean))

    z = derive_safety_stock_z(policy.service_level, policy.safety_stock_z_override)
    sigma = resolve_sigma(segment, risk_map, policy.fallback_sigma)
    safety_stock = compute_safety_stock(sigma, policy.lead_time_days, z)
    reorder_point = compute_reorder_point(forecast_daily_mean, policy.lead_time_days, safety_stock)
    order_up_to = compute_order_up_to_level(
        forecast_daily_mean, policy.lead_time_days, policy.review_period_days, safety_stock
    )

    # --- Edge case: no inventory information supplied ----------------------
    if inventory_position is None:
        return {
            "forecast_daily_mean": forecast_daily_mean,
            "lead_time_days": policy.lead_time_days,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "recommended_order_qty": None,
            "decision_reason": REASON_MISSING_INVENTORY,
            "risk_flag": RISK_NO_DATA,
        }

    # --- Edge case: zero/negative inventory position ------------------------
    # A negative position (backorders) is treated literally, not clamped to
    # zero, so the order quantity correctly accounts for the shortfall.
    inventory_position = float(inventory_position)

    if forecast_daily_mean == 0.0:
        # Edge case: zero-demand series. Only order if position has gone
        # negative (backorder) or below safety stock; otherwise no order.
        if inventory_position >= safety_stock:
            order_qty = 0.0
            reason = REASON_ZERO_DEMAND_SERIES
        else:
            order_qty = max(0.0, safety_stock - inventory_position)
            reason = REASON_ORDER_BELOW_ROP
    elif inventory_position < reorder_point:
        order_qty = max(0.0, order_up_to - inventory_position)
        reason = REASON_ORDER_BELOW_ROP
    else:
        order_qty = 0.0
        reason = REASON_NO_ORDER_NEEDED

    # --- Risk flags (informational, never alter the order quantity) --------
    risk_flag = RISK_LOW
    if inventory_position <= 0.0 or inventory_position < safety_stock:
        risk_flag = RISK_STOCKOUT
    elif segment in _HIGH_UNCERTAINTY_SEGMENTS:
        risk_flag = RISK_HIGH_UNCERTAINTY

    if historical_mean_sales is not None and historical_mean_sales > 0:
        if forecast_daily_mean > policy.outlier_forecast_multiplier * historical_mean_sales:
            risk_flag = RISK_HIGH_OUTLIER

    return {
        "forecast_daily_mean": forecast_daily_mean,
        "lead_time_days": policy.lead_time_days,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "recommended_order_qty": order_qty,
        "decision_reason": reason,
        "risk_flag": risk_flag,
    }


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def build_replenishment_recommendations(
    forecast_df: pd.DataFrame,
    cfg: Config,
    inventory_df: pd.DataFrame | None = None,
    risk_map: dict[str, float] | None = None,
    historical_mean_df: pd.DataFrame | None = None,
    key_cols: tuple[str, ...] = ("store_id", "item_id"),
) -> pd.DataFrame:
    """Turn a per-date forecast into per-series replenishment recommendations.

    Parameters
    ----------
    forecast_df : columns store_id, item_id, forecast_date, forecast_units,
                  horizon_day[, segment]. One row per series x forecast date.
    inventory_df : optional columns key_cols + inventory_position. If None,
                   inventory is SIMULATED from
                   replenishment.initial_inventory_days_of_cover (see
                   ``simulate_inventory_position``) and every output row is
                   labelled accordingly by the caller (REPLENISHMENT
                   SIMULATION mode) — see src/phase6_run.py.
    historical_mean_df : optional columns key_cols + historical_mean_sales,
                   used only for the HIGH_FORECAST_OUTLIER risk flag.

    Returns
    -------
    One row per series: store_id, item_id, forecast_horizon (days),
    forecast_daily_mean, inventory_position, lead_time_days, safety_stock,
    reorder_point, recommended_order_qty, decision_reason, risk_flag.
    """
    key_cols = list(key_cols)
    policy = ReplenishmentPolicy.from_config(cfg)
    risk_map = risk_map if risk_map is not None else load_segment_risk_proxy(cfg)

    if forecast_df.empty:
        return pd.DataFrame(columns=key_cols + [
            "forecast_horizon", "forecast_daily_mean", "inventory_position",
            "lead_time_days", "safety_stock", "reorder_point",
            "recommended_order_qty", "decision_reason", "risk_flag",
        ])

    agg = {"forecast_units": "mean", "horizon_day": "count"}
    seg_col = "segment" if "segment" in forecast_df.columns else None
    group_cols = key_cols + ([seg_col] if seg_col else [])
    per_series = forecast_df.groupby(key_cols, observed=True).agg(
        forecast_daily_mean=("forecast_units", "mean"),
        forecast_horizon=("horizon_day", "nunique"),
    ).reset_index()
    if seg_col:
        seg_lookup = forecast_df[key_cols + [seg_col]].drop_duplicates(subset=key_cols)
        per_series = per_series.merge(seg_lookup, on=key_cols, how="left")
    else:
        per_series[seg_col or "segment"] = None
        seg_col = "segment"

    if inventory_df is not None:
        per_series = per_series.merge(
            inventory_df[key_cols + ["inventory_position"]], on=key_cols, how="left"
        )
    else:
        per_series["inventory_position"] = per_series["forecast_daily_mean"].apply(
            lambda m: simulate_inventory_position(m, policy.initial_inventory_days_of_cover)
        )

    if historical_mean_df is not None:
        per_series = per_series.merge(
            historical_mean_df[key_cols + ["historical_mean_sales"]], on=key_cols, how="left"
        )
    else:
        per_series["historical_mean_sales"] = np.nan

    records = []
    for _, row in per_series.iterrows():
        decision = decide_one_series(
            forecast_daily_mean=row["forecast_daily_mean"],
            segment=row.get(seg_col),
            policy=policy,
            risk_map=risk_map,
            inventory_position=row.get("inventory_position"),
            historical_mean_sales=row.get("historical_mean_sales"),
        )
        record = {k: row[k] for k in key_cols}
        record["forecast_horizon"] = row["forecast_horizon"]
        record["inventory_position"] = row.get("inventory_position")
        record.update({k: v for k, v in decision.items() if k != "forecast_daily_mean"})
        record["forecast_daily_mean"] = decision["forecast_daily_mean"]
        records.append(record)

    out = pd.DataFrame(records)
    col_order = key_cols + [
        "forecast_horizon", "forecast_daily_mean", "inventory_position",
        "lead_time_days", "safety_stock", "reorder_point",
        "recommended_order_qty", "decision_reason", "risk_flag",
    ]
    return out[[c for c in col_order if c in out.columns]]


def build_forecast_risk_summary(
    forecast_df: pd.DataFrame,
    cfg: Config,
    risk_map: dict[str, float] | None = None,
    historical_mean_df: pd.DataFrame | None = None,
    key_cols: tuple[str, ...] = ("store_id", "item_id"),
) -> pd.DataFrame:
    """FORECAST-ONLY risk view: sigma/safety-stock/reorder-point WITHOUT any
    inventory assumption. Used by phase6_run.py in ``--mode forecast-only``
    so risk information can be reported without pretending real inventory
    is known. See module docstring for the uncertainty methodology.
    """
    key_cols = list(key_cols)
    policy = ReplenishmentPolicy.from_config(cfg)
    risk_map = risk_map if risk_map is not None else load_segment_risk_proxy(cfg)
    z = derive_safety_stock_z(policy.service_level, policy.safety_stock_z_override)

    seg_col = "segment" if "segment" in forecast_df.columns else None
    per_series = forecast_df.groupby(key_cols, observed=True).agg(
        forecast_daily_mean=("forecast_units", "mean"),
        forecast_horizon=("horizon_day", "nunique"),
    ).reset_index()
    if seg_col:
        seg_lookup = forecast_df[key_cols + [seg_col]].drop_duplicates(subset=key_cols)
        per_series = per_series.merge(seg_lookup, on=key_cols, how="left")
    else:
        per_series["segment"] = None

    if historical_mean_df is not None:
        per_series = per_series.merge(
            historical_mean_df[key_cols + ["historical_mean_sales"]], on=key_cols, how="left"
        )
    else:
        per_series["historical_mean_sales"] = np.nan

    per_series["sigma_daily"] = per_series["segment"].apply(
        lambda s: resolve_sigma(s, risk_map, policy.fallback_sigma)
    )
    per_series["safety_stock_z"] = z
    per_series["safety_stock"] = per_series["sigma_daily"].apply(
        lambda s: compute_safety_stock(s, policy.lead_time_days, z)
    )
    per_series["reorder_point_demand_only"] = per_series.apply(
        lambda r: compute_reorder_point(r["forecast_daily_mean"], policy.lead_time_days, r["safety_stock"]),
        axis=1,
    )

    def _risk_flag(row) -> str:
        if row["forecast_daily_mean"] is None or pd.isna(row["forecast_daily_mean"]):
            return RISK_NO_DATA
        hm = row.get("historical_mean_sales")
        if hm is not None and not pd.isna(hm) and hm > 0:
            if row["forecast_daily_mean"] > policy.outlier_forecast_multiplier * hm:
                return RISK_HIGH_OUTLIER
        if row.get("segment") in _HIGH_UNCERTAINTY_SEGMENTS:
            return RISK_HIGH_UNCERTAINTY
        return RISK_LOW

    per_series["risk_flag"] = per_series.apply(_risk_flag, axis=1)

    col_order = key_cols + [
        "segment", "forecast_horizon", "forecast_daily_mean", "sigma_daily",
        "safety_stock_z", "safety_stock", "reorder_point_demand_only", "risk_flag",
    ]
    return per_series[[c for c in col_order if c in per_series.columns]]
