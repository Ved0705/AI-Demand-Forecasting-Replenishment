"""M5 ingestion: raw CSVs -> tidy long table -> parquet + DuckDB star schema.

Design notes worth defending in an interview:

1.  M5 ships sales in WIDE form (one column per day, d_1..d_1941). Every
    downstream step - lag features, backtesting, segmentation - needs LONG
    form (one row per item-store-date). Reshaping once, at ingestion, means
    no later stage has to think about it.

2.  Leading zeros in M5 are ambiguous. A zero can mean "nobody bought it" or
    "the store did not stock it yet". Treating the second as real demand
    biases every forecast downward and corrupts intermittency metrics. We
    detect an activity window per series and flag it. See DECISION_LOG D-003.

3.  Prices join on (store_id, item_id, wm_yr_wk) - WEEKLY grain against DAILY
    sales. The many-to-one direction is validated explicitly rather than
    assumed, because a silent fan-out here would inflate row counts and
    quietly corrupt every aggregate in the project.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config

logger = logging.getLogger(__name__)

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def resolve_sales_path(cfg: Config) -> Path:
    """Prefer the evaluation file (1941 days); fall back to validation (1913)."""
    primary = cfg.raw_file(cfg["source"]["sales_file"])
    if primary.exists():
        return primary
    fallback = cfg.raw_file(cfg["source"]["sales_file_fallback"])
    if fallback.exists():
        logger.warning("Using fallback sales file %s", fallback.name)
        return fallback
    raise FileNotFoundError(
        f"No sales file found in {cfg.path('raw')}. Expected "
        f"{cfg['source']['sales_file']} or {cfg['source']['sales_file_fallback']}. "
        "Download from Kaggle: m5-forecasting-accuracy."
    )


def load_calendar(cfg: Config, raw_dir: Path | None = None) -> pd.DataFrame:
    path = (raw_dir / cfg["source"]["calendar_file"]) if raw_dir else cfg.raw_file(
        cfg["source"]["calendar_file"]
    )
    cal = pd.read_csv(path, parse_dates=["date"])
    cal["wm_yr_wk"] = cal["wm_yr_wk"].astype("int32")
    return cal


def load_prices(cfg: Config, raw_dir: Path | None = None) -> pd.DataFrame:
    path = (raw_dir / cfg["source"]["prices_file"]) if raw_dir else cfg.raw_file(
        cfg["source"]["prices_file"]
    )
    prices = pd.read_csv(path)
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")
    prices["sell_price"] = prices["sell_price"].astype("float32")
    return prices


def load_sales_wide(cfg: Config, raw_dir: Path | None = None) -> pd.DataFrame:
    path = (raw_dir / cfg["source"]["sales_file"]) if raw_dir else resolve_sales_path(cfg)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Subsetting
# ---------------------------------------------------------------------------

def apply_subset(sales_wide: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filter series before reshaping - reshaping is the expensive step."""
    sub = cfg["subset"]
    out = sales_wide
    before = len(out)

    if sub.get("stores"):
        out = out[out["store_id"].isin(sub["stores"])]
    if sub.get("categories"):
        out = out[out["cat_id"].isin(sub["categories"])]
    if sub.get("departments"):
        out = out[out["dept_id"].isin(sub["departments"])]

    cap = sub.get("max_items_per_store")
    if cap:
        # Deterministic cap: take the first N item_ids alphabetically per store
        # so reruns are reproducible without depending on a seed.
        out = (
            out.sort_values(["store_id", "item_id"])
            .groupby("store_id", group_keys=False)
            .head(cap)
        )

    logger.info("Subset: %d -> %d series", before, len(out))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------

def wide_to_long(sales_wide: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Melt d_* columns into rows and attach real dates."""
    d_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    if not d_cols:
        raise ValueError("No d_* columns found - is this the M5 sales file?")

    long = sales_wide.melt(
        id_vars=[c for c in ID_COLS if c in sales_wide.columns],
        value_vars=d_cols,
        var_name="d",
        value_name="sales",
    )
    long["sales"] = long["sales"].astype("int32")

    cal_slim = calendar[["d", "date", "wm_yr_wk"]]
    n_before = len(long)
    long = long.merge(cal_slim, on="d", how="left", validate="many_to_one")
    if len(long) != n_before:
        raise AssertionError("Calendar join changed row count - duplicate d values")
    if long["date"].isna().any():
        missing = long.loc[long["date"].isna(), "d"].unique()[:5]
        raise AssertionError(f"d values missing from calendar: {missing}")

    return long


def attach_prices(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join weekly prices onto daily sales.

    Validated as many_to_one: many daily rows -> one weekly price row. If the
    price table ever contains duplicate (store, item, week) keys this raises
    instead of silently multiplying rows.
    """
    n_before = len(long)
    out = long.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    if len(out) != n_before:
        raise AssertionError("Price join changed row count")
    return out


# ---------------------------------------------------------------------------
# Availability window
# ---------------------------------------------------------------------------

def flag_active_window(long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mark rows where the item was actually sellable in that store.

    Rationale (DECISION_LOG D-003): M5 zeros before an item is listed are
    structural, not demand. Including them makes a series look far more
    intermittent than it is and drags forecasts toward zero.

    method="price"      -> active from the first week with a non-null price.
                           Cleanest signal: M5 only publishes a price for
                           weeks the item was on sale in that store.
    method="first_sale" -> active from the first positive sale. Biased: it
                           discards genuine early zero-demand days.
    method="none"       -> no filtering.
    """
    method = cfg["availability"]["method"]
    if method == "none":
        long["is_active"] = True
        return long

    keys = ["store_id", "item_id"]
    if method == "price":
        priced = long.loc[long["sell_price"].notna()]
        start = priced.groupby(keys, observed=True)["date"].min().rename("active_from")
    elif method == "first_sale":
        sold = long.loc[long["sales"] > 0]
        start = sold.groupby(keys, observed=True)["date"].min().rename("active_from")
    else:
        raise ValueError(f"Unknown availability method: {method!r}")

    out = long.merge(start, on=keys, how="left", validate="many_to_one")
    # Series that never had a price/sale are never active.
    out["is_active"] = out["active_from"].notna() & (out["date"] >= out["active_from"])
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_long_table(cfg: Config, raw_dir: Path | None = None) -> pd.DataFrame:
    """Full ingestion: load -> subset -> reshape -> join -> flag availability."""
    calendar = load_calendar(cfg, raw_dir)
    prices = load_prices(cfg, raw_dir)
    sales_wide = load_sales_wide(cfg, raw_dir)

    sales_wide = apply_subset(sales_wide, cfg)
    long = wide_to_long(sales_wide, calendar)
    long = attach_prices(long, prices)
    long = flag_active_window(long, cfg)

    # Calendar context needed by later phases. Joined here so feature code
    # never has to re-open the calendar file.
    cal_ctx = calendar[
        [c for c in calendar.columns if c not in ("date", "wm_yr_wk", "weekday")]
    ]
    long = long.merge(cal_ctx, on="d", how="left", validate="many_to_one")

    if cfg["availability"]["drop_inactive"]:
        n_before = len(long)
        long = long.loc[long["is_active"]].copy()
        logger.info(
            "Dropped %d inactive rows (%.1f%%)",
            n_before - len(long),
            100 * (n_before - len(long)) / max(n_before, 1),
        )

    long = long.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    return long


def write_outputs(long: pd.DataFrame, cfg: Config) -> dict[str, Path]:
    """Persist the tidy table as parquet and build the DuckDB star schema."""
    import duckdb

    cfg.ensure_dirs()
    parquet_path = cfg.path("processed") / "sales_long.parquet"
    long.to_parquet(parquet_path, index=False)

    db_path = cfg.path("duckdb")
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.register("long_df", long)

    con.execute("""
        CREATE TABLE dim_product AS
        SELECT DISTINCT item_id, dept_id, cat_id FROM long_df
    """)
    con.execute("""
        CREATE TABLE dim_store AS
        SELECT DISTINCT store_id, state_id FROM long_df
    """)
    con.execute("""
        CREATE TABLE dim_calendar AS
        SELECT DISTINCT date, d, wm_yr_wk, wday, month, year,
               event_name_1, event_type_1, event_name_2, event_type_2,
               snap_CA, snap_TX, snap_WI
        FROM long_df
    """)
    con.execute("""
        CREATE TABLE fact_sales AS
        SELECT item_id, store_id, date, d, wm_yr_wk,
               sales, sell_price, is_active
        FROM long_df
    """)
    con.execute("""
        CREATE TABLE dim_price AS
        SELECT DISTINCT store_id, item_id, wm_yr_wk, sell_price
        FROM long_df WHERE sell_price IS NOT NULL
    """)
    con.close()

    return {"parquet": parquet_path, "duckdb": db_path}


def main(config_path: str | None = None, raw_dir: Path | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config_path)
    long = build_long_table(cfg, raw_dir)
    out = write_outputs(long, cfg)
    print(f"Rows: {len(long):,}")
    print(f"Series: {long.groupby(['store_id', 'item_id'], observed=True).ngroups:,}")
    print(f"Dates: {long['date'].min().date()} -> {long['date'].max().date()}")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
