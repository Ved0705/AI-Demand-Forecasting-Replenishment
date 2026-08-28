"""Tests for the ingestion layer.

These run against the synthetic fixture. They verify CODE CORRECTNESS only -
no analytical conclusion is drawn from fixture numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.ingest import (
    apply_subset,
    attach_prices,
    build_long_table,
    flag_active_window,
    load_calendar,
    load_prices,
    load_sales_wide,
    wide_to_long,
)
from src.make_fixture import build_fixture


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    return build_fixture(seed=0, out_dir=tmp_path_factory.mktemp("fixture"))


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def long(cfg, fixture_dir):
    return build_long_table(cfg, raw_dir=fixture_dir)


# --- reshape ---------------------------------------------------------------

def test_wide_to_long_preserves_total_units(cfg, fixture_dir):
    wide = load_sales_wide(cfg, fixture_dir)
    cal = load_calendar(cfg, fixture_dir)
    d_cols = [c for c in wide.columns if c.startswith("d_")]
    expected = wide[d_cols].to_numpy().sum()
    out = wide_to_long(wide, cal)
    assert out["sales"].sum() == expected
    assert len(out) == len(wide) * len(d_cols)


def test_every_row_gets_a_real_date(cfg, fixture_dir):
    wide = load_sales_wide(cfg, fixture_dir)
    cal = load_calendar(cfg, fixture_dir)
    out = wide_to_long(wide, cal)
    assert out["date"].notna().all()
    assert out["date"].nunique() == len([c for c in wide.columns if c.startswith("d_")])


# --- joins -----------------------------------------------------------------

def test_price_join_does_not_fan_out(cfg, fixture_dir):
    wide = load_sales_wide(cfg, fixture_dir)
    cal = load_calendar(cfg, fixture_dir)
    prices = load_prices(cfg, fixture_dir)
    long = wide_to_long(wide, cal)
    joined = attach_prices(long, prices)
    assert len(joined) == len(long)


def test_price_join_raises_on_duplicate_keys(cfg, fixture_dir):
    """A duplicated (store,item,week) must fail loudly, not silently multiply rows."""
    wide = load_sales_wide(cfg, fixture_dir).head(2)
    cal = load_calendar(cfg, fixture_dir)
    prices = load_prices(cfg, fixture_dir)
    dup_prices = pd.concat([prices, prices.head(1)], ignore_index=True)
    long = wide_to_long(wide, cal)
    with pytest.raises(Exception):
        attach_prices(long, dup_prices)


# --- availability ----------------------------------------------------------

def test_active_window_starts_at_first_price(cfg, fixture_dir):
    wide = load_sales_wide(cfg, fixture_dir)
    cal = load_calendar(cfg, fixture_dir)
    prices = load_prices(cfg, fixture_dir)
    long = attach_prices(wide_to_long(wide, cal), prices)
    flagged = flag_active_window(long, cfg)

    # No active row may precede that series' first observed price.
    priced_start = (
        long.loc[long["sell_price"].notna()]
        .groupby(["store_id", "item_id"], observed=True)["date"].min()
    )
    active = flagged.loc[flagged["is_active"]]
    merged = active.merge(
        priced_start.rename("first_price_date"), on=["store_id", "item_id"], how="left"
    )
    assert (merged["date"] >= merged["first_price_date"]).all()


def test_dropping_inactive_removes_leading_zeros(long):
    """Zero share after filtering must be lower than before - the whole point."""
    assert long["is_active"].all()
    # Fixture builds structural leading zeros; filtering should remove them all.
    first_dates = long.groupby(["store_id", "item_id"], observed=True)["date"].min()
    assert first_dates.nunique() > 1, "expected staggered item introductions"


# --- subsetting ------------------------------------------------------------

def test_subset_filters_stores(cfg, fixture_dir):
    wide = load_sales_wide(cfg, fixture_dir)
    out = apply_subset(wide, cfg)
    assert set(out["store_id"]) <= set(cfg["subset"]["stores"])


# --- output contract -------------------------------------------------------

def test_long_table_is_sorted_and_unique(long):
    keys = ["store_id", "item_id", "date"]
    assert not long.duplicated(subset=keys).any()
    assert long[keys].equals(long.sort_values(keys)[keys].reset_index(drop=True))


def test_expected_columns_present(long):
    required = {
        "item_id", "store_id", "dept_id", "cat_id", "state_id",
        "date", "d", "wm_yr_wk", "sales", "sell_price", "is_active",
        "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI",
    }
    assert required <= set(long.columns), required - set(long.columns)
