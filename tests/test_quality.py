"""Tests for quality checks - each check must actually catch its bug."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.ingest import build_long_table
from src.make_fixture import build_fixture
from src.quality import run_checks


@pytest.fixture(scope="module")
def long():
    cfg = load_config()
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    return build_long_table(cfg, raw_dir=build_fixture(seed=1, out_dir=d))


def test_clean_data_passes(long):
    rep = run_checks(long)
    assert rep.passed, str(rep)


def test_duplicate_rows_are_caught(long):
    bad = pd.concat([long, long.head(1)], ignore_index=True)
    rep = run_checks(bad)
    assert not rep.passed
    assert any(r.name == "no_duplicate_series_dates" and not r.passed for r in rep.results)


def test_negative_sales_are_caught(long):
    bad = long.copy()
    bad.loc[bad.index[0], "sales"] = -5
    rep = run_checks(bad)
    assert any(r.name == "no_negative_sales" and not r.passed for r in rep.results)


def test_date_gap_is_caught(long):
    """Drop a middle date from one series - lag features would silently break."""
    bad = long.copy()
    key = bad.iloc[len(bad) // 2]
    mask = (
        (bad["store_id"] == key["store_id"])
        & (bad["item_id"] == key["item_id"])
        & (bad["date"] == key["date"])
    )
    bad = bad.loc[~mask]
    rep = run_checks(bad)
    assert any(r.name == "series_are_contiguous_daily" and not r.passed for r in rep.results)
