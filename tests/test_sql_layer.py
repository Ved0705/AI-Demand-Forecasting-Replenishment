"""Tests for the Phase 2 SQL layer.

Executed against a fixture-built DuckDB. These verify SQL CORRECTNESS - that
aggregates reconcile, joins do not fan out, and the state-aware SNAP logic is
right. No analytical conclusion is drawn from fixture numbers (D-006).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.ingest import build_long_table, write_outputs
from src.make_fixture import build_fixture
from src.sql_runner import SQLLayer, load_queries, parse_sql_file, validate


@pytest.fixture(scope="module")
def layer(tmp_path_factory):
    cfg = load_config()
    work = tmp_path_factory.mktemp("sqldb")
    fx = build_fixture(seed=0, out_dir=work / "fixture")
    long = build_long_table(cfg, raw_dir=fx)

    import duckdb
    db = work / "retail.duckdb"
    con = duckdb.connect(str(db))
    con.register("long_df", long)
    con.execute("CREATE TABLE dim_product AS SELECT DISTINCT item_id, dept_id, cat_id FROM long_df")
    con.execute("CREATE TABLE dim_store AS SELECT DISTINCT store_id, state_id FROM long_df")
    con.execute("""CREATE TABLE dim_calendar AS SELECT DISTINCT date, d, wm_yr_wk, wday,
                   month, year, event_name_1, event_type_1, event_name_2, event_type_2,
                   snap_CA, snap_TX, snap_WI FROM long_df""")
    con.execute("""CREATE TABLE fact_sales AS SELECT item_id, store_id, date, d, wm_yr_wk,
                   sales, sell_price, is_active FROM long_df""")
    con.execute("""CREATE TABLE dim_price AS SELECT DISTINCT store_id, item_id, wm_yr_wk,
                   sell_price FROM long_df WHERE sell_price IS NOT NULL""")
    con.close()

    L = SQLLayer(db, cfg.root / "sql")
    L.create_views()
    yield L
    L.close()


# --- parsing ---------------------------------------------------------------

def test_all_sql_files_parse():
    cfg = load_config()
    registry = load_queries(cfg.root / "sql")
    assert len(registry) > 20
    assert all(q.sql.strip() for q in registry.values())


def test_query_names_are_unique():
    """load_queries raises on collision; this pins that behaviour."""
    cfg = load_config()
    load_queries(cfg.root / "sql")  # must not raise


def test_ddl_detection_ignores_leading_comments():
    """Every view block opens with comments - naive prefix checks miss them."""
    cfg = load_config()
    views = parse_sql_file(cfg.root / "sql" / "00_views.sql")
    assert views, "no view blocks parsed"
    assert all(q.is_ddl for q in views), [q.name for q in views if not q.is_ddl]


def test_params_directive_is_stripped_from_sql():
    cfg = load_config()
    registry = load_queries(cfg.root / "sql")
    q = registry["top_products_per_store"]
    assert "top_n" in q.params
    assert "-- params:" not in q.sql


# --- execution -------------------------------------------------------------

def test_every_analytical_query_executes(layer):
    failures = []
    for name in layer.analytical_queries():
        try:
            df = layer.run(name)
            assert isinstance(df, pd.DataFrame)
        except Exception as exc:  # noqa: BLE001
            failures.append((name, str(exc)[:120]))
    assert not failures, failures


def test_missing_parameter_raises(layer):
    with pytest.raises(ValueError, match="needs parameters"):
        layer.run("top_products_per_store", top_n=None)


def test_unknown_query_raises(layer):
    with pytest.raises(KeyError):
        layer.run("no_such_query")


def test_parameter_actually_changes_result(layer):
    """A parameter that silently does nothing is worse than no parameter."""
    small = layer.run("top_products_per_store", top_n=3)
    large = layer.run("top_products_per_store", top_n=5)
    assert len(large) > len(small)
    assert small["rank"].max() <= 3


# --- reconciliation --------------------------------------------------------

def test_full_validation_passes(layer):
    ok, report = validate(layer)
    assert ok, report.to_string(index=False)


def test_view_does_not_drop_or_duplicate_fact_rows(layer):
    fact = layer.con.execute("SELECT COUNT(*), SUM(sales) FROM fact_sales").fetchone()
    view = layer.con.execute("SELECT COUNT(*), SUM(sales) FROM v_sales").fetchone()
    assert fact == view


def test_category_and_store_totals_agree(layer):
    cat = layer.run("kpi_by_category")["total_units"].sum()
    store = layer.run("store_performance")["total_units"].sum()
    assert int(cat) == int(store)


def test_abc_classes_partition_the_assortment(layer):
    abc = layer.run("abc_classification")
    n_series = layer.con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT item_id, store_id FROM v_sales)"
    ).fetchone()[0]
    assert int(abc["n_series"].sum()) == n_series
    assert abs(float(abc["pct_of_units"].sum()) - 100.0) < 0.5


# --- SNAP correctness (the easiest thing in this schema to get wrong) ------

def test_snap_flag_resolves_to_the_stores_own_state(layer):
    """The bug this guards: reading snap_CA for a Wisconsin store.

    Compares v_sales.snap_active against the calendar column for that store's
    actual state. Any mismatch means the CASE in v_sales is wrong.
    """
    mismatches = layer.con.execute("""
        SELECT COUNT(*) FROM v_sales v
        JOIN dim_calendar c ON v.date = c.date
        WHERE v.snap_active <> CASE v.state_id
            WHEN 'CA' THEN c.snap_CA
            WHEN 'TX' THEN c.snap_TX
            WHEN 'WI' THEN c.snap_WI END
    """).fetchone()[0]
    assert mismatches == 0


def test_snap_flag_is_never_null(layer):
    """A state outside CA/TX/WI would fall through the CASE and yield NULL."""
    nulls = layer.con.execute(
        "SELECT COUNT(*) FROM v_sales WHERE snap_active IS NULL"
    ).fetchone()[0]
    assert nulls == 0


# --- revenue semantics -----------------------------------------------------

def test_revenue_is_null_not_zero_when_price_missing(layer):
    """Unpriced rows must not silently contribute zero revenue to any sum."""
    bad = layer.con.execute(
        "SELECT COUNT(*) FROM v_sales WHERE sell_price IS NULL AND revenue IS NOT NULL"
    ).fetchone()[0]
    assert bad == 0


def test_revenue_equals_units_times_price(layer):
    bad = layer.con.execute("""
        SELECT COUNT(*) FROM v_sales
        WHERE sell_price IS NOT NULL
          AND ABS(revenue - sales * sell_price) > 1e-6
    """).fetchone()[0]
    assert bad == 0


# --- window-frame safety ---------------------------------------------------

def test_rolling_windows_do_not_exceed_their_frame(layer):
    """ROWS-based frames are only valid on contiguous daily series.

    7d rolling sum can never exceed the 28d rolling sum on the same row; a
    violation would mean the frames are misaligned.
    """
    roll = layer.run("rolling_store_demand")
    assert (roll["units_7d"] <= roll["units_28d"]).all()
    assert (roll["units_7d"] >= roll["units"]).all()
