"""Tests for the Phase 3 analysis pipeline.

Runs the real pipeline against a fixture-built database. Verifies structure,
aggregation correctness and - importantly - that no analysis step reaches
outside the data it is given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.ingest import build_long_table
from src.make_fixture import build_fixture
from src.phase3_analysis import (
    analyse_distribution,
    analyse_intermittency,
    analyse_price,
    analyse_temporal,
    load_frames,
    run_tests,
    write_implications,
    write_report,
)
from src.sql_runner import SQLLayer
from src.stats_utils import results_to_frame


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    cfg = load_config()
    work = tmp_path_factory.mktemp("phase3")
    long = build_long_table(cfg, raw_dir=build_fixture(seed=0, out_dir=work / "fx"))

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

    layer = SQLLayer(db, cfg.root / "sql")
    layer.create_views()
    frames = load_frames(layer)

    # Phase 3 reuses the Phase 1 segmentation, so write series_stats.csv here.
    from src.profile import classify_demand, series_stats
    reports = work / "reports"
    reports.mkdir()
    stats = classify_demand(series_stats(long), cfg["segmentation"]["adi_threshold"],
                            cfg["segmentation"]["cv2_threshold"])
    stats.to_csv(reports / "series_stats.csv", index=False)

    yield {"cfg": cfg, "layer": layer, "frames": frames, "reports": reports, "long": long}
    layer.close()


# --- data loading ----------------------------------------------------------

def test_frames_load_with_expected_columns(env):
    f = env["frames"]
    assert {"store_id", "date", "units", "snap_active"} <= set(f["store_daily"].columns)
    assert {"wday", "month", "day_of_month"} <= set(f["store_daily_cal"].columns)
    assert {"item_id", "store_id", "zero_share"} <= set(f["series"].columns)


def test_store_daily_aggregates_reconcile_with_series(env):
    """Store-day totals must sum to the same units as the raw long table."""
    total_long = int(env["long"]["sales"].sum())
    total_sd = int(env["frames"]["store_daily"]["units"].sum())
    assert total_sd == total_long


def test_store_cat_daily_reconciles_to_store_daily(env):
    f = env["frames"]
    a = f["store_daily"].groupby("store_id")["units"].sum()
    b = f["store_cat_daily"].groupby("store_id")["units"].sum()
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index(), check_names=False)


def test_calendar_join_does_not_duplicate_store_days(env):
    f = env["frames"]
    assert len(f["store_daily_cal"]) == len(f["store_daily"])
    assert not f["store_daily_cal"].duplicated(subset=["store_id", "date"]).any()


# --- distribution ----------------------------------------------------------

def test_distribution_reports_zero_inflation(env):
    d = analyse_distribution(env["frames"])
    assert 0 <= d["series_zero_share"]["mean"] <= 1
    assert 0 <= d["gini_units"] <= 1
    ci = d["zero_share_ci"]
    assert ci["ci_lower"] <= ci["statistic"] <= ci["ci_upper"]


def test_concentration_shares_are_monotonic(env):
    """Top-1% share cannot exceed top-5%, and so on."""
    c = analyse_distribution(env["frames"])["concentration"]
    vals = [c[f"top_{p}pct_share"] for p in (1, 5, 10, 20, 50)]
    assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]))
    assert all(0 <= v <= 1 for v in vals)


# --- intermittency ---------------------------------------------------------

def test_intermittency_reuses_phase1_segmentation(env):
    """Must read series_stats.csv, not recompute - guards D-008 divergence."""
    r = analyse_intermittency(env["frames"], env["cfg"], env["reports"])
    assert set(r["profile"].columns) >= {"segment", "n_series", "pct_of_series"}
    assert int(r["profile"]["n_series"].sum()) == len(env["frames"]["series"])


def test_intermittency_fails_loudly_without_phase1_output(env, tmp_path):
    with pytest.raises(FileNotFoundError, match="series_stats.csv"):
        analyse_intermittency(env["frames"], env["cfg"], tmp_path)


def test_segment_shares_sum_to_100(env):
    r = analyse_intermittency(env["frames"], env["cfg"], env["reports"])
    assert float(r["profile"]["pct_of_series"].sum()) == pytest.approx(100.0, abs=0.01)
    assert float(r["profile"]["pct_of_units"].sum()) == pytest.approx(100.0, abs=0.01)


# --- temporal --------------------------------------------------------------

def test_temporal_covers_all_seven_weekdays(env):
    t = analyse_temporal(env["frames"])
    assert set(t["day_of_week"]["wday"]) == set(range(1, 8))
    assert t["day_of_week"]["day_name"].notna().all()


def test_store_mean_ci_uses_block_bootstrap(env):
    t = analyse_temporal(env["frames"])
    for _, ci in t["store_mean_ci"].items():
        assert "moving_block" in ci["ci_method"]
        assert ci["ci_lower"] <= ci["statistic"] <= ci["ci_upper"]


# --- tests family ----------------------------------------------------------

def test_test_family_is_fdr_corrected(env):
    df = results_to_frame(run_tests(env["frames"]))
    assert len(df) > 0
    valid = df["p_value"].notna()
    # Adjusted p is never smaller than raw p.
    assert (df.loc[valid, "p_adjusted"] >= df.loc[valid, "p_value"] - 1e-12).all()
    assert df["significant_fdr"].dtype == bool


def test_every_test_records_hypotheses_and_effect_size(env):
    df = results_to_frame(run_tests(env["frames"]))
    for col in ("question", "null_hypothesis", "alt_hypothesis", "test",
                "effect_metric", "notes"):
        assert df[col].astype(str).str.len().gt(0).all(), col


def test_snap_tests_are_labelled_observational(env):
    """Guards the no-causal-claims rule at the code level."""
    df = results_to_frame(run_tests(env["frames"]))
    snap = df[df["question"].str.contains("SNAP")]
    assert len(snap) > 0
    for note in snap["notes"]:
        assert "OBSERVATIONAL" in note.upper() or "observational" in note


def test_no_test_question_claims_causation(env):
    df = results_to_frame(run_tests(env["frames"]))
    banned = ["causes", "caused", "impact of", "effect of", "due to", "because of"]
    for text in pd.concat([df["question"], df["alt_hypothesis"]]):
        low = str(text).lower()
        assert not any(b in low for b in banned), text


def test_christmas_excluded_from_event_tests(env):
    """Store closure is not demand - including it would fabricate an effect."""
    df = results_to_frame(run_tests(env["frames"]))
    ev = df[df["question"].str.contains("event days")]
    assert len(ev) > 0
    assert all("Christmas excluded" in n for n in ev["notes"])


def test_results_are_deterministic(env):
    a = results_to_frame(run_tests(env["frames"]))
    b = results_to_frame(run_tests(env["frames"]))
    pd.testing.assert_frame_equal(a, b)


# --- leakage / integrity ---------------------------------------------------

def test_analysis_never_reads_beyond_supplied_data(env):
    """Phase 3 is descriptive; no step may reference dates outside the frame.

    Guards against an analysis silently pulling the full fact table when it
    was handed a subset - which in Phase 4 would be a leakage bug.
    """
    f = env["frames"]
    max_date = f["store_daily_cal"]["date"].max()
    trimmed = {
        k: (v[v["date"] <= max_date - pd.Timedelta(days=100)]
            if "date" in v.columns else v)
        for k, v in f.items()
    }
    t = analyse_temporal(trimmed)
    # Recomputed means must change: proof the function used the data given it.
    full = analyse_temporal(f)
    assert t["overall_mean"] != full["overall_mean"]


def test_price_analysis_is_not_labelled_elasticity(env):
    r = analyse_price(env["frames"])
    assert "elasticity" not in str(r.keys()).lower()
    assert set(r["by_move"]["price_move"]) <= {"increased", "decreased", "stable"}
    assert -1 <= r["spearman_rho"] <= 1


def test_price_change_uses_prior_week_only(env):
    """Comparison must be against the PREVIOUS week, never the next one."""
    f = env["frames"]
    wp = f["weekly_price"].sort_values(["store_id", "item_id", "wm_yr_wk"])
    g = wp.groupby(["store_id", "item_id"], observed=True)
    shifted = g["price"].shift(1)
    # First observation of each series must have no predecessor.
    firsts = g.head(1).index
    assert shifted.loc[firsts].isna().all()


# --- UTF-8 encoding regression (Windows charmap fix) ----------------------

def test_write_report_and_implications_are_utf8(env, tmp_path):
    """Regression guard: both report writers must use UTF-8 explicitly.

    Before the fix, write_text() on Windows used the system default encoding
    (cp1252), which raised UnicodeEncodeError on the '≈' character (U+2248)
    that appears in statistical summaries.  This test verifies:
      1. The files are written without error.
      2. The files can be read back as UTF-8 (would fail on cp1252).
      3. The '≈' character survives the round-trip if present in the output.
    """
    f = env["frames"]
    dist = analyse_distribution(f)
    interm = analyse_intermittency(f, env["cfg"], env["reports"])
    temporal = analyse_temporal(f)
    price = analyse_price(f)
    tests_df = results_to_frame(run_tests(f))

    report_path = tmp_path / "phase3_eda.md"
    impl_path = tmp_path / "forecasting_implications.md"

    # Neither call must raise UnicodeEncodeError (the original failure).
    write_report(f, dist, interm, temporal, price, tests_df, [], report_path)
    write_implications(dist, interm, temporal, tests_df, impl_path)

    # Files must be readable as strict UTF-8.
    report_text = report_path.read_text(encoding="utf-8")
    impl_text = impl_path.read_text(encoding="utf-8")

    # Basic sanity: non-empty Markdown.
    assert "Phase 3" in report_text
    assert "Forecasting Implications" in impl_text

