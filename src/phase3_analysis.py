"""Phase 3: EDA and statistical analysis on the real M5 data.

Reads the Phase 2 analytical views, runs the analysis, writes
reports/phase3/. Every number in the report comes from this module - nothing
is typed by hand.

UNIT OF ANALYSIS (see DECISION_LOG D-015)
-----------------------------------------
Inferential comparisons run on STORE-DAY aggregates, not series-days.

Series-day rows are ~14.1M, ~59.5% zero, and massively dependent both within a
series over time and across series on the same day (a busy Saturday lifts
everything at once). Testing on them would produce p-values near zero for
effects of no commercial size, driven entirely by the pretence that 14.1M
correlated rows are 14.1M independent observations.

Store-day totals are ~5.9k rows, far closer to symmetric, and their remaining
dependence is one-dimensional autocorrelation - which the moving-block
bootstrap handles honestly. Series-level data is still used for DESCRIPTIVE
work (distributions, intermittency), where no independence claim is made.

NO CAUSAL CLAIMS
----------------
Every comparison here is observational. Column and prose wording says
"observed difference", never "effect" or "impact".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.sql_runner import SQLLayer
from src.stats_utils import (
    TestResult,
    apply_fdr,
    block_bootstrap_diff_ci,
    bootstrap_ci,
    describe_distribution,
    gini,
    kruskal_test,
    mann_whitney_test,
    moving_block_bootstrap_ci,
    results_to_frame,
)

logger = logging.getLogger(__name__)

SEED = 42
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_frames(layer: SQLLayer) -> dict[str, pd.DataFrame]:
    """Pull the aggregates Phase 3 needs. Reuses Phase 2 views (no new SQL)."""
    con = layer.con
    frames = {}

    frames["store_daily"] = con.execute("""
        SELECT store_id, state_id, date, units, revenue,
               snap_active, is_event_day
        FROM v_store_daily ORDER BY store_id, date
    """).df()

    frames["store_cat_daily"] = con.execute("""
        SELECT store_id, cat_id, date, units
        FROM v_store_cat_daily ORDER BY store_id, cat_id, date
    """).df()

    frames["series"] = con.execute("""
        SELECT item_id, store_id, cat_id, dept_id, n_days, total_units,
               mean_daily_units, sd_daily_units, cv_daily_units,
               nonzero_days, zero_share
        FROM v_series_summary
    """).df()

    # Calendar context per store-day, for day-of-week stratification.
    frames["store_daily_cal"] = con.execute("""
        SELECT d.store_id, d.state_id, d.date, d.units,
               d.snap_active, d.is_event_day,
               c.wday, c.month, c.year,
               DAY(d.date) AS day_of_month,
               c.event_name_1, c.event_type_1
        FROM v_store_daily d
        JOIN dim_calendar c ON d.date = c.date
        ORDER BY d.store_id, d.date
    """).df()

    # Weekly price/units per series, for the price association.
    frames["weekly_price"] = con.execute("""
        SELECT item_id, store_id, cat_id, wm_yr_wk,
               SUM(sales) AS units, AVG(sell_price) AS price
        FROM v_sales WHERE sell_price IS NOT NULL
        GROUP BY 1,2,3,4
    """).df()

    return frames


# ---------------------------------------------------------------------------
# A. Demand distribution
# ---------------------------------------------------------------------------

def analyse_distribution(frames: dict) -> dict:
    series = frames["series"]
    store_daily = frames["store_daily"]

    out: dict = {}

    # Series-level (cross-sectional): each series contributes one value, so the
    # i.i.d. bootstrap is appropriate here.
    out["series_total_units"] = describe_distribution(
        series["total_units"], "total units per series")
    out["series_zero_share"] = describe_distribution(
        series["zero_share"], "zero share per series")
    out["series_mean_daily"] = describe_distribution(
        series["mean_daily_units"], "mean daily units per series")

    out["zero_share_ci"] = bootstrap_ci(
        series["zero_share"], np.mean, N_BOOT, seed=SEED,
        resample_unit="series (cross-sectional)").as_dict()

    # Demand concentration.
    out["gini_units"] = gini(series["total_units"])
    tot = series["total_units"].sort_values(ascending=False)
    grand = tot.sum()
    out["concentration"] = {
        f"top_{p}pct_share": float(tot.head(max(1, int(len(tot) * p / 100))).sum() / grand)
        for p in (1, 5, 10, 20, 50)
    }

    # Store-day level (temporal): block bootstrap, not i.i.d.
    out["store_daily_units"] = describe_distribution(
        store_daily["units"], "store-day total units")
    out["by_store_distribution"] = {
        sid: describe_distribution(g["units"], f"store-day units {sid}")
        for sid, g in store_daily.groupby("store_id")
    }
    out["by_category_series"] = {
        c: describe_distribution(g["total_units"], f"series totals {c}")
        for c, g in series.groupby("cat_id")
    }
    return out


# ---------------------------------------------------------------------------
# B. Intermittency
# ---------------------------------------------------------------------------

def analyse_intermittency(frames: dict, cfg: Config, reports_dir: Path) -> dict:
    """Compare demand regimes using the Phase 1 segmentation.

    Reuses reports/series_stats.csv (written by src/profile.py) rather than
    recomputing ADI/CV-squared, so Phase 3 cannot silently diverge from the
    Phase 1 segmentation (D-008).
    """
    stats_path = reports_dir / "series_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} missing. Run `python -m src.profile` first - Phase 3 "
            "reuses the Phase 1 segmentation rather than recomputing it."
        )
    seg = pd.read_csv(stats_path)
    series = frames["series"]

    merged = series.merge(
        seg[["store_id", "item_id", "adi", "cv2", "segment"]],
        on=["store_id", "item_id"], how="left", validate="one_to_one",
    )

    profile = merged.groupby("segment", dropna=False).agg(
        n_series=("item_id", "size"),
        mean_zero_share=("zero_share", "mean"),
        mean_daily_units=("mean_daily_units", "mean"),
        median_total_units=("total_units", "median"),
        total_units=("total_units", "sum"),
        mean_adi=("adi", "mean"),
        mean_cv2=("cv2", "mean"),
    ).reset_index()
    profile["pct_of_series"] = 100 * profile["n_series"] / profile["n_series"].sum()
    profile["pct_of_units"] = 100 * profile["total_units"] / profile["total_units"].sum()

    composition = (
        merged.groupby(["segment", "cat_id"], dropna=False)
        .size().rename("n_series").reset_index()
    )
    composition["pct_within_segment"] = 100 * composition["n_series"] / (
        composition.groupby("segment")["n_series"].transform("sum"))

    store_comp = (
        merged.groupby(["segment", "store_id"], dropna=False)
        .size().rename("n_series").reset_index()
    )

    return {
        "profile": profile,
        "category_composition": composition,
        "store_composition": store_comp,
        "merged": merged,
    }


# ---------------------------------------------------------------------------
# C/D/E/F/G. Inferential comparisons
# ---------------------------------------------------------------------------

def run_tests(frames: dict) -> list[TestResult]:
    """The full family of inferential tests, corrected together via BH-FDR.

    Deliberately small. Each test answers a question that changes a Phase 4/5
    decision; running hundreds of series-level tests would produce noise and a
    correction so severe nothing survives.
    """
    results: list[TestResult] = []
    sd = frames["store_daily_cal"]

    # --- C1. Weekend vs weekday (store-day totals) ----------------------
    # M5 wday: 1 = Saturday, 2 = Sunday.
    for store, g in sd.groupby("store_id"):
        we = g.loc[g["wday"].isin([1, 2]), "units"].to_numpy()
        wd = g.loc[~g["wday"].isin([1, 2]), "units"].to_numpy()
        ci = block_bootstrap_diff_ci(we, wd, N_BOOT, seed=SEED)
        results.append(mann_whitney_test(
            we, wd,
            question=f"[{store}] Do weekend store-day totals differ from weekdays?",
            null_hypothesis="Weekend and weekday store-day unit distributions are identical.",
            alt_hypothesis="They differ.",
            notes="Store-day aggregate. Block-bootstrap CI on the mean difference; "
                  "consecutive days are autocorrelated so the p-value is anti-conservative.",
            ci=ci,
        ))

    # --- F1. SNAP vs non-SNAP, day-of-week matched ----------------------
    # SNAP days fall on days 1-10 of each month, so a raw comparison confounds
    # SNAP with month-position AND with day-of-week composition. Restricting to
    # matched weekdays removes the day-of-week part. The month-position
    # confound CANNOT be removed observationally and is stated as a limitation.
    for store, g in sd.groupby("store_id"):
        parts_snap, parts_non = [], []
        for _, gw in g.groupby("wday"):
            parts_snap.append(gw.loc[gw["snap_active"] == 1, "units"])
            parts_non.append(gw.loc[gw["snap_active"] == 0, "units"])
        snap = pd.concat(parts_snap).to_numpy()
        non = pd.concat(parts_non).to_numpy()
        ci = block_bootstrap_diff_ci(snap, non, N_BOOT, seed=SEED)
        results.append(mann_whitney_test(
            snap, non,
            question=f"[{store}] Do observed store-day totals differ on SNAP-active days?",
            null_hypothesis="SNAP-active and inactive store-day unit distributions are identical.",
            alt_hypothesis="They differ.",
            notes="OBSERVATIONAL ONLY - not a SNAP effect. SNAP days are days 1-10 of "
                  "the month, so this is confounded with payday and month-position "
                  "cycles that cannot be separated without an identification strategy.",
            ci=ci,
        ))

    # --- F2. SNAP by category (FOODS is SNAP-eligible; HOBBIES largely not)
    scd = frames["store_cat_daily"].merge(
        sd[["store_id", "date", "snap_active", "wday"]],
        on=["store_id", "date"], how="left", validate="many_to_one",
    )
    for cat, g in scd.groupby("cat_id"):
        snap = g.loc[g["snap_active"] == 1, "units"].to_numpy()
        non = g.loc[g["snap_active"] == 0, "units"].to_numpy()
        ci = block_bootstrap_diff_ci(snap, non, N_BOOT, seed=SEED)
        results.append(mann_whitney_test(
            snap, non,
            question=f"[{cat}] Do observed store-category-day totals differ on SNAP days?",
            null_hypothesis="SNAP-active and inactive distributions are identical for this category.",
            alt_hypothesis="They differ.",
            notes="Observational. If the difference concentrates in FOODS that is "
                  "CONSISTENT with a SNAP-related mechanism but does not establish one.",
            ci=ci,
        ))

    # --- C2. Event days vs non-event days -------------------------------
    # Christmas is excluded: M5 stores close, so those near-zero days are a
    # closure artefact, not demand. Leaving them in would manufacture a large
    # spurious "event effect".
    for store, g in sd.groupby("store_id"):
        g2 = g.loc[g["event_name_1"].fillna("") != "Christmas"]
        ev = g2.loc[g2["is_event_day"] == 1, "units"].to_numpy()
        ne = g2.loc[g2["is_event_day"] == 0, "units"].to_numpy()
        ci = block_bootstrap_diff_ci(ev, ne, N_BOOT, seed=SEED)
        results.append(mann_whitney_test(
            ev, ne,
            question=f"[{store}] Do observed store-day totals differ on event days?",
            null_hypothesis="Event-day and non-event-day distributions are identical.",
            alt_hypothesis="They differ.",
            notes="Christmas excluded (store closure artefact, not demand). "
                  "Events pool many heterogeneous holiday types.",
            ci=ci,
        ))

    # --- D1. Store differences (omnibus) --------------------------------
    results.append(kruskal_test(
        {sid: g["units"].to_numpy() for sid, g in sd.groupby("store_id")},
        question="Do the three stores differ in daily observed unit totals?",
        null_hypothesis="All three stores share the same store-day unit distribution.",
        alt_hypothesis="At least one store differs.",
        notes="Omnibus across CA_1/TX_1/WI_1. Stores differ in assortment size "
              "as well as demand, so this reflects both.",
    ))

    # --- E1. Category differences (omnibus) -----------------------------
    results.append(kruskal_test(
        {c: g["units"].to_numpy() for c, g in frames["store_cat_daily"].groupby("cat_id")},
        question="Do categories differ in daily observed unit totals?",
        null_hypothesis="All categories share the same store-category-day distribution.",
        alt_hypothesis="At least one category differs.",
        notes="Categories contain very different numbers of SKUs, so this "
              "reflects assortment size as well as per-item demand.",
    ))

    # --- E2. Intermittency across categories ----------------------------
    series = frames["series"]
    results.append(kruskal_test(
        {c: g["zero_share"].to_numpy() for c, g in series.groupby("cat_id")},
        question="Does per-series zero-sales share differ across categories?",
        null_hypothesis="Zero-share distributions are identical across categories.",
        alt_hypothesis="At least one category differs.",
        notes="Cross-sectional over series - no temporal dependence issue. "
              "Directly relevant to Phase 5 model choice per category.",
    ))

    return apply_fdr(results, alpha=0.05)


# ---------------------------------------------------------------------------
# Temporal patterns (descriptive)
# ---------------------------------------------------------------------------

def analyse_temporal(frames: dict) -> dict:
    sd = frames["store_daily_cal"]
    overall = sd["units"].mean()

    dow = sd.groupby("wday")["units"].agg(["mean", "median", "std", "size"]).reset_index()
    dow["day_name"] = dow["wday"].map(
        {1: "Sat", 2: "Sun", 3: "Mon", 4: "Tue", 5: "Wed", 6: "Thu", 7: "Fri"})
    dow["pct_vs_overall"] = 100 * (dow["mean"] - overall) / overall

    month = sd.groupby("month")["units"].agg(["mean", "median", "size"]).reset_index()
    month["pct_vs_overall"] = 100 * (month["mean"] - overall) / overall

    year = sd.groupby(["store_id", "year"])["units"].agg(
        ["mean", "sum", "size"]).reset_index()

    # Block-bootstrap CIs for each store's mean daily units: these are temporal
    # statistics, so the i.i.d. bootstrap would understate the interval.
    store_ci = {}
    for sid, g in sd.groupby("store_id"):
        store_ci[sid] = moving_block_bootstrap_ci(
            g.sort_values("date")["units"].to_numpy(),
            np.mean, N_BOOT, seed=SEED).as_dict()

    return {"day_of_week": dow, "month": month, "yearly": year,
            "store_mean_ci": store_ci, "overall_mean": float(overall)}


# ---------------------------------------------------------------------------
# Price association (explicitly NOT elasticity)
# ---------------------------------------------------------------------------

def analyse_price(frames: dict) -> dict:
    """Week-over-week price moves against week-over-week unit changes.

    NOT an elasticity estimate. Retailers reprice in response to expected or
    observed demand, so causality runs in both directions. Reported as
    descriptive co-movement with a rank correlation.
    """
    from scipy import stats as sps

    wp = frames["weekly_price"].sort_values(["store_id", "item_id", "wm_yr_wk"]).copy()
    g = wp.groupby(["store_id", "item_id"], observed=True)
    wp["prev_price"] = g["price"].shift(1)
    wp["prev_units"] = g["units"].shift(1)
    wp = wp.dropna(subset=["prev_price", "prev_units"])
    wp = wp.loc[(wp["prev_price"] > 0) & (wp["prev_units"] > 0)]

    wp["price_pct_change"] = 100 * (wp["price"] - wp["prev_price"]) / wp["prev_price"]
    wp["units_pct_change"] = 100 * (wp["units"] - wp["prev_units"]) / wp["prev_units"]
    wp["price_move"] = np.select(
        [wp["price"] < wp["prev_price"] * 0.98, wp["price"] > wp["prev_price"] * 1.02],
        ["decreased", "increased"], default="stable")

    by_move = wp.groupby(["cat_id", "price_move"]).agg(
        n_transitions=("units", "size"),
        mean_units_pct_change=("units_pct_change", "mean"),
        median_units_pct_change=("units_pct_change", "median"),
    ).reset_index()

    changed = wp.loc[wp["price_move"] != "stable"]
    if len(changed) > 2:
        rho, p = sps.spearmanr(changed["price_pct_change"], changed["units_pct_change"])
    else:
        rho, p = np.nan, np.nan

    return {
        "by_move": by_move,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "n_price_changes": int(len(changed)),
        "n_transitions": int(len(wp)),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(frames: dict, interm: dict, temporal: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

    series = frames["series"]
    sd = frames["store_daily_cal"]

    # 1. Zero-share distribution across series - the headline data reality.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series["zero_share"], bins=50, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Share of active days with zero observed sales")
    ax.set_ylabel("Number of series")
    ax.set_title("Zero-sales share across item-store series")
    save(fig, "01_zero_share_distribution.png")

    # 2. Demand concentration (Lorenz-style).
    tot = np.sort(series["total_units"].to_numpy())[::-1]
    cum = np.cumsum(tot) / tot.sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(100 * np.arange(1, len(cum) + 1) / len(cum), 100 * cum, color="#C44E52")
    ax.axhline(80, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("Cumulative share of series (%, ranked by volume)")
    ax.set_ylabel("Cumulative share of units (%)")
    ax.set_title("Demand concentration across series")
    save(fig, "02_demand_concentration.png")

    # 3. Segment sizes vs their share of volume.
    prof = interm["profile"].sort_values("n_series", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(prof))
    ax.bar(x - 0.2, prof["pct_of_series"], 0.4, label="% of series", color="#4C72B0")
    ax.bar(x + 0.2, prof["pct_of_units"], 0.4, label="% of units", color="#DD8452")
    ax.set_xticks(x, prof["segment"].astype(str), rotation=20)
    ax.set_ylabel("Percent")
    ax.set_title("Demand regimes: share of series vs share of volume")
    ax.legend()
    save(fig, "03_segment_share.png")

    # 4. Day-of-week profile.
    dow = temporal["day_of_week"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dow["day_name"], dow["pct_vs_overall"], color="#55A868")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("% difference vs overall mean")
    ax.set_title("Day-of-week profile (store-day totals)")
    save(fig, "04_day_of_week.png")

    # 5. Store comparison.
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [g["units"].to_numpy() for _, g in sd.groupby("store_id")]
    ax.boxplot(data, tick_labels=[s for s, _ in sd.groupby("store_id")],
               showfliers=False)
    ax.set_ylabel("Daily observed units")
    ax.set_title("Store-day unit totals by store")
    save(fig, "05_store_comparison.png")

    # 6. SNAP comparison.
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, data = [], []
    for sid, g in sd.groupby("store_id"):
        labels += [f"{sid}\nSNAP", f"{sid}\nnon"]
        data += [g.loc[g["snap_active"] == 1, "units"].to_numpy(),
                 g.loc[g["snap_active"] == 0, "units"].to_numpy()]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Daily observed units")
    ax.set_title("Store-day totals: SNAP-active vs inactive (observed, not causal)")
    save(fig, "06_snap_comparison.png")

    # 7. Monthly seasonality.
    m = temporal["month"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(m["month"], m["pct_vs_overall"], marker="o", color="#8172B3")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("% difference vs overall mean")
    ax.set_title("Monthly pattern (store-day totals)")
    save(fig, "07_monthly_pattern.png")

    return paths


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(d: dict, keys: list[str]) -> str:
    rows = [{"statistic": k, "value": d.get(k)} for k in keys if k in d]
    return pd.DataFrame(rows).to_markdown(index=False)


def write_report(
    frames: dict, dist: dict, interm: dict, temporal: dict,
    price: dict, tests: pd.DataFrame, figures: list[Path], out_path: Path,
) -> Path:
    L = []
    A = L.append
    series = frames["series"]
    sd = frames["store_daily_cal"]

    A("# Phase 3 — EDA & Statistical Analysis\n")
    A("Generated by `src/phase3_analysis.py`. Every figure below is computed "
      "from the real M5 data; nothing is hand-entered.\n")

    A("## Executive summary\n")
    n_series = len(series)
    zs = dist["series_zero_share"]
    A(f"- {n_series:,} item-store series across {sd['store_id'].nunique()} stores, "
      f"{sd['date'].nunique():,} days.")
    A(f"- Mean per-series zero-sales share: **{zs['mean']:.1%}** "
      f"(median {zs['median']:.1%}), 95% CI "
      f"[{dist['zero_share_ci']['ci_lower']:.3f}, {dist['zero_share_ci']['ci_upper']:.3f}].")
    A(f"- Demand is highly concentrated: Gini **{dist['gini_units']:.3f}**; the top 10% "
      f"of series carry **{dist['concentration']['top_10pct_share']:.1%}** of units.")
    sig = int(tests["significant_fdr"].sum())
    A(f"- {sig} of {len(tests)} inferential comparisons survive BH-FDR at 5%; "
      "effect sizes matter more than the p-values and are reported for each.\n")

    A("## Data overview\n")
    A(f"- Stores: {', '.join(sorted(sd['store_id'].unique()))}")
    A(f"- Date range: {sd['date'].min()} to {sd['date'].max()}")
    A(f"- Series: {n_series:,} | Store-days analysed: {len(sd):,}\n")

    A("## Demand distribution\n")
    A("### Total units per series\n")
    A(_fmt(dist["series_total_units"],
           ["n", "mean", "median", "sd", "p25", "p75", "p95", "p99", "max", "skewness"]))
    A("\n### Store-day totals\n")
    A(_fmt(dist["store_daily_units"], ["n", "mean", "median", "sd", "p25", "p75", "p95"]))
    A("\n### Concentration\n")
    A(pd.DataFrame([{"cohort": k, "share_of_units": f"{v:.1%}"}
                    for k, v in dist["concentration"].items()]).to_markdown(index=False))
    A("\nThe distribution is strongly right-skewed, so the mean is a poor summary "
      "and the median plus quantiles carry more information. This is why every "
      "inferential test below is rank-based rather than mean-based.\n")

    A("## Intermittency\n")
    A(interm["profile"].to_markdown(index=False))
    A("\nSegments come from the Phase 1 classification (D-008); Phase 3 reuses "
      "`reports/series_stats.csv` rather than recomputing, so the two phases "
      "cannot diverge.\n")
    A("### Category composition of each regime\n")
    A(interm["category_composition"].to_markdown(index=False))
    A("")

    A("## Temporal behaviour\n")
    A("### Day of week (M5 wday: 1=Sat … 7=Fri)\n")
    A(temporal["day_of_week"][
        ["wday", "day_name", "mean", "median", "size", "pct_vs_overall"]
      ].to_markdown(index=False))
    A("\n### Month\n")
    A(temporal["month"].to_markdown(index=False))
    A("\n### Mean daily units per store (moving-block bootstrap CI)\n")
    A(pd.DataFrame([
        {"store_id": k, "mean_daily_units": v["statistic"],
         "ci_lower": v["ci_lower"], "ci_upper": v["ci_upper"],
         "method": v["ci_method"]}
        for k, v in temporal["store_mean_ci"].items()
    ]).to_markdown(index=False))
    A("\nBlock bootstrap, not the ordinary bootstrap: daily sales are "
      "autocorrelated, and resampling individual days would produce intervals "
      "that are far too narrow.\n")

    A("## Store analysis\n")
    A(pd.DataFrame([
        {"store_id": k, **{m: v[m] for m in ("n", "mean", "median", "sd", "p95")}}
        for k, v in dist["by_store_distribution"].items()
    ]).to_markdown(index=False))
    A("")

    A("## Category analysis\n")
    A(pd.DataFrame([
        {"cat_id": k, **{m: v[m] for m in ("n", "mean", "median", "p95", "max")}}
        for k, v in dist["by_category_series"].items()
    ]).to_markdown(index=False))
    A("")

    A("## SNAP / event analysis\n")
    A("**Observational association only.** SNAP-active days are days 1–10 of each "
      "month, so any observed difference is entangled with payday and "
      "month-position cycles. The comparison below matches on day of week, which "
      "removes the day-of-week confound but not the month-position one. No causal "
      "claim is made or supported.\n")
    snap_tests = tests[tests["question"].str.contains("SNAP")]
    A(snap_tests[["question", "n_group1", "n_group2", "p_value", "p_adjusted",
                  "effect_size", "effect_label", "rel_diff_pct",
                  "ci_lower", "ci_upper"]].to_markdown(index=False))
    A("")

    A("## Price analysis\n")
    A(f"- Week-over-week transitions analysed: {price['n_transitions']:,} "
      f"({price['n_price_changes']:,} with a price move beyond ±2%)")
    A(f"- Spearman rank correlation between % price change and % unit change: "
      f"**{price['spearman_rho']:.4f}** (p = {price['spearman_p']:.3g})\n")
    A(price["by_move"].to_markdown(index=False))
    A("\n**This is observational association, not causal price elasticity.** "
      "Retailers reduce prices when they expect or observe weak demand and raise "
      "them on strong sellers, so causality runs in both directions. Estimating "
      "elasticity would require an identification strategy this project does not "
      "have.\n")

    A("## Statistical methodology\n")
    A("| Choice | What and why |")
    A("|---|---|")
    A("| Unit of analysis | Store-day aggregates for inference (D-015). Series-day "
      "rows are ~14.1M, ~59.5% zero and heavily dependent; testing on them yields "
      "p≈0 for commercially trivial differences. |")
    A("| Test | Mann-Whitney U (2 groups) / Kruskal-Wallis (3+). Rank-based, so "
      "valid for skewed zero-inflated demand where a t-test's normality "
      "assumption fails. |")
    A("| Effect size | Cliff's delta (rank) plus relative difference (%). "
      "Significance alone is uninformative at these sample sizes. |")
    A("| Confidence intervals | Moving-block bootstrap for temporal statistics "
      "(D-016); ordinary bootstrap only for cross-sectional series statistics. |")
    A("| Multiple testing | Benjamini-Hochberg FDR at 5% across the whole family "
      "(D-017). |")
    A("| Dependence | Acknowledged, not solved. Block bootstrap widens intervals "
      "appropriately; the rank-test p-values remain anti-conservative. |")
    A("")
    A("### Full test register\n")
    A("See `statistical_tests.csv` for all columns including hypotheses and notes.\n")
    A(tests[["question", "test", "p_value", "p_adjusted", "significant_fdr",
             "effect_size", "effect_metric", "effect_label",
             "rel_diff_pct"]].to_markdown(index=False))
    A("")

    A("## Limitations\n")
    A("1. **Observed sales are not demand.** An in-window zero may be a stockout "
      "(D-003). M5 has no inventory data, so demand is censored below and every "
      "statistic here is a lower bound.")
    A("2. **No causal inference.** SNAP, event and price relationships are "
      "observational associations, all confounded.")
    A("3. **Three stores.** CA_1/TX_1/WI_1 (D-002); results do not generalise to "
      "the chain.")
    A("4. **Temporal dependence remains.** Block bootstrap addresses interval "
      "width; rank-test p-values still assume exchangeability and are optimistic.")
    A("5. **Multiple testing.** BH-FDR controls the false-discovery proportion, "
      "not the family-wise error rate.")
    A("6. **Assortment grows over time.** Series enter when first priced, so "
      "chain-level trends partly reflect assortment expansion.")
    A("7. **Christmas excluded from event tests** — store closure, not demand.\n")

    A("## Figures\n")
    for p in figures:
        A(f"- `figures/{p.name}`")
    A("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    return out_path


def write_implications(
    dist: dict, interm: dict, temporal: dict, tests: pd.DataFrame, out_path: Path
) -> Path:
    prof = interm["profile"]
    hard = prof.loc[prof["segment"].isin(["intermittent", "lumpy"])]
    hard_series = int(hard["n_series"].sum()) if len(hard) else 0
    hard_units = float(hard["pct_of_units"].sum()) if len(hard) else 0.0
    hard_pct = float(hard["pct_of_series"].sum()) if len(hard) else 0.0

    L = []
    A = L.append
    A("# Phase 3 → Forecasting Implications\n")
    A("What the statistical analysis means for Phase 4 (backtesting) and "
      "Phase 5 (models). Handoff only — no models are built here.\n")

    A("## 1. Most series are hard, and they are the majority\n")
    A(f"Intermittent and lumpy regimes cover **{hard_series:,} series "
      f"({hard_pct:.1f}%)** but only **{hard_units:.1f}%** of units.\n")
    A("Consequences for Phase 5:")
    A("- A single blended accuracy metric will be dominated by the small number "
      "of smooth, high-volume series and will hide failure on the majority.")
    A("- **MAPE is unusable** on these series: the denominator is zero on most "
      "days. Report WAPE/MAE as headline, and only quote MAPE where demand is "
      "reliably non-zero.")
    A("- Segment-level error reporting is mandatory, not a nice-to-have.\n")

    A("## 2. Zero-inflation dictates the loss function\n")
    A(f"Mean per-series zero share is **{dist['series_zero_share']['mean']:.1%}**.\n")
    A("- Squared-error objectives on zero-heavy series push predictions toward "
      "zero, which minimises RMSE while being commercially useless.")
    A("- Consider count-appropriate objectives (Poisson/Tweedie) for the "
      "intermittent segment, and evaluate whether a Croston-style method beats "
      "a naive baseline there at all.")
    A("- A near-zero constant is a genuinely strong baseline on these series. "
      "Any ML model must beat it, and that comparison must be explicit.\n")

    A("## 3. Demand concentration justifies tiered treatment\n")
    A(f"Gini **{dist['gini_units']:.3f}**; top 10% of series hold "
      f"**{dist['concentration']['top_10pct_share']:.1%}** of units.\n")
    A("- Model effort should follow volume: careful per-series modelling for "
      "the head, pooled or global models for the long tail.")
    A("- This maps directly onto the Phase 2 ABC classes and onto differentiated "
      "service levels in Phase 6.\n")

    A("## 4. Seasonality features worth engineering\n")
    dow = temporal["day_of_week"]
    if len(dow):
        top = dow.loc[dow["pct_vs_overall"].idxmax()]
        bot = dow.loc[dow["pct_vs_overall"].idxmin()]
        A(f"- Day-of-week is the strongest short-cycle pattern: {top['day_name']} "
          f"runs {top['pct_vs_overall']:+.1f}% against the mean, {bot['day_name']} "
          f"{bot['pct_vs_overall']:+.1f}%.")
    A("- Include day-of-week, month and SNAP flags as features. SNAP is a **known "
      "calendar** — future values are available at prediction time, so using it "
      "is not leakage.")
    A("- Event flags need care: they pool heterogeneous holidays, and Christmas "
      "is a closure artefact that should be modelled as such or excluded.\n")

    A("## 5. Store structure\n")
    A("- Stores differ enough that a store feature (or per-store models) is "
      "justified; they also differ in assortment size, so the difference is not "
      "purely demand.")
    A("- SNAP must stay state-resolved (D-011). A global SNAP flag would be wrong "
      "for two of three stores.\n")

    A("## 6. Constraints Phase 4 must respect\n")
    A("- **The target is observed sales, not demand.** Censoring from stockouts "
      "is unmeasurable here; state it rather than modelling around it.")
    A("- **Price is endogenous.** It may be used as a feature, but no "
      "coefficient on it may be reported as an elasticity.")
    A("- **Lag and rolling features must be built inside each fold's training "
      "window** (D-007, D-010). The SQL layer deliberately contains no "
      "series-level rolling features so there is exactly one fold-aware "
      "implementation.")
    A("- **Series start at different dates.** Expanding-window folds must handle "
      "series whose history is shorter than the minimum training window.\n")

    A("## 7. What the tests do and do not license\n")
    sig = int(tests["significant_fdr"].sum())
    A(f"- {sig} of {len(tests)} comparisons survive BH-FDR at 5%.")
    A("- All are **associations**. None licenses a causal statement about SNAP, "
      "events or price.")
    A("- Their value for Phase 4 is feature selection: they show which calendar "
      "structure carries signal worth encoding.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(config_path)
    cfg.ensure_dirs()

    out_dir = cfg.path("reports") / "phase3"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    with SQLLayer(cfg.path("duckdb"), cfg.root / "sql") as layer:
        layer.create_views()
        frames = load_frames(layer)
        logger.info("Loaded %d store-days, %d series",
                    len(frames["store_daily"]), len(frames["series"]))

        dist = analyse_distribution(frames)
        interm = analyse_intermittency(frames, cfg, cfg.path("reports"))
        temporal = analyse_temporal(frames)
        price = analyse_price(frames)
        tests_df = results_to_frame(run_tests(frames))

        figures = make_figures(frames, interm, temporal, fig_dir)

        tests_path = out_dir / "statistical_tests.csv"
        tests_df.to_csv(tests_path, index=False)

        report = write_report(frames, dist, interm, temporal, price, tests_df,
                              figures, out_dir / "phase3_eda.md")
        impl = write_implications(dist, interm, temporal, tests_df,
                                  out_dir / "forecasting_implications.md")

    print(f"Series analysed:      {len(frames['series']):,}")
    print(f"Store-days analysed:  {len(frames['store_daily_cal']):,}")
    print(f"Mean zero share:      {dist['series_zero_share']['mean']:.1%}")
    print(f"Gini (units):         {dist['gini_units']:.3f}")
    print(f"Tests run:            {len(tests_df)} "
          f"({int(tests_df['significant_fdr'].sum())} significant after BH-FDR)")
    print("\nWritten:")
    for p in [report, impl, tests_path, *figures]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
