"""Generate a small M5-SHAPED fixture for testing the pipeline.

=============================================================================
READ THIS BEFORE USING THE OUTPUT FOR ANYTHING.

This module produces synthetic files that MIMIC THE SCHEMA of the M5 dataset
so that ingestion, feature engineering, backtesting and replenishment code can
be unit-tested without the real ~450MB download.

The numbers it produces are FAKE. They must NEVER be used for:
  - EDA
  - hypothesis testing
  - model benchmarking
  - any result quoted in the README or an interview

Testing code correctness against a synthetic fixture is standard engineering
practice. Drawing analytical conclusions from generated sales is not, because
you would only be recovering the relationships you yourself wrote into the
generator. All analysis in this project runs on real observed M5 sales.

Files are written to data/fixture/ - deliberately NOT data/raw/ - so a fixture
can never be silently mistaken for the real download.
=============================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

FIXTURE_DIR = PROJECT_ROOT / "data" / "fixture"

STORES = [("CA_1", "CA"), ("TX_1", "TX"), ("WI_1", "WI")]
DEPTS = [
    ("FOODS_1", "FOODS"),
    ("HOBBIES_1", "HOBBIES"),
    ("HOUSEHOLD_1", "HOUSEHOLD"),
]
N_ITEMS_PER_DEPT = 4
N_DAYS = 800
START_DATE = "2011-01-29"  # real M5 start date


def _wm_yr_wk(dates: pd.Series) -> pd.Series:
    """Replicate M5's wm_yr_wk key (Walmart weeks run Saturday->Friday)."""
    # Days since M5 epoch, bucketed into 7-day weeks starting on the epoch.
    epoch = pd.Timestamp(START_DATE)
    week_idx = ((dates - epoch).dt.days // 7).astype(int)
    year = 11 + (week_idx // 52)
    week = (week_idx % 52) + 1
    return (year * 1000 + week).astype(int)


def build_fixture(seed: int = 0, out_dir: Path | None = None) -> Path:
    """Write calendar.csv, sales_train_evaluation.csv, sell_prices.csv."""
    out_dir = out_dir or FIXTURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---------------- calendar ----------------
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": _wm_yr_wk(pd.Series(dates)),
            "weekday": dates.day_name(),
            "wday": ((dates.dayofweek + 2) % 7) + 1,  # M5: Saturday = 1
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i + 1}" for i in range(N_DAYS)],
        }
    )
    # Sparse events, mirroring M5's mostly-null event columns.
    cal["event_name_1"] = pd.NA
    cal["event_type_1"] = pd.NA
    holiday_mask = (cal["month"] == 12) & (pd.to_datetime(cal["date"]).dt.day == 25)
    cal.loc[holiday_mask, "event_name_1"] = "Christmas"
    cal.loc[holiday_mask, "event_type_1"] = "National"
    cal["event_name_2"] = pd.NA
    cal["event_type_2"] = pd.NA
    for state in ("CA", "TX", "WI"):
        # SNAP runs on the first ~10 days of each month in the real data.
        cal[f"snap_{state}"] = (
            pd.to_datetime(cal["date"]).dt.day <= 10
        ).astype(int)

    # ---------------- items ----------------
    rows = []
    for dept_id, cat_id in DEPTS:
        for i in range(1, N_ITEMS_PER_DEPT + 1):
            rows.append({"item_id": f"{dept_id}_{i:03d}", "dept_id": dept_id, "cat_id": cat_id})
    items = pd.DataFrame(rows)

    # ---------------- sales (wide) ----------------
    d_cols = [f"d_{i + 1}" for i in range(N_DAYS)]
    dow = np.array(((dates.dayofweek + 2) % 7) + 1)
    weekend_lift = np.where(np.isin(dow, [1, 2]), 1.35, 1.0)  # Sat/Sun

    sales_rows = []
    for _, item in items.iterrows():
        for store_id, state_id in STORES:
            # Category drives the demand regime, so the fixture exercises both
            # the fast-moving and the intermittent code paths.
            if item["cat_id"] == "FOODS":
                base, intermittent = rng.uniform(4, 12), False
            elif item["cat_id"] == "HOUSEHOLD":
                base, intermittent = rng.uniform(1, 3), False
            else:
                base, intermittent = rng.uniform(0.2, 0.8), True

            lam = base * weekend_lift
            series = rng.poisson(lam).astype(float)
            if intermittent:
                series *= rng.random(N_DAYS) < 0.35  # long zero runs

            # Item not listed yet -> leading zeros, exactly like real M5.
            start = int(rng.integers(0, 120))
            series[:start] = 0

            rec = {
                "id": f"{item['item_id']}_{store_id}_evaluation",
                "item_id": item["item_id"],
                "dept_id": item["dept_id"],
                "cat_id": item["cat_id"],
                "store_id": store_id,
                "state_id": state_id,
                "_start": start,
            }
            rec.update(dict(zip(d_cols, series.astype(int))))
            sales_rows.append(rec)

    sales = pd.DataFrame(sales_rows)
    starts = sales[["item_id", "store_id", "_start"]].copy()
    sales = sales.drop(columns="_start")

    # ---------------- prices ----------------
    # Price is null before the item is listed. This is what makes the
    # availability detection in ingest.py testable.
    date_to_wk = dict(zip(cal["d"], cal["wm_yr_wk"]))
    price_rows = []
    for _, r in starts.iterrows():
        start_wk = date_to_wk[f"d_{r['_start'] + 1}"]
        weeks = sorted({w for w in cal["wm_yr_wk"].unique() if w >= start_wk})
        price = float(rng.uniform(1.0, 9.0))
        for w in weeks:
            if rng.random() < 0.03:  # occasional repricing
                price = max(0.5, price * rng.uniform(0.8, 1.2))
            price_rows.append(
                {
                    "store_id": r["store_id"],
                    "item_id": r["item_id"],
                    "wm_yr_wk": w,
                    "sell_price": round(price, 2),
                }
            )
    prices = pd.DataFrame(price_rows)

    cal.to_csv(out_dir / "calendar.csv", index=False)
    sales.to_csv(out_dir / "sales_train_evaluation.csv", index=False)
    prices.to_csv(out_dir / "sell_prices.csv", index=False)

    marker = out_dir / "SYNTHETIC_DO_NOT_ANALYSE.txt"
    marker.write_text(
        "These files are synthetic, generated by src/make_fixture.py.\n"
        "They exist only to unit-test the pipeline. Any statistic computed\n"
        "from them describes the generator, not retail behaviour.\n"
    )
    return out_dir


if __name__ == "__main__":
    path = build_fixture()
    print(f"Fixture written to {path}")
