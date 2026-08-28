"""Data-quality checks.

These run after ingestion and fail loudly. The point is that a silent data
bug - a duplicated key, a gap in a series, a negative sale - will otherwise
surface much later as a confusing model result, and you will waste a day
blaming the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class QualityReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name, passed, detail))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"check": r.name, "passed": r.passed, "detail": r.detail} for r in self.results]
        )

    def __str__(self) -> str:
        lines = []
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"[{mark}] {r.name}" + (f" - {r.detail}" if r.detail else ""))
        return "\n".join(lines)


def run_checks(long: pd.DataFrame) -> QualityReport:
    rep = QualityReport()
    keys = ["store_id", "item_id", "date"]

    # --- structural -------------------------------------------------------
    dup = long.duplicated(subset=keys).sum()
    rep.add("no_duplicate_series_dates", dup == 0, f"{dup} duplicate rows")

    missing_core = long[["store_id", "item_id", "date", "sales"]].isna().sum().sum()
    rep.add("no_nulls_in_core_columns", missing_core == 0, f"{missing_core} nulls")

    neg = (long["sales"] < 0).sum()
    rep.add("no_negative_sales", neg == 0, f"{neg} negative rows")

    # --- temporal continuity ---------------------------------------------
    # Every active series should be a contiguous daily run. A gap breaks lag
    # features silently, because shift(1) would jump across the hole.
    g = long.groupby(["store_id", "item_id"], observed=True)["date"]
    span = (g.max() - g.min()).dt.days + 1
    gapped = (g.count() != span).sum()
    rep.add("series_are_contiguous_daily", gapped == 0, f"{gapped} series with date gaps")

    # --- prices -----------------------------------------------------------
    if "sell_price" in long.columns:
        bad_price = (long["sell_price"] <= 0).sum()
        rep.add("prices_positive", bad_price == 0, f"{bad_price} non-positive prices")

        # After availability filtering, active rows should almost always be priced.
        if "is_active" in long.columns:
            active = long.loc[long["is_active"]]
            unpriced = active["sell_price"].isna().mean() if len(active) else 0.0
            rep.add(
                "active_rows_mostly_priced",
                unpriced < 0.01,
                f"{unpriced:.2%} of active rows have no price",
            )

    # --- sanity -----------------------------------------------------------
    all_zero = (
        long.groupby(["store_id", "item_id"], observed=True)["sales"].sum().eq(0).sum()
    )
    rep.add("few_all_zero_series", all_zero == 0, f"{all_zero} series never sell")

    return rep
