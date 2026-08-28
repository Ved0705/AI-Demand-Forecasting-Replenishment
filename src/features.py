"""Feature engineering for Phase 4 backtesting.

LEAKAGE ARCHITECTURE
--------------------
The single most important design decision in this module: every function that
computes a feature from historical sales accepts a ``cutoff`` parameter and
**slices the data to date <= cutoff as its very first operation**.

This means:
  1. No rolling window can ever see a date beyond the cutoff, even if the
     caller passes the full dataset accidentally.
  2. The slice precedes ALL computation — no ordering dependency between the
     leakage guard and the feature logic.
  3. The pattern is testable: the adversarial test in tests/test_leakage.py
     mutates only post-cutoff rows and verifies that features are bit-identical.

See DECISION_LOG D-022.

KNOWN-FUTURE vs HISTORICAL
---------------------------
Features are divided into two explicit groups documented in ``FeatureSet``:

  historical_cols : computed from past observations (y[t-k], rolling windows).
                    Available only for training rows; cannot be used for future
                    dates without carrying forward the last-known value.

  known_future_cols : available at forecast creation time for any future date.
                      Calendar, day-of-week, month, SNAP flags, static metadata.
                      Safe to use at any horizon without leakage.

See DECISION_LOG D-026.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LAGS: list[int] = [1, 7, 14, 28]
DEFAULT_WINDOWS: list[int] = [7, 28]

KEY_COLS = ("store_id", "item_id")


# ---------------------------------------------------------------------------
# FeatureSet contract
# ---------------------------------------------------------------------------

@dataclass
class FeatureSet:
    """Holds feature columns for one fold half (train or forecast window).

    The split into historical vs known-future columns is explicit so that
    Phase 5 model code can inspect it and downstream reviewers can verify it.
    Nothing in historical_cols may be used on forecast rows without being
    explicitly carried forward (e.g. the last-training-day lag value).

    Attributes
    ----------
    df : DataFrame with keys (store_id, item_id, date), target ``y``
         (NaN for forecast rows), and all feature columns.
    historical_cols : feature names computed from past y values.
    known_future_cols : feature names available at prediction time.
    """

    df: pd.DataFrame
    historical_cols: list[str] = field(default_factory=list)
    known_future_cols: list[str] = field(default_factory=list)

    @property
    def all_feature_cols(self) -> list[str]:
        return self.historical_cols + self.known_future_cols


# ---------------------------------------------------------------------------
# Leakage-safe lag features
# ---------------------------------------------------------------------------

def make_lag_features(
    series_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    lags: Sequence[int] = DEFAULT_LAGS,
    date_col: str = "date",
    target_col: str = "sales",
    key_cols: Sequence[str] = KEY_COLS,
) -> pd.DataFrame:
    """Compute lag features restricted to the training window (date <= cutoff).

    For each series and each date t in the training window:
        lag_k(t) = y(t - k days)

    Parameters
    ----------
    series_df : Long-format DataFrame with at least (store_id, item_id, date, sales).
    cutoff    : Training cutoff date (inclusive).  Rows after this date are
                ignored — this is the leakage guard.
    lags      : Integer lag offsets (in days).

    Returns
    -------
    DataFrame with columns [store_id, item_id, date, lag_1, lag_7, ...].
    Values are NaN when fewer than k historical observations exist.

    Leakage guarantee
    -----------------
    ``df = series_df[date <= cutoff]`` is the FIRST line of computation.
    The adversarial test in test_leakage.py verifies this by mutating post-
    cutoff values and confirming that lag features are bit-identical.
    """
    key_cols = list(key_cols)

    # =========================================================================
    # LEAKAGE GUARD: slice first, compute second.  This line is the boundary.
    # =========================================================================
    df = series_df.loc[series_df[date_col] <= cutoff].copy()

    if df.empty:
        empty = pd.DataFrame(columns=key_cols + [date_col] + [f"lag_{k}" for k in lags])
        return empty

    df = df.sort_values(key_cols + [date_col]).reset_index(drop=True)
    result = df[key_cols + [date_col]].copy()

    # groupby preserves the per-series index so shift aligns correctly.
    grp = df.groupby(key_cols, observed=True)[target_col]
    for k in lags:
        result[f"lag_{k}"] = grp.shift(k).values

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Leakage-safe rolling features
# ---------------------------------------------------------------------------

def make_rolling_features(
    series_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    date_col: str = "date",
    target_col: str = "sales",
    key_cols: Sequence[str] = KEY_COLS,
) -> pd.DataFrame:
    """Compute rolling window features restricted to the training window.

    For each series and each date t:
        rolling_mean_w(t)      = mean(y[t-w .. t-1])
        rolling_std_w(t)       = std(y[t-w .. t-1])   (NaN if < 2 obs)
        rolling_zero_rate_w(t) = share of zeros in [t-w .. t-1]

    y[t] is **never** included.  This is enforced by shift(1) before the
    rolling window — the test in test_features.py verifies it explicitly.

    Leakage guarantee
    -----------------
    Same as make_lag_features: slice to date <= cutoff before any computation.
    """
    key_cols = list(key_cols)

    # =========================================================================
    # LEAKAGE GUARD
    # =========================================================================
    df = series_df.loc[series_df[date_col] <= cutoff].copy()

    if df.empty:
        cols = [date_col] + key_cols
        for w in windows:
            cols += [f"rolling_mean_{w}", f"rolling_std_{w}", f"rolling_zero_rate_{w}"]
        return pd.DataFrame(columns=cols)

    df = df.sort_values(key_cols + [date_col]).reset_index(drop=True)
    result = df[key_cols + [date_col]].copy()

    # shift(1) within each series group moves y[t] to the t+1 position.
    # rolling(w) ending at position t then sees y[t-w .. t-1] — never y[t].
    grp = df.groupby(key_cols, observed=True)[target_col]

    for w in windows:
        result[f"rolling_mean_{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        result[f"rolling_std_{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=2).std()
        )
        result[f"rolling_zero_rate_{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).apply(
                lambda x: float((x == 0).mean()), raw=True
            )
        )

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Known-future features (no past y required)
# ---------------------------------------------------------------------------

def make_known_future_features(
    dates: pd.DatetimeIndex | pd.Series,
    store_id: str,
    state_id: str,
    snap_ca: pd.Series | None = None,
    snap_tx: pd.Series | None = None,
    snap_wi: pd.Series | None = None,
    event_name_1: pd.Series | None = None,
    cat_id: str | None = None,
    dept_id: str | None = None,
) -> pd.DataFrame:
    """Build calendar and metadata features available at forecast creation time.

    These features require NO past sales values and are therefore available
    for any future forecast date at any horizon.

    SNAP treatment (D-026)
    ----------------------
    SNAP is a known-future calendar feature.  The SNAP disbursement schedule
    is published in advance and is always available at forecast creation time.
    It is NOT a historical feature.

    Parameters
    ----------
    dates   : Dates to build features for.
    state_id: Determines which SNAP column applies (CA / TX / WI).

    Returns
    -------
    DataFrame with columns: date, store_id, state_id, wday, month, year,
    week_of_year, is_weekend, snap_active, is_event, [cat_id], [dept_id].
    """
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)

    df = pd.DataFrame({
        "date": dates,
        "store_id": store_id,
        "state_id": state_id,
        "wday": dates.dt.dayofweek + 1,          # 1=Monday … 7=Sunday
        "month": dates.dt.month,
        "year": dates.dt.year,
        "week_of_year": dates.dt.isocalendar().week.astype(int).values,
        "is_weekend": dates.dt.dayofweek.isin([5, 6]).astype(int),
    })

    # State-aware SNAP resolution (mirrors D-011 in the SQL layer)
    snap_map = {"CA": snap_ca, "TX": snap_tx, "WI": snap_wi}
    snap_series = snap_map.get(state_id)
    df["snap_active"] = (
        np.asarray(snap_series, dtype=float) if snap_series is not None else np.nan
    )

    df["is_event"] = (
        pd.Series(event_name_1).fillna("").ne("").astype(int).values
        if event_name_1 is not None else 0
    )

    if cat_id is not None:
        df["cat_id"] = cat_id
    if dept_id is not None:
        df["dept_id"] = dept_id

    return df


# ---------------------------------------------------------------------------
# Fold-level feature orchestration
# ---------------------------------------------------------------------------

KNOWN_FUTURE_BASE = [
    "wday", "month", "year", "week_of_year", "is_weekend",
    "snap_active", "is_event",
]


def build_fold_features(
    long_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    date_col: str = "date",
    target_col: str = "sales",
    key_cols: Sequence[str] = KEY_COLS,
) -> tuple[FeatureSet, FeatureSet]:
    """Build train and forecast FeatureSets for a single backtest fold.

    Parameters
    ----------
    long_df : Full long-format DataFrame (may contain rows beyond cutoff;
              the leakage guard in the sub-functions will filter them out).
    cutoff  : Last training date (inclusive).
    horizon : Forecast horizon in days (forecast window = cutoff+1 .. cutoff+horizon).

    Returns
    -------
    (train_fs, forecast_fs)

    train_fs  — historical + known-future features for all active training rows.
    forecast_fs — known-future features for each (series × forecast date),
                  plus last-training-day lag/rolling values carried forward.
                  The ``y`` column is NaN (target unknown at forecast time).

    Leakage guarantee
    -----------------
    All historical features are computed by make_lag_features and
    make_rolling_features, both of which slice to date <= cutoff first.
    The only training-derived values that appear in forecast_fs are the
    tail values of lag/rolling computed from the training window only.
    """
    key_cols = list(key_cols)
    lags = list(lags)
    windows = list(windows)

    # ------------------------------------------------------------------
    # Training half
    # ------------------------------------------------------------------
    train_df = long_df.loc[long_df[date_col] <= cutoff].copy()

    lag_df = make_lag_features(train_df, cutoff, lags, date_col, target_col, key_cols)
    roll_df = make_rolling_features(train_df, cutoff, windows, date_col, target_col, key_cols)

    merge_on = key_cols + [date_col]
    train_feat = train_df[key_cols + [date_col, target_col]].copy()
    train_feat = train_feat.merge(lag_df, on=merge_on, how="left")

    roll_cols_only = [c for c in roll_df.columns if c not in key_cols + [date_col, target_col]]
    train_feat = train_feat.merge(roll_df[key_cols + [date_col] + roll_cols_only],
                                  on=merge_on, how="left")
    train_feat = train_feat.rename(columns={target_col: "y"})

    lag_col_names = [f"lag_{k}" for k in lags]
    roll_col_names = [c for c in train_feat.columns if c.startswith("rolling_")]
    hist_cols = lag_col_names + roll_col_names

    # Known-future features for training rows (calendar columns already in long_df)
    cal_cols = [c for c in long_df.columns if c in KNOWN_FUTURE_BASE]
    if cal_cols:
        cal_train = long_df.loc[long_df[date_col] <= cutoff, key_cols + [date_col] + cal_cols]
        train_feat = train_feat.merge(cal_train, on=merge_on, how="left")

    train_fs = FeatureSet(
        df=train_feat,
        historical_cols=hist_cols,
        known_future_cols=cal_cols,
    )

    # ------------------------------------------------------------------
    # Forecast half
    # ------------------------------------------------------------------
    forecast_start = cutoff + pd.Timedelta(days=1)
    forecast_end = cutoff + pd.Timedelta(days=horizon)
    forecast_dates = pd.date_range(forecast_start, forecast_end, freq="D")

    series_keys = train_df[key_cols].drop_duplicates().copy()
    forecast_rows = series_keys.assign(_tmp=1).merge(
        pd.DataFrame({date_col: forecast_dates, "_tmp": 1}), on="_tmp"
    ).drop(columns="_tmp").reset_index(drop=True)
    forecast_rows["y"] = np.nan

    # Carry forward the last training-day lag/rolling values per series.
    # These are computed entirely from training data, so not leakage.
    last_lag = lag_df[lag_df[date_col] == cutoff][key_cols + lag_col_names]
    last_roll = roll_df[roll_df[date_col] == cutoff][key_cols + roll_col_names]
    forecast_rows = forecast_rows.merge(last_lag, on=key_cols, how="left")
    forecast_rows = forecast_rows.merge(last_roll, on=key_cols, how="left")

    # Known-future features for forecast dates (from long_df if those dates exist,
    # otherwise from calendar columns already in long_df for those dates)
    if cal_cols and len(long_df.loc[long_df[date_col].isin(forecast_dates)]) > 0:
        # Use one series as the calendar source (calendar is store-agnostic for
        # wday/month/year; snap_active is per-store, so we join per store)
        cal_fcst = long_df.loc[long_df[date_col].isin(forecast_dates),
                                key_cols + [date_col] + cal_cols].drop_duplicates()
        forecast_rows = forecast_rows.merge(cal_fcst, on=key_cols + [date_col], how="left")

    forecast_fs = FeatureSet(
        df=forecast_rows,
        historical_cols=hist_cols,   # carried-forward values; label preserved
        known_future_cols=cal_cols,
    )

    logger.debug(
        "build_fold_features: cutoff=%s train=%d rows forecast=%d rows",
        cutoff.date(), len(train_feat), len(forecast_rows),
    )
    return train_fs, forecast_fs
