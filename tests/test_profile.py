"""Tests for profiling and demand segmentation.

`series_stats` was vectorised for scale. These tests pin it against a slow,
obviously-correct reference implementation so the optimisation cannot silently
change any number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.profile import classify_demand, series_stats


def _reference_stats(long: pd.DataFrame) -> pd.DataFrame:
    """Slow but transparent. The definition of correct."""
    rows = []
    for (store, item), grp in long.groupby(["store_id", "item_id"], observed=True):
        s = grp["sales"]
        nz = s[s > 0]
        n, k = len(s), len(nz)
        rows.append(
            {
                "store_id": store,
                "item_id": item,
                "n_days": n,
                "n_nonzero_days": k,
                "adi": n / k if k else np.nan,
                "cv2": float((nz.std() / nz.mean()) ** 2)
                if k >= 2 and nz.mean() != 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["store_id", "item_id"]).reset_index(drop=True)


@pytest.fixture
def toy() -> pd.DataFrame:
    """Hand-built series covering every edge case that matters."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    cases = {
        "smooth": [5, 6, 5, 7, 6, 5, 6, 6, 5, 7],   # dense, low variability
        "intermit": [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],  # long zero runs
        "lumpy": [0, 0, 1, 0, 0, 0, 40, 0, 0, 0],    # sparse AND wild sizes
        "single": [0, 0, 0, 0, 7, 0, 0, 0, 0, 0],    # one sale -> cv2 undefined
        "allzero": [0] * 10,                          # never sells -> adi undefined
    }
    frames = []
    for item, vals in cases.items():
        frames.append(
            pd.DataFrame(
                {"store_id": "S1", "item_id": item, "date": dates, "sales": vals}
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_vectorised_matches_reference(toy):
    got = series_stats(toy).sort_values(["store_id", "item_id"]).reset_index(drop=True)
    want = _reference_stats(toy)
    for col in ("n_days", "n_nonzero_days", "adi", "cv2"):
        pd.testing.assert_series_equal(
            got[col].astype(float), want[col].astype(float),
            check_names=False, rtol=1e-9,
        )


def test_single_sale_series_has_undefined_cv2(toy):
    """One observation has no measurable size variability - must be NaN, not 0."""
    stats = series_stats(toy).set_index("item_id")
    assert stats.loc["single", "n_nonzero_days"] == 1
    assert np.isnan(stats.loc["single", "cv2"])


def test_never_selling_series_has_undefined_adi(toy):
    stats = series_stats(toy).set_index("item_id")
    assert stats.loc["allzero", "n_nonzero_days"] == 0
    assert np.isnan(stats.loc["allzero", "adi"])


def test_zero_share_is_consistent(toy):
    stats = series_stats(toy).set_index("item_id")
    assert stats.loc["smooth", "zero_share"] == pytest.approx(0.0)
    assert stats.loc["allzero", "zero_share"] == pytest.approx(1.0)


def test_segmentation_assigns_expected_quadrants(toy):
    seg = classify_demand(series_stats(toy), adi_thr=1.32, cv2_thr=0.49)
    seg = seg.set_index("item_id")["segment"]
    assert seg["smooth"] == "smooth"
    # Sparse with erratic sizes -> lumpy; sparse with steady sizes -> intermittent.
    assert seg["lumpy"] == "lumpy"
    assert seg["intermit"] in {"intermittent", "lumpy"}


def test_undefined_stats_do_not_become_a_real_segment(toy):
    """NaN adi/cv2 must fall through to 'unknown', not silently land in a quadrant."""
    seg = classify_demand(series_stats(toy), adi_thr=1.32, cv2_thr=0.49)
    seg = seg.set_index("item_id")["segment"]
    assert seg["allzero"] == "unknown"
    assert seg["single"] == "unknown"
