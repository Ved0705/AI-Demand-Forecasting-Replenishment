# SQL KPI Definitions

Exact definitions for every metric produced by the Phase 2 SQL layer. The
point of this document is that "units" and "revenue" mean one thing in this
project, and anyone reading a number can check what was computed.

## Observed vs derived

This distinction runs through everything below.

| Quantity | Status | Notes |
|---|---|---|
| `sales` (units) | **Observed** | M5 records daily units sold per item-store. |
| `sell_price` | **Observed** | Weekly shelf price per item-store. |
| `revenue` | **Derived** | `sales × sell_price`. See caveat below. |
| SNAP / event flags | **Observed** | From the M5 calendar. |
| Inventory, lead time | **Not present** | Simulated in Phase 6 (D-004). Absent here. |

**Revenue caveat.** M5 does not record transaction value. `revenue` multiplies
units by the *weekly shelf price*, which ignores basket discounts, coupons,
loyalty pricing and intra-week markdowns. It is a consistent value proxy for
ranking and mix analysis, not a financial figure. Rows without a price yield
`NULL` revenue rather than zero, so unpriced units never silently depress a
revenue total — `price_coverage` quantifies that gap.

---

## Volume metrics

**Total units** — `SUM(sales)` over the grain in question. Active rows only:
Phase 1 (D-003) excludes pre-listing rows, so this counts units over periods
when the item was actually sellable.

**Average daily units** — `SUM(sales) / COUNT(DISTINCT date)`.

Note the denominator. Dividing by `COUNT(*)` gives mean units *per row*, which
is a per-series-day figure, not a daily rate. Both appear in the layer and are
named distinctly: `avg_units_per_day` (date-denominated) versus
`mean_units_per_row` (row-denominated). Mixing them is a real error — with
9,147 series they differ by roughly four orders of magnitude.

**Zero share** — `AVG(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END)`, the fraction
of active series-days with no observed sale.

---

## Rolling metrics

**7-day / 28-day rolling demand**

```sql
SUM(units) OVER (PARTITION BY store_id ORDER BY date
                 ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
```

Trailing and inclusive of the current row: a 7-day window covers today plus the
six prior days.

`ROWS` (not `RANGE`) is safe **only because** Phase 1 guarantees contiguous
daily rows per series, verified by `series_date_continuity`. With date gaps,
`ROWS BETWEEN 6 PRECEDING` would span more than seven calendar days without
error. If that validation ever fails, these frames must move to `RANGE ...
INTERVAL`.

**Rolling volatility** — `STDDEV_SAMP(units)` over the 28-day frame.

**Momentum** — 7-day average minus 28-day average. Positive means recent demand
is running above the monthly baseline.

---

## Growth metrics

**28-day growth**

```
growth_pct = 100 × (units_last_28d − units_prior_28d) / units_prior_28d
```

Both windows anchor on `MAX(date)` in the data, never a hardcoded date.

*Interpretation limit:* one 28-day pair confounds trend with seasonality. A
window straddling a seasonal peak produces a decline that is not a trend. Treat
as a flag to investigate.

**Year-over-year** — `LAG(units, 12)` over monthly totals, i.e. same month last
year. 2011 and 2016 are partial years in M5, so `days_observed` is reported
alongside; comparing a partial year to a full one without it is misleading.

At series level, `declining_products` requires `units_prior_28d >=
min_prior_units` (default 30). Without that floor the ranking fills with series
that fell from 3 units to 1 — a −67% change carrying no information.

---

## Volatility

**Coefficient of variation** — `STDDEV_SAMP(sales) / AVG(sales)` over daily
observed sales per series. `NULL` when the mean is zero.

*This is a coarse screen, not a segmentation.* CV here conflates three distinct
things: genuine demand variability, intermittency (long zero runs inflate the
standard deviation), and unobservable stockouts. Phase 5 separates them using
ADI and CV² computed on **non-zero demand sizes** — a different and more
defensible measure. Do not quote CV from this layer as an intermittency metric.

---

## Ranking

**Product rank within store** — `ROW_NUMBER() OVER (PARTITION BY store_id ORDER
BY total_units DESC, item_id)`. `ROW_NUMBER` rather than `RANK` so a top-N
request returns exactly N rows; `item_id` breaks ties deterministically, which
makes reruns reproducible.

**Store rank** — `RANK() OVER (ORDER BY total_units DESC)`. Ties share a rank
here because tied stores genuinely are tied.

**ABC classification** — series sorted by total units descending, cumulative
share computed with an unbounded-preceding window:

- **A** — cumulative share ≤ 80%
- **B** — 80% to 95%
- **C** — above 95%

Feeds Phase 6: A-items justify tighter service levels than C-items.

**Index vs chain** — `share_in_store / chain_share`. Above 1.10 is
over-indexed, below 0.90 under-indexed. Identifies genuine local strength
rather than simply large categories.

---

## SNAP and event metrics

**`snap_active`** — resolved against the store's **own state**:

```sql
CASE state_id WHEN 'CA' THEN snap_CA
              WHEN 'TX' THEN snap_TX
              WHEN 'WI' THEN snap_WI END
```

M5 stores SNAP as three wide per-state columns. Reading `snap_CA` for every
store would compare a Wisconsin store against California's SNAP calendar —
silent, plausible-looking, and wrong. Resolved once in `v_sales`; guarded by
`test_snap_flag_resolves_to_the_stores_own_state`.

**`observed_gap_pct`** — the difference in mean units between flagged and
unflagged days, as a percentage of the unflagged mean.

**This is an association, not an effect.** The column is deliberately named
`observed_gap_pct` rather than `snap_lift` or `snap_effect`. SNAP days fall in
the first ~10 days of the month, so any gap is entangled with payday cycles,
day-of-week composition and month-position effects. Event days coincide with
holidays that shift traffic for many simultaneous reasons. Phase 3 addresses
the inferential question; this layer only describes what was observed.

**Christmas is a closure artefact.** M5 stores close on 25 December, so those
dates show near-zero sales. That is not depressed demand and must not be read
as an event effect.

---

## Price-change metrics

`price_change_association` compares week-over-week price moves against
week-over-week unit changes, bucketed by ±2%.

**Not an elasticity estimate.** Retailers cut prices when they expect or
observe weak demand and raise them on strong sellers, so causality runs in both
directions. Presenting these numbers as elasticity would be a serious error.
The output columns are named `mean_unit_change` and `mean_pct_change` — never
"elasticity".

---

## Grain reference

| View | Grain | Row order of magnitude |
|---|---|---|
| `v_sales` | item × store × date | ~14.1M |
| `v_store_daily` | store × date | ~5.9k |
| `v_store_cat_daily` | store × category × date | ~18k |
| `v_series_summary` | item × store | ~9.1k |

Rolling windows run over the aggregated views, never over `v_sales`.

---

## Universal caveats

1. **Observed sales, not demand.** An in-window zero may be a stockout (D-003).
   M5 has no inventory data, so demand is censored below and every metric here
   is a lower bound on true demand.
2. **Three stores.** `CA_1`, `TX_1`, `WI_1` (D-002). Chain-level figures
   describe these three stores, not Walmart.
3. **Assortment grows over time.** Series enter the panel when first priced, so
   chain-level growth partly reflects assortment expansion rather than per-item
   demand growth. `series_activity_timeline` quantifies this.
4. **No causal claims.** Everything in this layer is descriptive.
