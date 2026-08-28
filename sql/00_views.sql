-- =============================================================================
-- 00 - ANALYTICAL VIEWS
-- Created once, reused by every later file. Views (not materialised tables)
-- because DuckDB pushes predicates through them and the storage cost of
-- duplicating a 14M-row fact table is not worth paying. See DECISION_LOG D-010.
-- =============================================================================


-- name: create_v_sales
-- The denormalised analytical spine: fact + product + store + calendar.
--
-- The SNAP columns are the reason this view exists. M5 stores SNAP as three
-- wide per-state flags (snap_CA, snap_TX, snap_WI). A query that joins the
-- calendar and reads snap_CA for every store would be comparing a Wisconsin
-- store against California's SNAP calendar. Resolving the flag against the
-- store's own state, once, here, means no downstream query can get it wrong.
CREATE OR REPLACE VIEW v_sales AS
SELECT
    f.item_id,
    f.store_id,
    f.date,
    f.wm_yr_wk,
    f.sales,
    f.sell_price,
    -- Revenue is a DEFINED quantity, not an observed one: M5 records units
    -- sold and a weekly shelf price, never transaction value. NULL where the
    -- price is unknown, so revenue never silently reads as zero.
    CASE WHEN f.sell_price IS NULL THEN NULL
         ELSE f.sales * f.sell_price END        AS revenue,
    p.dept_id,
    p.cat_id,
    s.state_id,
    c.wday,
    c.month,
    c.year,
    c.event_name_1,
    c.event_type_1,
    -- State-aware SNAP resolution.
    CASE s.state_id
        WHEN 'CA' THEN c.snap_CA
        WHEN 'TX' THEN c.snap_TX
        WHEN 'WI' THEN c.snap_WI
    END                                          AS snap_active,
    -- Any national/religious/sporting/cultural event on that date.
    CASE WHEN c.event_name_1 IS NOT NULL THEN 1 ELSE 0 END AS is_event_day
FROM fact_sales f
JOIN dim_product  p ON f.item_id  = p.item_id
JOIN dim_store    s ON f.store_id = s.store_id
JOIN dim_calendar c ON f.date     = c.date;


-- name: create_v_store_daily
-- Store x date aggregate. Rolling windows over 14M rows are expensive; over
-- ~5.9k store-days they are instant. Aggregate first, then window.
CREATE OR REPLACE VIEW v_store_daily AS
SELECT
    store_id,
    state_id,
    date,
    SUM(sales)                        AS units,
    SUM(revenue)                      AS revenue,
    COUNT(DISTINCT item_id)           AS active_skus,
    MAX(snap_active)                  AS snap_active,
    MAX(is_event_day)                 AS is_event_day
FROM v_sales
GROUP BY 1, 2, 3;


-- name: create_v_store_cat_daily
-- Store x category x date aggregate for category-level trend work.
CREATE OR REPLACE VIEW v_store_cat_daily AS
SELECT
    store_id,
    cat_id,
    date,
    SUM(sales)   AS units,
    SUM(revenue) AS revenue
FROM v_sales
GROUP BY 1, 2, 3;


-- name: create_v_series_summary
-- One row per item-store series: lifetime totals and dispersion. The base for
-- ranking, volatility and ABC analysis.
--
-- NOTE ON VOLATILITY: stddev/mean is computed over ACTIVE days only, and an
-- in-window zero may be a stockout rather than zero demand (D-003). This
-- measures variability of OBSERVED SALES, not of true demand.
CREATE OR REPLACE VIEW v_series_summary AS
SELECT
    item_id,
    store_id,
    cat_id,
    dept_id,
    COUNT(*)                                        AS n_days,
    SUM(sales)                                      AS total_units,
    SUM(revenue)                                    AS total_revenue,
    AVG(sales)                                      AS mean_daily_units,
    STDDEV_SAMP(sales)                              AS sd_daily_units,
    CASE WHEN AVG(sales) > 0
         THEN STDDEV_SAMP(sales) / AVG(sales) END   AS cv_daily_units,
    SUM(CASE WHEN sales > 0 THEN 1 ELSE 0 END)      AS nonzero_days,
    AVG(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END)  AS zero_share,
    MIN(date)                                       AS first_active_date,
    MAX(date)                                       AS last_active_date
FROM v_sales
GROUP BY 1, 2, 3, 4;
