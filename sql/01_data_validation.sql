-- =============================================================================
-- 01 - DATA VALIDATION
-- Structural checks against the real database. These are not analysis; they
-- establish that the analytical queries downstream are trustworthy.
-- Query blocks are delimited by "-- name:" and executed by src/sql_runner.py.
-- =============================================================================


-- name: fact_overview
-- Headline shape of the fact table. Every other Phase 2 number should
-- reconcile against these totals.
SELECT
    COUNT(*)                                   AS n_rows,
    COUNT(DISTINCT item_id || '|' || store_id) AS n_series,
    COUNT(DISTINCT item_id)                    AS n_items,
    COUNT(DISTINCT store_id)                   AS n_stores,
    COUNT(DISTINCT date)                       AS n_dates,
    MIN(date)                                  AS date_min,
    MAX(date)                                  AS date_max,
    SUM(sales)                                 AS total_units,
    ROUND(AVG(sales), 4)                       AS mean_daily_units,
    SUM(CASE WHEN sales = 0 THEN 1 ELSE 0 END) AS zero_rows,
    ROUND(AVG(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END), 4) AS zero_share
FROM fact_sales;


-- name: grain_uniqueness
-- The declared grain is one row per (item, store, date). A non-zero result
-- here invalidates every aggregate in the project.
SELECT COUNT(*) AS duplicate_keys
FROM (
    SELECT item_id, store_id, date
    FROM fact_sales
    GROUP BY 1, 2, 3
    HAVING COUNT(*) > 1
);


-- name: join_fanout_check
-- Joining all three dimensions must not change the row count. If it does, a
-- dimension is not unique on its business key and every JOIN in this folder
-- is quietly inflating totals.
WITH joined AS (
    SELECT f.item_id
    FROM fact_sales f
    JOIN dim_product  p ON f.item_id  = p.item_id
    JOIN dim_store    s ON f.store_id = s.store_id
    JOIN dim_calendar c ON f.date     = c.date
)
SELECT
    (SELECT COUNT(*) FROM fact_sales) AS fact_rows,
    (SELECT COUNT(*) FROM joined)     AS joined_rows,
    (SELECT COUNT(*) FROM joined) - (SELECT COUNT(*) FROM fact_sales) AS fanout_delta;


-- name: dimension_key_uniqueness
-- Explicitly proves the business keys are unique, which is what makes the
-- fan-out check above meaningful rather than coincidental.
SELECT 'dim_product' AS table_name, COUNT(*) AS n_rows,
       COUNT(DISTINCT item_id) AS n_keys
FROM dim_product
UNION ALL
SELECT 'dim_store', COUNT(*), COUNT(DISTINCT store_id) FROM dim_store
UNION ALL
SELECT 'dim_calendar', COUNT(*), COUNT(DISTINCT date) FROM dim_calendar
UNION ALL
SELECT 'dim_price', COUNT(*),
       COUNT(DISTINCT store_id || '|' || item_id || '|' || wm_yr_wk)
FROM dim_price;


-- name: price_coverage
-- Revenue is defined as sales * sell_price, so unpriced rows are rows where
-- revenue is unknowable. Quantifying that gap is a precondition for reporting
-- any revenue figure.
SELECT
    COUNT(*)                                              AS n_rows,
    SUM(CASE WHEN sell_price IS NULL THEN 1 ELSE 0 END)   AS unpriced_rows,
    ROUND(AVG(CASE WHEN sell_price IS NULL THEN 1.0 ELSE 0.0 END), 6)
                                                          AS unpriced_share,
    SUM(CASE WHEN sell_price IS NULL THEN sales ELSE 0 END)
                                                          AS units_without_price
FROM fact_sales;


-- name: series_date_continuity
-- Lag and rolling features assume contiguous daily runs per series. A gap
-- makes LAG(1) silently jump across a hole. Phase 4 depends on this holding.
WITH bounds AS (
    SELECT item_id, store_id,
           COUNT(*)                                  AS n_days,
           DATE_DIFF('day', MIN(date), MAX(date)) + 1 AS span_days
    FROM fact_sales
    GROUP BY 1, 2
)
SELECT
    COUNT(*)                                                AS n_series,
    SUM(CASE WHEN n_days <> span_days THEN 1 ELSE 0 END)    AS series_with_gaps
FROM bounds;


-- name: active_flag_check
-- Phase 1 (D-003) drops inactive rows at ingestion, so is_active should be
-- uniformly true here. A false row means the ingestion contract changed.
SELECT
    SUM(CASE WHEN is_active THEN 1 ELSE 0 END)     AS active_rows,
    SUM(CASE WHEN NOT is_active THEN 1 ELSE 0 END) AS inactive_rows
FROM fact_sales;
