-- =============================================================================
-- 05 - DEMAND TRENDS
-- Rolling windows and trend decomposition.
--
-- PERFORMANCE NOTE: every rolling window here runs over a pre-aggregated view
-- (store x date, or store x category x date), never over the 14M-row fact
-- table. Windowing 5.9k rows is instant; windowing 14M rows partitioned by
-- series is not, and would buy nothing at this level of analysis.
--
-- FRAME NOTE: ROWS BETWEEN n PRECEDING is safe here only because Phase 1
-- guarantees contiguous daily rows (validated by series_date_continuity).
-- With gaps, ROWS would silently span the wrong number of calendar days.
-- =============================================================================


-- name: rolling_store_demand
-- 7-day and 28-day trailing windows plus a rolling volatility measure.
SELECT
    store_id,
    date,
    units,
    SUM(units) OVER (PARTITION BY store_id ORDER BY date
                     ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)   AS units_7d,
    ROUND(AVG(units) OVER (PARTITION BY store_id ORDER BY date
                     ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1)
                                                                 AS avg_7d,
    SUM(units) OVER (PARTITION BY store_id ORDER BY date
                     ROWS BETWEEN 27 PRECEDING AND CURRENT ROW)  AS units_28d,
    ROUND(AVG(units) OVER (PARTITION BY store_id ORDER BY date
                     ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 1)
                                                                 AS avg_28d,
    ROUND(STDDEV_SAMP(units) OVER (PARTITION BY store_id ORDER BY date
                     ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 1)
                                                                 AS sd_28d,
    -- Short-vs-long crossover: a standard momentum read.
    ROUND(
        AVG(units) OVER (PARTITION BY store_id ORDER BY date
                         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
      - AVG(units) OVER (PARTITION BY store_id ORDER BY date
                         ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 1)
                                                                 AS momentum_7d_vs_28d
FROM v_store_daily
ORDER BY store_id, date;


-- name: rolling_category_demand
SELECT
    store_id,
    cat_id,
    date,
    units,
    SUM(units) OVER (PARTITION BY store_id, cat_id ORDER BY date
                     ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)   AS units_7d,
    ROUND(AVG(units) OVER (PARTITION BY store_id, cat_id ORDER BY date
                     ROWS BETWEEN 27 PRECEDING AND CURRENT ROW), 1)
                                                                 AS avg_28d,
    LAG(units, 7) OVER (PARTITION BY store_id, cat_id ORDER BY date)
                                                                 AS units_same_day_last_week
FROM v_store_cat_daily
ORDER BY store_id, cat_id, date;


-- name: monthly_demand_trend
-- Month-over-month and same-month-last-year comparisons.
WITH monthly AS (
    SELECT
        store_id,
        year,
        month,
        DATE_TRUNC('month', date)  AS month_start,
        SUM(sales)                 AS units,
        COUNT(DISTINCT date)       AS days_observed
    FROM v_sales
    GROUP BY 1, 2, 3, 4
)
SELECT
    store_id,
    month_start,
    days_observed,
    units,
    ROUND(units * 1.0 / days_observed, 1)                        AS units_per_day,
    LAG(units) OVER (PARTITION BY store_id ORDER BY month_start)  AS prev_month_units,
    LAG(units, 12) OVER (PARTITION BY store_id ORDER BY month_start)
                                                                  AS same_month_last_year,
    ROUND(100.0 * (units - LAG(units, 12) OVER
              (PARTITION BY store_id ORDER BY month_start))
          / NULLIF(LAG(units, 12) OVER
              (PARTITION BY store_id ORDER BY month_start), 0), 1) AS yoy_pct
FROM monthly
ORDER BY store_id, month_start;


-- name: weekday_weekend_profile
-- Comparison only. Whether the gap is statistically meaningful is a Phase 3
-- question, not something this aggregate can answer.
SELECT
    store_id,
    CASE WHEN wday IN (1, 2) THEN 'weekend' ELSE 'weekday' END AS day_type,
    COUNT(DISTINCT date)                                        AS n_days,
    SUM(sales)                                                  AS total_units,
    ROUND(SUM(sales) * 1.0 / COUNT(DISTINCT date), 1)           AS units_per_day
FROM v_sales
GROUP BY store_id, day_type
ORDER BY store_id, day_type;
