-- =============================================================================
-- 02 - SALES KPIs
-- Headline volume and revenue metrics. Every figure here is observed units;
-- revenue is derived (units x weekly shelf price), never observed.
-- =============================================================================


-- name: kpi_overall
SELECT
    SUM(sales)                              AS total_units,
    ROUND(SUM(revenue), 2)                  AS total_revenue_derived,
    COUNT(DISTINCT date)                    AS n_days,
    ROUND(SUM(sales) * 1.0
          / COUNT(DISTINCT date), 1)        AS avg_units_per_day,
    COUNT(DISTINCT item_id)                 AS n_items,
    COUNT(DISTINCT store_id)                AS n_stores
FROM v_sales;


-- name: kpi_by_category
-- Conditional aggregation (FILTER) splits SNAP and event days without a
-- self-join, and window functions give each category its share of the total.
SELECT
    cat_id,
    COUNT(DISTINCT item_id)                             AS n_items,
    SUM(sales)                                          AS total_units,
    ROUND(SUM(revenue), 2)                              AS revenue_derived,
    ROUND(AVG(sales), 3)                                AS mean_daily_units,
    SUM(sales) FILTER (WHERE snap_active = 1)           AS units_snap_days,
    SUM(sales) FILTER (WHERE is_event_day = 1)          AS units_event_days,
    ROUND(100.0 * SUM(sales)
          / SUM(SUM(sales)) OVER (), 2)                 AS pct_of_total_units,
    ROUND(100.0 * SUM(revenue)
          / SUM(SUM(revenue)) OVER (), 2)               AS pct_of_total_revenue
FROM v_sales
GROUP BY cat_id
ORDER BY total_units DESC;


-- name: kpi_by_department
SELECT
    cat_id,
    dept_id,
    COUNT(DISTINCT item_id)                    AS n_items,
    SUM(sales)                                 AS total_units,
    ROUND(SUM(revenue), 2)                     AS revenue_derived,
    -- Rank departments inside their own category, not globally.
    RANK() OVER (PARTITION BY cat_id
                 ORDER BY SUM(sales) DESC)     AS rank_in_category,
    ROUND(100.0 * SUM(sales)
          / SUM(SUM(sales)) OVER (PARTITION BY cat_id), 2)
                                               AS pct_of_category
FROM v_sales
GROUP BY cat_id, dept_id
ORDER BY cat_id, rank_in_category;


-- name: kpi_store_category_matrix
-- Store x category performance, with each cell expressed relative to its
-- store so that stores of different sizes are comparable.
SELECT
    store_id,
    state_id,
    cat_id,
    SUM(sales)                                          AS total_units,
    ROUND(SUM(revenue), 2)                              AS revenue_derived,
    ROUND(100.0 * SUM(sales)
          / SUM(SUM(sales)) OVER (PARTITION BY store_id), 2)
                                                        AS pct_of_store_units,
    RANK() OVER (PARTITION BY cat_id
                 ORDER BY SUM(sales) DESC)              AS store_rank_in_category
FROM v_sales
GROUP BY store_id, state_id, cat_id
ORDER BY store_id, total_units DESC;


-- name: kpi_day_of_week_profile
-- M5 wday: 1 = Saturday ... 7 = Friday.
SELECT
    wday,
    CASE wday
        WHEN 1 THEN 'Sat' WHEN 2 THEN 'Sun' WHEN 3 THEN 'Mon'
        WHEN 4 THEN 'Tue' WHEN 5 THEN 'Wed' WHEN 6 THEN 'Thu'
        WHEN 7 THEN 'Fri'
    END                                          AS day_name,
    SUM(sales)                                   AS total_units,
    ROUND(AVG(sales), 4)                         AS mean_units_per_row,
    ROUND(100.0 * SUM(sales)
          / SUM(SUM(sales)) OVER (), 2)          AS pct_of_total
FROM v_sales
GROUP BY wday
ORDER BY wday;
