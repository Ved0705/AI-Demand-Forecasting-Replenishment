-- =============================================================================
-- 04 - PRODUCT ANALYSIS
-- Rankings, decline detection, volatility and assortment structure.
-- Parameterised queries use DuckDB named parameters ($name).
-- =============================================================================


-- name: top_products_per_store
-- params: top_n
-- ROW_NUMBER (not RANK) because we want exactly N rows per store with no ties
-- expanding the result set.
WITH ranked AS (
    SELECT
        store_id,
        item_id,
        cat_id,
        total_units,
        total_revenue,
        ROW_NUMBER() OVER (PARTITION BY store_id
                           ORDER BY total_units DESC, item_id) AS rn,
        ROUND(100.0 * total_units
              / SUM(total_units) OVER (PARTITION BY store_id), 3)
                                                               AS pct_of_store
    FROM v_series_summary
)
SELECT store_id, rn AS rank, item_id, cat_id, total_units,
       ROUND(total_revenue, 2) AS revenue_derived, pct_of_store
FROM ranked
WHERE rn <= $top_n
ORDER BY store_id, rn;


-- name: declining_products
-- params: min_prior_units, top_n
-- Trailing 28d vs prior 28d at series level.
--
-- The min_prior_units filter matters: without it the list is dominated by
-- series that went from 3 units to 1, a -67% "decline" that is statistically
-- meaningless. Requiring a minimum base makes the result actionable.
WITH anchor AS (
    SELECT MAX(date) AS max_date FROM v_sales
),
windowed AS (
    SELECT
        v.item_id,
        v.store_id,
        v.cat_id,
        SUM(v.sales) FILTER (
            WHERE v.date > a.max_date - INTERVAL 28 DAY
        )                                                       AS units_last_28d,
        SUM(v.sales) FILTER (
            WHERE v.date > a.max_date - INTERVAL 56 DAY
              AND v.date <= a.max_date - INTERVAL 28 DAY
        )                                                       AS units_prior_28d
    FROM v_sales v
    CROSS JOIN anchor a
    WHERE v.date > a.max_date - INTERVAL 56 DAY
    GROUP BY 1, 2, 3
)
SELECT
    item_id,
    store_id,
    cat_id,
    units_prior_28d,
    units_last_28d,
    units_last_28d - units_prior_28d                            AS units_delta,
    ROUND(100.0 * (units_last_28d - units_prior_28d)
          / NULLIF(units_prior_28d, 0), 1)                      AS change_pct
FROM windowed
WHERE units_prior_28d >= $min_prior_units
  AND units_last_28d < units_prior_28d
ORDER BY change_pct ASC, units_prior_28d DESC
LIMIT $top_n;


-- name: product_volatility
-- params: min_total_units, top_n
-- Coefficient of variation of daily observed sales.
--
-- IMPORTANT CAVEAT: high CV here conflates genuine demand variability with
-- intermittency (long zero runs inflate the standard deviation) and with
-- unobservable stockouts. Phase 5 separates these using ADI/CV-squared on
-- non-zero demand sizes; this query is a coarse screen, not a segmentation.
SELECT
    item_id,
    store_id,
    cat_id,
    total_units,
    ROUND(mean_daily_units, 4)                    AS mean_daily_units,
    ROUND(sd_daily_units, 4)                      AS sd_daily_units,
    ROUND(cv_daily_units, 3)                      AS cv_daily_units,
    ROUND(zero_share, 3)                          AS zero_share,
    NTILE(4) OVER (ORDER BY cv_daily_units)       AS volatility_quartile
FROM v_series_summary
WHERE total_units >= $min_total_units
  AND cv_daily_units IS NOT NULL
ORDER BY cv_daily_units DESC
LIMIT $top_n;


-- name: persistent_low_demand
-- params: max_mean_daily, min_zero_share
-- Series that sell almost nothing almost all the time. These are the
-- assortment-rationalisation candidates, and they are also where forecasting
-- will be hardest (Phase 5).
SELECT
    store_id,
    cat_id,
    COUNT(*)                                  AS n_series,
    ROUND(AVG(mean_daily_units), 4)           AS avg_mean_daily,
    ROUND(AVG(zero_share), 3)                 AS avg_zero_share,
    SUM(total_units)                          AS units_contributed,
    ROUND(100.0 * SUM(total_units)
          / SUM(SUM(total_units)) OVER (), 3) AS pct_of_all_units
FROM v_series_summary
WHERE mean_daily_units <= $max_mean_daily
  AND zero_share >= $min_zero_share
GROUP BY store_id, cat_id
ORDER BY n_series DESC;


-- name: abc_classification
-- Pareto structure of the assortment using a cumulative-share window.
-- Feeds Phase 6: A-items justify tighter service levels than C-items.
WITH ranked AS (
    SELECT
        item_id,
        store_id,
        cat_id,
        total_units,
        SUM(total_units) OVER (
            ORDER BY total_units DESC, item_id, store_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                     AS cum_units,
        SUM(total_units) OVER ()              AS grand_total
    FROM v_series_summary
),
classed AS (
    SELECT
        item_id, store_id, cat_id, total_units,
        1.0 * cum_units / grand_total AS cum_share,
        CASE
            WHEN 1.0 * cum_units / grand_total <= 0.80 THEN 'A'
            WHEN 1.0 * cum_units / grand_total <= 0.95 THEN 'B'
            ELSE 'C'
        END AS abc_class
    FROM ranked
)
SELECT
    abc_class,
    COUNT(*)                                          AS n_series,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_series,
    SUM(total_units)                                  AS total_units,
    ROUND(100.0 * SUM(total_units)
          / SUM(SUM(total_units)) OVER (), 2)         AS pct_of_units
FROM classed
GROUP BY abc_class
ORDER BY abc_class;
