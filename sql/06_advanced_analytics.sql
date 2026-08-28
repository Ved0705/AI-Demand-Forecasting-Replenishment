-- =============================================================================
-- 06 - ADVANCED ANALYTICS
-- SNAP, events and price movement.
--
-- STATISTICAL CAUTION (applies to this entire file): every result here is an
-- ASSOCIATION measured on observational data. None of it is causal.
--
-- Concretely: SNAP days fall on the first ~10 days of the month, so a
-- SNAP-vs-non-SNAP gap is entangled with payday cycles, day-of-week
-- composition and month-position effects. Event days coincide with holidays
-- that change store traffic for many reasons at once. Price changes are set
-- by the retailer in response to expected demand, which is reverse causality.
--
-- Column names say "on SNAP days", never "because of SNAP". Phase 3 handles
-- the inferential question; this file only describes what was observed.
-- =============================================================================


-- name: snap_day_comparison
-- Uses the state-aware snap_active flag from v_sales (see 00_views.sql). A
-- naive query reading snap_CA for all stores would compare Wisconsin stores
-- against California's SNAP calendar.
SELECT
    store_id,
    state_id,
    COUNT(DISTINCT date) FILTER (WHERE snap_active = 1)      AS snap_days,
    COUNT(DISTINCT date) FILTER (WHERE snap_active = 0)      AS non_snap_days,
    ROUND(SUM(sales) FILTER (WHERE snap_active = 1) * 1.0
          / NULLIF(COUNT(DISTINCT date)
                   FILTER (WHERE snap_active = 1), 0), 1)    AS units_per_snap_day,
    ROUND(SUM(sales) FILTER (WHERE snap_active = 0) * 1.0
          / NULLIF(COUNT(DISTINCT date)
                   FILTER (WHERE snap_active = 0), 0), 1)    AS units_per_non_snap_day,
    ROUND(100.0 * (
        SUM(sales) FILTER (WHERE snap_active = 1) * 1.0
        / NULLIF(COUNT(DISTINCT date) FILTER (WHERE snap_active = 1), 0)
      - SUM(sales) FILTER (WHERE snap_active = 0) * 1.0
        / NULLIF(COUNT(DISTINCT date) FILTER (WHERE snap_active = 0), 0)
    ) / NULLIF(SUM(sales) FILTER (WHERE snap_active = 0) * 1.0
        / NULLIF(COUNT(DISTINCT date) FILTER (WHERE snap_active = 0), 0), 0), 1)
                                                             AS observed_gap_pct
FROM v_sales
GROUP BY store_id, state_id
ORDER BY store_id;


-- name: snap_by_category
-- FOODS is SNAP-eligible; HOBBIES largely is not. If the observed gap is
-- concentrated in FOODS that is at least consistent with a SNAP-related
-- mechanism, but this query cannot establish one.
SELECT
    cat_id,
    state_id,
    ROUND(AVG(sales) FILTER (WHERE snap_active = 1), 4)  AS mean_units_snap,
    ROUND(AVG(sales) FILTER (WHERE snap_active = 0), 4)  AS mean_units_non_snap,
    ROUND(100.0 * (AVG(sales) FILTER (WHERE snap_active = 1)
                 - AVG(sales) FILTER (WHERE snap_active = 0))
          / NULLIF(AVG(sales) FILTER (WHERE snap_active = 0), 0), 1)
                                                          AS observed_gap_pct
FROM v_sales
GROUP BY cat_id, state_id
ORDER BY cat_id, state_id;


-- name: event_day_comparison
-- Christmas is the notable case: M5 stores are closed, so those dates show
-- near-zero sales. That is a closure artefact, not depressed demand.
SELECT
    COALESCE(event_type_1, 'no_event')          AS event_type,
    COUNT(DISTINCT date)                        AS n_days,
    ROUND(AVG(sales), 4)                        AS mean_units_per_row,
    SUM(sales)                                  AS total_units,
    ROUND(100.0 * (AVG(sales)
          - (SELECT AVG(sales) FROM v_sales WHERE event_name_1 IS NULL))
          / NULLIF((SELECT AVG(sales) FROM v_sales
                    WHERE event_name_1 IS NULL), 0), 1)  AS observed_gap_vs_normal_pct
FROM v_sales
GROUP BY event_type
ORDER BY n_days DESC;


-- name: named_event_impact
-- params: min_occurrences
SELECT
    event_name_1                                AS event_name,
    event_type_1                                AS event_type,
    COUNT(DISTINCT date)                        AS n_occurrences,
    ROUND(AVG(sales), 4)                        AS mean_units_per_row,
    ROUND(100.0 * (AVG(sales)
          - (SELECT AVG(sales) FROM v_sales WHERE event_name_1 IS NULL))
          / NULLIF((SELECT AVG(sales) FROM v_sales
                    WHERE event_name_1 IS NULL), 0), 1)  AS observed_gap_vs_normal_pct
FROM v_sales
WHERE event_name_1 IS NOT NULL
GROUP BY event_name_1, event_type_1
HAVING COUNT(DISTINCT date) >= $min_occurrences
ORDER BY observed_gap_vs_normal_pct DESC;


-- name: price_change_association
-- Weekly price changes vs the change in weekly units for the same series.
--
-- This is NOT a price elasticity estimate. Retailers cut prices when they
-- expect or observe weak demand and raise them on strong sellers, so causality
-- runs in both directions. Reporting this as elasticity would be a serious
-- error; it is a descriptive co-movement.
WITH weekly AS (
    SELECT
        item_id, store_id, cat_id, wm_yr_wk,
        SUM(sales)      AS units,
        AVG(sell_price) AS price
    FROM v_sales
    WHERE sell_price IS NOT NULL
    GROUP BY 1, 2, 3, 4
),
changes AS (
    SELECT
        item_id, store_id, cat_id, wm_yr_wk, units, price,
        LAG(price) OVER (PARTITION BY item_id, store_id ORDER BY wm_yr_wk) AS prev_price,
        LAG(units) OVER (PARTITION BY item_id, store_id ORDER BY wm_yr_wk) AS prev_units
    FROM weekly
),
classified AS (
    SELECT
        cat_id,
        CASE
            WHEN price < prev_price * 0.98 THEN 'price_decreased'
            WHEN price > prev_price * 1.02 THEN 'price_increased'
            ELSE 'price_stable'
        END                                                     AS price_move,
        units,
        prev_units,
        units - prev_units                                      AS unit_delta
    FROM changes
    WHERE prev_price IS NOT NULL AND prev_units IS NOT NULL
)
SELECT
    cat_id,
    price_move,
    COUNT(*)                                        AS n_week_transitions,
    ROUND(AVG(prev_units), 2)                       AS mean_units_before,
    ROUND(AVG(units), 2)                            AS mean_units_after,
    ROUND(AVG(unit_delta), 3)                       AS mean_unit_change,
    ROUND(100.0 * AVG(unit_delta)
          / NULLIF(AVG(prev_units), 0), 1)          AS mean_pct_change
FROM classified
GROUP BY cat_id, price_move
ORDER BY cat_id, price_move;


-- name: series_activity_timeline
-- How the active assortment grew over time. Directly reflects D-003: series
-- enter the panel when they are first priced, so the count rises over time.
-- Useful context for any trend read, since chain-level growth partly reflects
-- assortment expansion rather than per-item demand growth.
WITH bounds AS (
    SELECT item_id, store_id, MIN(date) AS first_active
    FROM v_sales GROUP BY 1, 2
)
SELECT
    DATE_TRUNC('month', first_active)   AS month_started,
    COUNT(*)                            AS series_activated,
    SUM(COUNT(*)) OVER (
        ORDER BY DATE_TRUNC('month', first_active)
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                   AS cumulative_active_series
FROM bounds
GROUP BY 1
ORDER BY 1;
