-- =============================================================================
-- 03 - STORE ANALYSIS
-- Store performance, growth and peer-relative position.
--
-- All period comparisons are anchored on MAX(date) in the data rather than a
-- hardcoded date, so these queries stay correct if the ingestion window moves.
-- =============================================================================


-- name: store_performance
-- Peer comparison via window aggregates: each store against the 3-store mean.
WITH totals AS (
    SELECT
        store_id,
        state_id,
        SUM(units)              AS total_units,
        SUM(revenue)            AS total_revenue,
        COUNT(DISTINCT date)    AS n_days,
        AVG(active_skus)        AS avg_active_skus
    FROM v_store_daily
    GROUP BY 1, 2
)
SELECT
    store_id,
    state_id,
    total_units,
    ROUND(total_revenue, 2)                              AS revenue_derived,
    ROUND(total_units * 1.0 / n_days, 1)                 AS avg_units_per_day,
    ROUND(avg_active_skus, 0)                            AS avg_active_skus,
    RANK() OVER (ORDER BY total_units DESC)              AS units_rank,
    ROUND(100.0 * total_units
          / SUM(total_units) OVER (), 2)                 AS pct_of_chain_units,
    ROUND(100.0 * (total_units - AVG(total_units) OVER ())
          / AVG(total_units) OVER (), 2)                 AS pct_vs_peer_mean
FROM totals
ORDER BY units_rank;


-- name: store_growth_28d
-- Trailing 28 days vs the immediately preceding 28 days.
--
-- Caveat for interpretation: this is a single 28-day pair, so it mixes trend
-- with seasonality. A store can look like it is "declining" purely because the
-- comparison window straddles a seasonal peak. Treat it as a flag to
-- investigate, not a conclusion.
WITH anchor AS (
    SELECT MAX(date) AS max_date FROM v_store_daily
),
windowed AS (
    SELECT
        d.store_id,
        d.state_id,
        SUM(d.units) FILTER (
            WHERE d.date > a.max_date - INTERVAL 28 DAY
        )                                                   AS units_last_28d,
        SUM(d.units) FILTER (
            WHERE d.date > a.max_date - INTERVAL 56 DAY
              AND d.date <= a.max_date - INTERVAL 28 DAY
        )                                                   AS units_prior_28d
    FROM v_store_daily d
    CROSS JOIN anchor a
    GROUP BY 1, 2
)
SELECT
    store_id,
    state_id,
    units_last_28d,
    units_prior_28d,
    units_last_28d - units_prior_28d                        AS units_delta,
    ROUND(100.0 * (units_last_28d - units_prior_28d)
          / NULLIF(units_prior_28d, 0), 2)                  AS growth_pct,
    RANK() OVER (ORDER BY (units_last_28d - units_prior_28d)
                          / NULLIF(units_prior_28d, 0) DESC) AS growth_rank
FROM windowed
ORDER BY growth_rank;


-- name: store_yearly_trend
-- Year-over-year using LAG. 2011 and 2016 are partial years in M5, so the
-- query reports days_observed alongside the totals; comparing a partial year
-- against a full one without that context would be misleading.
WITH yearly AS (
    SELECT
        store_id,
        year,
        SUM(sales)              AS units,
        COUNT(DISTINCT date)    AS days_observed
    FROM v_sales
    GROUP BY 1, 2
)
SELECT
    store_id,
    year,
    days_observed,
    units,
    ROUND(units * 1.0 / days_observed, 1)                       AS units_per_day,
    LAG(units) OVER (PARTITION BY store_id ORDER BY year)       AS prev_year_units,
    ROUND(100.0 * (units - LAG(units) OVER (PARTITION BY store_id ORDER BY year))
          / NULLIF(LAG(units) OVER (PARTITION BY store_id ORDER BY year), 0), 2)
                                                                AS yoy_growth_pct
FROM yearly
ORDER BY store_id, year;


-- name: store_category_strength
-- Which store-category cells over- or under-index relative to how that
-- category performs across the chain. Highlights genuine local strength
-- rather than just large categories.
WITH cell AS (
    SELECT store_id, cat_id, SUM(units) AS units
    FROM v_store_cat_daily
    GROUP BY 1, 2
),
shares AS (
    SELECT
        store_id,
        cat_id,
        units,
        1.0 * units / SUM(units) OVER (PARTITION BY store_id) AS share_in_store,
        1.0 * SUM(units) OVER (PARTITION BY cat_id)
              / SUM(units) OVER ()                            AS chain_share
    FROM cell
)
SELECT
    store_id,
    cat_id,
    units,
    ROUND(100.0 * share_in_store, 2)              AS pct_of_store,
    ROUND(100.0 * chain_share, 2)                 AS pct_chain_benchmark,
    ROUND(share_in_store / chain_share, 3)        AS index_vs_chain,
    CASE
        WHEN share_in_store / chain_share >= 1.10 THEN 'over-indexed'
        WHEN share_in_store / chain_share <= 0.90 THEN 'under-indexed'
        ELSE 'in line'
    END                                           AS assessment
FROM shares
ORDER BY store_id, index_vs_chain DESC;
