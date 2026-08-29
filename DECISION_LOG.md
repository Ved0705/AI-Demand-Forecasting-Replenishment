# Decision Log

Every non-obvious technical choice, recorded when it was made. Format:
**what / why / alternatives / why rejected / assumptions / limitations / how to defend it.**

The purpose is interview defensibility. Reconstructing reasoning three weeks
later is harder than it sounds, and "why did you choose X?" is asked far more
often than "what is X?".

---

## D-001 — DuckDB as the SQL engine

**Chose.** DuckDB, embedded, with a star schema persisted to a single file.

**Why.** The JD asks for advanced SQL. DuckDB runs real analytical SQL (CTEs,
window functions, QUALIFY) over 17M+ rows on a laptop with no server to
install, and reads Parquet directly. The project stays a `git clone` away from
reproducible.

**Alternatives.** PostgreSQL; SQLite; pandas only.

**Rejected because.** Postgres adds setup friction and a service dependency for
zero analytical gain at this scale. SQLite's window-function and analytical
performance is materially weaker. Doing it all in pandas would mean the SQL
requirement is met by nothing at all.

**Assumptions.** Single-user, single-machine, read-heavy workload.

**Limitations.** Not a concurrent multi-user warehouse. No row-level security,
no orchestration.

**Defend it as.** "The SQL is genuinely analytical rather than a syntax demo.
DuckDB is the right engine for a single-node analytical workload; the queries
themselves are standard and would move to Snowflake or BigQuery unchanged,
which matters more than the engine choice."

---

## D-002 — Subset: 3 stores, one per state, full assortment

**Chose.** `CA_1`, `TX_1`, `WI_1` — all items, all categories, all dates.
~9,100 series, ~17.7M rows. Configurable in `config.yaml`.

**Why.** Three constraints had to hold simultaneously:
- **SNAP variation is preserved.** M5's SNAP flags are per-state
  (`snap_CA/TX/WI`) and fire on different days. Sampling within one state
  would destroy a real demand driver.
- **All demand regimes are preserved.** FOODS is fast-moving; HOBBIES is
  heavily intermittent. Keeping the full assortment is what makes the
  segmentation and Croston work in Phase 5 possible.
- **It fits in memory.** Full M5 is ~59M rows.

**Alternatives.** Full 10 stores; one store; random SKU sample; single category.

**Rejected because.** Full M5 costs hours per experiment for no methodological
gain — the pipeline is written to scale to it by editing one config line, which
is the part that matters. One store loses regional and SNAP variation. Random
SKU sampling breaks category-level and store-level aggregate analysis. A single
category would eliminate exactly the intermittent demand this project is
supposed to handle well.

**Assumptions.** Store-level demand behaviour in these three stores is broadly
representative of their states.

**Limitations.** Cross-store effects (cannibalisation, regional promotions) are
under-observed. Store-level conclusions do not generalise to all 10 stores.

**Defend it as.** "I subset on the store dimension rather than the product
dimension deliberately. Cutting products would have destroyed the intermittency
distribution, which is the hardest and most interesting part of the problem.
Cutting stores costs me some regional generalisation, and the pipeline runs on
the full dataset by changing one config value."

---

## D-003 — Leading zeros treated as unavailability, not demand

**Chose.** Per-series *active window*, starting at the first week the item has a
non-null `sell_price` in that store. Rows before it are flagged `is_active=False`
and excluded from training and evaluation by default.

**Why.** This is the single most consequential data decision in the project. In
M5 a zero is ambiguous: it can mean "nobody bought it" or "the store did not
stock it yet". M5 only publishes `sell_price` for weeks an item was actually on
sale in that store, so the first priced week is a clean availability signal.
Treating structural zeros as observed demand would (a) bias every forecast
downward, (b) inflate ADI and make ordinary items look intermittent, corrupting
the segmentation, and (c) reward a model that predicts zero.

**Alternatives.** First non-zero sale as the start; no filtering; imputation.

**Rejected because.** First-sale anchoring throws away genuine early zero-demand
days and is circular — it defines availability using the target. No filtering
keeps a known bias. Imputation invents demand that was never observable.

**Assumptions.** Price presence is a reliable proxy for shelf availability. Once
active, a series stays active (no mid-life delisting is modelled).

**Limitations.** Mid-series stockouts are still invisible — a zero during the
active window may be a stockout rather than zero demand, which means observed
demand is censored below true demand. This is a known and unsolved issue in the
M5 data and should be stated, not papered over.

**Defend it as.** "Leading zeros in M5 are structural, not behavioural. I anchor
availability on the first priced week because price presence is exogenous to the
target, unlike first-sale anchoring which is circular. The residual limitation
is censoring — I can't distinguish an in-window zero from a stockout — so my
forecasts estimate observed sales, not true demand."

---

## D-004 — Inventory state will be simulated and labelled as such

**Chose.** Recorded now, implemented in Phase 6. On-hand inventory, lead time
and service level are **assumptions declared in `config.yaml`**, never presented
as M5 observations.

**Why.** M5 contains no inventory or lead-time data. The replenishment engine
needs an inventory position to produce an order quantity. Simulating it is a
legitimate engineering choice; presenting simulated values as observed data is
not.

**Alternatives.** Drop replenishment entirely; find a dataset with real
inventory; leave assumptions implicit.

**Rejected because.** Dropping replenishment removes the prescriptive layer that
distinguishes this project from a standard forecasting notebook. Public
retail datasets with real on-hand inventory and lead times are effectively
unavailable. Implicit assumptions are the version that gets found in an
interview rather than disclosed.

**Assumptions.** Fixed 7-day lead time, 95% service level, weekly review,
14 days of initial cover. All in config, all varied in sensitivity analysis.

**Limitations.** Results are directionally meaningful, not a validated inventory
outcome. No supplier variability, no MOQs, no shelf capacity.

**Defend it as.** "The forecasting is on real observed data. The inventory layer
is explicitly simulated because M5 has no on-hand data, and I state that up
front rather than letting it be discovered. Every assumption is a config value
and I ran sensitivity analysis on lead time and service level."

---

## D-005 — Reshape to long format once, at ingestion

**Chose.** Melt `d_1..d_1941` into one row per (item, store, date) during
ingestion; persist as Parquet.

**Why.** Every downstream stage — lag features, temporal splitting, backtesting,
segmentation — requires long form. Doing it once means no later stage carries
reshape logic, and the reshape is validated by a test asserting total units are
conserved.

**Alternatives.** Keep wide and reshape per stage; reshape lazily in SQL.

**Rejected because.** Repeated reshaping is the kind of duplicated logic where
an off-by-one date mapping enters and silently shifts the whole series.

**Assumptions.** The long table fits in memory at the chosen subset size.

**Limitations.** At full-M5 scale this would need chunking or a pure-SQL path.

**Defend it as.** "One reshape, validated by a conservation test. Date mapping
bugs in time series are silent and catastrophic, so I asserted total units are
preserved rather than eyeballing the output."

---

## D-006 — Synthetic fixture for tests only, never for analysis

**Chose.** `src/make_fixture.py` generates M5-*shaped* files in `data/fixture/`
for unit tests. All analysis runs on the real M5 download in `data/raw/`.

**Why.** The pipeline needs testing without a 450MB dependency in CI, and tests
need deterministic edge cases (staggered item introductions, intermittent
series, duplicate price keys). Testing code against synthetic data is standard
practice.

**Alternatives.** Test on real M5; no tests.

**Rejected because.** Real M5 makes tests slow and undownloadable in a sandbox.
No tests means leakage and join bugs surface as confusing model results.

**Assumptions.** The fixture reproduces M5's *schema and pathologies*, not its
statistics.

**Limitations.** Passing tests prove the code is correct, not that the modelling
choices suit real M5 demand.

**Defend it as.** "The fixture tests code paths — reshape conservation, join
fan-out, availability logic. I never compute a statistic from it, because you'd
only be recovering the relationships you wrote into the generator. Every number
in the README comes from observed M5 sales."

---

## D-007 — Backtest contract defined before any model is built

**Chose.** Horizon, folds, scheme and minimum training window are fixed in
`config.yaml` *now*, in Phase 1, before a single model exists.

**Why.** If the evaluation protocol is defined after seeing model results, it
gets tuned — consciously or not — until the favoured model wins. Fixing it
first makes every model comparable under identical conditions and makes the
comparison credible.

**Assumptions.** 28-day horizon and 5 non-overlapping folds give enough
evaluation coverage; ≥730 training days ensures every fold sees full yearly
seasonality.

**Limitations.** Five folds is a modest sample for variance estimates.

**Defend it as.** "I wrote the evaluation harness before the models, so no model
got a protocol tuned in its favour. Every result in the comparison table came
from the same five folds and the same feature cutoffs."

---

## D-008 — Undefined demand statistics are `unknown`, not a real segment

**Chose.** Series whose ADI or CV² is undefined are classified `unknown` and
excluded from the four Syntetos-Boylan quadrants.

**Why.** Found by a test, not by reading the code. ADI is undefined for a series
that never sells (division by zero non-zero days); CV² is undefined for a series
with a single sale (sample standard deviation of one observation). In NumPy,
`NaN >= threshold` is `False`, so both "is it high?" comparisons returned False
and the series satisfied the *low-ADI, low-CV²* condition — landing in `smooth`,
the segment for dense, stable, easy-to-forecast demand. The precise opposite of
the truth.

**Impact if unfixed.** The rarest, hardest series would have been silently
pooled with the easiest ones. Segment-level error metrics in Phase 5 would be
contaminated, and the headline claim — "I treat intermittent demand separately"
— would have been false in exactly the cases that matter most.

**Alternatives.** Impute CV²=0 for single-sale series; drop undefined series
silently.

**Rejected because.** Imputing zero asserts *measured* stability where there is
no measurement — the same error in a different costume. Dropping silently hides
how much of the assortment is too sparse to classify, which is itself a finding.

**Assumptions.** A series needs ≥2 non-zero observations before its demand
variability means anything.

**Limitations.** `unknown` is a genuine gap, not a solved case. Those series
still need a forecasting policy in Phase 5 — most likely a conservative
constant or a pooled category-level estimate.

**Defend it as.** "A NaN-comparison bug was classifying never-selling series as
smooth. I caught it with an edge-case test, not by inspection. It's a good
example of why I test the boundary cases in segmentation logic — the failure
was silent and would have quietly invalidated my headline result about
intermittent demand."

---

## D-009 — Views over the existing schema; no rebuild

**Chose.** Keep the Phase 1 star schema exactly as ingested. Phase 2 adds four
analytical **views** on top and changes no table.

**Why.** The Phase 1 schema is already correct for retail analytics: a fact at
item × store × date with conformed product, store, calendar and price
dimensions. Rebuilding would invalidate the ingestion tests and the Phase 1
validation results for no analytical gain. Views keep one definition of
"revenue" and "SNAP active" rather than repeating the logic in every query.

**Alternatives.** Materialised summary tables; rebuild the schema; put the
logic in each query.

**Rejected because.** Materialising duplicates a 14M-row fact table for a
query set that runs in seconds; DuckDB pushes predicates through views anyway.
Repeating logic per query is how two queries end up with two definitions of
revenue and nobody notices.

**Assumptions.** Read-heavy, single-user workload. View overhead is negligible
at this scale.

**Limitations.** Views recompute on each call. If Phase 5 needs repeated heavy
scans, `v_series_summary` is the first candidate to materialise.

**Defend it as.** "I inspected the existing schema before changing anything and
concluded it was already right. Phase 2 is additive — four views — so no Phase 1
test or result was invalidated."

---

## D-010 — Aggregate before windowing

**Chose.** Rolling windows run over pre-aggregated views (`v_store_daily`,
`v_store_cat_daily`), never over the 14M-row fact table.

**Why.** A rolling 28-day window partitioned by series over 14M rows sorts and
frames ~9,100 partitions. The same window over ~5,900 store-days is instant,
and store- and category-level trend analysis is what these queries are for.

**Alternatives.** Window over the raw fact; materialise a series-level rolling
table.

**Rejected because.** Windowing the raw fact costs minutes for output nobody
reads at that grain. Series-level rolling features *are* needed — but as
**model features in Phase 4**, computed inside each fold's training boundary.
Building them here would create a second, leakage-unaware code path for the
same quantity, which is exactly how leakage enters a project.

**Assumptions.** Store and category grain suffices for descriptive trends.

**Limitations.** No series-level rolling metrics in the SQL layer, by design.

**Defend it as.** "Rolling features exist twice in most projects — once for
reporting, once for modelling — and the reporting version leaks into the model.
I kept them out of the SQL layer entirely so Phase 4 has exactly one
fold-aware implementation."

---

## D-011 — State-aware SNAP resolution in the view

**Chose.** Resolve `snap_active` once in `v_sales` via `CASE state_id`, guarded
by a test asserting it matches the store's own state column.

**Why.** M5 stores SNAP as three wide per-state flags (`snap_CA`, `snap_TX`,
`snap_WI`). The natural query — join the calendar, read `snap_CA` — compares
every store against California's SNAP calendar. The result is plausible-looking
and wrong, and nothing in the output signals it. Since our subset is one store
per state (D-002), the error would corrupt two of three stores.

**Alternatives.** Unpivot SNAP into a long table; resolve per query.

**Rejected because.** Unpivoting means another table to maintain and join.
Per-query resolution means every future query, including Phase 7's agent tools,
can get it wrong independently.

**Assumptions.** Store state determines the applicable SNAP calendar.

**Limitations.** The `CASE` enumerates CA/TX/WI. A fourth state yields `NULL` —
caught by `test_snap_flag_is_never_null` rather than passing silently.

**Defend it as.** "SNAP is per-state in M5's wide format. Reading one state's
column for all stores is the kind of bug that produces a believable number, so
I resolved it once in a view and wrote a test that compares the resolved flag
against the store's own state column."

---

## D-012 — Revenue is derived and NULL-safe

**Chose.** `revenue = sales × sell_price`, `NULL` when price is unknown.
Documented as derived, never observed.

**Why.** M5 records units and a weekly shelf price, never transaction value.
Revenue is useful for ranking and mix, but calling it revenue without
qualification overstates what the data supports. `NULL` rather than `0` means
unpriced units cannot silently depress a revenue total — a zero would look like
a real, low value.

**Alternatives.** Skip revenue; treat missing price as zero; impute price.

**Rejected because.** Units alone cannot compare a $0.50 and a $15 item.
Zero-filling produces silently wrong totals. Imputation invents prices.

**Assumptions.** Weekly shelf price approximates realised unit price.

**Limitations.** Ignores basket discounts, coupons, loyalty pricing, intra-week
markdowns. Not a financial figure.

**Defend it as.** "M5 has no transaction value, so revenue is explicitly a
derived proxy — units times weekly shelf price. I made it NULL-safe so unpriced
rows can't masquerade as zero-revenue rows, and `price_coverage` quantifies how
many rows that affects."

---

## D-013 — Named query registry instead of loose SQL files

**Chose.** Queries live in `sql/` as `-- name:` blocks, loaded into a registry
by `src/sql_runner.py` and called by name with DuckDB named parameters.

**Why.** Phase 7 needs the agent to call **controlled, parameterised** queries
rather than generating SQL. A registry loaded from disk is that interface, and
it means the queries the agent runs are the exact text reviewed and tested
here. It also makes "every query executes and reconciles" a testable property
rather than a claim.

**Alternatives.** Loose `.sql` files run by hand; SQL embedded in Python;
free-form text-to-SQL for the agent.

**Rejected because.** Hand-run files drift out of test coverage. Embedded SQL
is unreadable and un-lintable. Free-form generation was rejected in the project
plan — an LLM writing SQL against a schema it half-remembers produces
confident, wrong numbers, and there is no downstream check that catches it.

**Assumptions.** The query set is known ahead of time. Parameters cover the
variation that matters.

**Limitations.** Genuinely novel questions need a new named query, not an
ad-hoc one. That is the intended trade-off: determinism over flexibility.

**Defend it as.** "The SQL layer is the agent's tool interface, built in Phase
2 rather than retrofitted in Phase 7. Parameterised named queries mean the
agent selects and fills a reviewed query instead of authoring one, so it cannot
invent a join or silently change a metric definition."

---

## D-014 — Aggregates validated by independent recomputation

**Chose.** `validate()` recomputes headline figures a second way and compares:
view totals against fact totals, category and store and ABC sums against the
grand total, and per-state-date SNAP consistency.

**Why.** Column-presence checks pass on corrupted output. A JOIN that fans out
returns the right columns and the wrong numbers. The only check that catches it
is computing the number twice by different routes.

**Alternatives.** Column/type checks only; eyeball the output; no validation.

**Rejected because.** The failure mode being guarded against — silent
duplication through a dimension join — is invisible to every one of those.

**Assumptions.** Fact-table totals are ground truth for reconciliation.

**Limitations.** Confirms internal consistency, not that a metric is the right
metric for the business question.

**Defend it as.** "Every aggregate in the layer reconciles against the fact
table by an independent path, and report export is blocked if validation fails.
Join fan-out doesn't announce itself — it just gives you a bigger number."

---

## D-020 — Expanding-window backtesting

**Chose.** Expanding window (grow-forward): all folds share the same
`train_start`; only `train_end` advances by `step_days` per fold.

**Why.** In an expanding window, each fold is trained on strictly more data
than the previous one. This matches the production scenario: as time passes
the model has access to more history. A sliding window (fixed-width) is simpler
to reason about but makes earlier folds train on less data than production
models will ever have — it tests a suboptimal configuration.

**Alternatives.** Sliding window (fixed width); blocked time-series CV;
single train/test split.

**Rejected because.** A single split has high variance (unlucky fold). Blocked
CV was considered but adds complexity without benefit at this dataset size.
Sliding window was considered but is harder to justify for a model that will
always have access to the full history in production.

**Assumptions.** The data distribution does not shift so dramatically over the
evaluation period that early and late folds are measuring incomparable things.
M5 spans 2011-2016; some structural breaks exist (SNAP policy) but they are
captured in the calendar features.

**Limitations.** Expanding window estimates get noisier as folds diverge in
training size. At 5 folds the variance is meaningful; report fold-by-fold
numbers alongside the aggregate.

**Defend it as.** "Production models always have access to all historical data,
so every test fold should also. Expanding window is the honest emulation of
that — sliding window tests a deliberately impoverished version of the model."

---

## D-021 — 28-day forecast horizon

**Chose.** 28 days (4 weeks) as the primary forecast horizon.

**Why.** Standard replenishment cycles in grocery retail are 4 weeks. Shorter
horizons (7 days) would make even naive baselines look good. Longer horizons
(90 days) are dominated by uncertainty the model cannot reduce. 28 days is
the right commercial target for this problem and matches M5 competition practice.

**Alternatives.** 7-day, 14-day, 56-day, multi-horizon.

**Rejected because.** 7-day is too short to expose the weaknesses of naive
baselines. 56-day degrades gracefully into the demand signal becoming noise.
Multi-horizon evaluation is Phase 5 work, not Phase 4 infrastructure.

**Assumptions.** Replenishment decisions are made on a 4-week cycle.

**Defend it as.** "28 days matches the M5 competition horizon, the standard
retail replenishment cycle, and is short enough that calendar effects
(seasonality, SNAP) are meaningful but long enough that naive baselines fail."

---

## D-022 — Feature leakage architecture (slice-first)

**Chose.** Every function that computes historical features takes a `cutoff`
parameter and slices to `date <= cutoff` as its FIRST operation.

**Why.** The most common leakage bug in time-series ML is computing rolling
windows or lags over the full dataset and then subsetting. The window silently
reaches into the future. By slicing first, the operation is structurally
impossible: there are no future rows to read from.

**Alternatives.** Compute globally, then filter features that touch future rows;
use a flag column to mark training rows; trust the caller.

**Rejected because.** Post-hoc filtering is error-prone: a function that
computes mean over a grouped window will include future rows before the filter
reaches them. Trusting the caller is not testable. A flag column requires
discipline in every consumer — the slice-first approach requires discipline
in exactly one place (the feature function) and is testable by the adversarial
test in test_leakage.py.

**Assumptions.** Callers may accidentally pass the full dataset (including
future rows). The leakage guard must not rely on caller discipline.

**Defend it as.** "The adversarial test in test_leakage.py demonstrates this
empirically: we mutate post-cutoff values to 7777 and show that every feature
and forecast is bit-identical to the clean data. If leakage existed, those
tests would fail."

---

## D-023 — Baseline selection

**Chose.** Five baselines: naive, seasonal naive (7-day), MA7, MA28, zero.

**Why.** These cover the full difficulty spectrum for retail demand:
  - `naive` is the minimum reasonable forecast.
  - `seasonal_naive` captures the weekly demand cycle identified in Phase 3.
  - `ma7` and `ma28` smooth over noise at different scales.
  - `zero` is the maximum-pessimism baseline; any model that doesn't beat it
    for intermittent/lumpy series should not be deployed.

Phase 5 models must demonstrate improvement over ALL five baselines,
not just the weakest one.

**Alternatives.** Exponential smoothing (Holt-Winters), SARIMA, STL decomposition.

**Rejected because.** These are models, not baselines. They belong in Phase 5
where they can be properly evaluated and compared. A baseline is something
a junior analyst could implement in an afternoon.

**Defend it as.** "A model that doesn't beat MA28 on a 28-day horizon is not
worth deploying. The zero baseline is the right floor for intermittent series
— 91% of series are intermittent or lumpy."

---

## D-024 — Metric selection (WAPE/MAE primary, MAPE not primary)

**Chose.** Primary: MAE and WAPE. Secondary: RMSE, bias. Informational: sMAPE,
MAPE (where actuals > 0 only).

**Why.** Phase 3 established that 59.9% of series have mean zero-share > 50%.
MAPE is undefined when the actual is 0, and reporting it only over non-zero
rows produces a biased view of performance (you're optimizing for the easy part
of the distribution). WAPE avoids this by weighting by absolute demand volume:
high-volume series (which are also the commercially important ones) dominate
appropriately.

**Alternatives.** MAPE, sMAPE, pinball loss (for probabilistic forecasts).

**MAPE rejected as primary because.** Excludes the majority of rows for
intermittent/lumpy series, producing a metric that describes only the
non-zero demand subset. That subset is not randomly sampled from the full
distribution.

**sMAPE rejected as primary because.** The 2/(|y|+|ŷ|) denominator has
unintuitive behaviour near zero and is not standard in retail forecasting.

**Pinball loss.** Phase 5 work if probabilistic forecasts are added.

**Defend it as.** "WAPE is the industry standard for retail demand forecasting.
MAE is always defined. MAPE is in the output for completeness but the footnote
says clearly why it should not be used as a decision metric for this dataset."

---

## D-025 — Segment-level mandatory evaluation

**Chose.** Every metric table is broken out by Phase 1 segment (smooth /
erratic / intermittent / lumpy). The overall aggregate is never the sole
reported number.

**Why.** Phase 3 confirmed that smooth and intermittent series have completely
different demand profiles (Gini = 0.643; 59.9% mean zero-share). A model could
achieve a good overall MAE by performing well only on smooth (high-volume)
series while performing no better than zero on intermittent ones. Without
segment-level reporting, this failure is invisible.

**Alternatives.** Report overall only; report by category; report by store.

**All are reported** (overall, segment, store, category, horizon). But segment
is the most important because the segment defines the problem type.

**Defend it as.** "A single headline metric hides the model's failure mode.
The zero baseline achieves WAPE=1.0 for every series by construction. If the
model only beats zero for smooth series but matches it for intermittent series,
the overall WAPE might still look good while 91% of the series see no benefit."

---

## D-026 — Known-future feature policy (SNAP and calendar)

**Chose.** SNAP flags, day-of-week, month, year, week-of-year, is_weekend, and
event flags are classified as `known_future` features. They are never placed in
`historical_cols`.

**Why.** SNAP (Supplemental Nutrition Assistance Program) disbursement dates
are published by the US government in advance. They are known at forecast
creation time for any future date. Treating them as historical features is
technically wrong: it implies the feature is derived from past sales, which
would require carry-forward logic at inference time and invite leakage bugs.

Calendar and static metadata features (DOW, month, store_id, cat_id) are
always known for future dates and require no past sales to compute.

**The `FeatureSet` dataclass** makes this distinction explicit in code. Phase 5
model code must inspect `FeatureSet.known_future_cols` to confirm it is not
accidentally using historical features on the forecast horizon.

**Defend it as.** "In production, the model scores on future dates where y is
unknown. Known-future features are the only safe inputs at inference time beyond
the model's own carried-forward predictions. Marking them explicitly in the
FeatureSet prevents a class of leakage bugs from ever reaching production."

---

## D-027 — Production model reuses the Phase 5 XGBoost architecture unmodified

**Chose.** Phase 6 trains ONE global `GlobalXGBoostForecaster` (unmodified
class from `src/forecasting_models.py`) on all permitted data up to an
explicit training cutoff, and persists it (`models/phase6_xgboost_production.json`
+ a metadata sidecar) so subsequent runs reuse it instead of retraining.

**Why.** Phase 5 already ran a fair, leakage-safe, 5-fold comparison and
selected XGBoost (WAPE 0.776962, see `reports/phase5/model_selection.md`).
Retuning hyperparameters or changing the architecture for "production" would
silently invalidate that selection — a different model would be shipped than
the one that was actually evaluated. Persisting the trained model makes
repeated Phase 6 runs (forecast-only vs replenishment mode, different
inventory scenarios) cheap and reproducible without retraining.

**Alternatives.** Retrain from scratch on every run; tune hyperparameters for
production; train per-series models; use a different global architecture.

**Rejected because.** Per-series models were explicitly rejected in Phase 5
(9,147 models vs one). Hyperparameter tuning here would mean the shipped
model was never actually benchmarked against the baselines. Retraining every
run is wasteful and makes two runs on the same data non-comparable if the
random seed or environment drifts.

**Assumptions.** The persisted model is only reused if its saved
`xgboost_params` and `feature_cols` exactly match the current Phase 5
architecture (`src/phase6_run.py::train_or_load_production_model`);
otherwise it is retrained automatically and a warning is logged.

**Limitations.** The training cutoff is fixed at persist time. A model
trained on data through a given date is not automatically refreshed as new
sales data arrives — `--force-retrain` must be run explicitly (this is a
deliberate choice: silent retraining would make forecast provenance
un-auditable).

**Defend it as.** "I didn't get to invent a new production model — Phase 5
already ran the fair comparison. Phase 6 trains and persists exactly that
architecture once, with the training cutoff and full hyperparameter set
written into the report, so every forecast is traceable to a specific model
artifact."

---

## D-028 — Demand-uncertainty proxy: Phase 5 backtest RMSE by segment

**Chose.** Safety stock uses `z * sigma_daily * sqrt(lead_time_days)`, where
`sigma_daily` is the per-segment RMSE of the Phase 5 XGBoost backtest
(`reports/phase5/segment_model_comparison.csv`, averaged across the 5
out-of-sample folds) and `z` is derived from `replenishment.service_level`
via `scipy.stats.norm.ppf`.

**Why.** XGBoost's point predictions carry no native uncertainty estimate.
Fabricating a per-series confidence interval would overstate what the model
actually provides. The Phase 5 backtest already measured out-of-sample error
by segment across 5 non-overlapping folds — reusing it as a risk proxy is
honest about what it is (a historical error magnitude) and what it is not (a
calibrated prediction interval).

**Alternatives.** Quantile regression / prediction intervals from XGBoost;
bootstrap the training residuals; assume a fixed CV of demand; no
uncertainty at all (deterministic reorder point only).

**Rejected because.** Quantile regression means training additional models
per quantile — out of scope and would compete with the Phase 5 selection.
Bootstrapping training residuals (in-sample) understates true error, exactly
the mistake D-022's leakage architecture exists to prevent. No uncertainty at
all would silently order zero safety stock for every series, which is worse
than an honest proxy.

**Assumptions.** RMSE approximates the error standard deviation under the
(measured, not verified further) small-bias condition reported in
`reports/phase5/phase5_forecasting.md`. Segment-level granularity is coarse
— sigma is shared by ~thousands of series in the same segment.

**Limitations.** This is explicitly a RISK PROXY, not a calibrated interval.
No coverage/calibration study has been run. It uses the OUT-OF-SAMPLE Phase 5
backtest (never the actuals of the specific decision being scored), but it is
still a historical average, not a per-series or per-date estimate.

**Defend it as.** "I do not claim XGBoost gives me calibrated uncertainty. I
reused the one thing I actually measured out-of-sample — segment RMSE from
the Phase 5 backtest — as an ordering signal, and I say explicitly in the
report and the code that it is a proxy, not a validated interval."

---

## D-029 — Two explicit Phase 6 modes: forecast-only vs replenishment simulation

**Chose.** `src/phase6_run.py --mode forecast-only` produces forecasts and a
risk summary with no inventory assumption at all. `--mode replenishment`
additionally requires an inventory position — either supplied via
`--inventory-csv` (real data) or SIMULATED from
`replenishment.initial_inventory_days_of_cover` (config assumption) — and
every replenishment output is labelled as scenario/simulation.

**Why.** M5 has no real inventory, on-hand, or open-purchase-order data
(D-004). Collapsing "forecast" and "replenishment recommendation" into one
undifferentiated output would make it easy to present a simulated order
quantity as if it were a measured business outcome. Splitting the modes
forces every consumer of `--mode replenishment` output to confront that its
inventory input is either user-supplied or explicitly simulated.

**Alternatives.** One mode that always simulates inventory silently; require
real inventory data and refuse to run without it.

**Rejected because.** Silent simulation is exactly the kind of undisclosed
assumption D-004 was written to prevent. Refusing to run without real
inventory data would make Phase 6 undemonstrable against the only dataset
available (M5), which has none.

**Assumptions.** When inventory is simulated, `initial_inventory_days_of_cover`
(config, currently 14 days) times the forecast daily mean is a reasonable
starting position for demonstration purposes only.

**Limitations.** No stockout reduction, service-level improvement, or cost
saving is claimed anywhere in Phase 6 output — there is no real inventory
ground truth to measure those against.

**Defend it as.** "I built two modes on purpose so nobody could mistake a
simulated order quantity for a measured result. Forecast-only mode never
touches inventory assumptions at all; replenishment mode labels every row
with where its inventory number came from."

---

## D-030 — Genuinely future known-future features sourced from calendar.csv

**Chose.** When the production forecast horizon extends beyond the last date
present in `sales_long.parquet` (the normal case — the model forecasts the
28 days after the most recent sales date), known-future calendar columns for
those dates are filled from `data/raw/calendar.csv` directly, which M5
publishes 28 days beyond the last day of sales (through 2016-06-19, exactly
covering the D-021 horizon). If `calendar.csv` is unavailable the columns are
left NaN and a warning is logged — not fabricated.

**Why.** `sales_long.parquet` only contains rows for dates with a sales
observation, so `src/features.py::build_fold_features`'s known-future merge
(designed for Phase 4/5 backtesting, where forecast dates are always
historical test dates already present in the data) finds no calendar
columns for a genuinely future production forecast. Without this fallback,
production forecasting beyond the training data's last date would either
crash (missing feature columns) or silently train on all-NaN calendar
features.

**Alternatives.** Restrict Phase 6 to only forecasting within already-observed
historical dates (as Phase 5 does); hardcode/guess future calendar values;
require the caller to supply future calendar data.

**Rejected because.** Restricting to historical dates defeats the purpose of
a *production* forecasting layer. Guessing calendar values (SNAP, events)
would be a fabrication of exactly the kind D-004 warns against. calendar.csv
already contains the real published values for exactly this window, so using
it is not an assumption — it is the correct data source.

**Assumptions.** `calendar.csv`'s SNAP and event columns for the forecast
window are the actual published values, not placeholders (verified: M5's
calendar.csv spans 1,969 days = 1,941 sales days + 28 future days).

**Limitations.** If the forecast horizon ever exceeded calendar.csv's
28-day lookahead, or calendar.csv is not present in `data/raw/`, the
fallback leaves NaN — XGBoost handles NaN natively in tree splits (see
`src/forecasting_models.py`), so this degrades gracefully rather than
crashing, but accuracy for those rows is unverified.

**Defend it as.** "The 28-day M5 calendar lookahead isn't a coincidence — it
matches the competition's own forecast horizon. I use the real published
calendar for genuinely future dates rather than reusing Phase 5's
backtest-only feature path unchanged, and if that file isn't there I leave
the features as NaN and say so in the log rather than inventing values."
