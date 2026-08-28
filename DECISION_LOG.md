Decision Log
Every non-obvious technical choice, recorded when it was made. Format:
what / why / alternatives / why rejected / assumptions / limitations / how to defend it.
The purpose is interview defensibility. Reconstructing reasoning three weeks
later is harder than it sounds, and "why did you choose X?" is asked far more
often than "what is X?".
---
D-001 — DuckDB as the SQL engine
Chose. DuckDB, embedded, with a star schema persisted to a single file.
Why. The JD asks for advanced SQL. DuckDB runs real analytical SQL (CTEs,
window functions, QUALIFY) over 17M+ rows on a laptop with no server to
install, and reads Parquet directly. The project stays a `git clone` away from
reproducible.
Alternatives. PostgreSQL; SQLite; pandas only.
Rejected because. Postgres adds setup friction and a service dependency for
zero analytical gain at this scale. SQLite's window-function and analytical
performance is materially weaker. Doing it all in pandas would mean the SQL
requirement is met by nothing at all.
Assumptions. Single-user, single-machine, read-heavy workload.
Limitations. Not a concurrent multi-user warehouse. No row-level security,
no orchestration.
Defend it as. "The SQL is genuinely analytical rather than a syntax demo.
DuckDB is the right engine for a single-node analytical workload; the queries
themselves are standard and would move to Snowflake or BigQuery unchanged,
which matters more than the engine choice."
---
D-002 — Subset: 3 stores, one per state, full assortment
Chose. `CA_1`, `TX_1`, `WI_1` — all items, all categories, all dates.
~9,100 series, ~17.7M rows. Configurable in `config.yaml`.
Why. Three constraints had to hold simultaneously:
SNAP variation is preserved. M5's SNAP flags are per-state
(`snap_CA/TX/WI`) and fire on different days. Sampling within one state
would destroy a real demand driver.
All demand regimes are preserved. FOODS is fast-moving; HOBBIES is
heavily intermittent. Keeping the full assortment is what makes the
segmentation and Croston work in Phase 5 possible.
It fits in memory. Full M5 is ~59M rows.
Alternatives. Full 10 stores; one store; random SKU sample; single category.
Rejected because. Full M5 costs hours per experiment for no methodological
gain — the pipeline is written to scale to it by editing one config line, which
is the part that matters. One store loses regional and SNAP variation. Random
SKU sampling breaks category-level and store-level aggregate analysis. A single
category would eliminate exactly the intermittent demand this project is
supposed to handle well.
Assumptions. Store-level demand behaviour in these three stores is broadly
representative of their states.
Limitations. Cross-store effects (cannibalisation, regional promotions) are
under-observed. Store-level conclusions do not generalise to all 10 stores.
Defend it as. "I subset on the store dimension rather than the product
dimension deliberately. Cutting products would have destroyed the intermittency
distribution, which is the hardest and most interesting part of the problem.
Cutting stores costs me some regional generalisation, and the pipeline runs on
the full dataset by changing one config value."
---
D-003 — Leading zeros treated as unavailability, not demand
Chose. Per-series active window, starting at the first week the item has a
non-null `sell_price` in that store. Rows before it are flagged `is_active=False`
and excluded from training and evaluation by default.
Why. This is the single most consequential data decision in the project. In
M5 a zero is ambiguous: it can mean "nobody bought it" or "the store did not
stock it yet". M5 only publishes `sell_price` for weeks an item was actually on
sale in that store, so the first priced week is a clean availability signal.
Treating structural zeros as observed demand would (a) bias every forecast
downward, (b) inflate ADI and make ordinary items look intermittent, corrupting
the segmentation, and (c) reward a model that predicts zero.
Alternatives. First non-zero sale as the start; no filtering; imputation.
Rejected because. First-sale anchoring throws away genuine early zero-demand
days and is circular — it defines availability using the target. No filtering
keeps a known bias. Imputation invents demand that was never observable.
Assumptions. Price presence is a reliable proxy for shelf availability. Once
active, a series stays active (no mid-life delisting is modelled).
Limitations. Mid-series stockouts are still invisible — a zero during the
active window may be a stockout rather than zero demand, which means observed
demand is censored below true demand. This is a known and unsolved issue in the
M5 data and should be stated, not papered over.
Defend it as. "Leading zeros in M5 are structural, not behavioural. I anchor
availability on the first priced week because price presence is exogenous to the
target, unlike first-sale anchoring which is circular. The residual limitation
is censoring — I can't distinguish an in-window zero from a stockout — so my
forecasts estimate observed sales, not true demand."
---
D-004 — Inventory state will be simulated and labelled as such
Chose. Recorded now, implemented in Phase 6. On-hand inventory, lead time
and service level are assumptions declared in `config.yaml`, never presented
as M5 observations.
Why. M5 contains no inventory or lead-time data. The replenishment engine
needs an inventory position to produce an order quantity. Simulating it is a
legitimate engineering choice; presenting simulated values as observed data is
not.
Alternatives. Drop replenishment entirely; find a dataset with real
inventory; leave assumptions implicit.
Rejected because. Dropping replenishment removes the prescriptive layer that
distinguishes this project from a standard forecasting notebook. Public
retail datasets with real on-hand inventory and lead times are effectively
unavailable. Implicit assumptions are the version that gets found in an
interview rather than disclosed.
Assumptions. Fixed 7-day lead time, 95% service level, weekly review,
14 days of initial cover. All in config, all varied in sensitivity analysis.
Limitations. Results are directionally meaningful, not a validated inventory
outcome. No supplier variability, no MOQs, no shelf capacity.
Defend it as. "The forecasting is on real observed data. The inventory layer
is explicitly simulated because M5 has no on-hand data, and I state that up
front rather than letting it be discovered. Every assumption is a config value
and I ran sensitivity analysis on lead time and service level."
---
D-005 — Reshape to long format once, at ingestion
Chose. Melt `d_1..d_1941` into one row per (item, store, date) during
ingestion; persist as Parquet.
Why. Every downstream stage — lag features, temporal splitting, backtesting,
segmentation — requires long form. Doing it once means no later stage carries
reshape logic, and the reshape is validated by a test asserting total units are
conserved.
Alternatives. Keep wide and reshape per stage; reshape lazily in SQL.
Rejected because. Repeated reshaping is the kind of duplicated logic where
an off-by-one date mapping enters and silently shifts the whole series.
Assumptions. The long table fits in memory at the chosen subset size.
Limitations. At full-M5 scale this would need chunking or a pure-SQL path.
Defend it as. "One reshape, validated by a conservation test. Date mapping
bugs in time series are silent and catastrophic, so I asserted total units are
preserved rather than eyeballing the output."
---
D-006 — Synthetic fixture for tests only, never for analysis
Chose. `src/make_fixture.py` generates M5-shaped files in `data/fixture/`
for unit tests. All analysis runs on the real M5 download in `data/raw/`.
Why. The pipeline needs testing without a 450MB dependency in CI, and tests
need deterministic edge cases (staggered item introductions, intermittent
series, duplicate price keys). Testing code against synthetic data is standard
practice.
Alternatives. Test on real M5; no tests.
Rejected because. Real M5 makes tests slow and undownloadable in a sandbox.
No tests means leakage and join bugs surface as confusing model results.
Assumptions. The fixture reproduces M5's schema and pathologies, not its
statistics.
Limitations. Passing tests prove the code is correct, not that the modelling
choices suit real M5 demand.
Defend it as. "The fixture tests code paths — reshape conservation, join
fan-out, availability logic. I never compute a statistic from it, because you'd
only be recovering the relationships you wrote into the generator. Every number
in the README comes from observed M5 sales."
---
D-007 — Backtest contract defined before any model is built
Chose. Horizon, folds, scheme and minimum training window are fixed in
`config.yaml` now, in Phase 1, before a single model exists.
Why. If the evaluation protocol is defined after seeing model results, it
gets tuned — consciously or not — until the favoured model wins. Fixing it
first makes every model comparable under identical conditions and makes the
comparison credible.
Assumptions. 28-day horizon and 5 non-overlapping folds give enough
evaluation coverage; ≥730 training days ensures every fold sees full yearly
seasonality.
Limitations. Five folds is a modest sample for variance estimates.
Defend it as. "I wrote the evaluation harness before the models, so no model
got a protocol tuned in its favour. Every result in the comparison table came
from the same five folds and the same feature cutoffs."
---
D-008 — Undefined demand statistics are `unknown`, not a real segment
Chose. Series whose ADI or CV² is undefined are classified `unknown` and
excluded from the four Syntetos-Boylan quadrants.
Why. Found by a test, not by reading the code. ADI is undefined for a series
that never sells (division by zero non-zero days); CV² is undefined for a series
with a single sale (sample standard deviation of one observation). In NumPy,
`NaN >= threshold` is `False`, so both "is it high?" comparisons returned False
and the series satisfied the low-ADI, low-CV² condition — landing in `smooth`,
the segment for dense, stable, easy-to-forecast demand. The precise opposite of
the truth.
Impact if unfixed. The rarest, hardest series would have been silently
pooled with the easiest ones. Segment-level error metrics in Phase 5 would be
contaminated, and the headline claim — "I treat intermittent demand separately"
— would have been false in exactly the cases that matter most.
Alternatives. Impute CV²=0 for single-sale series; drop undefined series
silently.
Rejected because. Imputing zero asserts measured stability where there is
no measurement — the same error in a different costume. Dropping silently hides
how much of the assortment is too sparse to classify, which is itself a finding.
Assumptions. A series needs ≥2 non-zero observations before its demand
variability means anything.
Limitations. `unknown` is a genuine gap, not a solved case. Those series
still need a forecasting policy in Phase 5 — most likely a conservative
constant or a pooled category-level estimate.
Defend it as. "A NaN-comparison bug was classifying never-selling series as
smooth. I caught it with an edge-case test, not by inspection. It's a good
example of why I test the boundary cases in segmentation logic — the failure
was silent and would have quietly invalidated my headline result about
intermittent demand."