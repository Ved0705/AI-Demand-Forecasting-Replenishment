# AI-Powered Demand Forecasting & Intelligent Replenishment System

Retail demand forecasting on the M5 (Walmart) dataset, extended into a
prescriptive replenishment engine and a tool-using analytics agent.

**Forecast → Evaluate Risk → Recommend Action → Explain Decision**

---

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | Data foundation, ingestion, quality checks | **Code complete — awaiting M5 download** |
| 2 | SQL layer & business analytics | **Code complete — run `python -m src.sql_runner`** |
| 3 | EDA & statistics | **Code complete — run `python -m src.phase3_analysis`** |
| 4 | Backtesting framework | Contract fixed in config (D-007) |
| 5 | Forecasting models & error analysis | Not started |
| 6 | Risk & replenishment engine | Assumptions recorded (D-004) |
| 7 | Retail analytics agent | Not started |
| 8 | Productionisation & packaging | Not started |

---

## Setup

```bash
pip install -r requirements.txt
```

### Get the data

The M5 dataset is on Kaggle and must be downloaded manually:

```bash
# https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
# or, with the Kaggle CLI:
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/
```

`data/raw/` should then contain:

```
calendar.csv                  1,969 rows   — dates, events, SNAP flags
sales_train_evaluation.csv   30,490 rows   — d_1..d_1941 daily units
sell_prices.csv           ~6,841,121 rows  — weekly price per store-item
```

### Run ingestion

```bash
python -m src.ingest
```

Produces `data/processed/sales_long.parquet` and `data/processed/retail.duckdb`
(star schema: `fact_sales`, `dim_product`, `dim_store`, `dim_calendar`,
`dim_price`).

### Run tests

```bash
pytest tests/ -q
```

Tests run against a synthetic fixture (`src/make_fixture.py`) — no download
needed. See D-006 on why fixture data is never analysed.

### Run the SQL analytics layer

```bash
python -m src.sql_runner
```

Creates the analytical views, runs 12 reconciliation checks, and writes
`reports/phase2/`. Export is blocked if validation fails. KPI definitions are
in [`docs/sql_kpis.md`](docs/sql_kpis.md).

### Run the Phase 3 statistical analysis

```bash
python -m src.profile          # must run first: Phase 3 reuses its segmentation
python -m src.phase3_analysis
```

Writes `reports/phase3/` — EDA report, test register, forecasting implications
and figures.

---

## Ground rules

These are non-negotiable and are what the project is defending in an interview.

- **Real data only.** All analysis runs on observed M5 sales. Synthetic data
  exists solely to unit-test code paths.
- **No random train/test split.** Expanding-window backtesting, 28-day horizon,
  5 folds, fixed before any model was built (D-007).
- **Every feature respects the forecast cutoff.** Lags and rolling statistics
  are computed inside the training window only.
- **Inventory is explicitly simulated.** M5 has no on-hand or lead-time data;
  every assumption is a config value, never presented as observation (D-004).
- **Intermittent demand gets real treatment.** Segmented and diagnosed rather
  than hidden inside one blended MAPE.
- **Every major decision is logged** with alternatives considered — see
  [`DECISION_LOG.md`](DECISION_LOG.md).

---

## Layout

```
config.yaml           all tunable parameters — nothing hardcoded in src/
DECISION_LOG.md       what / why / alternatives / limitations, per decision
docs/
  sql_kpis.md         exact KPI formulas; observed vs derived quantities
sql/
  00_views.sql        analytical views (incl. state-aware SNAP resolution)
  01_data_validation.sql   grain, fan-out and coverage checks
  02_sales_kpis.sql        volume & revenue KPIs
  03_store_analysis.sql    store performance, growth, peer comparison
  04_product_analysis.sql  rankings, decline, volatility, ABC
  05_demand_trends.sql     rolling windows and trends
  06_advanced_analytics.sql SNAP, events, price association
src/
  config.py           config loading and path resolution
  ingest.py           M5 → long format → parquet + DuckDB
  quality.py          data-quality checks that fail loudly
  profile.py          dataset profiling & demand segmentation
  sql_runner.py       named-query registry, validation, report export
  stats_utils.py      bootstrap, effect sizes, FDR correction
  phase3_analysis.py  EDA + statistical analysis + figures
  make_fixture.py     synthetic M5-shaped fixture (TESTS ONLY)
tests/                pytest suite
reports/              generated profiles and results
```

---

## Known limitations

- **Demand is censored.** An in-window zero may be a stockout rather than zero
  demand. M5 cannot distinguish these, so models estimate *observed sales*, not
  true demand.
- **Three stores, not ten.** Regional generalisation is limited by design
  (D-002); the pipeline scales to full M5 via config.
- **No real inventory data.** The replenishment layer is a simulation with
  declared assumptions (D-004).
