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
| 4 | Backtesting framework | **Code complete — run `python -m src.backtest`** |
| 5 | Forecasting models & error analysis | **Code complete — XGBoost selected, see `reports/phase5/model_selection.md`** |
| 6 | Production forecasting & replenishment decision engine | **Code complete — run `python -m src.phase6_run`** |
| 7 | Retail intelligence interface (CLI over Phase 6) | **Code complete — run `python -m src.phase7_interface`** |
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

### Run the Phase 4 backtest

```bash
python -m src.profile          # if not already run
python -m src.backtest         # rolling-origin baseline evaluation
```

Generates `reports/phase4/` — fold metadata, per-method metrics by segment,
store, category, and horizon day.  All five baselines (naive, seasonal naive,
MA7, MA28, zero) are evaluated against 5 expanding-window folds of the real
M5 data.  Leakage prevention is structurally enforced and adversarially tested
in `tests/test_leakage.py`.

### Run the Phase 5 forecasting benchmark

```bash
python -m src.profile          # if not already run
python -m src.phase5_run       # Croston/SBA/TSB/SES + global XGBoost vs Phase 4 baselines
```

Generates `reports/phase5/` — model comparison by fold/segment/store/category/
horizon, plus `model_selection.md`, which selects XGBoost as the primary
forecasting engine (WAPE 0.776962 across 9,147 series / 5 folds). Leakage is
adversarially tested end-to-end in `tests/test_leakage_phase5.py`.

### Run Phase 6 — production forecasting & replenishment

```bash
python -m src.phase6_run --mode forecast-only        # forecast + risk only, no inventory assumption
python -m src.phase6_run --mode replenishment         # + simulated or supplied inventory decisions
python -m src.phase6_run --mode replenishment --inventory-csv path/to/inventory.csv
python -m src.phase6_run --fixture                    # fast fixture smoke test
python -m src.phase6_run --subset 300                 # quick real-data run on 300 series
```

Trains (once) and persists the Phase 5 `GlobalXGBoostForecaster` architecture
unmodified (`models/phase6_xgboost_production.json`) with an explicit training
cutoff, forecasts the next `backtest.horizon_days` days, and — in
`replenishment` mode — turns those forecasts into auditable per-series
recommendations (safety stock, reorder point, recommended order quantity)
using config-driven business assumptions (`replenishment:` / `phase6:` in
`config.yaml`). Forecasting (`src/phase6_run.py`) and the replenishment
decision engine (`src/replenishment.py`) are kept in separate modules.

**M5 has no real inventory data** — replenishment output is always either
user-supplied or explicitly labelled as a simulation/scenario, never a
measured outcome. See `reports/phase6/phase6_forecasting.md` and
DECISION_LOG D-027..D-030 for the full methodology and limitations.

### Query the system — Phase 7 interface

A read-only CLI over the Phase 6 outputs above — never retrains a model,
never reloads the full dataset:

```bash
python -m src.phase7_interface health
python -m src.phase7_interface metadata
python -m src.phase7_interface forecast CA_1 FOODS_1_001
python -m src.phase7_interface replenishment CA_1 FOODS_1_001
python -m src.phase7_interface risk --top 20 --store CA_1
python -m src.phase7_interface summary
python -m src.phase7_interface --fixture summary          # query the fixture run instead
```

Every command prints JSON and distinguishes FORECAST-ONLY from
REPLENISHMENT-SIMULATION responses, explains each recommendation in business
language (never claiming the model itself predicts a stockout), and traces
every forecast to a model name, training cutoff, and Phase 5 selection
metric. See `reports/phase7/phase7_interface.md` and DECISION_LOG
D-031..D-034 for the full interface contract and its known limitations.

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
  backtest.py          Phase 4 expanding-window backtesting + baselines
  features.py          Phase 4/5/6 leakage-safe feature engineering
  forecasting_models.py Phase 5 models (Croston/SBA/TSB/SES + GlobalXGBoostForecaster)
  model_runner.py       Phase 5 benchmark orchestration
  phase5_run.py         Phase 5 CLI entrypoint
  replenishment.py      Phase 6 replenishment DECISION layer (no model code)
  phase6_run.py         Phase 6 production forecast + CLI orchestration
  phase7_interface.py   Phase 7 read-only CLI over Phase 6 outputs
  make_fixture.py     synthetic M5-shaped fixture (TESTS ONLY)
tests/                pytest suite
models/               persisted Phase 6 production model artifact + metadata
reports/              generated profiles and results (incl. phase6/, phase7/)
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
