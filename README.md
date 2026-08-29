# AI-Powered Demand Forecasting & Intelligent Replenishment System

Retail demand forecasting on the M5 (Walmart) dataset, extended into a prescriptive replenishment engine and a tool-using analytics agent.

**Forecast → Evaluate Risk → Recommend Action → Explain Decision**

---

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | Data foundation, ingestion, quality checks | **Code complete — awaiting M5 download** |
| 2 | SQL layer & business analytics | **Code complete — run `python -m src.sql_runner`** |
| 3 | EDA & statistics | **Code complete — run `python -m src.phase3_analysis`** |
| 4 | Backtesting framework | **Code complete — run `python -m src.backtest`** |
| 5 | Forecasting models & error benchmark | **Code complete — XGBoost selected (WAPE 0.776)** |
| 6 | Production forecasting & replenishment | **Code complete — run `python -m src.phase6_run`** |
| 7 | Retail intelligence interface | **Code complete — run `python -m src.phase7_interface`** |
| 8 | Productionisation & reproducibility | **Code complete — Architecture and Documentation finalized** |

---

## Setup & Reproducibility

This project is built to be fully reproducible on standard hardware without relying on cloud clusters or GPUs. For full instructions, refer to the [Reproducibility Guide](reports/phase8/reproducibility.md).

```bash
git clone https://github.com/Ved0705/AI-Demand-Forecasting-Replenishment.git
cd AI-Demand-Forecasting-Replenishment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset
The M5 dataset is on Kaggle and must be downloaded manually:
```bash
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/
```

Verify that `data/raw/` contains `calendar.csv`, `sales_train_evaluation.csv`, and `sell_prices.csv`.

---

## Pipeline Architecture

The system is constructed as a strict unidirectional pipeline, processing 9,147 item-store time series across 3 Walmart stores (CA_1, TX_1, WI_1) over 5 years. For the complete diagram, see [Final Architecture](reports/phase8/final_architecture.md).

### 1. Data Foundation & Analytics (Phases 1-3)
*   **Ingestion (`src/ingest.py`)**: Transforms 6.8 million price records and 30,000 daily sales series into an analytical Star Schema powered by embedded DuckDB and partitioned Parquet files.
*   **SQL Analytics (`src/sql_runner.py`)**: Computes analytical views and reconciles KPIs directly inside DuckDB.
*   **EDA (`src/phase3_analysis.py`)**: Segments series into Smooth, Lumpy, Erratic, and Intermittent buckets to prevent standard metrics from masking intermittent demand characteristics.

### 2. Leakage-Safe Backtesting & Modeling (Phases 4-5)
*   **Backtesting Framework (`src/backtest.py`)**: Structurally enforces an expanding-window rolling-origin evaluation across 5 folds and a 28-day horizon. Evaluates classic baselines (MA28, Seasonal Naive, Naive, Zero).
*   **Model Benchmark (`src/phase5_run.py`)**: Introduces per-series Exponential Smoothing (TSB, SES, Croston, SBA) and a global XGBoost model.
*   **Model Selection**: XGBoost narrowly outperformed TSB and SES, achieving a WAPE of **0.776962** across all folds. It was selected as the production model due to its cross-learning capabilities on calendar events and operational simplicity (managing 1 model vs 9,147). See [Project Summary](reports/phase8/project_summary.md) for detailed segment-level breakdowns.

### 3. Production Forecasting & Replenishment (Phases 6)
*   **Orchestration (`src/phase6_run.py`)**: Trains the persisted XGBoost artifact and generates point forecasts for the target horizon.
*   **Replenishment Logic (`src/replenishment.py`)**: Simulates safety-stock boundaries and reorder points based on configurable lead-time targets. Flags high-risk stockouts and outliers.

### 4. Retail Intelligence Interface (Phase 7)
*   **Presentation Layer (`src/phase7_interface.py`)**: A lightweight JSON CLI that queries Phase 6 artifacts without retraining models or loading heavy dataframes. It maps point forecasts and business thresholds into human-readable explanations.

```bash
python -m src.phase7_interface forecast CA_1 FOODS_3_090
python -m src.phase7_interface replenishment CA_1 FOODS_3_090
python -m src.phase7_interface summary
python -m src.phase7_interface risk --top 5
```

---

## Testing

The project maintains a rigorous automated test suite covering analytical data quality, feature leakage prevention, and model equivalence.

```bash
pytest tests/ -q
```
*Current Status: 303 tests passing in ~23 seconds.*

To perform a fast, end-to-end pipeline verification without touching the multi-gigabyte M5 dataset:
```bash
python -m src.phase6_run --fixture --mode replenishment
python -m src.phase7_interface --fixture summary
```

---

## Known Limitations & Business Assumptions

*   **Observed Sales vs True Demand**: The M5 dataset records units sold. It does not record missed sales due to stockouts. Our models therefore predict observed sales, which may underestimate true unconstrained demand.
*   **Simulated Inventory**: The M5 dataset has no on-hand inventory levels or supplier lead times. To build the prescriptive replenishment engine, inventory was explicitly simulated via assumptions defined in `config.yaml`. Output from the replenishment layer should be interpreted as a simulated scenario analysis. We do not claim measured inventory savings or a production deployment.
*   **Decision Logging**: Every non-obvious architecture or modeling decision was rigorously logged with alternatives considered. See [`DECISION_LOG.md`](DECISION_LOG.md) for full defensibility.
