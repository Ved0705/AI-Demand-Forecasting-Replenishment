"""Phase 5 — Execute Forecasting Benchmark."""

import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import load_config
from src.model_runner import run_phase5_backtest, write_outputs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true", help="Run on fixture data")
    parser.add_argument("--subset", type=int, default=None, help="Run on N series")
    args = parser.parse_args()
    
    cfg = load_config()
    cfg.ensure_dirs()
    
    if args.fixture:
        # Load fixture data
        parquet = Path("data/fixture/processed/sales_long.parquet")
        # Ensure we have sensible backtest parameters for fixture (override config)
        cfg.raw["backtest"]["min_train_days"] = 30
        cfg.raw["backtest"]["step_days"] = 7
        cfg.raw["backtest"]["horizon_days"] = 7
        out_dir = Path("reports/fixture/phase5")
    else:
        parquet = cfg.path("processed") / "sales_long.parquet"
        out_dir = cfg.path("reports") / "phase5"
        
    if not parquet.exists():
        if args.fixture:
            raise SystemExit(
                "Fixture processed data not found. To generate it:\n"
                "1. Run `python -m src.make_fixture` to create raw fixture data.\n"
                "2. Run a custom script to ingest the raw fixture into `data/fixture/processed/sales_long.parquet` using `src.ingest.build_long_table`.\n"
                "Or run `pytest` which generates the fixture in-memory."
            )
        raise SystemExit(f"No processed data at {parquet}.")
        
    # Column projection to prevent ArrowMemoryError
    req_cols = [
        "store_id", "item_id", "date", "sales", 
        "cat_id", "dept_id", "state_id",
        "wday", "month", "year",
        "event_name_1", "snap_CA", "snap_TX", "snap_WI"
    ]
    
    # Read only required columns and use pyarrow strings/categoricals if possible
    # We will cast to category later, but reading less columns saves RAM instantly.
    long_df = pd.read_parquet(parquet, columns=req_cols)
    logger.info("Loaded %d rows from %s (with column projection)", len(long_df), parquet.name)
    
    # Generate known-future features if they don't exist
    if "is_event" not in long_df.columns:
        long_df["is_event"] = long_df["event_name_1"].notna().astype(np.int8)
    if "snap_active" not in long_df.columns:
        snap_CA_mask = (long_df["state_id"] == "CA") & (long_df["snap_CA"] == 1)
        snap_TX_mask = (long_df["state_id"] == "TX") & (long_df["snap_TX"] == 1)
        snap_WI_mask = (long_df["state_id"] == "WI") & (long_df["snap_WI"] == 1)
        long_df["snap_active"] = (snap_CA_mask | snap_TX_mask | snap_WI_mask).astype(np.int8)
        
    # Early datetime conversion
    long_df["date"] = pd.to_datetime(long_df["date"])
    
    if "is_weekend" not in long_df.columns:
        long_df["is_weekend"] = long_df["date"].dt.dayofweek.isin([5, 6]).astype(np.int8)
    if "week_of_year" not in long_df.columns:
        long_df["week_of_year"] = long_df["date"].dt.isocalendar().week.astype(np.int8)
        
    # Early categorical conversion to drastically reduce memory footprint
    for c in ["store_id", "item_id", "cat_id", "dept_id", "state_id"]:
        if c in long_df.columns:
            long_df[c] = long_df[c].astype("category")
    
    if args.subset:
        series_keys = long_df[["store_id", "item_id"]].drop_duplicates().head(args.subset)
        long_df = long_df.merge(series_keys, on=["store_id", "item_id"], how="inner")
        logger.info("Subset to %d series (%d rows)", args.subset, len(long_df))
        out_dir = out_dir.parent / "phase5_subset"
        
    # Load segmentation
    seg_path = cfg.path("reports") / "series_stats.csv"
    if args.fixture:
        seg_path = Path("reports/fixture/series_stats.csv")
        
    seg_df = None
    if seg_path.exists():
        seg_df = pd.read_csv(seg_path)[["store_id", "item_id", "segment"]]
        logger.info("Loaded segmentation for %d series", len(seg_df))
        
    import time
    start = time.time()
    results = run_phase5_backtest(long_df, cfg, seg_df)
    elapsed = time.time() - start
    logger.info("Phase 5 Backtest took %.2f seconds", elapsed)
    
    written = write_outputs(results, out_dir)
    
    print("\n=== Phase 5 Backtest Complete ===")
    print(f"Runtime: {elapsed:.2f}s")
    print(f"Folds:   {len(results['fold_meta'])}")
    print(f"Series:  {results['all_predictions'][['store_id','item_id']].drop_duplicates().shape[0]:,}")
    print("\nOverall MAE by method:")
    print(results["summary"].groupby("method")["mae"].mean().sort_values().to_string())
    print("\nOverall WAPE by method:")
    print(results["summary"].groupby("method")["wape"].mean().sort_values().to_string())
    print("\nWritten:")
    for k, p in written.items():
        print(f"  {p}")
        
    # Create Markdown report
    from src.backtest import FoldSpec
    folds = [FoldSpec(**d) for d in results["fold_meta"].to_dict("records")]
    report_path = out_dir / "phase5_forecasting.md"
    write_report(results, cfg, folds, elapsed, report_path)

def write_report(results, cfg, folds, elapsed, out_path):
    """Write Phase 5 report."""
    bt = cfg["backtest"]
    L = []
    A = L.append
    
    A("# Phase 5 — Forecasting Model Benchmark\n")
    
    A("## Fold Summary\n")
    A(results["fold_meta"].to_markdown(index=False))
    A(f"\nRuntime: {elapsed:.2f} seconds\n")
    
    A("## Overall Metrics\n")
    overall = results["summary"].groupby("method")[
        ["mae", "wape", "rmse", "bias"]
    ].mean().reset_index()
    A(overall.sort_values("wape").to_markdown(index=False))
    A("")
    
    A("## Metrics by Segment\n")
    if "segment_results" in results:
        seg = results["segment_results"].groupby(["method", "segment"])[
            ["mae", "wape"]
        ].mean().reset_index()
        A(seg.to_markdown(index=False))
    A("")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")

if __name__ == "__main__":
    main()
