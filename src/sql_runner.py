"""SQL layer runner.

Parses the `-- name:` blocks in sql/, executes them against the real DuckDB
database, and exports selected results to reports/phase2/.

Why a runner rather than loose .sql files: Phase 7's agent needs to call these
queries as controlled, parameterised tools (never free-form generated SQL). A
named registry loaded from disk is that interface, and it means the queries the
agent runs are literally the same text reviewed here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Config, load_config

logger = logging.getLogger(__name__)

SQL_DIR_NAME = "sql"
NAME_RE = re.compile(r"^--\s*name:\s*(\S+)\s*$", re.MULTILINE)
PARAM_RE = re.compile(r"^--\s*params:\s*(.+)$", re.MULTILINE)


@dataclass
class Query:
    name: str
    sql: str
    params: list[str]
    source_file: str

    @property
    def is_ddl(self) -> bool:
        # Every block opens with explanatory comments, so the first non-comment
        # line is what determines the statement type.
        for line in self.sql.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            return stripped.upper().startswith(("CREATE", "DROP", "ALTER"))
        return False


def parse_sql_file(path: Path) -> list[Query]:
    """Split a .sql file into named query blocks."""
    text = path.read_text()
    matches = list(NAME_RE.finditer(text))
    queries = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        pm = PARAM_RE.search(body)
        params = [p.strip() for p in pm.group(1).split(",")] if pm else []
        # Strip the params directive; keep other comments as documentation.
        body = PARAM_RE.sub("", body).strip()

        if body:
            queries.append(Query(m.group(1), body, params, path.name))
    return queries


def load_queries(sql_dir: Path) -> dict[str, Query]:
    """Load every named query, in filename order. Duplicate names are an error."""
    registry: dict[str, Query] = {}
    for path in sorted(sql_dir.glob("*.sql")):
        for q in parse_sql_file(path):
            if q.name in registry:
                raise ValueError(
                    f"Duplicate query name {q.name!r} in {q.source_file} "
                    f"(already defined in {registry[q.name].source_file})"
                )
            registry[q.name] = q
    return registry


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, Any] = {
    "top_n": 10,
    "min_prior_units": 30,
    "min_total_units": 100,
    "max_mean_daily": 0.1,
    "min_zero_share": 0.9,
    "min_occurrences": 3,
}


class SQLLayer:
    """Opens the DuckDB database and runs registered queries."""

    def __init__(self, db_path: Path, sql_dir: Path, read_only: bool = False):
        import duckdb

        if not db_path.exists():
            raise FileNotFoundError(
                f"No database at {db_path}. Run `python -m src.ingest` first."
            )
        self.con = duckdb.connect(str(db_path), read_only=read_only)
        self.queries = load_queries(sql_dir)

    def create_views(self) -> list[str]:
        """Run the DDL blocks that define the analytical views."""
        created = []
        for name, q in self.queries.items():
            if q.is_ddl:
                self.con.execute(q.sql)
                created.append(name)
        return created

    def run(self, name: str, **params: Any) -> pd.DataFrame:
        if name not in self.queries:
            raise KeyError(f"Unknown query {name!r}. Known: {sorted(self.queries)}")
        q = self.queries[name]
        merged = {k: DEFAULT_PARAMS.get(k) for k in q.params}
        merged.update({k: v for k, v in params.items() if k in q.params})

        missing = [k for k, v in merged.items() if v is None]
        if missing:
            raise ValueError(f"Query {name!r} needs parameters: {missing}")

        return (
            self.con.execute(q.sql, merged).df() if merged
            else self.con.execute(q.sql).df()
        )

    def analytical_queries(self) -> list[str]:
        return [n for n, q in self.queries.items() if not q.is_ddl]

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> SQLLayer:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(layer: SQLLayer) -> tuple[bool, pd.DataFrame]:
    """Structural validation plus independent cross-checks of key aggregates.

    The cross-checks matter more than the structural ones: they recompute a
    headline number a second way and compare. A JOIN that silently fans out
    would pass every column-presence check and still corrupt every total.
    """
    rows = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        rows.append({"check": name, "passed": bool(passed), "detail": detail})

    overview = layer.run("fact_overview").iloc[0]
    fact_rows = int(overview["n_rows"])
    fact_units = int(overview["total_units"])

    dup = int(layer.run("grain_uniqueness").iloc[0]["duplicate_keys"])
    check("grain_is_unique", dup == 0, f"{dup} duplicated (item,store,date) keys")

    fan = layer.run("join_fanout_check").iloc[0]
    check("no_join_fanout", int(fan["fanout_delta"]) == 0,
          f"delta={int(fan['fanout_delta'])}")

    keys = layer.run("dimension_key_uniqueness")
    bad = keys.loc[keys["n_rows"] != keys["n_keys"], "table_name"].tolist()
    check("dimension_keys_unique", not bad, f"non-unique: {bad}" if bad else "")

    gaps = int(layer.run("series_date_continuity").iloc[0]["series_with_gaps"])
    check("series_contiguous", gaps == 0, f"{gaps} series with date gaps")

    act = layer.run("active_flag_check").iloc[0]
    check("all_rows_active", int(act["inactive_rows"]) == 0,
          f"{int(act['inactive_rows'])} inactive rows present")

    # --- independent reconciliation -------------------------------------
    # Units summed through the denormalised view must equal units summed
    # straight off the fact table.
    view_units = int(layer.con.execute("SELECT SUM(sales) FROM v_sales").fetchone()[0])
    check("view_units_match_fact", view_units == fact_units,
          f"view={view_units:,} fact={fact_units:,}")

    view_rows = int(layer.con.execute("SELECT COUNT(*) FROM v_sales").fetchone()[0])
    check("view_rows_match_fact", view_rows == fact_rows,
          f"view={view_rows:,} fact={fact_rows:,}")

    # Category totals must sum back to the grand total.
    cat = layer.run("kpi_by_category")
    check("category_units_reconcile", int(cat["total_units"].sum()) == fact_units,
          f"sum={int(cat['total_units'].sum()):,} vs {fact_units:,}")
    check("category_shares_sum_to_100",
          abs(float(cat["pct_of_total_units"].sum()) - 100.0) < 0.5,
          f"sum={float(cat['pct_of_total_units'].sum()):.2f}%")

    # Store totals must too.
    store = layer.run("store_performance")
    check("store_units_reconcile", int(store["total_units"].sum()) == fact_units,
          f"sum={int(store['total_units'].sum()):,} vs {fact_units:,}")

    # ABC classes partition the assortment exactly once.
    abc = layer.run("abc_classification")
    check("abc_units_reconcile", int(abc["total_units"].sum()) == fact_units,
          f"sum={int(abc['total_units'].sum()):,} vs {fact_units:,}")

    # SNAP flag must be resolved per state, not read from one wide column.
    snap = layer.con.execute("""
        SELECT COUNT(*) FROM (
            SELECT state_id, date, COUNT(DISTINCT snap_active) AS n
            FROM v_sales GROUP BY 1, 2 HAVING n > 1
        )
    """).fetchone()[0]
    check("snap_flag_consistent_per_state_date", int(snap) == 0,
          f"{int(snap)} state-dates with conflicting SNAP flags")

    df = pd.DataFrame(rows)
    return bool(df["passed"].all()), df


# ---------------------------------------------------------------------------
# Report export
# ---------------------------------------------------------------------------

# Only outputs later phases or an interview would actually use.
EXPORTS: dict[str, dict[str, Any]] = {
    "store_kpis": {"query": "store_performance"},
    "store_growth": {"query": "store_growth_28d"},
    "category_kpis": {"query": "kpi_by_category"},
    "store_category_matrix": {"query": "kpi_store_category_matrix"},
    "product_rankings": {"query": "top_products_per_store", "params": {"top_n": 20}},
    "declining_products": {"query": "declining_products", "params": {"top_n": 50}},
    "volatility_analysis": {"query": "product_volatility", "params": {"top_n": 200}},
    "abc_classification": {"query": "abc_classification"},
    "rolling_demand": {"query": "rolling_store_demand"},
    "monthly_trend": {"query": "monthly_demand_trend"},
    "snap_comparison": {"query": "snap_day_comparison"},
    "snap_by_category": {"query": "snap_by_category"},
    "price_change_association": {"query": "price_change_association"},
}


def export_reports(layer: SQLLayer, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, spec in EXPORTS.items():
        df = layer.run(spec["query"], **spec.get("params", {}))
        path = out_dir / f"{stem}.csv"
        df.to_csv(path, index=False)
        written.append(path)
        logger.info("%s -> %d rows", path.name, len(df))
    return written


def main(config_path: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg: Config = load_config(config_path)
    sql_dir = cfg.root / SQL_DIR_NAME

    with SQLLayer(cfg.path("duckdb"), sql_dir) as layer:
        created = layer.create_views()
        print(f"Views created: {', '.join(created)}\n")

        ok, report = validate(layer)
        print(report.to_string(index=False))
        print(f"\nValidation: {'PASS' if ok else 'FAIL'}\n")
        if not ok:
            raise SystemExit("SQL validation failed - not exporting reports.")

        out_dir = cfg.path("reports") / "phase2"
        written = export_reports(layer, out_dir)

        qc_path = out_dir / "sql_validation.md"
        qc_path.write_text(
            "# Phase 2 - SQL Layer Validation\n\n"
            f"Overall: {'PASS' if ok else 'FAIL'}\n\n"
            + report.to_markdown(index=False) + "\n"
        )

        print(f"\n{len(written)} report files written to {out_dir}")
        print(f"Analytical queries available: {len(layer.analytical_queries())}")


if __name__ == "__main__":
    main()
