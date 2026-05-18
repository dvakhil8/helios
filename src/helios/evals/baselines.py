"""Baseline cache — the original (unoptimized) job's runtime, DBU, and output
hash. Keyed by (fixture_id, fixture.version) so a fixture version bump
invalidates stale entries.

Stored as JSON under evals/baselines.json (committed to the repo). Small
file; fine for git. If this grows past ~MB scale, move to a Delta table.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..tools.databricks import execute_sql, wait_for_job_run
from .fixtures import Fixture


# Spark types that need rounding to avoid float-reorder false positives
# in the equivalence hash. Stored as a set of uppercase substrings.
_FLOAT_TYPES = {"DOUBLE", "FLOAT", "REAL"}
_FLOAT_ROUND_DIGITS = 6


BASELINES_PATH: Path = (
    Path(os.environ.get("HELIOS_EVAL_BASELINES_PATH")
         or Path(__file__).resolve().parents[3] / "evals" / "baselines.json")
)


@dataclass(frozen=True)
class Baseline:
    fixture_id: str
    fixture_version: int
    duration_ms: int                       # wall-clock for the whole DAG
    output_row_count: int                  # primary output (back-compat); use `outputs` for multi-table
    output_hash: str                       # primary output (back-compat)
    captured_at: int                       # unix seconds
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per-task wall-clock durations from the baseline run, keyed by task_key.
    # Empty for single-task fixtures (or older cached baselines).
    task_durations_ms: dict[str, int] = field(default_factory=dict)
    # Per-output-table {row_count, hash, stats}, for scoped multi-output fixtures.
    # Keyed by the (rendered) fully-qualified table name. Empty when only the
    # primary output is tracked.
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)


def _load_all() -> dict[str, dict]:
    if not BASELINES_PATH.exists():
        return {}
    return json.loads(BASELINES_PATH.read_text())


def _save_all(data: dict[str, dict]) -> None:
    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def _key(fixture: Fixture) -> str:
    return f"{fixture.id}@v{fixture.version}"


def get(fixture: Fixture) -> Baseline | None:
    raw = _load_all().get(_key(fixture))
    if raw is None:
        return None
    return Baseline(**raw)


def put(baseline: Baseline) -> None:
    data = _load_all()
    data[f"{baseline.fixture_id}@v{baseline.fixture_version}"] = asdict(baseline)
    _save_all(data)


# ---- Capture from a completed run --------------------------------------------


def capture_from_run(
    fixture: Fixture,
    *,
    run_id: int,
    output_table_fqn: str | None,
    extra_output_tables: list[str] | None = None,
) -> Baseline:
    """Wait for the run to terminate, then record duration + output stats.

    Caller is responsible for triggering the run; this only waits and measures.

    Single-task fixtures pass `output_table_fqn` (the table the notebook wrote
    to via the {{output_table}} placeholder). Scoped multi-task fixtures pass
    `None` here — there's no "primary" output, only the per-table list in
    `extra_output_tables` (rendered from `scope.output_tables`).
    """
    duration_ms = _wait_for_run(run_id)
    if output_table_fqn:
        row_count, output_hash = _hash_output_table(output_table_fqn)
        stats = compute_stats_fingerprint(output_table_fqn)
    else:
        row_count, output_hash, stats = 0, "", {}
    task_durations = _capture_task_durations(run_id)
    outputs: dict[str, dict[str, Any]] = {}
    for tbl in (extra_output_tables or []):
        try:
            t_rows, t_hash = _hash_output_table(tbl)
            t_stats = compute_stats_fingerprint(tbl)
            outputs[tbl] = {"row_count": t_rows, "hash": t_hash, "stats": t_stats}
        except Exception as e:
            outputs[tbl] = {"error": str(e)}
    return Baseline(
        fixture_id=fixture.id,
        fixture_version=fixture.version,
        duration_ms=duration_ms,
        output_row_count=row_count,
        output_hash=output_hash,
        captured_at=int(time.time()),
        stats=stats,
        task_durations_ms=task_durations,
        outputs=outputs,
    )


def _capture_task_durations(run_id: int) -> dict[str, int]:
    """Return {task_key: execution_duration_ms} for each task in a run.
    Returns {} if the run is single-task or doesn't expose per-task info."""
    from ..tools.databricks import get_job_run
    try:
        run = get_job_run(run_id)
    except Exception:
        return {}
    out: dict[str, int] = {}
    for task in run.get("tasks") or []:
        key = task.get("task_key")
        dur = task.get("execution_duration")
        if key is not None and dur is not None:
            out[str(key)] = int(dur)
    return out


def _wait_for_run(run_id: int, timeout_s: int = 3600) -> int:
    """Block until the run terminates. Returns execution_duration_ms.

    Raises if the run didn't succeed; baseline only makes sense on a clean run.
    """
    result = wait_for_job_run(run_id, timeout_seconds=timeout_s, poll_interval_seconds=10)
    if result.get("timed_out"):
        raise TimeoutError(f"run {run_id} exceeded {timeout_s}s")
    if result.get("result_state") != "SUCCESS":
        raise RuntimeError(
            f"run {run_id} ended in {result.get('state')}/{result.get('result_state')}"
        )
    return int(result.get("execution_duration_ms") or 0)


def _table_columns(fqn: str) -> list[tuple[str, str]]:
    """Return [(name, type), ...] for the table. Uses DESCRIBE TABLE and stops
    at the first separator row (Spark's DESCRIBE includes # Partition Information)."""
    rows = execute_sql(f"DESCRIBE TABLE {fqn}", row_limit=500)["rows"]
    out: list[tuple[str, str]] = []
    for r in rows:
        name = (r.get("col_name") or "").strip()
        dtype = (r.get("data_type") or "").strip()
        if not name or name.startswith("#"):
            break
        out.append((name, dtype.upper()))
    return out


def _is_float_type(dtype: str) -> bool:
    return any(t in dtype for t in _FLOAT_TYPES)


def _struct_expr_with_rounding(cols: list[tuple[str, str]]) -> str:
    """Build a `struct(col_or_rounded AS col, ...)` expression for hashing.

    DOUBLE / FLOAT columns get rounded to _FLOAT_ROUND_DIGITS dp so SUM-reorder
    drift (machine epsilon) doesn't break equivalence. All other types pass through.
    """
    parts: list[str] = []
    for name, dtype in cols:
        ident = f"`{name}`"
        if _is_float_type(dtype):
            parts.append(f"ROUND({ident}, {_FLOAT_ROUND_DIGITS}) AS {ident}")
        else:
            parts.append(f"{ident} AS {ident}")
    return f"struct({', '.join(parts)})"


def _hash_output_table(fqn: str, timeout_seconds: int = 300) -> tuple[int, str]:
    """Order-independent content hash. Schema-aware: rounds floats so SUM-reorder
    drift doesn't break the hash. Sensitive to schema (column names + non-float values).

    `timeout_seconds` controls the SQL execution wait. For large prod tables, set
    to 1800+ (30 min) since the per-row JSON serialization is expensive.
    """
    row_count = int(
        execute_sql(f"SELECT COUNT(*) AS c FROM {fqn}", timeout_seconds=timeout_seconds)["rows"][0]["c"]
    )
    cols = _table_columns(fqn)
    struct_expr = _struct_expr_with_rounding(cols) if cols else "struct(*)"
    result = execute_sql(
        f"SELECT hex(BIT_XOR(xxhash64(to_json({struct_expr})))) AS h FROM {fqn}",
        row_limit=1, timeout_seconds=timeout_seconds,
    )
    h = result["rows"][0]["h"] if result["rows"] else ""
    return row_count, str(h or "")


def compute_stats_fingerprint(fqn: str, timeout_seconds: int = 300) -> dict[str, dict[str, Any]]:
    """Per-column aggregate stats. For each column reports null_count and (for
    numerics) sum/min/max. Cheap (single SQL query), gives the correctness scorer
    a usable diff when the hash disagrees.

    Returned shape: {col_name: {"type": "...", "null_count": N, "sum": ..., ...}}
    """
    cols = _table_columns(fqn)
    if not cols:
        return {}

    # Build a single SELECT with per-column aggregates. Avoids N round-trips.
    selects: list[str] = []
    for name, dtype in cols:
        ident = f"`{name}`"
        prefix = f"`stat_{name}_"
        selects.append(f"SUM(CASE WHEN {ident} IS NULL THEN 1 ELSE 0 END) AS {prefix}null_count`")
        if any(t in dtype for t in ("INT", "LONG", "BIGINT", "DECIMAL", "DOUBLE", "FLOAT", "REAL", "SHORT", "BYTE", "TINYINT", "SMALLINT")):
            selects.append(f"CAST(SUM({ident}) AS DOUBLE) AS {prefix}sum`")
            selects.append(f"CAST(MIN({ident}) AS DOUBLE) AS {prefix}min`")
            selects.append(f"CAST(MAX({ident}) AS DOUBLE) AS {prefix}max`")

    sql = f"SELECT {', '.join(selects)} FROM {fqn}"
    row = execute_sql(sql, row_limit=1, timeout_seconds=timeout_seconds)["rows"]
    if not row:
        return {}
    raw = row[0]

    stats: dict[str, dict[str, Any]] = {}
    for name, dtype in cols:
        prefix = f"stat_{name}_"
        col_stats: dict[str, Any] = {"type": dtype}
        for k, v in raw.items():
            if k.startswith(prefix):
                stat_name = k[len(prefix):]
                col_stats[stat_name] = v
        stats[name] = col_stats
    return stats
