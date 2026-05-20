"""All Databricks LLM tools — SQL, table diagnostics, jobs, runs, permissions.

Each tool is a function + a `*_SCHEMA` dict in Anthropic tool-use format.
The REGISTRY at the bottom maps tool name -> (schema, handler); the package-level
`tools/__init__.py` merges this with other integrations' REGISTRYs.
"""

from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from typing import Any, Callable

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import JobSettings, Task, ViewsToExport
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service.workspace import ExportFormat, ImportFormat, Language


# ---- Client / auth -----------------------------------------------------------
# Self-contained: this file owns its own SDK construction. Auth is auto-resolved
# by databricks-sdk from DATABRICKS_HOST/DATABRICKS_TOKEN env vars or ~/.databrickscfg.


@lru_cache(maxsize=1)
def workspace() -> WorkspaceClient:
    return WorkspaceClient()


def default_warehouse_id() -> str | None:
    return os.environ.get("DATABRICKS_WAREHOUSE_ID")


_TERMINAL_SQL_STATES = {
    StatementState.SUCCEEDED,
    StatementState.FAILED,
    StatementState.CANCELED,
    StatementState.CLOSED,
}


# =============================================================================
# SQL & Data
# =============================================================================

EXECUTE_SQL_SCHEMA: dict[str, Any] = {
    "name": "execute_sql",
    "description": (
        "Execute a single SQL statement against a Databricks SQL Warehouse and return the rows. "
        "Use for arbitrary read or DDL/DML queries — including system tables "
        "(system.billing.usage, system.compute.node_timeseries, system.access.column_lineage, etc.) "
        "and metadata commands like DESCRIBE HISTORY. Returns columns + rows (capped by row_limit)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The SQL statement to execute. Single statement only.",
            },
            "warehouse_id": {
                "type": "string",
                "description": "SQL warehouse ID. If omitted, uses DATABRICKS_WAREHOUSE_ID env var.",
            },
            "row_limit": {
                "type": "integer",
                "description": "Maximum rows to return. Defaults to 1000.",
                "default": 1000,
            },
            "catalog": {"type": "string", "description": "Default catalog for the statement."},
            "schema": {"type": "string", "description": "Default schema for the statement."},
            "timeout_seconds": {
                "type": "integer",
                "default": 300,
                "description": "Max wall-clock seconds to wait. Default 5 min. Use 1800+ for full-table hashes on large tables.",
            },
        },
        "required": ["sql"],
    },
}


def execute_sql(
    sql: str,
    warehouse_id: str | None = None,
    row_limit: int = 1000,
    catalog: str | None = None,
    schema: str | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    wh = warehouse_id or default_warehouse_id()
    if not wh:
        raise ValueError("No SQL warehouse: pass warehouse_id or set DATABRICKS_WAREHOUSE_ID.")
    w = workspace()
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=wh,
        wait_timeout="30s",
        row_limit=row_limit,
        catalog=catalog,
        schema=schema,
    )

    deadline = time.monotonic() + timeout_seconds
    while resp.status and resp.status.state not in _TERMINAL_SQL_STATES:
        if time.monotonic() > deadline:
            w.statement_execution.cancel_execution(statement_id=resp.statement_id)
            raise TimeoutError(
                f"Statement {resp.statement_id} exceeded {timeout_seconds}s wall time"
            )
        time.sleep(2)
        resp = w.statement_execution.get_statement(statement_id=resp.statement_id)

    state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        err = (
            resp.status.error.message
            if resp.status and resp.status.error
            else f"Statement ended in state {state}"
        )
        raise RuntimeError(err)

    manifest = resp.manifest
    columns = (
        [c.name for c in manifest.schema.columns]
        if manifest and manifest.schema and manifest.schema.columns
        else []
    )
    data_array = resp.result.data_array if resp.result else None
    rows = [dict(zip(columns, row)) for row in (data_array or [])]
    return {
        "statement_id": resp.statement_id,
        "columns": columns,
        "row_count": len(rows),
        "rows": rows,
        "truncated": bool(manifest.truncated) if manifest else False,
    }


# =============================================================================
# Table Diagnostics
# =============================================================================

CHECK_TABLE_HEALTH_SCHEMA: dict[str, Any] = {
    "name": "check_table_health",
    "description": (
        "Diagnose the health of a Delta table. Steps:\n"
        "  1. Look up upstream sources via system.access.column_lineage.\n"
        "  2. Run DESCRIBE HISTORY on the target and each source to compare freshness.\n"
        "  3. Identify recent entities (jobs / notebooks / pipelines / queries) that wrote to "
        "the target, using column_lineage's entity_type/entity_id columns.\n"
        "Returns a structured report with sources, histories, and refresh entities."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "table": {
                "type": "string",
                "description": "Fully qualified table name: catalog.schema.table.",
            },
            "lookback_days": {
                "type": "integer",
                "default": 7,
                "description": "How many days back to look for lineage and refresh activity.",
            },
            "history_limit": {
                "type": "integer",
                "default": 5,
                "description": "Number of DESCRIBE HISTORY rows to return per table.",
            },
            "warehouse_id": {
                "type": "string",
                "description": "SQL warehouse to use. Defaults to DATABRICKS_WAREHOUSE_ID env var.",
            },
        },
        "required": ["table"],
    },
}


def _parse_fqn(table: str) -> tuple[str, str, str]:
    parts = table.split(".")
    if len(parts) != 3:
        raise ValueError(f"Expected catalog.schema.table, got {table!r}")
    for p in parts:
        if not p or not all(c.isalnum() or c == "_" for c in p):
            raise ValueError(f"Identifier {p!r} contains characters other than alnum/underscore")
    return parts[0], parts[1], parts[2]


def check_table_health(
    table: str,
    lookback_days: int = 7,
    history_limit: int = 5,
    warehouse_id: str | None = None,
) -> dict[str, Any]:
    catalog, schema, name = _parse_fqn(table)

    def sql(q: str, row_limit: int = 200) -> dict[str, Any]:
        return execute_sql(q, warehouse_id=warehouse_id, row_limit=row_limit)

    upstream_q = f"""
        SELECT DISTINCT
            source_table_catalog AS catalog,
            source_table_schema  AS schema,
            source_table_name    AS name
        FROM system.access.column_lineage
        WHERE target_table_catalog = '{catalog}'
          AND target_table_schema  = '{schema}'
          AND target_table_name    = '{name}'
          AND source_table_full_name IS NOT NULL
          AND event_date >= current_date() - INTERVAL {lookback_days} DAYS
    """
    upstream_rows = sql(upstream_q)["rows"]
    sources = [
        f"{r['catalog']}.{r['schema']}.{r['name']}"
        for r in upstream_rows
        if r.get("catalog") and r.get("schema") and r.get("name")
    ]

    histories: dict[str, Any] = {}
    for t in [table, *sources]:
        try:
            h = sql(f"DESCRIBE HISTORY {t} LIMIT {history_limit}", row_limit=history_limit)
            histories[t] = {"row_count": h["row_count"], "rows": h["rows"]}
        except Exception as e:
            histories[t] = {"error": str(e)}

    refresh_q = f"""
        SELECT
            entity_type,
            entity_id,
            COUNT(*) AS write_events,
            MAX(event_time) AS last_event
        FROM system.access.column_lineage
        WHERE target_table_catalog = '{catalog}'
          AND target_table_schema  = '{schema}'
          AND target_table_name    = '{name}'
          AND event_date >= current_date() - INTERVAL {lookback_days} DAYS
          AND entity_type IS NOT NULL
        GROUP BY entity_type, entity_id
        ORDER BY last_event DESC
        LIMIT 50
    """
    try:
        refresh_entities = sql(refresh_q)["rows"]
    except Exception as e:
        refresh_entities = [{"error": str(e)}]

    return {
        "target": table,
        "lookback_days": lookback_days,
        "upstream_sources": sources,
        "histories": histories,
        "refresh_entities": refresh_entities,
    }


# =============================================================================
# Job Management
# =============================================================================

LIST_JOBS_SCHEMA: dict[str, Any] = {
    "name": "list_jobs",
    "description": (
        "List Databricks jobs in the workspace. Paginated server-side: returns "
        "ALL matching jobs across all pages by default, not just the first 100. "
        "Use max_results only when you want a preview or to cap the response "
        "size; omit it for a full audit. Response includes `truncated=True` if "
        "max_results was reached (there may be more jobs)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name_filter": {
                "type": "string",
                "description": "Case-insensitive substring filter applied to job name.",
            },
            "max_results": {
                "type": "integer",
                "description": "Cap on jobs returned. Omit (or set 0) for ALL jobs across all pages.",
            },
        },
    },
}


def list_jobs(name_filter: str | None = None, max_results: int | None = None) -> dict[str, Any]:
    w = workspace()
    out: list[dict[str, Any]] = []
    # API caps page size at 100; the SDK iterator auto-paginates beyond that.
    for job in w.jobs.list(name=name_filter, limit=100):
        out.append(
            {
                "job_id": job.job_id,
                "name": job.settings.name if job.settings else None,
                "created_time": job.created_time,
                "creator_user_name": job.creator_user_name,
                "schedule": job.settings.schedule.as_dict()
                if job.settings and job.settings.schedule
                else None,
            }
        )
        if max_results and len(out) >= max_results:
            break
    truncated = bool(max_results) and len(out) >= max_results
    return {"count": len(out), "truncated": truncated, "jobs": out}


GET_JOB_SCHEMA: dict[str, Any] = {
    "name": "get_job",
    "description": "Get full settings and metadata for a single job by ID.",
    "input_schema": {
        "type": "object",
        "properties": {"job_id": {"type": "integer", "description": "The job ID."}},
        "required": ["job_id"],
    },
}


def get_job(job_id: int) -> dict[str, Any]:
    return workspace().jobs.get(job_id=job_id).as_dict()


CREATE_JOB_SCHEMA: dict[str, Any] = {
    "name": "create_job",
    "description": (
        "Create a new Databricks job from a Jobs API 2.1 settings payload. "
        "The 'settings' object must include at minimum 'name' and 'tasks'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "settings": {
                "type": "object",
                "description": "Jobs API 2.1 JobSettings object (name, tasks, schedule, job_clusters, etc.).",
            }
        },
        "required": ["settings"],
    },
}


def create_job(settings: dict[str, Any]) -> dict[str, Any]:
    w = workspace()
    js = JobSettings.from_dict(settings)
    response = w.jobs.create(
        name=js.name,
        tasks=js.tasks,
        job_clusters=js.job_clusters,
        email_notifications=js.email_notifications,
        webhook_notifications=js.webhook_notifications,
        notification_settings=js.notification_settings,
        timeout_seconds=js.timeout_seconds,
        schedule=js.schedule,
        max_concurrent_runs=js.max_concurrent_runs,
        tags=js.tags,
        parameters=js.parameters,
        run_as=js.run_as,
        git_source=js.git_source,
        trigger=js.trigger,
        continuous=js.continuous,
        environments=js.environments,
        queue=js.queue,
        description=js.description,
    )
    return {"job_id": response.job_id}


RUN_JOB_NOW_SCHEMA: dict[str, Any] = {
    "name": "run_job_now",
    "description": "Trigger a one-off run of an existing job. Does not wait for completion. Returns run_id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "job_id": {"type": "integer", "description": "The job ID to trigger."},
            "notebook_params": {
                "type": "object",
                "description": "Override notebook parameters (map of name → value).",
            },
            "python_params": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Override python script parameters.",
            },
            "jar_params": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Override JAR parameters.",
            },
            "job_parameters": {
                "type": "object",
                "description": "Override job-level parameters (map of name → value).",
            },
        },
        "required": ["job_id"],
    },
}


def run_job_now(
    job_id: int,
    notebook_params: dict[str, str] | None = None,
    python_params: list[str] | None = None,
    jar_params: list[str] | None = None,
    job_parameters: dict[str, str] | None = None,
) -> dict[str, Any]:
    w = workspace()
    result = w.jobs.run_now(
        job_id=job_id,
        notebook_params=notebook_params,
        python_params=python_params,
        jar_params=jar_params,
        job_parameters=job_parameters,
    )
    rn = getattr(result, "response", result)
    return {"run_id": rn.run_id, "number_in_job": getattr(rn, "number_in_job", None)}


ADD_JOB_TASKS_SCHEMA: dict[str, Any] = {
    "name": "add_job_tasks",
    "description": (
        "Append one or more tasks to an existing multi-task job. "
        "Each item in 'tasks' must be a Jobs API 2.1 Task object — "
        "each must have a unique task_key not used by existing tasks."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "job_id": {"type": "integer", "description": "The job ID to modify."},
            "tasks": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Jobs API 2.1 Task objects to append.",
            },
        },
        "required": ["job_id", "tasks"],
    },
}


def add_job_tasks(job_id: int, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    w = workspace()
    job = w.jobs.get(job_id=job_id)
    settings = job.settings
    if settings is None:
        raise RuntimeError(f"Job {job_id} has no settings to modify")

    existing = list(settings.tasks or [])
    existing_keys = {t.task_key for t in existing if t.task_key}
    new_tasks = [Task.from_dict(t) for t in tasks]
    duplicates = [t.task_key for t in new_tasks if t.task_key in existing_keys]
    if duplicates:
        raise ValueError(f"task_key already exists on job {job_id}: {duplicates}")

    settings.tasks = existing + new_tasks
    w.jobs.reset(job_id=job_id, new_settings=settings)
    return {
        "job_id": job_id,
        "added": len(new_tasks),
        "total_tasks": len(settings.tasks),
        "added_task_keys": [t.task_key for t in new_tasks],
    }


# =============================================================================
# Job Runs & Output
# =============================================================================

LIST_JOB_RUNS_SCHEMA: dict[str, Any] = {
    "name": "list_job_runs",
    "description": (
        "List job runs. Optional filters: job_id, active-only, completed-only, "
        "and a start-time window (Unix milliseconds)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "job_id": {"type": "integer", "description": "Filter to a single job."},
            "active_only": {"type": "boolean", "description": "Only return runs that are still active."},
            "completed_only": {"type": "boolean", "description": "Only return runs that have completed."},
            "limit": {"type": "integer", "default": 25, "description": "Max runs to return. Defaults to 25."},
            "start_time_from_ms": {
                "type": "integer",
                "description": "Filter to runs starting at or after this Unix ms timestamp.",
            },
            "start_time_to_ms": {
                "type": "integer",
                "description": "Filter to runs starting at or before this Unix ms timestamp.",
            },
        },
    },
}


def list_job_runs(
    job_id: int | None = None,
    active_only: bool = False,
    completed_only: bool = False,
    limit: int = 25,
    start_time_from_ms: int | None = None,
    start_time_to_ms: int | None = None,
) -> dict[str, Any]:
    w = workspace()
    out: list[dict[str, Any]] = []
    for run in w.jobs.list_runs(
        job_id=job_id,
        active_only=active_only or None,
        completed_only=completed_only or None,
        limit=limit,
        start_time_from=start_time_from_ms,
        start_time_to=start_time_to_ms,
    ):
        out.append(
            {
                "run_id": run.run_id,
                "job_id": run.job_id,
                "run_name": run.run_name,
                "state": run.state.life_cycle_state.value
                if run.state and run.state.life_cycle_state
                else None,
                "result_state": run.state.result_state.value
                if run.state and run.state.result_state
                else None,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "execution_duration_ms": run.execution_duration,
                "creator_user_name": run.creator_user_name,
                "run_page_url": run.run_page_url,
            }
        )
        if len(out) >= limit:
            break
    return {"count": len(out), "runs": out}


GET_JOB_RUN_SCHEMA: dict[str, Any] = {
    "name": "get_job_run",
    "description": (
        "Get detailed metadata for a single run by run_id. For multi-task jobs, the response "
        "includes each task's run_id (use those for get_job_run_output)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "run_id": {"type": "integer", "description": "The run ID."},
            "include_history": {"type": "boolean", "description": "Include repair history for the run."},
        },
        "required": ["run_id"],
    },
}


def get_job_run(run_id: int, include_history: bool = False) -> dict[str, Any]:
    w = workspace()
    return w.jobs.get_run(run_id=run_id, include_history=include_history or None).as_dict()


GET_JOB_RUN_OUTPUT_SCHEMA: dict[str, Any] = {
    "name": "get_job_run_output",
    "description": (
        "Get the output of a single task run — notebook output, exit value, error, "
        "and truncated logs. For multi-task jobs pass a task-level run_id (from "
        "get_job_run's tasks[].run_id), not the top-level job run_id."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"run_id": {"type": "integer", "description": "Task-level run ID."}},
        "required": ["run_id"],
    },
}


def get_job_run_output(run_id: int) -> dict[str, Any]:
    return workspace().jobs.get_run_output(run_id=run_id).as_dict()


WAIT_FOR_JOB_RUN_SCHEMA: dict[str, Any] = {
    "name": "wait_for_job_run",
    "description": (
        "Block until a job run reaches a terminal state (TERMINATED / "
        "INTERNAL_ERROR / SKIPPED) or `timeout_seconds` elapses. Returns the "
        "final state, result_state, durations, and run page URL. Use this "
        "instead of shell `sleep` + repeated get_job_run — it blocks once "
        "without burning agent iterations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "run_id": {"type": "integer", "description": "Run ID to wait on."},
            "timeout_seconds": {
                "type": "integer",
                "default": 1800,
                "description": "Max wait in seconds. Default 30 min.",
            },
            "poll_interval_seconds": {
                "type": "integer",
                "default": 15,
                "description": "Seconds between status polls.",
            },
        },
        "required": ["run_id"],
    },
}


def wait_for_job_run(
    run_id: int, timeout_seconds: int = 1800, poll_interval_seconds: int = 15
) -> dict[str, Any]:
    w = workspace()
    deadline = time.monotonic() + timeout_seconds
    while True:
        run = w.jobs.get_run(run_id=run_id)
        life = (
            run.state.life_cycle_state.value
            if run.state and run.state.life_cycle_state else None
        )
        if life in {"TERMINATED", "INTERNAL_ERROR", "SKIPPED"}:
            result = (
                run.state.result_state.value
                if run.state and run.state.result_state else None
            )
            return {
                "run_id": run_id,
                "state": life,
                "result_state": result,
                "execution_duration_ms": run.execution_duration,
                "setup_duration_ms": run.setup_duration,
                "cleanup_duration_ms": run.cleanup_duration,
                "run_page_url": run.run_page_url,
                "state_message": run.state.state_message if run.state else None,
            }
        if time.monotonic() > deadline:
            return {
                "run_id": run_id,
                "state": life,
                "result_state": None,
                "timed_out": True,
                "elapsed_seconds": timeout_seconds,
                "run_page_url": run.run_page_url,
            }
        time.sleep(poll_interval_seconds)


EXPLAIN_QUERY_SCHEMA: dict[str, Any] = {
    "name": "explain_query",
    "description": (
        "Run EXPLAIN on a SQL query and return both the FORMATTED physical plan "
        "(join strategies + shuffle count) AND the COST logical plan (cardinality "
        "estimates) in one call. Use this BEFORE proposing an optimization (to "
        "know where the bottleneck is) AND BEFORE triggering an expensive "
        "sandbox run (to verify your optimization produces a better plan).\n"
        "\n"
        "Default mode is 'ALL' which runs FORMATTED + COST and combines the "
        "summaries. Pass a specific mode to run only one:\n"
        "  - FORMATTED  — physical plan with numbered nodes (join strategies + shuffles)\n"
        "  - COST       — logical plan with cardinality stats (rowCount + sizeInBytes)\n"
        "  - EXTENDED   — all 4 plans (parsed/analyzed/optimized/physical) for deep debug\n"
        "\n"
        "Pass the SELECT body of a CTAS, not the full CREATE TABLE AS — EXPLAIN "
        "doesn't always accept the CTAS wrapper."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The SELECT (or INSERT/MERGE) to EXPLAIN. Don't include the CTAS wrapper.",
            },
            "mode": {
                "type": "string",
                "enum": ["ALL", "FORMATTED", "COST", "EXTENDED"],
                "default": "ALL",
                "description": "EXPLAIN mode. Default ALL = FORMATTED + COST combined.",
            },
            "warehouse_id": {
                "type": "string",
                "description": "SQL warehouse ID. Defaults to DATABRICKS_WAREHOUSE_ID env.",
            },
            "timeout_seconds": {
                "type": "integer",
                "default": 120,
                "description": "Wait time per EXPLAIN call. Default 120s.",
            },
        },
        "required": ["sql"],
    },
}


# Databricks SQL warehouses use the Photon engine, which prefixes operator
# names (PhotonScan, PhotonShuffleExchangeSink, etc.). Patterns accept both
# the Photon-prefixed and unprefixed forms.
_JOIN_PATTERN = re.compile(
    r"\b(?:Photon)?(BroadcastHashJoin|SortMergeJoin|ShuffledHashJoin|ShuffleHashJoin|"
    r"BroadcastNestedLoopJoin|CartesianProduct|HashJoin)\b"
)
# Generic `Join (N)` operator in the FORMATTED physical plan when Photon uses
# the abstract Join wrapper; or `Join Inner/LeftOuter/...` in COST logical plan.
_GENERIC_JOIN_PATTERN = re.compile(r"^\s*\+?-?\s*Join\b", re.MULTILINE)
# Each logical shuffle decomposes in Photon into a Sink + MapStage + Source —
# count only the Sink to avoid double-counting. For non-Photon plans, plain
# `Exchange` is the marker.
_EXCHANGE_PATTERN = re.compile(
    r"\b(?:PhotonShuffleExchangeSink|Exchange)\b"
)
_SCAN_PATTERN = re.compile(
    r"\b(?:Photon)?(?:FileScan|Scan|LocalTableScan)\b[^\n]*"
)
# Simple independent extractors — robust to nested parens.
_ROWCOUNT_PATTERN = re.compile(r"rowCount=([0-9.eE+-]+|[0-9.]+[KMBT]?)")
_SIZEBYTES_PATTERN = re.compile(r"sizeInBytes=([0-9.eE+-]+\s*(?:B|KiB|MiB|GiB|TiB)?)")


def explain_query(
    sql: str,
    mode: str = "ALL",
    warehouse_id: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    mode_clean = (mode or "ALL").strip().upper()
    if mode_clean == "ALL":
        formatted = _explain_one_mode(sql, "FORMATTED", warehouse_id, timeout_seconds)
        cost = _explain_one_mode(sql, "COST", warehouse_id, timeout_seconds)
        # Dedupe warnings across both views
        all_warnings: list[str] = []
        for w in (formatted.get("summary", {}).get("warnings", [])
                  + cost.get("summary", {}).get("warnings", [])):
            if w not in all_warnings:
                all_warnings.append(w)
        return {
            "sql": sql[:500],
            "mode": "ALL (formatted + cost)",
            "formatted": formatted,
            "cost": cost,
            "combined_warnings": all_warnings,
        }
    return _explain_one_mode(sql, mode_clean, warehouse_id, timeout_seconds)


def _explain_one_mode(
    sql: str, mode_clean: str, warehouse_id: str | None, timeout_seconds: int,
) -> dict[str, Any]:
    if mode_clean and mode_clean not in {"FORMATTED", "COST", "EXTENDED", "CODEGEN"}:
        raise ValueError(f"unknown EXPLAIN mode: {mode_clean!r}")
    explain_sql = f"EXPLAIN {mode_clean} {sql}".strip() if mode_clean else f"EXPLAIN {sql}"

    # EXPLAIN returns a single column (typically "plan") with one row whose value
    # is the full multi-line plan text.
    result = execute_sql(
        explain_sql, warehouse_id=warehouse_id, row_limit=1, timeout_seconds=timeout_seconds
    )
    rows = result["rows"]
    if not rows:
        return {"sql": sql, "mode": mode_clean, "plan_text": "", "summary": {}, "error": "EXPLAIN returned no rows"}
    # First column whatever its name is
    plan_text = list(rows[0].values())[0] or ""

    # Parse summary from the plan text
    joins = _JOIN_PATTERN.findall(plan_text)
    generic_join_count = len(_GENERIC_JOIN_PATTERN.findall(plan_text))
    exchanges = _EXCHANGE_PATTERN.findall(plan_text)
    scans = _SCAN_PATTERN.findall(plan_text)
    # Independent extractors — robust against nested parens like ColumnStat(...)
    row_counts_seen = _ROWCOUNT_PATTERN.findall(plan_text)
    sizes_seen = _SIZEBYTES_PATTERN.findall(plan_text)

    from collections import Counter
    join_counts = Counter(joins)
    # If we found generic `Join` operators that didn't match the specific
    # strategy regex (common in COST mode logical plan or Photon's abstract
    # form), record them as `Join (strategy unknown)`.
    unknown_joins = max(0, generic_join_count - sum(join_counts.values()))
    if unknown_joins:
        join_counts["Join (strategy unspecified)"] = unknown_joins

    warnings: list[str] = []
    if "SortMergeJoin" in join_counts:
        warnings.append(
            f"{join_counts['SortMergeJoin']} SortMergeJoin(s) — if one side is small, "
            "consider /*+ BROADCAST(...) */ hint to avoid a shuffle."
        )
    if "CartesianProduct" in join_counts:
        warnings.append(
            f"{join_counts['CartesianProduct']} CartesianProduct — these explode "
            "rows by N×M. Almost always wrong unless intentional."
        )
    if "BroadcastNestedLoopJoin" in join_counts:
        # BNLJ ≈ "join condition has no equality key, only inequality / interval
        # predicate" (BETWEEN, <, >, overlap). Without a hint Spark falls back
        # to O(n·m). Databricks Photon often emits exactly this hint
        # suggestion in run insights — mirror it here.
        warnings.append(
            f"{join_counts['BroadcastNestedLoopJoin']} BroadcastNestedLoopJoin "
            "— this is a non-equi (range/interval) join. Add a "
            "/*+ RANGE_JOIN(rel, bin_size) */ hint with bin_size ≈ the typical "
            "interval length in the range column's units (e.g. 3600 for "
            "seconds with ~1h intervals). Same results, ~linear instead of "
            "O(n·m)."
        )
    if len(exchanges) >= 5:
        warnings.append(
            f"{len(exchanges)} Exchange (shuffle) operators — heavy shuffling. "
            "Each Exchange is a stage boundary that materializes data on disk."
        )

    summary: dict[str, Any] = {
        "join_strategies": dict(join_counts),
        "num_shuffles": len(exchanges),
        "scan_count": len(scans),
        "scans": scans[:10],  # first 10 (truncate to keep size bounded)
        "estimated_row_counts": row_counts_seen[:15],
        "estimated_sizes": sizes_seen[:15],
        "warnings": warnings,
    }

    # Keep plan_text bounded so a 50KB plan doesn't blow up the agent's context.
    if len(plan_text) > 8000:
        plan_text = plan_text[:8000] + f"\n...[truncated, full plan was {len(plan_text)} chars]"

    return {
        "sql": sql[:500],  # echo back truncated query for clarity
        "mode": mode_clean or "default",
        "plan_text": plan_text,
        "summary": summary,
    }


# Module-level run-stamp heuristic (used by diff_tables AND by propose.py's
# self-authorizing safety gate — both need the same definition of "this
# column is a write-time stamp, never part of the data's identity").
_RUN_STAMP_NAMES: frozenset[str] = frozenset({
    "last_refresh_time", "last_refresh_date", "refresh_time", "refresh_timestamp",
    "last_updated", "last_updated_at", "updated_at", "updated_time", "update_timestamp",
    "loaded_at", "load_timestamp", "load_time", "etl_timestamp", "etl_time",
    "etl_loaded_at", "ingested_at", "ingestion_time", "ingestion_timestamp",
    "inserted_at", "insert_timestamp", "processed_at", "process_timestamp",
    "run_timestamp", "run_time", "_run_id", "batch_id", "batch_timestamp",
    "dbt_updated_at", "dbt_loaded_at", "_loaded_at", "record_loaded_time",
    "created_at", "create_time", "created_timestamp",
})
_RUN_STAMP_SUFFIXES: tuple[str, ...] = (
    "_refresh_time", "_refresh_date", "_loaded_at",
    "_etl_timestamp", "_run_timestamp", "_ingested_at",
)


def looks_run_stamp(name: str) -> bool:
    """True iff `name` matches a known ETL run-stamp pattern. Type-agnostic;
    callers gate further on TIMESTAMP/DATE typing if they need to be strict."""
    ln = name.lower().strip("`")
    if ln in _RUN_STAMP_NAMES:
        return True
    return ln.endswith(_RUN_STAMP_SUFFIXES)


DIFF_TABLES_SCHEMA: dict[str, Any] = {
    "name": "diff_tables",
    "description": (
        "Row-level equivalence check between two Delta tables. Use this BEFORE "
        "declaring a query optimization correct — coarse checks (COUNT(*), "
        "SUM(col)) miss bugs where individual rows are wrong but totals match.\n"
        "\n"
        "Performs a FULL OUTER JOIN on the natural key (auto-detected from schema "
        "if not provided) and categorizes each row: identical / extra_in_b / "
        "missing_from_b / same_key_drifted_metric. For drifted metrics, reports "
        "which COLUMNS drift and the total magnitude.\n"
        "\n"
        "Auto-detection of natural_key vs metric_columns:\n"
        "  - STRING / DATE / TIMESTAMP / BOOLEAN  → dimension\n"
        "  - DOUBLE / FLOAT / DECIMAL              → metric (you SUM these)\n"
        "  - INT / BIGINT / SMALLINT               → metric IF name looks count-like\n"
        "        (starts with count_, n_, num_, total_, sum_, p_keys, qty_, etc.)\n"
        "        otherwise dimension.\n"
        "If the auto-detection looks wrong for your table (e.g. there's a column\n"
        "called `customer_age` that's INT but is a metric), pass either\n"
        "`natural_key` or `metric_columns` explicitly. The other list is auto-\n"
        "derived from whichever you pass.\n"
        "\n"
        "Cost: O(rows_a + rows_b) full table scans of both tables. Budget 5–30 "
        "minutes for tables with tens of millions of rows."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "table_a": {
                "type": "string",
                "description": "Fully-qualified name of the 'reference' (e.g. prod) table.",
            },
            "table_b": {
                "type": "string",
                "description": "Fully-qualified name of the 'optimized' (e.g. sandbox) table.",
            },
            "natural_key": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional natural-key columns. If omitted, auto-detected from schema (all non-numeric columns).",
            },
            "metric_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional metric columns to compare. If omitted, auto-detected (all numeric columns).",
            },
            "ignore_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Columns to exclude from BOTH the natural key and metric "
                    "comparison — use for ETL run-stamps (last_refresh_time, "
                    "loaded_at, batch_id, etc.) that differ every run by "
                    "construction. Common refresh/load/etl timestamp names are "
                    "auto-detected and excluded even if not listed here; the "
                    "result reports `auto_ignored_columns` so you can verify."
                ),
            },
            "rtol": {
                "type": "number",
                "default": 1e-9,
                "description": (
                    "Relative tolerance for DOUBLE/FLOAT columns. Two float "
                    "cells are equal iff |a-b| <= atol + rtol*max(|a|,|b|). "
                    "1e-9 is ~4 orders of magnitude above IEEE-754 reduction "
                    "noise (~1e-13) and far below any real logic bug. "
                    "DECIMAL/INT/string columns are ALWAYS compared exactly "
                    "(tolerance never applies to them)."
                ),
            },
            "atol": {
                "type": "number",
                "default": 1e-9,
                "description": "Absolute tolerance for DOUBLE/FLOAT columns (handles values near zero). See rtol.",
            },
            "reorder_rel_threshold": {
                "type": "number",
                "default": 1e-6,
                "description": (
                    "If float cells exceed rtol/atol but the worst relative "
                    "diff stays under this, the verdict is FLOAT_REORDER_ONLY "
                    "(large-magnitude reorder, still negligible) rather than "
                    "REAL_DIFFERENCE."
                ),
            },
            "top_k_drift_dims": {
                "type": "integer",
                "default": 10,
                "description": "Number of top dimension-value groups to return for each non-identical bucket.",
            },
            "timeout_seconds": {
                "type": "integer",
                "default": 1800,
                "description": "Max SQL wait per query. Default 30 min — full-table scans on large tables are slow.",
            },
        },
        "required": ["table_a", "table_b"],
    },
}


def diff_tables(
    table_a: str,
    table_b: str,
    natural_key: list[str] | None = None,
    metric_columns: list[str] | None = None,
    ignore_columns: list[str] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-9,
    reorder_rel_threshold: float = 1e-6,
    float_round_digits: int | None = None,  # DEPRECATED: superseded by rtol/atol
    top_k_drift_dims: int = 10,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    # Resolve schema from table_a (assume both have the same shape).
    # DESCRIBE TABLE does NOT accept Delta time-travel (`VERSION/TIMESTAMP AS
    # OF`) — that clause is only valid in SELECT ... FROM. Strip it for the
    # schema lookup; the column list is version-independent for our purposes.
    schema_ref = re.sub(
        r"\s+(?:VERSION|TIMESTAMP)\s+AS\s+OF\s+.*$", "", table_a, flags=re.IGNORECASE
    ).strip()
    cols_resp = execute_sql(
        f"DESCRIBE TABLE {schema_ref}", row_limit=500, timeout_seconds=120
    )
    cols: list[tuple[str, str]] = []
    for row in cols_resp["rows"]:
        name = (row.get("col_name") or "").strip()
        dtype = (row.get("data_type") or "").strip().upper()
        if not name or name.startswith("#"):
            break
        cols.append((name, dtype))

    _integer_substrings = ("INT", "BIGINT", "LONG", "SHORT", "TINYINT", "SMALLINT")
    _float_substrings = ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC")

    def is_integer(t: str) -> bool:
        return any(s in t for s in _integer_substrings) and not any(s in t for s in ("DECIMAL", "NUMERIC"))

    def is_float(t: str) -> bool:
        return any(s in t for s in ("DOUBLE", "FLOAT", "REAL"))

    def is_decimal(t: str) -> bool:
        return any(s in t for s in ("DECIMAL", "NUMERIC"))

    # Auto-detection contract:
    #   FLOAT / DOUBLE / DECIMAL  → metric  (you SUM these)
    #   INT / BIGINT / SMALLINT   → metric IF name looks count-like (count, n_*, p_keys,
    #                                       num_*, total_*, sum_*) — else dimension
    #   STRING / DATE / TIMESTAMP / BOOLEAN → dimension
    # Caller can override either list explicitly. When in doubt, pass both lists.
    _count_like_prefixes = ("count", "n_", "num_", "total_", "sum_", "p_keys", "qty_", "amt_")
    def looks_count_like(name: str) -> bool:
        ln = name.lower()
        return any(ln.startswith(p) or ln.endswith("_" + p) or ln.endswith("_count")
                   or ln == "count" for p in _count_like_prefixes)

    # ETL run-stamp columns: written with the job's run time / batch id, so
    # they DIFFER on every run by construction. Including them in the natural
    # key makes the FULL OUTER JOIN match nothing (100% extra/missing). They
    # must be dropped from BOTH key and metrics — they're not part of the
    # data's identity and aren't meaningful to compare. Matched conservatively:
    # only well-known refresh/load/etl stamp names, not generic *_date / *_time
    # (those are usually business dates that SHOULD be in the key).
    # _RUN_STAMP_NAMES / looks_run_stamp are at module scope (below).

    ignored = set(ignore_columns or [])
    auto_ignored: list[str] = []
    for n, t in cols:
        if n in ignored:
            continue
        # Only auto-ignore temporal-typed columns whose name screams run-stamp.
        if looks_run_stamp(n) and any(s in t for s in ("TIMESTAMP", "DATE")):
            ignored.add(n)
            auto_ignored.append(n)

    usable_cols = [(n, t) for n, t in cols if n not in ignored]

    if natural_key is None and metric_columns is None:
        natural_key = []
        metric_columns = []
        for n, t in usable_cols:
            if is_float(t) or is_decimal(t):
                metric_columns.append(n)
            elif is_integer(t):
                if looks_count_like(n):
                    metric_columns.append(n)
                else:
                    natural_key.append(n)
            else:
                natural_key.append(n)
    elif natural_key is None:
        natural_key = [n for n, t in usable_cols if n not in metric_columns]
    elif metric_columns is None:
        metric_columns = [n for n, t in usable_cols if n not in natural_key]
    # Honour explicit ignore even when caller passed key/metrics lists
    natural_key = [c for c in natural_key if c not in ignored]
    metric_columns = [c for c in metric_columns if c not in ignored]

    col_types = dict(cols)

    # IEEE-754 double addition is non-associative: a correct query rewrite that
    # changes join/shuffle order produces bit-different DOUBLE sums (~1e-13
    # relative). Fixed-decimal rounding is the wrong instrument — FP error is
    # RELATIVE to magnitude, not absolute. Use a magnitude-aware tolerance
    # (the numpy.isclose criterion) on DOUBLE/FLOAT only; DECIMAL/INT/string
    # are compared EXACTLY (Spark DECIMAL arithmetic is exact & order-stable).
    def _num(x: float) -> str:
        return repr(float(x))

    def cell_equal(c: str) -> str:
        """SQL boolean: TRUE iff a.c and b.c are equal (tolerant for floats)."""
        a, b = f"a.`{c}`", f"b.`{c}`"
        if not is_float(col_types.get(c, "")):
            return f"({a} <=> {b})"  # exact, null-safe (DECIMAL/INT/string)
        return (
            f"( ({a} IS NULL AND {b} IS NULL) "
            f"OR ({a} IS NOT NULL AND {b} IS NOT NULL AND ( "
            f"(isnan({a}) AND isnan({b})) "
            f"OR abs({a} - {b}) <= "
            f"({_num(atol)} + {_num(rtol)} * greatest(abs({a}), abs({b}))) )) )"
        )

    def cell_diff(c: str) -> str:
        return f"(NOT {cell_equal(c)})"

    join_cond = " AND ".join(f"a.`{k}` <=> b.`{k}`" for k in natural_key)

    # Wrap both tables in `(SELECT * FROM ...)` so a Delta time-travel suffix
    # (`VERSION/TIMESTAMP AS OF`) can be aliased. `FROM tbl VERSION AS OF 5 a`
    # mis-parses the alias; `FROM (SELECT * FROM tbl VERSION AS OF 5) a` is
    # robust. Spark's optimizer flattens the trivial subquery — no perf cost.
    ta = f"(SELECT * FROM {table_a})"
    tb = f"(SELECT * FROM {table_b})"

    # 1. Categorize every row
    metric_diff_clauses = " OR ".join(
        cell_diff(c) for c in metric_columns
    ) or "false"

    bucket_case = f"""
        CASE
            WHEN a.`{natural_key[0]}` IS NULL THEN 'extra_in_b'
            WHEN b.`{natural_key[0]}` IS NULL THEN 'missing_from_b'
            WHEN {metric_diff_clauses} THEN 'same_key_drifted_metric'
            ELSE 'identical'
        END
    """
    bucket_sql = f"""
    WITH joined AS (
      SELECT {bucket_case} AS bucket
      FROM {ta} a FULL OUTER JOIN {tb} b ON {join_cond}
    )
    SELECT bucket, COUNT(*) AS n FROM joined GROUP BY 1
    """
    bucket_rows = execute_sql(bucket_sql, timeout_seconds=timeout_seconds)["rows"]
    buckets = {r["bucket"]: int(r["n"]) for r in bucket_rows}
    for key in ("identical", "same_key_drifted_metric", "extra_in_b", "missing_from_b"):
        buckets.setdefault(key, 0)

    # 2. Per-metric drift summary (total magnitude — distinguishes float reorder
    # vs real semantic drift)
    metric_summary: dict[str, dict[str, Any]] = {}
    if metric_columns:
        sums_a = ", ".join(
            f"CAST(SUM(`{c}`) AS DOUBLE) AS a_sum_{c}" for c in metric_columns
        )
        sums_b = ", ".join(
            f"CAST(SUM(`{c}`) AS DOUBLE) AS b_sum_{c}" for c in metric_columns
        )
        # Per-column drift count + (float only) drift PROFILE: worst absolute
        # and worst RELATIVE diff among the rows that actually exceeded
        # tolerance. This makes the verdict evidence-based — "max rel 3e-13"
        # is obviously reorder noise; "max rel 0.4" is a real bug.
        _diff_parts: list[str] = []
        for c in metric_columns:
            _diff_parts.append(
                f"SUM(CASE WHEN {cell_diff(c)} THEN 1 ELSE 0 END) AS diff_{c}"
            )
            if is_float(col_types.get(c, "")):
                a, b = f"a.`{c}`", f"b.`{c}`"
                _diff_parts.append(
                    f"MAX(CASE WHEN {cell_diff(c)} THEN abs({a} - {b}) END) "
                    f"AS maxabs_{c}"
                )
                _diff_parts.append(
                    f"MAX(CASE WHEN {cell_diff(c)} "
                    f"AND greatest(abs({a}), abs({b})) > 0 "
                    f"THEN abs({a} - {b}) / greatest(abs({a}), abs({b})) END) "
                    f"AS maxrel_{c}"
                )
        per_col_diff_counts = ", ".join(_diff_parts)
        # Per-table sums + per-column drift counts (kept as separate queries).
        sums_a_sql = f"SELECT {sums_a} FROM {ta}"
        sums_b_sql = f"SELECT {sums_b} FROM {tb}"
        diff_counts_sql = (
            f"SELECT {per_col_diff_counts} "
            f"FROM {ta} a JOIN {tb} b ON {join_cond}"
        )
        a_totals = execute_sql(sums_a_sql, timeout_seconds=timeout_seconds)["rows"][0]
        b_totals = execute_sql(sums_b_sql, timeout_seconds=timeout_seconds)["rows"][0]
        diff_counts = execute_sql(diff_counts_sql, timeout_seconds=timeout_seconds)["rows"][0]

        for c in metric_columns:
            a_sum = a_totals.get(f"a_sum_{c}")
            b_sum = b_totals.get(f"b_sum_{c}")
            try:
                a_f = float(a_sum) if a_sum is not None else None
                b_f = float(b_sum) if b_sum is not None else None
                delta = (b_f - a_f) if (a_f is not None and b_f is not None) else None
                rel = (delta / a_f) if (delta is not None and a_f) else None
            except (TypeError, ValueError):
                a_f = b_f = delta = rel = None
            entry: dict[str, Any] = {
                "type": col_types.get(c),
                "is_float": is_float(col_types.get(c, "")),
                "rows_drifted": int(diff_counts.get(f"diff_{c}") or 0),
                "a_total": a_f,
                "b_total": b_f,
                "total_delta": delta,
                "total_delta_relative": rel,
            }
            if entry["is_float"]:
                ma = diff_counts.get(f"maxabs_{c}")
                mr = diff_counts.get(f"maxrel_{c}")
                try:
                    entry["max_abs_diff"] = float(ma) if ma is not None else None
                except (TypeError, ValueError):
                    entry["max_abs_diff"] = None
                try:
                    entry["max_rel_diff"] = float(mr) if mr is not None else None
                except (TypeError, ValueError):
                    entry["max_rel_diff"] = None
            metric_summary[c] = entry

    # 3. Top-K dimension concentrations for non-identical rows
    drift_concentration: dict[str, list[dict[str, Any]]] = {}
    if natural_key and (buckets["extra_in_b"] or buckets["same_key_drifted_metric"]):
        first_dim = f"`{natural_key[0]}`"  # group by first key column as a quick locator
        for which_bucket, where_clause in (
            ("extras_in_b", "a.`" + natural_key[0] + "` IS NULL"),
            ("drifted_metrics",
             f"a.`{natural_key[0]}` IS NOT NULL AND b.`{natural_key[0]}` IS NOT NULL "
             f"AND ({metric_diff_clauses})"),
        ):
            ref_table = "b" if which_bucket == "extras_in_b" else "a"
            try:
                top = execute_sql(
                    f"""
                    SELECT {ref_table}.{first_dim} AS dim_value, COUNT(*) AS n
                    FROM {ta} a FULL OUTER JOIN {tb} b ON {join_cond}
                    WHERE {where_clause}
                    GROUP BY {ref_table}.{first_dim}
                    ORDER BY n DESC
                    LIMIT {top_k_drift_dims}
                    """,
                    timeout_seconds=timeout_seconds,
                )["rows"]
                drift_concentration[which_bucket] = [
                    {"dim_value": r["dim_value"], "rows": int(r["n"])} for r in top
                ]
            except Exception as e:
                drift_concentration[which_bucket] = [{"error": str(e)}]

    # 4. Interpretation — evidence-based, driven by the drift profile.
    #   rows_drifted now means "cells that EXCEEDED rtol/atol tolerance"
    #   (sub-tolerance FP-reorder noise is absorbed into 'identical').
    real_drift_cols = [
        c for c, m in metric_summary.items()
        if (not m["is_float"]) and m["rows_drifted"] > 0
    ]
    float_drift_cols = [
        c for c, m in metric_summary.items()
        if m["is_float"] and m["rows_drifted"] > 0
    ]
    # Worst relative diff among float columns that exceeded tolerance.
    worst_float_rel = max(
        (metric_summary[c].get("max_rel_diff") or 0.0) for c in float_drift_cols
    ) if float_drift_cols else 0.0
    structural_drift = buckets["extra_in_b"] > 0 or buckets["missing_from_b"] > 0

    # Compact profile for humans / proposal.md.
    drift_profile = [
        {
            "column": c,
            "type": metric_summary[c]["type"],
            "rows_drifted": metric_summary[c]["rows_drifted"],
            "max_abs_diff": metric_summary[c].get("max_abs_diff"),
            "max_rel_diff": metric_summary[c].get("max_rel_diff"),
        }
        for c in (real_drift_cols + float_drift_cols)
    ]

    if structural_drift or real_drift_cols:
        verdict = "REAL_DIFFERENCE"
        interp = (
            "Real semantic drift detected: "
            + (f"{buckets['extra_in_b']} extra rows in B, "
               f"{buckets['missing_from_b']} missing. " if structural_drift else "")
            + (f"Exact (DECIMAL/INT/string) columns drift: {real_drift_cols}. "
               if real_drift_cols else "")
            + "Tables are NOT equivalent."
        )
    elif float_drift_cols and worst_float_rel > reorder_rel_threshold:
        verdict = "REAL_DIFFERENCE"
        interp = (
            f"Float columns {float_drift_cols} exceed tolerance with worst "
            f"relative diff {worst_float_rel:.3e} (> reorder threshold "
            f"{reorder_rel_threshold:.0e}). Too large for FP-reorder noise — "
            "the optimized query's arithmetic is genuinely different."
        )
    elif float_drift_cols:
        verdict = "FLOAT_REORDER_ONLY"
        interp = (
            f"Only DOUBLE/FLOAT columns {float_drift_cols} differ, worst "
            f"relative diff {worst_float_rel:.3e} (<= {reorder_rel_threshold:.0e}). "
            "Consistent with IEEE-754 aggregation-reorder noise — functionally "
            "equivalent."
        )
    else:
        verdict = "IDENTICAL"
        interp = (
            f"Equivalent within tolerance (rtol={rtol:.0e}, atol={atol:.0e}); "
            "any sub-tolerance float drift is IEEE-754 reorder noise."
        )

    return {
        "table_a": table_a,
        "table_b": table_b,
        "natural_key": natural_key,
        "metric_columns": metric_columns,
        "ignored_columns": sorted(ignored),
        "auto_ignored_columns": auto_ignored,
        "tolerance": {
            "rtol": rtol, "atol": atol,
            "reorder_rel_threshold": reorder_rel_threshold,
        },
        "buckets": buckets,
        "metric_summary": metric_summary,
        "drift_profile": drift_profile,
        "worst_float_rel_diff": worst_float_rel,
        "drift_concentration": drift_concentration,
        "verdict": verdict,
        "interpretation": interp,
    }


GET_NOTEBOOK_SOURCE_SCHEMA: dict[str, Any] = {
    "name": "get_notebook_source",
    "description": (
        "Read a notebook's source code from a Databricks workspace path. "
        "Returns the raw `# Databricks notebook source`-formatted text. Use this "
        "instead of `export_job_run` when you just need to see the code — it "
        "returns the source directly without HTML wrapping or base64 encoding."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "workspace_path": {
                "type": "string",
                "description": "Workspace path of the notebook (e.g. /Shared/team/notebook).",
            },
        },
        "required": ["workspace_path"],
    },
}


def get_notebook_source(workspace_path: str) -> dict[str, Any]:
    import base64
    w = workspace()
    response = w.workspace.export(path=workspace_path, format=ExportFormat.SOURCE)
    content = base64.b64decode(response.content).decode("utf-8") if response.content else ""
    # file_type may come back as the Enum or as a plain string depending on SDK
    # version; coerce defensively.
    ft = response.file_type
    if hasattr(ft, "value"):
        ft = ft.value
    return {
        "workspace_path": workspace_path,
        "file_type": ft,
        "content": content,
        "length": len(content),
    }


UPLOAD_NOTEBOOK_SCHEMA: dict[str, Any] = {
    "name": "upload_notebook",
    "description": (
        "Upload a notebook to a Databricks workspace path. Replaces the file at "
        "`workspace_path` if it already exists.\n"
        "\n"
        "Content format is auto-detected:\n"
        "  - If `content` parses as JSON with a top-level `cells` key, it's "
        "treated as a Jupyter `.ipynb` notebook and uploaded as JUPYTER format. "
        "The `language` argument is ignored (Databricks reads language from the "
        "notebook metadata).\n"
        "  - Otherwise, content is treated as Databricks SOURCE format text "
        "(`# Databricks notebook source` + `# MAGIC` prefixes) and uploaded as "
        "SOURCE format with the caller-specified `language`.\n"
        "\n"
        "Use this instead of shelling out to curl/python with the workspace "
        "REST API."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "workspace_path": {
                "type": "string",
                "description": "Target workspace path (e.g. /Shared/team/notebook).",
            },
            "content": {
                "type": "string",
                "description": "Raw notebook source. Include the `# Databricks notebook source` header.",
            },
            "language": {
                "type": "string",
                "enum": ["PYTHON", "SQL", "SCALA", "R"],
                "default": "PYTHON",
                "description": "Notebook language.",
            },
        },
        "required": ["workspace_path", "content"],
    },
}


def upload_notebook(
    workspace_path: str, content: str, language: str = "PYTHON"
) -> dict[str, Any]:
    # Auto-detect .ipynb JSON: content that parses as a JSON object with a top-
    # level "cells" key is a Jupyter notebook. Upload it as JUPYTER format so
    # Databricks parses the cells, languages, and outputs correctly. Otherwise
    # treat the content as raw SOURCE format text (the `# Databricks notebook
    # source` + `# MAGIC`-prefixed format) with the caller-specified language.
    import json as _json
    w = workspace()
    parent = workspace_path.rsplit("/", 1)[0]
    if parent:
        w.workspace.mkdirs(parent)

    is_ipynb = False
    if content.lstrip().startswith("{"):
        try:
            parsed = _json.loads(content)
            is_ipynb = isinstance(parsed, dict) and "cells" in parsed
        except (ValueError, TypeError):
            pass

    if is_ipynb:
        w.workspace.upload(
            path=workspace_path,
            content=content.encode("utf-8"),
            format=ImportFormat.JUPYTER,
            overwrite=True,
        )
        fmt = "JUPYTER"
    else:
        w.workspace.upload(
            path=workspace_path,
            content=content.encode("utf-8"),
            format=ImportFormat.SOURCE,
            language=Language(language.upper()),
            overwrite=True,
        )
        fmt = "SOURCE"
    return {
        "workspace_path": workspace_path,
        "bytes_written": len(content.encode("utf-8")),
        "format_used": fmt,
    }


EXPORT_JOB_RUN_SCHEMA: dict[str, Any] = {
    "name": "export_job_run",
    "description": (
        "Export a job run's notebook view as HTML — CODE, DASHBOARDS, or ALL. "
        "Only supported for notebook-task runs; will error on python/SQL/JAR/wheel tasks. "
        "For multi-task jobs pass the task-level run_id of a notebook task."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "run_id": {"type": "integer", "description": "The run ID to export."},
            "views_to_export": {
                "type": "string",
                "enum": ["CODE", "DASHBOARDS", "ALL"],
                "default": "CODE",
                "description": "Which views to export.",
            },
        },
        "required": ["run_id"],
    },
}


def export_job_run(run_id: int, views_to_export: str = "CODE") -> dict[str, Any]:
    w = workspace()
    response = w.jobs.export_run(run_id=run_id, views_to_export=ViewsToExport(views_to_export))
    views = response.views or []
    return {
        "view_count": len(views),
        "views": [
            {
                "name": v.name,
                "type": v.type.value if v.type else None,
                "content": v.content,
            }
            for v in views
        ],
    }


# =============================================================================
# Job Permissions
# =============================================================================

GET_JOB_PERMISSIONS_SCHEMA: dict[str, Any] = {
    "name": "get_job_permissions",
    "description": "Get the current permission assignments on a Databricks job.",
    "input_schema": {
        "type": "object",
        "properties": {"job_id": {"type": "integer", "description": "The job ID."}},
        "required": ["job_id"],
    },
}


def get_job_permissions(job_id: int) -> dict[str, Any]:
    w = workspace()
    return w.permissions.get(request_object_type="jobs", request_object_id=str(job_id)).as_dict()


GET_JOB_PERMISSION_LEVELS_SCHEMA: dict[str, Any] = {
    "name": "get_job_permission_levels",
    "description": "List the permission levels that can be granted on a Databricks job.",
    "input_schema": {
        "type": "object",
        "properties": {"job_id": {"type": "integer", "description": "The job ID."}},
        "required": ["job_id"],
    },
}


def get_job_permission_levels(job_id: int) -> dict[str, Any]:
    w = workspace()
    levels = w.permissions.get_permission_levels(
        request_object_type="jobs", request_object_id=str(job_id)
    )
    return {
        "permission_levels": [
            {
                "level": pl.permission_level.value if pl.permission_level else None,
                "description": pl.description,
            }
            for pl in (levels.permission_levels or [])
        ]
    }


# =============================================================================
# Registry
# =============================================================================

Tool = tuple[dict[str, Any], Callable[..., Any]]

REGISTRY: dict[str, Tool] = {
    "execute_sql": (EXECUTE_SQL_SCHEMA, execute_sql),
    "check_table_health": (CHECK_TABLE_HEALTH_SCHEMA, check_table_health),
    "list_jobs": (LIST_JOBS_SCHEMA, list_jobs),
    "get_job": (GET_JOB_SCHEMA, get_job),
    "create_job": (CREATE_JOB_SCHEMA, create_job),
    "run_job_now": (RUN_JOB_NOW_SCHEMA, run_job_now),
    "add_job_tasks": (ADD_JOB_TASKS_SCHEMA, add_job_tasks),
    "list_job_runs": (LIST_JOB_RUNS_SCHEMA, list_job_runs),
    "get_job_run": (GET_JOB_RUN_SCHEMA, get_job_run),
    "wait_for_job_run": (WAIT_FOR_JOB_RUN_SCHEMA, wait_for_job_run),
    "upload_notebook": (UPLOAD_NOTEBOOK_SCHEMA, upload_notebook),
    "get_notebook_source": (GET_NOTEBOOK_SOURCE_SCHEMA, get_notebook_source),
    "diff_tables": (DIFF_TABLES_SCHEMA, diff_tables),
    "explain_query": (EXPLAIN_QUERY_SCHEMA, explain_query),
    "get_job_run_output": (GET_JOB_RUN_OUTPUT_SCHEMA, get_job_run_output),
    "export_job_run": (EXPORT_JOB_RUN_SCHEMA, export_job_run),
    "get_job_permissions": (GET_JOB_PERMISSIONS_SCHEMA, get_job_permissions),
    "get_job_permission_levels": (GET_JOB_PERMISSION_LEVELS_SCHEMA, get_job_permission_levels),
}


