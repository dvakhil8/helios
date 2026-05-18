"""Comprehensive read-only smoke test for helios Databricks tools.

Chains tool calls — uses list_jobs to find a real job_id, then exercises the
per-job tools (get_job, permissions, runs, run output, export) on it.

Required env vars:
  DATABRICKS_HOST
  DATABRICKS_TOKEN
  DATABRICKS_WAREHOUSE_ID   (optional — skips execute_sql tests if missing)

Optional env var:
  TEST_TABLE                (fully qualified catalog.schema.table for check_table_health)

WRITE tools (create_job, run_job_now, add_job_tasks) are NOT exercised here.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from helios.tools import all_schemas, call_tool


PASS: list[str] = []
FAIL: list[tuple[str, str]] = []
SKIP: list[tuple[str, str]] = []


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n {title}\n{'=' * 72}")


def show(obj: Any, limit: int = 2000) -> None:
    s = json.dumps(obj, indent=2, default=str)
    print(s if len(s) <= limit else s[:limit] + f"\n... (truncated, {len(s)} chars)")


def call(label: str, name: str, **kwargs: Any) -> Any | None:
    section(f"{label}  ::  {name}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})")
    try:
        result = call_tool(name, **kwargs)
        show(result)
        PASS.append(label)
        return result
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        FAIL.append((label, f"{type(e).__name__}: {e}"))
        return None


def skip(label: str, reason: str) -> None:
    section(f"{label}  ::  SKIPPED — {reason}")
    SKIP.append((label, reason))


def main() -> int:
    for v in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        if not os.environ.get(v):
            print(f"Missing required env var: {v}", file=sys.stderr)
            return 2

    section(f"Registry — {len(all_schemas())} tools loaded")
    for s in all_schemas():
        print(f"  - {s['name']}")

    # --- 1. list_jobs ---
    jobs_result = call("list_jobs", "list_jobs", limit=10)
    jobs = (jobs_result or {}).get("jobs", []) if isinstance(jobs_result, dict) else []
    job_id = jobs[0]["job_id"] if jobs else None

    # --- 2. job-scoped tools ---
    if job_id is None:
        for t in ("get_job", "get_job_permissions", "get_job_permission_levels"):
            skip(t, "no jobs found in workspace")
    else:
        call("get_job", "get_job", job_id=job_id)
        call("get_job_permissions", "get_job_permissions", job_id=job_id)
        call(
            "get_job_permission_levels",
            "get_job_permission_levels",
            job_id=job_id,
        )

    # --- 3. list_job_runs (completed) ---
    runs_result = call("list_job_runs", "list_job_runs", limit=5, completed_only=True)
    runs = (runs_result or {}).get("runs", []) if isinstance(runs_result, dict) else []
    run_id = runs[0]["run_id"] if runs else None

    # --- 4. run-scoped tools ---
    if run_id is None:
        for t in ("get_job_run", "get_job_run_output", "export_job_run"):
            skip(t, "no completed runs found")
    else:
        run_detail = call("get_job_run", "get_job_run", run_id=run_id, include_history=True)

        # Find a task-level run_id for get_job_run_output
        task_run_id = None
        if isinstance(run_detail, dict):
            tasks = run_detail.get("tasks") or []
            for t in tasks:
                if isinstance(t, dict) and t.get("run_id"):
                    task_run_id = t["run_id"]
                    break
        if task_run_id is None:
            task_run_id = run_id

        call("get_job_run_output", "get_job_run_output", run_id=task_run_id)

        # export_job_run only works on notebook-task runs — scan completed runs
        # to find one whose top-level task is a notebook task.
        notebook_run_id = None
        for r in runs:
            detail = call_tool("get_job_run", run_id=r["run_id"])
            for t in detail.get("tasks") or []:
                if t.get("notebook_task") and t.get("run_id"):
                    notebook_run_id = t["run_id"]
                    break
            if notebook_run_id:
                break
        if notebook_run_id:
            call(
                "export_job_run",
                "export_job_run",
                run_id=notebook_run_id,
                views_to_export="CODE",
            )
        else:
            skip(
                "export_job_run",
                "no completed notebook-task runs in the first batch — tool is "
                "notebook-only by Databricks API design",
            )

    # --- 5. execute_sql ---
    if os.environ.get("DATABRICKS_WAREHOUSE_ID"):
        call(
            "execute_sql:whoami",
            "execute_sql",
            sql="SELECT current_timestamp() AS now, current_user() AS who",
        )
        call("execute_sql:catalogs", "execute_sql", sql="SHOW CATALOGS")
        # Best-effort: system tables may not be enabled
        call(
            "execute_sql:billing_peek",
            "execute_sql",
            sql=(
                "SELECT usage_date, sku_name, usage_quantity "
                "FROM system.billing.usage "
                "WHERE usage_date >= current_date() - INTERVAL 3 DAYS "
                "LIMIT 3"
            ),
        )
    else:
        for t in ("execute_sql:whoami", "execute_sql:catalogs", "execute_sql:billing_peek"):
            skip(t, "DATABRICKS_WAREHOUSE_ID not set")

    # --- 6. check_table_health (only if user provided a target) ---
    table = os.environ.get("TEST_TABLE")
    if table:
        call("check_table_health", "check_table_health", table=table, lookback_days=14)
    else:
        skip("check_table_health", "TEST_TABLE env var not set — pass a catalog.schema.table")

    # --- 7. WRITE tools — never auto-exercised ---
    for t in ("create_job", "run_job_now", "add_job_tasks"):
        skip(t, "write tool — requires explicit confirmation + sandbox target")

    # --- Summary ---
    section("Summary")
    print(f"  PASS  ({len(PASS)})")
    for n in PASS:
        print(f"    + {n}")
    print(f"  FAIL  ({len(FAIL)})")
    for n, err in FAIL:
        print(f"    - {n}  ::  {err}")
    print(f"  SKIP  ({len(SKIP)})")
    for n, reason in SKIP:
        print(f"    . {n}  ::  {reason}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
