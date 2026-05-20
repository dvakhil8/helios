"""Propose mode — point Helios at a real production job and produce an
optimization proposal *without* ever modifying the prod job.

Differs from eval-mode harness in five places:
  1. No fixture.yaml. Inputs are a live prod job_id + a task_key to scope to.
  2. No seed.sql. Source data is real prod (the cloned task reads real prod tables).
  3. Notebook source is pulled from GitHub (Pocket-Fm/de_databricks) because
     prod tasks are git-sourced. Reads-vs-writes are determined by regex; only
     the WRITE target is rewritten to point at the sandbox catalog.
  4. Baseline is history-based — pulled from prior `get_job_run.tasks[].
     execution_duration`. No baseline re-run.
  5. Tool guard hard-blocks any mutation tool whose `job_id == prod_job_id`.

Output is `proposal.md` — a self-contained markdown document a human reviews
before deciding whether to apply the change to prod. The sandbox clone job
remains in the workspace as proof; teardown is opt-in.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from ..tools.databricks import (
    create_job,
    execute_sql,
    get_job,
    get_job_run,
    list_job_runs,
    run_job_now,
    upload_notebook,
    wait_for_job_run,
    workspace,
)
from . import baselines, sandbox
from .baselines import _hash_output_table, compute_stats_fingerprint
from .runner import run_agent
from .scorers.diagnosis import score as score_tier2_eval


# Where proposal artifacts live locally
PROPOSALS_ROOT: Path = Path(__file__).resolve().parents[3] / "evals" / "proposals"

# GitHub repo that holds the prod notebooks. The job's git_source field tells us
# this at run time, but we hard-code the org as a fallback for now.
DEFAULT_REPO = "Pocket-Fm/de_databricks"


@dataclass
class TaskCloneResult:
    """Everything captured when cloning one prod task into the sandbox."""

    task_key: str
    original_notebook_path: str               # repo-relative path
    sandbox_notebook_path: str                # workspace path
    original_notebook_source: str             # raw source from GitHub
    sandbox_notebook_source: str              # rewritten source (write-target → sandbox)
    write_targets_original: list[str]         # FQNs the original wrote
    write_targets_sandbox: list[str]          # FQNs the sandbox version writes
    read_sources: list[str]                   # FQNs the notebook reads (unchanged)
    # Sources are pinned via TIMESTAMP AS OF to the moment the prod TASK
    # STARTED (Delta snapshot isolation captures source state at query start,
    # not at commit). Sandbox reads sources as the prod task saw them.
    source_alignment_timestamp: str | None = None
    source_alignment_basis: str = "unset"  # 'prod_task_run_start' | 'boundary_commit_fallback' | 'unset'
    source_pinned_fqns: dict[str, str] = field(default_factory=dict)  # {fqn: timestamp_used}
    # Prod boundary itself is pinned via VERSION AS OF (we know the exact
    # version at clone time) when diff_tables compares for equivalence.
    prod_boundary_versions: dict[str, int] = field(default_factory=dict)
    # Per-write-target mode: "full_rewrite" (CTAS / CREATE OR REPLACE / INSERT
    # OVERWRITE) vs "incremental" (INSERT INTO / MERGE INTO appending to an
    # accumulating table). Drives whether diff_tables compares the full prod
    # table or just the prod increment for this task's commit.
    write_target_modes: dict[str, str] = field(default_factory=dict)
    # For incremental targets only: the prod table's Delta version JUST BEFORE
    # the prod task's WRITE/MERGE commit. The increment = post EXCEPT pre.
    prod_pre_versions: dict[str, int] = field(default_factory=dict)
    # Partition columns of each write target (from DESCRIBE DETAIL). Used to
    # bound the prod-increment EXCEPT to just the affected partitions —
    # critical for avoiding archived-partition access and full-table scans.
    partition_columns_by_target: dict[str, list[str]] = field(default_factory=dict)
    unpinnable_sources: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Notebook source acquisition + rewriting
# =============================================================================

_WRITE_VERB_RE = re.compile(
    r"(?:CREATE\s+(?:OR\s+REPLACE\s+)?TABLE|REPLACE\s+TABLE|INSERT\s+(?:OVERWRITE|INTO)|"
    r"MERGE\s+INTO|TRUNCATE\s+TABLE|DROP\s+TABLE|ALTER\s+TABLE)"
    r"\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?",
    re.IGNORECASE,
)
_FQN_RE = re.compile(r"([a-zA-Z_]\w*\.[a-zA-Z_]\w*\.[a-zA-Z_]\w*)")
_READ_RE = re.compile(
    r"(?:FROM|JOIN)\s+([a-zA-Z_]\w*\.[a-zA-Z_]\w*\.[a-zA-Z_]\w*)",
    re.IGNORECASE,
)


def extract_tables(source: str) -> tuple[set[str], set[str]]:
    """Return (write_targets, read_sources) found in the notebook.

    Write detection: looks for a write verb followed within a few words by an
    FQN. Read detection: FROM/JOIN <FQN>.
    """
    writes: set[str] = set()
    for m in _WRITE_VERB_RE.finditer(source):
        tail = source[m.end():m.end() + 200]
        fqn_match = _FQN_RE.search(tail)
        if fqn_match:
            writes.add(fqn_match.group(1))
    reads = {m.group(1) for m in _READ_RE.finditer(source)} - writes
    return writes, reads


def get_prod_task_run_start_time(prod_job_id: int, task_key: str) -> str | None:
    """Return the START timestamp of the prod task's most recent SUCCESS run.

    Delta uses snapshot isolation — a query reads source tables at the version
    current when the query STARTED, not when it committed. So the correct
    alignment point for source-table pinning is the task run's start_time,
    NOT the boundary table's commit timestamp.

    Walks back through recent job runs (looking inside each for the named
    task) until it finds one where the task itself succeeded — even if the
    parent run as a whole failed OR is still RUNNING. A task that has already
    reached SUCCESS has committed its write target; downstream tasks in the
    same DAG write elsewhere and won't mutate it, so a finished task inside an
    in-flight parent run is a valid (and fresher) alignment point. Hence we
    list with completed_only=False and key off the *task's* terminal state,
    not the parent run's lifecycle. Returns the ISO-formatted UTC timestamp
    Spark TIMESTAMP AS OF accepts, or None if no successful task run found.
    """
    seen_run_ids: set[int] = set()
    cursor_ms: int | None = None
    pages = 0
    while pages < 6:
        kwargs: dict[str, Any] = {
            # completed_only=False so an in-flight parent run whose target
            # task has *already* finished SUCCESS is still considered.
            "job_id": prod_job_id, "completed_only": False, "limit": 25,
        }
        if cursor_ms is not None:
            kwargs["start_time_to_ms"] = cursor_ms - 1
        runs = list_job_runs(**kwargs).get("runs") or []
        if not runs:
            break
        for r in runs:
            rid = r["run_id"]
            if rid in seen_run_ids:
                continue
            seen_run_ids.add(rid)
            try:
                detail = get_job_run(rid)
            except Exception:
                continue
            for t in (detail.get("tasks") or []):
                if t.get("task_key") != task_key:
                    continue
                state = (t.get("state") or {}).get("result_state")
                # Accept the task only once it has terminally SUCCEEDED.
                # A still-RUNNING task in the newest run yields state in
                # {None, "RUNNING"} → skip, fall through to older runs.
                if state == "SUCCESS":
                    start_ms = t.get("start_time") or r.get("start_time")
                    if start_ms:
                        dt = datetime.fromtimestamp(int(start_ms) / 1000, tz=timezone.utc)
                        # Strip TZ — Spark TIMESTAMP AS OF interprets bare ts
                        # in session TZ, which is UTC by default on Databricks.
                        return _format_ts_for_spark(dt.replace(tzinfo=None))
                break  # only one task per task_key per run
        cursor_ms = min(r["start_time"] for r in runs)
        pages += 1
    return None


def get_prod_boundary_timestamp(write_targets: list[str]) -> str | None:
    """Return ISO timestamp of the most recent commit across all prod boundary
    write targets. This is the FALLBACK alignment point when no successful
    task run is found — it's the boundary's WRITE-COMMIT time, which is
    `start_time + duration` (potentially HOURS after the snapshot Delta
    captured for source reads). Prefer `get_prod_task_run_start_time`.

    Returns None if no Delta history is available on any write target.
    """
    max_ts_str: str | None = None
    for fqn in write_targets:
        try:
            r = execute_sql(f"DESCRIBE HISTORY {fqn} LIMIT 1", timeout_seconds=60)
            rows = r.get("rows") or []
            if not rows:
                continue
            ts = rows[0]["timestamp"]
            # Normalize to ISO-like string Spark TIMESTAMP AS OF accepts.
            if hasattr(ts, "isoformat"):
                ts_str = ts.isoformat(sep=" ")
            else:
                ts_str = str(ts)
            # Strip trailing timezone if present — Spark TIMESTAMP AS OF is
            # picky about the format.
            if "+" in ts_str:
                ts_str = ts_str.split("+", 1)[0]
            if max_ts_str is None or ts_str > max_ts_str:
                max_ts_str = ts_str
        except Exception:
            continue
    return max_ts_str


def _parse_spark_ts(ts: Any) -> datetime | None:
    """Best-effort parse of a Spark timestamp value (datetime object or string
    in various ISO-ish formats) to a Python datetime."""
    if isinstance(ts, datetime):
        return ts
    if ts is None:
        return None
    s = str(ts).rstrip("Z").replace("T", " ")
    # Strip timezone offset if present
    for tz_marker in ("+", "-"):
        idx = s.rfind(tz_marker)
        # Don't match the date separator dashes
        if idx > 10:
            s = s[:idx]
            break
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _format_ts_for_spark(dt: datetime) -> str:
    """Format datetime as 'YYYY-MM-DD HH:MM:SS.fff' (millisecond precision) —
    a format Spark TIMESTAMP AS OF reliably accepts."""
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def pin_sources_to_timestamp(
    source: str,
    fqns: list[str],
    boundary_timestamp: str,
    *,
    skip: set[str] | None = None,
) -> tuple[str, dict[str, str], list[dict[str, Any]]]:
    """Rewrite source-table references to `FROM <fqn> TIMESTAMP AS OF '<T>'`
    where T is per-source = MIN(boundary_timestamp, source.latest_commit_ts).

    Why per-source MIN: Spark refuses TIMESTAMP AS OF for a timestamp AFTER
    the table's latest commit — even though semantically it should resolve
    to "the version current at that time." So if a source hasn't been
    updated since the boundary was written, we pin to the source's own
    latest commit (which IS the version that existed at boundary time).

    Returns (rewritten_source, {fqn: timestamp_used}, [warnings]).
    """
    skip = skip or set()
    pinned: dict[str, str] = {}
    warnings: list[dict[str, Any]] = []
    rewritten = source
    boundary_dt = _parse_spark_ts(boundary_timestamp)
    if boundary_dt is None:
        return source, {}, [{"reason": f"could not parse boundary timestamp {boundary_timestamp!r}"}]

    for fqn in fqns:
        if fqn in skip:
            continue
        if fqn.lower().startswith("system."):
            warnings.append({"fqn": fqn, "reason": "system.* tables not pinned"})
            continue
        # Look up source's latest commit time
        try:
            r = execute_sql(f"DESCRIBE HISTORY {fqn} LIMIT 1", timeout_seconds=60)
            rows = r.get("rows") or []
            if not rows:
                warnings.append({"fqn": fqn, "reason": "no Delta history (not a Delta table?)"})
                continue
            source_latest_dt = _parse_spark_ts(rows[0]["timestamp"])
            if source_latest_dt is None:
                warnings.append({"fqn": fqn, "reason": f"could not parse source timestamp {rows[0]['timestamp']!r}"})
                continue
        except Exception as e:
            warnings.append({"fqn": fqn, "reason": f"DESCRIBE HISTORY failed: {type(e).__name__}"})
            continue

        # Pin to MIN(boundary, source.latest_commit) — Spark requires
        # TIMESTAMP AS OF <= source's latest commit.
        pin_dt = min(boundary_dt, source_latest_dt)
        pin_ts_str = _format_ts_for_spark(pin_dt)

        # Idempotent substitution — skip already-pinned occurrences
        pattern = re.compile(
            r"\b(" + re.escape(fqn) + r")\b(?!\s+(?:VERSION|TIMESTAMP)\s+AS\s+OF)",
            re.IGNORECASE,
        )
        rewritten = pattern.sub(f"\\1 TIMESTAMP AS OF '{pin_ts_str}'", rewritten)
        pinned[fqn] = pin_ts_str
    return rewritten, pinned, warnings


# Legacy alias for backwards compatibility — the old VERSION AS OF behavior
# kept available but no longer used in the propose flow.
def pin_table_versions(
    source: str, fqns: list[str], *, skip: set[str] | None = None,
) -> tuple[str, dict[str, int], list[dict[str, Any]]]:
    """DEPRECATED. Pins to CURRENT version (wrong reference point for propose
    mode — use pin_sources_to_timestamp instead). Kept for any caller that
    explicitly wants "pin to whatever's current now."
    """
    skip = skip or set()
    version_map: dict[str, int] = {}
    warnings: list[dict[str, Any]] = []
    rewritten = source
    for fqn in fqns:
        if fqn in skip:
            continue
        if fqn.lower().startswith("system."):
            warnings.append({"fqn": fqn, "reason": "system.* tables not pinned"})
            continue
        try:
            r = execute_sql(f"DESCRIBE HISTORY {fqn} LIMIT 1", timeout_seconds=60)
            rows = r.get("rows") or []
            if not rows:
                warnings.append({"fqn": fqn, "reason": "no Delta history"})
                continue
            version_map[fqn] = int(rows[0]["version"])
        except Exception as e:
            warnings.append({"fqn": fqn, "reason": f"DESCRIBE HISTORY failed: {type(e).__name__}"})
            continue
        pattern = re.compile(
            r"\b(" + re.escape(fqn) + r")\b(?!\s+(?:VERSION|TIMESTAMP)\s+AS\s+OF)",
            re.IGNORECASE,
        )
        rewritten = pattern.sub(f"\\1 VERSION AS OF {version_map[fqn]}", rewritten)
    return rewritten, version_map, warnings


def rewrite_write_targets(
    source: str, mapping: dict[str, str]
) -> str:
    """Replace each FQN in `mapping` (original → sandbox) wherever it appears in
    the source. Word-boundary aware so we don't rewrite substrings.

    NOTE: we rewrite EVERY occurrence (including any reads of the same FQN within
    this notebook). That's deliberate: if the notebook writes table T and also
    reads T's previous state for an UPSERT, both should go to the sandbox copy.
    Cross-notebook reads of T from OTHER tasks still hit prod (they were never
    in this notebook's source).
    """
    out = source
    for orig, new in mapping.items():
        # Word-boundary on either side, case-insensitive to be safe
        pattern = r"\b" + re.escape(orig) + r"\b"
        out = re.sub(pattern, new, out, flags=re.IGNORECASE)
    return out


# =============================================================================
# Cloning a prod task into the sandbox
# =============================================================================


def detect_incremental(notebook_sql: str, write_target_fqn: str) -> str:
    """Classify a write target's write mode by scanning the notebook SQL.

    `full_rewrite`: CREATE OR REPLACE TABLE <t>, INSERT OVERWRITE <t>, CTAS.
    `incremental`:  INSERT INTO <t> / MERGE INTO <t> with no enclosing full
                    rewrite of the same target. Daily ETLs appending rows.

    Returns `"full_rewrite"` or `"incremental"`. The default on ambiguity is
    `full_rewrite` (safer — diff_tables will compare the full table; if the
    sandbox is actually only an increment, the result will be a visibly
    extreme mismatch the human notices, vs. a silently-wrong comparison).
    """
    if not notebook_sql:
        return "full_rewrite"
    # Plain table name (drop catalog/schema qualifier); SQL identifiers
    # commonly refer to write targets either fully-qualified or by name.
    bare = write_target_fqn.rsplit(".", 1)[-1].lower()
    fq_l = write_target_fqn.lower()

    def _refs_target(stmt_re: re.Pattern[str]) -> bool:
        for m in stmt_re.finditer(notebook_sql):
            tail = notebook_sql[m.end(): m.end() + 400].lower()
            # First identifier-ish token after the verb
            id_match = re.match(r"\s*([`a-zA-Z0-9_.]+)", tail)
            if not id_match:
                continue
            ident = id_match.group(1).strip("`").lower()
            if ident == fq_l or ident == bare or ident.endswith("." + bare):
                return True
        return False

    full_re = re.compile(
        r"(?:CREATE\s+(?:OR\s+REPLACE\s+)?TABLE|REPLACE\s+TABLE|"
        r"INSERT\s+OVERWRITE(?:\s+TABLE)?)\s+",
        re.IGNORECASE,
    )
    inc_re = re.compile(r"(?:INSERT\s+INTO|MERGE\s+INTO)\s+", re.IGNORECASE)

    if _refs_target(full_re):
        return "full_rewrite"
    if _refs_target(inc_re):
        return "incremental"
    return "full_rewrite"


def clone_task_from_prod(
    *,
    prod_job_id: int,
    task_key: str,
    ctx: sandbox.RunContext,
    repo: str = DEFAULT_REPO,
) -> tuple[int, TaskCloneResult]:
    """Clone a single prod task into the sandbox. Returns (sandbox_job_id, clone_info).

    Steps:
      1. Read prod job spec, locate the named task + its job_cluster_key.
      2. Resolve the notebook path against the job's git_source (or fall back to
         workspace path if the task isn't git-sourced).
      3. Fetch the notebook source.
      4. Extract write targets, build orig→sandbox mapping, rewrite.
      5. Upload rewritten source to sandbox workspace path.
      6. Create a one-task sandbox job that references the rewritten notebook
         and the cluster spec extracted from prod. Tagged for cleanup.
    """
    prod_spec = get_job(prod_job_id)
    settings = prod_spec.get("settings", {})
    tasks = settings.get("tasks") or []
    matching = next((t for t in tasks if t.get("task_key") == task_key), None)
    if matching is None:
        raise ValueError(f"task_key {task_key!r} not found in job {prod_job_id}")

    nb_task = matching.get("notebook_task")
    if not nb_task:
        raise ValueError(f"task {task_key} is not a notebook task")
    notebook_path = nb_task.get("notebook_path", "")
    source_type = nb_task.get("source")  # 'GIT' or 'WORKSPACE'

    cluster_key = matching.get("job_cluster_key")
    job_clusters = settings.get("job_clusters") or []
    cluster_def = next((c for c in job_clusters if c.get("job_cluster_key") == cluster_key), None)
    if cluster_def is None and not matching.get("existing_cluster_id") and not matching.get("new_cluster"):
        raise ValueError(f"task {task_key} has no resolvable cluster spec")

    # Fetch notebook source. For git-sourced tasks the path is repo-relative.
    if source_type == "GIT" or settings.get("git_source"):
        source_text = _fetch_notebook_from_git(repo, notebook_path)
    else:
        from ..tools.databricks import get_notebook_source
        source_text = get_notebook_source(notebook_path)["content"]

    # Extract write targets and build the orig→sandbox mapping for rewriting.
    writes, reads = extract_tables(source_text)
    write_mapping: dict[str, str] = {}
    for fqn in writes:
        # Encode the original FQN into a sandbox table name: catalog__schema__table.
        _, schema, table = fqn.split(".")
        sandbox_fqn = f"{ctx.run_catalog}.{ctx.run_schema}.{schema}__{table}"
        write_mapping[fqn] = sandbox_fqn
    rewritten = rewrite_write_targets(source_text, write_mapping)

    # Pin source-table reads to the moment the prod TASK STARTED (Delta
    # snapshot isolation captures source state at query start, not commit).
    # Fall back to the boundary's commit timestamp only if no successful
    # task run is recoverable (rare — e.g., first-ever run of this task).
    skip_pinning = set(write_mapping.values()) | set(writes)
    align_ts = get_prod_task_run_start_time(prod_job_id, task_key)
    align_source = "prod_task_run_start"
    if align_ts is None:
        align_ts = get_prod_boundary_timestamp(list(writes))
        align_source = "boundary_commit_fallback"
    source_pinned_at: dict[str, str] = {}
    pin_warnings: list[dict[str, Any]] = []
    if align_ts:
        rewritten, source_pinned_at, pin_warnings = pin_sources_to_timestamp(
            rewritten, sorted(reads), align_ts, skip=skip_pinning,
        )
        if align_source == "boundary_commit_fallback":
            pin_warnings.append({
                "reason": "no successful prod task run found; using boundary commit time instead "
                          "of task start time (may pin sources to NEWER state than prod actually read)",
            })
    else:
        pin_warnings.append({
            "reason": "could not determine prod task start time NOR boundary commit time; sources not pinned",
            "write_targets": list(writes),
        })

    # Capture the current version of each prod boundary write target — this is
    # the snapshot we'll compare the sandbox output against for equivalence.
    # For INCREMENTAL targets (INSERT INTO / MERGE INTO), also capture:
    #   * write_target_modes[fqn] — classify full_rewrite vs incremental.
    #   * prod_pre_versions[fqn] — version JUST BEFORE the prod task's commit,
    #       so the prod-side increment can be computed as (post EXCEPT pre).
    #   * partition_columns_by_target[fqn] — used to bound the increment view
    #       to the affected partitions (cheap, avoids archived-partition reads).
    prod_boundary_versions: dict[str, int] = {}
    write_target_modes: dict[str, str] = {}
    prod_pre_versions: dict[str, int] = {}
    partition_columns_by_target: dict[str, list[str]] = {}
    for fqn in writes:
        try:
            r = execute_sql(f"DESCRIBE HISTORY {fqn} LIMIT 1", timeout_seconds=60)
            if r["rows"]:
                prod_boundary_versions[fqn] = int(r["rows"][0]["version"])
        except Exception:
            pass
        # Classify mode from the canonical notebook source.
        mode = detect_incremental(source_text, fqn)
        write_target_modes[fqn] = mode
        if mode != "incremental":
            continue
        # Pre-version: lowest commit version in the prod task's time window
        # whose operation modifies data → minus 1. Falls back to (post - 1)
        # if we can't find a tight window. Bounded scan of the last ~50
        # commits — enough for any daily ETL.
        post = prod_boundary_versions.get(fqn)
        if post is None:
            continue
        try:
            hist = execute_sql(
                f"SELECT version, timestamp, operation FROM (DESCRIBE HISTORY {fqn}) "
                f"WHERE version BETWEEN {max(0, post - 50)} AND {post} "
                f"AND operation IN ('WRITE','MERGE','UPDATE','DELETE','INSERT') "
                f"ORDER BY version ASC",
                timeout_seconds=60,
            )["rows"]
            in_window: list[int] = []
            # Define the prod task time window: [align_ts - 30min, now]. align_ts
            # is the task START; commits typically land near task END. We accept
            # any data-modifying commit in this generous window; the lowest is
            # the start of the prod task's write sequence.
            if align_ts and hist:
                from datetime import datetime, timedelta
                start = _parse_spark_ts(align_ts)
                if start is not None:
                    lo = start - timedelta(minutes=30)
                    for h in hist:
                        ts = h.get("timestamp")
                        if isinstance(ts, str):
                            ts_p = _parse_spark_ts(ts)
                        elif hasattr(ts, "year"):
                            ts_p = ts.replace(tzinfo=None) if getattr(ts, "tzinfo", None) else ts
                        else:
                            ts_p = None
                        if ts_p is not None and ts_p >= lo:
                            in_window.append(int(h["version"]))
            min_v = min(in_window) if in_window else post
            prod_pre_versions[fqn] = max(0, min_v - 1)
        except Exception:
            prod_pre_versions[fqn] = max(0, post - 1)
        # Partition columns — drives bounded prod-increment view.
        try:
            det = execute_sql(f"DESCRIBE DETAIL {fqn}", timeout_seconds=60)["rows"]
            if det:
                pcols = det[0].get("partitionColumns")
                if isinstance(pcols, str):
                    # Some workspaces return JSON string; parse defensively.
                    try:
                        pcols = json.loads(pcols)
                    except Exception:
                        pcols = []
                if pcols:
                    partition_columns_by_target[fqn] = list(pcols)
        except Exception:
            pass

    # Upload to sandbox workspace path.
    sandbox_notebook_path = f"{ctx.workspace_dir}/proposal_{task_key}"
    upload_notebook(
        workspace_path=sandbox_notebook_path,
        content=rewritten,
        language="PYTHON",  # works for SQL too — Databricks accepts both as SOURCE format
    )

    # Build a one-task job spec. Reuse the original task spec but swap the
    # notebook_path and pull the matching cluster into job_clusters.
    new_task = dict(matching)
    new_task["notebook_task"] = dict(nb_task)
    new_task["notebook_task"]["notebook_path"] = sandbox_notebook_path
    new_task["notebook_task"]["source"] = "WORKSPACE"
    # Drop dependencies — single-task sandbox clone has no upstream in scope
    new_task.pop("depends_on", None)

    job_settings: dict[str, Any] = {
        "name": f"helios_proposal__{task_key}__{ctx.run_id}",
        "tags": {
            "helios_eval": "true",
            "helios_eval_run_id": ctx.run_id,
            "helios_eval_role": "proposal",
            "helios_eval_source_job_id": str(prod_job_id),
            "helios_eval_source_task_key": task_key,
        },
        "tasks": [new_task],
        "job_clusters": [cluster_def] if cluster_def else [],
        "max_concurrent_runs": 1,
    }
    response = create_job(settings=job_settings)
    sandbox_job_id = int(response["job_id"])

    clone = TaskCloneResult(
        task_key=task_key,
        original_notebook_path=notebook_path,
        sandbox_notebook_path=sandbox_notebook_path,
        original_notebook_source=source_text,
        sandbox_notebook_source=rewritten,
        write_targets_original=sorted(writes),
        write_targets_sandbox=sorted(write_mapping.values()),
        read_sources=sorted(reads),
        source_alignment_timestamp=align_ts,
        source_alignment_basis=align_source,
        source_pinned_fqns=source_pinned_at,
        prod_boundary_versions=prod_boundary_versions,
        write_target_modes=write_target_modes,
        prod_pre_versions=prod_pre_versions,
        partition_columns_by_target=partition_columns_by_target,
        unpinnable_sources=pin_warnings,
    )
    return sandbox_job_id, clone


def _fetch_notebook_from_git(repo: str, repo_relative_path: str) -> str:
    """Look up the notebook source in GitHub. Tries common extensions if the
    spec's path lacks one (Databricks job specs store paths without ext)."""
    from ..tools.github import get_file
    candidates = (
        [repo_relative_path] if "." in repo_relative_path.split("/")[-1]
        else [f"{repo_relative_path}.sql", f"{repo_relative_path}.py", f"{repo_relative_path}.ipynb"]
    )
    last_err: Exception | None = None
    for candidate in candidates:
        try:
            r = get_file(repo=repo, path=candidate)
            content = r.get("content") or ""
            if content:
                return content
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"could not fetch notebook from {repo}: tried {candidates}. last error: {last_err}"
    )


# =============================================================================
# History-based baseline + prod output snapshot
# =============================================================================


@dataclass
class HistoryBaseline:
    """Task-level baseline derived from prior job runs."""

    task_key: str
    samples: int
    median_duration_ms: int
    min_duration_ms: int
    max_duration_ms: int
    most_recent_run_id: int
    output_table_fqns: list[str] = field(default_factory=list)
    output_table_stats: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_history_baseline(
    *,
    prod_job_id: int,
    task_key: str,
    write_targets: list[str],
    samples_to_pull: int = 10,
) -> HistoryBaseline:
    """Pull recent job runs and extract task-level durations for `task_key`,
    even if the parent run failed (some other task may have failed; this task
    may still have succeeded). Returns the median.

    Then snapshot each `write_targets` table for equivalence checking (hash + stats).
    """
    durations: list[int] = []
    most_recent_run_id = 0
    # API caps limit at ~25; iterate pages by start_time_to_ms if needed.
    pages = 0
    seen_run_ids: set[int] = set()
    cursor_ms: int | None = None
    while len(durations) < samples_to_pull and pages < 6:
        kwargs: dict[str, Any] = {"job_id": prod_job_id, "completed_only": True, "limit": 25}
        if cursor_ms is not None:
            kwargs["start_time_to_ms"] = cursor_ms - 1
        runs = list_job_runs(**kwargs)["runs"]
        if not runs:
            break
        for r in runs:
            rid = r["run_id"]
            if rid in seen_run_ids:
                continue
            seen_run_ids.add(rid)
            if not most_recent_run_id:
                most_recent_run_id = rid
            try:
                detail = get_job_run(rid)
            except Exception:
                continue
            for t in detail.get("tasks") or []:
                if t.get("task_key") != task_key:
                    continue
                state = (t.get("state") or {}).get("result_state")
                dur = t.get("execution_duration") or 0
                if state == "SUCCESS" and dur > 0:
                    durations.append(int(dur))
                break
            if len(durations) >= samples_to_pull:
                break
        cursor_ms = min(r["start_time"] for r in runs)
        pages += 1

    if not durations:
        raise RuntimeError(
            f"no successful task-level runs of {task_key!r} found for job {prod_job_id}"
        )

    # Snapshot prod output tables. Read-only — does NOT modify prod.
    # Prod tables can be huge; the full content hash (JSON-serialize + hash
    # every row) is expensive. Strategy:
    #   1. COUNT(*) + stats fingerprint (cheap, single aggregate)  — always succeeds
    #   2. Full content hash (expensive)                           — best-effort with 30-min timeout
    # If (2) times out we proceed with stats-only. The scorer falls back to
    # stats-only equivalence in that case (weaker than byte-exact, but still
    # catches any real semantic divergence — sum/min/max/null_count on every column).
    stats: dict[str, dict[str, Any]] = {}
    HASH_TIMEOUT_S = 1800
    STATS_TIMEOUT_S = 600
    for fqn in write_targets:
        entry: dict[str, Any] = {}
        try:
            rc_result = execute_sql(
                f"SELECT COUNT(*) AS c FROM {fqn}", timeout_seconds=STATS_TIMEOUT_S
            )
            entry["row_count"] = int(rc_result["rows"][0]["c"])
            entry["stats"] = compute_stats_fingerprint(fqn, timeout_seconds=STATS_TIMEOUT_S)
            entry["snapshot_mode"] = "stats_only"
        except Exception as e:
            stats[fqn] = {"error": f"stats fingerprint failed: {type(e).__name__}: {e}"}
            continue
        # Best-effort full hash; on timeout, keep stats-only.
        try:
            _, table_hash = _hash_output_table(fqn, timeout_seconds=HASH_TIMEOUT_S)
            entry["hash"] = table_hash
            entry["snapshot_mode"] = "full_hash"
        except Exception as e:
            entry["hash"] = None
            entry["hash_error"] = (
                f"full hash skipped ({type(e).__name__}); equivalence will use stats-only"
            )
        stats[fqn] = entry

    return HistoryBaseline(
        task_key=task_key,
        samples=len(durations),
        median_duration_ms=int(statistics.median(durations)),
        min_duration_ms=min(durations),
        max_duration_ms=max(durations),
        most_recent_run_id=most_recent_run_id,
        output_table_fqns=list(write_targets),
        output_table_stats=stats,
    )


# =============================================================================
# Orchestrator
# =============================================================================


def propose(
    *,
    prod_job_id: int,
    task_key: str,
    console: Console | None = None,
    samples_to_pull: int = 10,
) -> dict[str, Any]:
    """Run propose mode against a real prod job. Returns the summary dict
    (also written to disk as proposal.md + proposal.json)."""
    console = console or Console()

    run_id = uuid.uuid4().hex[:12]
    run_schema = f"proposal_{run_id}"

    # Build a minimal RunContext compatible with the sandbox helpers.
    ctx = _make_propose_run_context(run_id=run_id, run_schema=run_schema)

    proposal_dir = PROPOSALS_ROOT / run_id
    proposal_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold cyan]helios propose[/] prod_job=[yellow]{prod_job_id}[/] "
        f"task=[yellow]{task_key}[/] run_id=[yellow]{run_id}[/]"
    )
    summary: dict[str, Any] = {
        "mode": "propose",
        "prod_job_id": prod_job_id,
        "task_key": task_key,
        "run_id": run_id,
        "started_at": int(time.time()),
    }

    try:
        sandbox.ensure_catalogs_exist(ctx.seed_catalog, ctx.run_catalog)
        sandbox.create_run_schema(ctx)

        # 1. Clone the prod task into sandbox
        console.print(f"[cyan]→ cloning task {task_key} from prod job {prod_job_id}[/]")
        sandbox_job_id, clone = clone_task_from_prod(
            prod_job_id=prod_job_id, task_key=task_key, ctx=ctx,
        )
        summary["sandbox_job_id"] = sandbox_job_id
        summary["clone"] = {
            "original_notebook_path": clone.original_notebook_path,
            "sandbox_notebook_path": clone.sandbox_notebook_path,
            "write_targets_original": clone.write_targets_original,
            "write_targets_sandbox": clone.write_targets_sandbox,
            "read_sources": clone.read_sources,
        }
        # Persist clone metadata + notebooks IMMEDIATELY so a partial / interrupted
        # run can still be finalized later via `helios propose-finalize <run_id>`.
        (proposal_dir / "notebook_original.txt").write_text(clone.original_notebook_source)
        (proposal_dir / "notebook_sandbox_pre_agent.txt").write_text(clone.sandbox_notebook_source)
        (proposal_dir / "clone.json").write_text(json.dumps({
            "prod_job_id": prod_job_id,
            "task_key": task_key,
            "sandbox_job_id": sandbox_job_id,
            "original_notebook_path": clone.original_notebook_path,
            "sandbox_notebook_path": clone.sandbox_notebook_path,
            "write_targets_original": clone.write_targets_original,
            "write_targets_sandbox": clone.write_targets_sandbox,
            "read_sources": clone.read_sources,
            "source_alignment_timestamp": clone.source_alignment_timestamp,
            "source_alignment_basis": clone.source_alignment_basis,
            "source_pinned_fqns": clone.source_pinned_fqns,
            "prod_boundary_versions": clone.prod_boundary_versions,
            "write_target_modes": clone.write_target_modes,
            "prod_pre_versions": clone.prod_pre_versions,
            "partition_columns_by_target": clone.partition_columns_by_target,
            "unpinnable_sources": clone.unpinnable_sources,
            "run_catalog": ctx.run_catalog,
            "run_schema": ctx.run_schema,
        }, indent=2, default=str))

        # Surface the pinning result to the user
        if clone.source_alignment_timestamp:
            basis_label = {
                "prod_task_run_start": "prod task last SUCCESS run START time (matches Delta snapshot isolation)",
                "boundary_commit_fallback": "prod boundary COMMIT time (FALLBACK — may pin sources newer than prod actually read)",
            }.get(clone.source_alignment_basis, clone.source_alignment_basis)
            console.print(
                f"  source-alignment timestamp:  {clone.source_alignment_timestamp}\n"
                f"  basis:                       {basis_label}\n"
                f"  pinned {len(clone.source_pinned_fqns)} source tables "
                f"`TIMESTAMP AS OF '<per-source MIN(align_ts, source.latest_commit)>'` "
                f"(sandbox sees same input state prod did)"
            )
        else:
            console.print(
                f"  [yellow]could not determine alignment timestamp — sources NOT pinned. "
                f"Equivalence may drift if upstream refreshes during the experiment.[/]"
            )
        if clone.unpinnable_sources:
            unpinnable_fqns = [w['fqn'] for w in clone.unpinnable_sources if 'fqn' in w][:3]
            if unpinnable_fqns:
                console.print(
                    f"  [yellow]could not pin {len(clone.unpinnable_sources)} source(s) "
                    f"(non-Delta or system tables): {unpinnable_fqns}...[/]"
                )
        console.print(
            f"  sandbox job:          {sandbox_job_id}\n"
            f"  write targets:        {clone.write_targets_original}\n"
            f"  → rewritten to:       {clone.write_targets_sandbox}"
        )

        # 2. History-based baseline + prod output snapshot
        console.print("[cyan]→ building history baseline + prod output snapshot[/]")
        history = build_history_baseline(
            prod_job_id=prod_job_id, task_key=task_key,
            write_targets=clone.write_targets_original,
            samples_to_pull=samples_to_pull,
        )
        summary["baseline"] = asdict(history)
        # Persist baseline immediately so resume / finalize have it without re-pulling.
        (proposal_dir / "baseline.json").write_text(json.dumps(asdict(history), indent=2, default=str))
        console.print(
            f"  baseline median: [bold]{history.median_duration_ms/1000:.0f}s[/] "
            f"(from {history.samples} samples; range "
            f"{history.min_duration_ms/1000:.0f}–{history.max_duration_ms/1000:.0f}s)"
        )
        for fqn, info in history.output_table_stats.items():
            if "error" in info:
                console.print(f"  [red]snapshot {fqn}: {info['error']}[/]")
            else:
                console.print(f"  prod snapshot: {fqn} rows={info['row_count']} hash={info['hash']}")

        # 3. Invoke agent with frozen prod job_id
        baseline_seconds = history.median_duration_ms / 1000

        # Build the EXACT diff_tables command(s) the agent must run. The prod
        # side MUST be pinned to the boundary version captured at clone time
        # (`prod_boundary_versions`), NOT the live table — prod is a daily
        # full-overwrite ETL, so the live table will have moved past the
        # snapshot the sandbox was computed against. Injecting the concrete
        # command removes all guessing (the agent previously hallucinated a
        # non-existent `__boundary` table and fell back to unpinned live prod).
        def _pinned_boundary_ref(fqn: str) -> str:
            ver = clone.prod_boundary_versions.get(fqn)
            return f"{fqn} VERSION AS OF {ver}" if ver is not None else fqn

        _diff_pairs = list(zip(
            clone.write_targets_original, clone.write_targets_sandbox
        ))
        diff_commands = "\n".join(
            f'    diff_tables(table_a="{_pinned_boundary_ref(o)}", '
            f'table_b="{s}")'
            for o, s in _diff_pairs
        )
        extra = _PROPOSE_INSTRUCTIONS.format(
            prod_job_id=prod_job_id, task_key=task_key,
            baseline_seconds=baseline_seconds,
            baseline_timeout_hint=max(3600, baseline_seconds * 2),
            write_targets_original=", ".join(clone.write_targets_original),
            write_targets_sandbox=", ".join(clone.write_targets_sandbox),
            sandbox_notebook_path=clone.sandbox_notebook_path,
            diff_commands=diff_commands,
        )
        sandbox_output_fqn = clone.write_targets_sandbox[0] if clone.write_targets_sandbox else ""
        live_trace_path = proposal_dir / "trace.live.jsonl"
        message_log_path = proposal_dir / "messages.json"
        console.print(
            f"[cyan]→ invoking agent (frozen prod job {prod_job_id})[/]\n"
            f"  [dim]live trace: tail -f {live_trace_path}[/]\n"
            f"  [dim]resumable:  helios propose-resume {ctx.run_id}[/]"
        )
        agent_result = run_agent(
            ctx, sandbox_job_id, sandbox_output_fqn,
            console=console,
            frozen_job_ids=frozenset({prod_job_id}),
            extra_instructions=extra,
            live_trace_path=live_trace_path,
            message_log_path=message_log_path,
        )
        summary["agent"] = {
            "final_job_id": agent_result.final_job_id,
            "iterations_used": agent_result.iterations_used,
            "tool_calls": len(agent_result.trace),
            "sandbox_violations": len(agent_result.sandbox_violations),
            "final_text": agent_result.final_text,
            "failed": agent_result.failed,
            "failure_reason": agent_result.failure_reason,
        }
        _write_trace(proposal_dir, agent_result)

        # 4. Run the agent's final job to score
        if not agent_result.failed:
            console.print(
                f"[cyan]→ running optimized sandbox job (id={agent_result.final_job_id})[/]"
            )
            opt_run = run_job_now(agent_result.final_job_id)
            opt_result = wait_for_job_run(int(opt_run["run_id"]), timeout_seconds=14400, poll_interval_seconds=15)
            optimized_succeeded = (
                (not opt_result.get("timed_out")) and opt_result.get("result_state") == "SUCCESS"
            )
            optimized_ms = int(opt_result.get("execution_duration_ms") or 0)
            summary["optimized_run"] = {
                "run_id": opt_result.get("run_id"),
                "succeeded": optimized_succeeded,
                "duration_ms": optimized_ms,
                "run_page_url": opt_result.get("run_page_url"),
            }
            _sc = _score_proposal(
                history=history, clone=clone, optimized_succeeded=optimized_succeeded,
                optimized_ms=optimized_ms, agent_result=agent_result,
                proposal_dir=proposal_dir, console=console,
            )
            summary["scores"] = {"tier1": _sc["tier1"], "tier3": _sc["tier3"]}
            summary["nondeterminism"] = _sc["nondeterminism"]
            summary["tier1_full_detail"] = _sc["tier1_full_detail"]
        else:
            summary["optimized_run"] = {"skipped": "agent_failed"}
            summary["scores"] = {
                "tier1": {"tier": 1, "passed": False, "skipped": "agent_failed"},
                "tier3": {"tier": 3, "passed": False, "skipped": "agent_failed"},
            }

        # 5. Render proposal.md and persist scores
        (proposal_dir / "proposal.json").write_text(json.dumps(summary, indent=2, default=str))
        proposal_md = _render_proposal_md(summary=summary, clone=clone, history=history)
        (proposal_dir / "proposal.md").write_text(proposal_md)
        console.print(
            f"\n[bold green]proposal written:[/] {proposal_dir / 'proposal.md'}\n"
            f"[dim]sandbox job retained:[/] {sandbox_job_id} "
            f"(use `helios eval cleanup {run_id}` to remove)"
        )
        return summary

    except Exception:
        console.print("[red]propose mode failed; artifacts retained for inspection[/]")
        raise


_PROPOSE_INSTRUCTIONS = """
=== PROPOSE MODE — production optimization proposal ===

You are proposing a real performance optimization for prod job {prod_job_id},
task `{task_key}`. The sandbox clone (job_id passed above as the job to modify)
is ALREADY set up — its notebook has prod write targets re-mapped to sandbox.
That setup is sandboxing, NOT optimization. Removing prod writes / prod paths
is required for safety; it does not count as a performance improvement.

HARD CONSTRAINTS (do not violate these):
  - NEVER call run_job_now / add_job_tasks / any mutating tool against
    job_id={prod_job_id}. Harness will refuse.
  - You may READ prod tables (`silver_catalog.prod.*`, `spice_catalog.prod.*`,
    `gold_catalog.*`, etc.) — those are your real source data.
  - You may MODIFY only the sandbox notebook at:
        {sandbox_notebook_path}
    via `upload_notebook` (overwrite is fine).
  - Equivalence is ALWAYS checked against the prod boundary pinned with
    `VERSION AS OF` (the exact command is given verbatim in step 5 below).
    Never diff against the live/unpinned prod table and never invent a
    `*__boundary` / snapshot table — no such table exists.
  - **Preserve the write mode.** If the original notebook uses `INSERT INTO`
    or `MERGE INTO`, the task is INCREMENTAL — it appends/updates rows on
    top of prod's accumulated history. DO NOT rewrite `INSERT INTO` into
    `CREATE OR REPLACE TABLE AS SELECT` (or `INSERT OVERWRITE`); that would
    REPLACE prod's entire historical table with just today's rows when the
    PR ships. The harness scoring already compares against the prod-side
    INCREMENT for these tasks — your sandbox output IS the day's increment
    by design, not a full table. Keep `INSERT INTO`/`MERGE INTO` as-is.
  - Write targets {write_targets_original} are re-mapped to {write_targets_sandbox}
    in the sandbox notebook. Output must produce equivalent content to the
    cached prod snapshot. DO NOT change the SELECT columns / aggregations /
    GROUP BY structure — equivalence is checked strictly.
  - The sandbox notebook ALREADY contains `TIMESTAMP AS OF '<T>'` clauses on
    every Delta source table. T is the prod boundary's last-write timestamp —
    i.e., the moment the baseline you're trying to beat was actually produced.
    Pinning aligns the sandbox's input data to that state, so upstream
    refreshes during your experiment don't cause spurious equivalence
    failures. DO NOT REMOVE these clauses. If you introduce a NEW source
    reference in your rewrite, pin it to the SAME timestamp T (you can see it
    in the existing pinned references).

BASELINE (the bar you must beat):
  - Median runtime of `{task_key}` from the last 10 prod runs: {baseline_seconds:.0f}s.

GOAL — actual performance optimization (NOT sandboxing):
  Achieve a measurable runtime reduction on the sandbox clone vs the {baseline_seconds:.0f}s baseline.
  An accepted proposal must reduce runtime by AT LEAST 15% (target: 30%+).

REQUIRED FIRST STEP — understand the existing plan via EXPLAIN:
  Before proposing ANY optimization, you MUST call `explain_query` on the
  original (unmodified) notebook's main SELECT. Without this, you're guessing.

  Workflow:
    1. get_notebook_source on the sandbox notebook
    2. Extract the SELECT body (the body of the CREATE OR REPLACE TABLE ... AS).
       Do not include the CTAS wrapper — EXPLAIN may reject it.
    3. explain_query(sql=<SELECT body>)
       → returns BOTH the FORMATTED physical plan (join strategies + shuffle
       count) AND the COST logical plan (cardinality estimates: rowCount,
       sizeInBytes per operator). One call, both views.
    4. Read result['combined_warnings'] first — it auto-flags common issues
       like SortMergeJoin where a Broadcast would work, or cartesian products,
       or heavy shuffling.
    5. Read result['cost']['summary']['estimated_row_counts'] and
       result['cost']['summary']['estimated_sizes'] — these are the optimizer's
       predicted cardinalities at each plan node. The BIGGEST numbers point
       at the bottleneck.
    6. ONLY THEN reason about which optimization category to apply.

OPTIMIZATION PRIORITY ORDER — try in this sequence, stop when you find a real win:

  Categories 1-5 don't change WHAT the query computes; only category 6 does.
  Most successful prod optimizations live in 1-5 — try them FIRST.
  Reaching category 6 should be a last resort, not a first instinct.

  ── Category 1: CLUSTER + SPARK CONFIG (safest — no algebra change) ──
    • spark.sql.adaptive.enabled = true (leave AQE on; only disable for
      reproducible bugs)
    • spark.sql.adaptive.skewJoin.enabled = true
    • spark.databricks.adaptive.autoBroadcastJoinThreshold: raise to 256MB or
      512MB if you have a side that's ~100MB-500MB and SortMergeJoin is being
      picked
    • spark.sql.shuffle.partitions: default 200 might be wrong. Rule of thumb:
      target ~128MB per shuffle partition (total_shuffle_bytes / 128MB).
    • If executor OOM or heavy spill: bump driver/executor memory, or enable
      off-heap (spark.memory.offHeap.enabled).

  ── Category 2: SPARK HINTS (still no algebra change) ──
    • /*+ BROADCAST(small_table) */ — when one join side is ≤512MB after
      pruning AND AQE isn't broadcasting it automatically (verify via
      explain_query: SortMergeJoin where you expect BroadcastHashJoin).
      Don't broadcast tables over 1GB — you'll OOM the executors.
    • /*+ REPARTITION(N, key) */ — when you need stable partitioning before
      a multi-stage operation.
    • /*+ COALESCE(N) */ — to merge small partitions, reducing task overhead.
    • /*+ RANGE_JOIN(rel, bin_size) */ — when a join has an INTERVAL /
      INEQUALITY condition (BETWEEN, <, >, point-in-range, interval overlap)
      with no equality key. Without the hint, Spark falls back to
      BroadcastNestedLoopJoin / O(n·m) scan; with it, Databricks bins both
      sides along the range column and turns the work into ~linear. Same
      results — diff_tables stays IDENTICAL. `bin_size` matches the typical
      interval length in the COLUMN's units: timestamps-as-seconds with
      ~1-hour intervals → 3600; dates with ~month ranges → 30; numeric
      ranges → typical span of [low, high]. Order-of-magnitude is enough.
      Databricks' optimizer often suggests this directly in the run hints
      ("This query has a join condition that can benefit from range join
      optimization"); if you see that, apply it.

  ── Category 3: CACHING (no algebra change, but uses memory) ──
    • CACHE TABLE / .cache() / .persist() — when the SAME intermediate result
      is read 3+ times in the same job (subqueries / CTEs that unfold to
      repeated scans of the same base data). Verify reuse via explain_query
      first: if you see the same scan repeated multiple times, caching wins.
    • Don't cache something read only once — pure overhead.
    • Don't cache a result larger than the cluster's executor memory — it'll
      spill to disk and you've gained nothing.
    • UNCACHE / unpersist when the data is no longer needed if memory is tight.

  ── Category 4: PREDICATE PUSHDOWN + PARTITION PRUNING (filters, not algebra) ──
    • Push WHERE clauses INSIDE source-table scans, not after a JOIN.
    • For daily ETLs reading full history when only a date range is needed:
      add a date filter on the source. (Check: is the source date-partitioned?
      Does adding `WHERE date >= ...` create partition pruning?)
    • Replace UDFs with native SQL functions if the UDF is on the hot path.

  ── Category 5: TABLE MAINTENANCE (one-time, no query change) ──
    • OPTIMIZE ZORDER BY (cols) — when query filters frequently on `cols`
      and they're not already in the table's CLUSTER BY / ZORDER list.
    • ANALYZE TABLE — when the optimizer is making bad join-strategy choices
      because stats are stale. Run on tables in complex joins.
    • OPTIMIZE (compaction) — when DESCRIBE DETAIL shows avg file size <16MB
      ("small-files problem"). Don't bother on tables with ≥128MB avg file size.

  ── Category 6: ALGORITHMIC REWRITE (CHANGES THE ALGEBRA — HIGH CORRECTNESS RISK) ──
    Only after categories 1-5 are exhausted AND explain_query shows the plan
    is the bottleneck (not the cluster). Examples:
    • Splitting NULL-skewed joins into NULL-branch + non-NULL UNION
    • Pre-aggregation before cross-joining
    • Replacing LEFT JOIN + GROUP BY with INNER JOIN + filter
    • Restructuring cumulative-window CTEs

    ** STRICT RULE FOR CATEGORY 5: **
       - Make ONE algorithmic change per iteration, not several stacked.
       - After each change: run explain_query on the new plan. If cardinality
         or shuffle count is WORSE than the original, revert and try another
         approach. Do NOT trigger the sandbox job until explain_query confirms
         the new plan is at least as good as the original.
       - After the sandbox runs: ALWAYS run diff_tables. If REAL_DIFFERENCE,
         the algebra is broken — revert this change and try a different
         category-6 approach (or revert to your last category-1-5 state and
         report partial proposal).

  ── Common mistakes to avoid ──
    • Jumping to category 5 first. The most expensive failure mode.
    • Making multiple category-6 changes in one iteration. You won't know
      which broke things.
    • Disabling AQE (it's almost always the right default).
    • Broadcasting a table that's actually large (causes OOM, not speedup).
    • "Optimizing" by just removing prod-side effects (ANALYZE on prod
      tables, hardcoded S3 paths). Those are sandboxing, NOT optimization.

REQUIRED REPORTING:
  Before applying any change, state explicitly in your reasoning:
    a) What you measured (which probe / EXPLAIN summary fields)
    b) What you hypothesize is the bottleneck (one sentence)
    c) Which CATEGORY (1-6) the change belongs to
    d) Why you expect a measurable speedup

  If after honest investigation you cannot find a high-confidence
  category-1-5 win, and category 6 doesn't preserve equivalence either,
  say so explicitly:
      "After investigating X, Y, Z, I could not produce a byte-equivalent
       optimization. Reporting partial proposal: <safe changes> with
       explain_query / diff_tables output for human review."
  Honesty beats a fake win that fails equivalence.

WHEN DONE — MANDATORY verification before declaring success:
  1. Modify the sandbox notebook with your real optimization.
  1a. *** REQUIRED *** Before triggering the sandbox job, call `explain_query`
      on your OPTIMIZED SELECT body (one call, default mode returns both views).
      Compare to the original plan from the REQUIRED FIRST STEP above. Verify:
        - join_strategies improved (e.g., SortMergeJoin → BroadcastHashJoin) OR
          stayed the same — should NOT regress
        - num_shuffles did not INCREASE — if it did, your "optimization" is
          probably worse than the original
        - estimated_row_counts and estimated_sizes on intermediate steps are
          NOT larger (especially watch for any value jumping to 10× or more)
      If the new plan looks WORSE on any of these dimensions, abort and try
      a different approach BEFORE paying for a 20-minute cluster run.
  2. Trigger the sandbox job via run_job_now.
  3. Wait for terminal state. CRITICAL: pass an explicit `timeout_seconds`
     to wait_for_job_run that's generous relative to the baseline runtime.
     A good default is:
            timeout_seconds = max(3600, baseline_seconds * 2)
     Baseline median for this task = {baseline_seconds:.0f}s. So pick at
     least {baseline_timeout_hint:.0f}s. The tool's built-in default of
     30 min is for small jobs — DO NOT rely on it for prod-scale tasks.
     If wait_for_job_run returns `timed_out: True`, the job is still
     RUNNING (not failed); call wait_for_job_run again with a longer
     timeout.
  4. If the job FAILED (result_state="FAILED" / "INTERNAL_ERROR"), call
     get_job_run_output on the task run_id to read the error, fix the
     notebook, and retry from step 2. Do not give up on a transient parse
     error or schema bug — those are part of the work.
  5. *** REQUIRED *** — Verify row-level equivalence by calling `diff_tables`
     with EXACTLY this command (copy it verbatim — do not invent table names,
     do not add suffixes like `__boundary`, do not look for a snapshot table):

{diff_commands}

     `table_a` is the prod boundary PINNED to the exact Delta version the
     baseline was produced at (`... VERSION AS OF <n>`). This is mandatory.
       - DO NOT compare against the live/unpinned prod table (e.g.
         `spice_catalog.prod.<t>` with no `VERSION AS OF`). Prod is a daily
         full-overwrite ETL — the live table has already moved past the
         snapshot your sandbox was computed from, so an unpinned compare is
         apples-to-oranges and will report a FALSE `REAL_DIFFERENCE`.
       - There is NO materialized `*__boundary` table. The pinned reference
         above IS the boundary. If `diff_tables` returns TABLE_OR_VIEW_NOT_FOUND
         you mistyped the command — re-copy it verbatim; do not go hunting
         with SHOW TABLES / SHOW SCHEMAS.

     Read the returned `verdict`:
       - "IDENTICAL"            → safe to declare success (any sub-tolerance
                                  float drift was IEEE-754 reorder noise,
                                  already absorbed).
       - "FLOAT_REORDER_ONLY"   → safe; only DOUBLE/FLOAT columns drift and
                                  `worst_float_rel_diff` is within the reorder
                                  threshold. Cite it in your summary.
       - "REAL_DIFFERENCE"      → DO NOT declare success. The output is
                                  semantically wrong. Read `drift_profile`
                                  (per-column rows_drifted / max_rel_diff),
                                  `buckets`, and `drift_concentration` to
                                  localize the broken CTE/join, fix your
                                  notebook, and re-run from step 2.

     Coarse checks (COUNT(*), SUM of one column) are NOT sufficient — they
     miss bugs where total row count is close but individual rows are wrong.
     `diff_tables` is the only check that catches this category of bug.

  6. Respond with a one-line summary of the optimization + the `diff_tables`
     verdict you got, then:
        FINAL_JOB_ID=<sandbox_job_id>

  If diff_tables returns "REAL_DIFFERENCE" and you cannot find a fix after
  reasonable investigation, respond honestly: "After investigating X, Y, Z,
  I could not produce a byte-equivalent optimization. Reporting partial
  proposal with diff_tables output for human review." — and still emit
  FINAL_JOB_ID so the harness can capture the artifacts.
"""


def _aggregate_verdict(verdicts: list[str]) -> str:
    """Worst-of across per-table verdicts. REAL_DIFFERENCE dominates;
    UNKNOWN is treated as REAL_DIFFERENCE for safety."""
    if not verdicts:
        return "UNKNOWN"
    if any(v in ("REAL_DIFFERENCE", "UNKNOWN") for v in verdicts):
        return "REAL_DIFFERENCE"
    if any(v == "FLOAT_REORDER_ONLY" for v in verdicts):
        return "FLOAT_REORDER_ONLY"
    return "IDENTICAL"


def _build_prod_increment_view(
    *,
    orig_fqn: str,
    post_v: int,
    pre_v: int | None,
    partition_cols: list[str],
    sandbox_fqn: str,
    sandbox_schema: str,
    console: Console,
) -> tuple[str, dict[str, Any]]:
    """Materialize a view in the sandbox schema that exposes ONLY the prod-side
    increment for an incremental write target (INSERT INTO / MERGE INTO).

    Algorithm (from cheap to expensive):
      * Partitioned + sandbox covers a subset of partition values
          → bound `prod VERSION AS OF post` by those partition predicates.
          If pre had 0 rows in that range → no EXCEPT needed (pure new-partition INSERT).
          Otherwise → bounded EXCEPT.
      * Non-partitioned → unbounded EXCEPT on full table versions (slow; fallback).

    Returns (view_fqn, metadata) where metadata describes which path was taken.
    """
    from ..tools.databricks import execute_sql

    bare = orig_fqn.rsplit(".", 1)[-1].replace("`", "")
    view_name = f"prod_increment__{bare}"
    view_fqn = f"{sandbox_schema}.{view_name}"
    meta: dict[str, Any] = {
        "post_version": post_v, "pre_version": pre_v,
        "partition_cols": partition_cols, "where_clause": None, "used_except": False,
    }

    # Discover the sandbox-side range for each partition column (the set of
    # partition values the sandbox actually wrote to — defines the scope).
    where_clauses: list[str] = []
    if partition_cols:
        for pc in partition_cols:
            r = execute_sql(
                f"SELECT MIN(`{pc}`) AS lo, MAX(`{pc}`) AS hi, "
                f"COUNT(DISTINCT `{pc}`) AS n FROM {sandbox_fqn}",
                timeout_seconds=120,
            )["rows"][0]
            lo, hi, n = r.get("lo"), r.get("hi"), int(r.get("n") or 0)
            if lo is None or hi is None or n == 0:
                continue
            # Quote string-y values; cast through string for safety on dates.
            if str(lo) == str(hi):
                where_clauses.append(f"`{pc}` = '{lo}'")
            else:
                where_clauses.append(f"`{pc}` BETWEEN '{lo}' AND '{hi}'")

    if where_clauses:
        where = " AND ".join(where_clauses)
        meta["where_clause"] = where
        # If pre had 0 rows in this scope, the entire scope IS the new
        # increment — no EXCEPT needed (typical daily new-partition INSERT).
        pre_count = 0
        if pre_v is not None:
            try:
                pre_count = int(execute_sql(
                    f"SELECT COUNT(*) AS c FROM {orig_fqn} VERSION AS OF {pre_v} "
                    f"WHERE {where}",
                    timeout_seconds=180,
                )["rows"][0]["c"])
            except Exception:
                pre_count = 1  # be safe: assume pre had rows → use EXCEPT
        if pre_count == 0:
            view_sql = (
                f"SELECT * FROM {orig_fqn} VERSION AS OF {post_v} WHERE {where}"
            )
        else:
            meta["used_except"] = True
            view_sql = (
                f"SELECT * FROM {orig_fqn} VERSION AS OF {post_v} WHERE {where} "
                f"EXCEPT "
                f"SELECT * FROM {orig_fqn} VERSION AS OF {pre_v} WHERE {where}"
            )
    else:
        # Non-partitioned: unbounded EXCEPT. Slow on large tables but correct.
        meta["used_except"] = True
        pre_ref = pre_v if pre_v is not None else max(0, post_v - 1)
        view_sql = (
            f"SELECT * FROM {orig_fqn} VERSION AS OF {post_v} "
            f"EXCEPT "
            f"SELECT * FROM {orig_fqn} VERSION AS OF {pre_ref}"
        )

    execute_sql(
        f"CREATE OR REPLACE VIEW {view_fqn} AS {view_sql}",
        timeout_seconds=300,
    )
    console.print(
        f"    [dim]prod-increment view materialized: {view_fqn} "
        f"(post=v{post_v}, pre={'v'+str(pre_v) if pre_v is not None else '∅'}, "
        f"where={meta['where_clause'] or 'all rows'}, "
        f"EXCEPT={meta['used_except']})[/]"
    )
    return view_fqn, meta


def _value_preservation_check(
    *,
    table_a: str,
    table_b: str,
    natural_key: list[str],
    probe_required: list[str],
    columns_detail: dict[str, dict[str, Any]],
    console: Console,
) -> list[dict[str, Any]]:
    """Empirical tie-break corroboration. For each probe-required column with
    a declared `deterministic_sibling` (the ORDER BY key of the untied
    ROW_NUMBER), join prod-side vs sandbox-side on the stable natural key and
    classify the discrepancy:

      diff_total == 0          → DETERMINISTIC_EMPIRICALLY (never differed)
      diff>0  &  sibling_diff==0 → TIE_BREAK_CONFIRMED (safe to exclude)
      diff>0  &  sibling_diff>0  → POSSIBLE_REAL_BUG (do not exclude)

    Converts the manual "is this a real bug or just a tied argmax?" diagnostic
    into a permanent, automated part of every proposal.
    """
    from ..tools.databricks import execute_sql

    if not probe_required:
        return []
    # Build per-column entries. Columns WITH a declared sibling get the full
    # tie-break vs real-bug discriminator (diff_total + sibling_SAME +
    # sibling_DIFF). Columns WITHOUT a sibling still get diff_total — the
    # LLM may have skipped the sibling field (stochastic), but diff=0 alone
    # is enough to mark the column DETERMINISTIC_EMPIRICALLY; diff>0 without
    # a sibling is surfaced as DIFFERS_NO_SIBLING_TO_CHECK so the human knows
    # the LLM left an evidence gap.
    entries: list[dict[str, Any]] = []
    siblings: set[str] = set()
    for c in probe_required:
        sib = (columns_detail.get(c, {}) or {}).get("deterministic_sibling")
        # Forbid self-reference (a column can't be its own ORDER BY key).
        if sib == c:
            sib = None
        entries.append({"column": c, "sibling": sib})
        if sib:
            siblings.add(sib)
    # Build a JOIN key that EXCLUDES both probe cols and their siblings (we
    # measure their differences/matches; they can't also be join keys).
    excluded = set(probe_required) | siblings
    join_keys = [k for k in natural_key if k not in excluded]
    if not join_keys:
        console.print(
            "[yellow]value-preservation skipped: no usable join key left "
            "after excluding probe columns and their siblings[/]"
        )
        return []
    join_cond = " AND ".join(f"p.`{k}` <=> s.`{k}`" for k in join_keys)

    # One aggregated query covering every column. With-sibling columns get
    # 3 measurements; without-sibling get 1.
    def _alias(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", s)
    select_parts: list[str] = ["COUNT(*) AS paired_rows"]
    for e in entries:
        c = e["column"]; sib = e["sibling"]
        ca = _alias(c)
        select_parts.append(
            f"SUM(CASE WHEN NOT (p.`{c}` <=> s.`{c}`) THEN 1 ELSE 0 END) "
            f"AS `{ca}__diff_total`"
        )
        if sib:
            select_parts.extend([
                f"SUM(CASE WHEN NOT (p.`{c}` <=> s.`{c}`) "
                f"AND (p.`{sib}` <=> s.`{sib}`) THEN 1 ELSE 0 END) "
                f"AS `{ca}__sibling_SAME`",
                f"SUM(CASE WHEN NOT (p.`{c}` <=> s.`{c}`) "
                f"AND NOT (p.`{sib}` <=> s.`{sib}`) THEN 1 ELSE 0 END) "
                f"AS `{ca}__sibling_DIFF`",
            ])
    sql = (
        f"SELECT {', '.join(select_parts)} "
        f"FROM (SELECT * FROM {table_a}) p "
        f"JOIN (SELECT * FROM {table_b}) s ON {join_cond}"
    )
    with_sib = sum(1 for e in entries if e["sibling"])
    console.print(
        f"[cyan]→ value-preservation check on {len(entries)} probe-required "
        f"column(s) ({with_sib} with sibling, {len(entries)-with_sib} "
        f"diff-only); join key={join_keys}[/]"
    )
    try:
        row = execute_sql(sql, timeout_seconds=1800)["rows"][0]
    except Exception as ex:
        console.print(
            f"[yellow]value-preservation check failed: "
            f"{type(ex).__name__}: {ex}[/]"
        )
        return []

    paired = int(row.get("paired_rows") or 0)
    results: list[dict[str, Any]] = []
    for e in entries:
        c = e["column"]; sib = e["sibling"]
        ca = _alias(c)
        total = int(row.get(f"{ca}__diff_total") or 0)
        same = int(row.get(f"{ca}__sibling_SAME") or 0) if sib else 0
        diff = int(row.get(f"{ca}__sibling_DIFF") or 0) if sib else 0

        if total == 0:
            verdict = "DETERMINISTIC_EMPIRICALLY"
            note = (
                "Column matched on every paired row — LLM was overcautious. "
                "Safe to drop from `equivalence_ignore_columns` (no effect "
                "either way; just less noise)."
            )
        elif sib is None:
            verdict = "DIFFERS_NO_SIBLING_TO_CHECK"
            note = (
                f"Column differs on {total:,} rows, but the LLM did not "
                "identify a deterministic ORDER BY sibling — can't auto-"
                "discriminate tie-break vs real-bug. Inspect manually, or "
                "set the sibling explicitly in `clone.json` and re-finalize."
            )
        elif diff == 0:
            verdict = "TIE_BREAK_CONFIRMED"
            note = (
                f"All {total:,} differences occur on rows where the ORDER BY "
                f"sibling `{sib}` matches between prod and sandbox → arbitrary "
                "pick among rows tied on the ranking key. Safe to add to "
                "`equivalence_ignore_columns` with rationale."
            )
        else:
            verdict = "POSSIBLE_REAL_BUG"
            note = (
                f"{diff:,} of {total:,} differences have a DIFFERING sibling "
                f"`{sib}` — at least some picks landed on rows with truly "
                "different ordering values. NOT a pure tie-break; do NOT "
                "exclude without investigating the algebra."
            )
        results.append({
            "column": c, "sibling": sib,
            "paired_rows": paired,
            "diff_total": total,
            "sibling_SAME": same,
            "sibling_DIFF": diff,
            "verdict": verdict, "note": note,
        })
        console.print(
            f"    [dim]{c}"
            + (f" (sibling {sib})" if sib else " (no sibling)")
            + f": {verdict} (diff={total}"
            + (f", sibling_same={same}, sibling_diff={diff}" if sib else "")
            + ")[/]"
        )
    return results


def _equivalence_and_nd(
    *,
    write_orig: list[str],
    write_sb: list[str],
    prod_boundary_versions: dict[str, int],
    explicit_ignore: list[str],
    proposal_dir: Path,
    console: Console,
    optimized_succeeded: bool = True,
    write_target_modes: dict[str, str] | None = None,
    prod_pre_versions: dict[str, int] | None = None,
    partition_columns_by_target: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Single source of truth for Tier-1 equivalence across propose /
    propose-resume / propose-finalize.

    1. LLM nondeterminism analysis on the ORIGINAL canonical notebook.
    2. Effective ignore = explicit ∪ LLM self-authorizing (gated to the
       known-safe run-stamp class). probe_required is NEVER auto-excluded
       (data-derived; indistinguishable from a real bug) — only surfaced
       for the determinism probe / human sign-off.
    3. For each boundary table:
         * full_rewrite mode → diff against the pinned prod table directly.
         * incremental mode → materialize a prod-increment view (post EXCEPT
           pre, partition-bounded) and diff against that.

    Returns {per_table, all_eq, nondeterminism}.
    """
    from ..tools.databricks import diff_tables, execute_sql, looks_run_stamp

    write_target_modes = write_target_modes or {}
    prod_pre_versions = prod_pre_versions or {}
    partition_columns_by_target = partition_columns_by_target or {}

    if not optimized_succeeded:
        return {"per_table": {}, "all_eq": False, "nondeterminism": {}}

    # 1. Nondeterminism analysis. We also capture each column's type from the
    # DESCRIBE so the self_authorizing safety gate can check TIMESTAMP/DATE.
    nd_analysis: dict[str, Any] = {}
    col_types_by_orig: dict[str, dict[str, str]] = {}
    orig_sql_path = proposal_dir / "notebook_original.txt"
    if orig_sql_path.exists():
        try:
            from .nondeterminism import detect_nondeterministic_columns
            orig_sql = orig_sql_path.read_text()
            for orig in write_orig:
                schema_ref = re.sub(
                    r"\s+(?:VERSION|TIMESTAMP)\s+AS\s+OF\s+.*$", "", orig,
                    flags=re.IGNORECASE,
                ).strip()
                drows = execute_sql(
                    f"DESCRIBE TABLE {schema_ref}", row_limit=500,
                    timeout_seconds=120,
                )["rows"]
                ocols: list[str] = []
                ctypes: dict[str, str] = {}
                for r in drows:
                    n = (r.get("col_name") or "").strip()
                    if not n or n.startswith("#"):
                        break
                    ocols.append(n)
                    ctypes[n] = (r.get("data_type") or "").upper()
                col_types_by_orig[orig] = ctypes
                console.print(
                    f"[cyan]→ analyzing nondeterminism on {len(ocols)} "
                    f"output columns of {orig}[/]"
                )
                nd_analysis[orig] = detect_nondeterministic_columns(
                    orig_sql, ocols
                )
        except Exception as e:
            console.print(
                f"[yellow]nondeterminism analysis skipped: "
                f"{type(e).__name__}: {e}[/]"
            )
            nd_analysis = {}
    else:
        console.print(
            "[yellow]notebook_original.txt missing — skipping nondeterminism "
            "analysis[/]"
        )

    # Sandbox schema for materializing prod-increment views (derived from the
    # first sandbox FQN: <catalog>.<schema>.<table>).
    sandbox_schema = ".".join(write_sb[0].split(".")[:2]) if write_sb else ""

    # 2 + 3. Effective ignore + diff_tables per boundary table.
    per_table: dict[str, Any] = {}
    all_eq = True
    for orig, sb in zip(write_orig, write_sb):
        mode = write_target_modes.get(orig, "full_rewrite")
        post_v = prod_boundary_versions.get(orig)
        pre_v = prod_pre_versions.get(orig)
        partition_cols = partition_columns_by_target.get(orig) or []

        # Choose the prod-side reference: incremental view if applicable,
        # else the pinned full table.
        increment_meta: dict[str, Any] = {}
        if mode == "incremental" and post_v is not None and sandbox_schema:
            try:
                view_fqn, increment_meta = _build_prod_increment_view(
                    orig_fqn=orig, post_v=post_v, pre_v=pre_v,
                    partition_cols=partition_cols, sandbox_fqn=sb,
                    sandbox_schema=sandbox_schema, console=console,
                )
                pinned_orig = view_fqn
                pinned_label = (
                    f"{view_fqn}  [incremental: v{post_v}"
                    f"{' EXCEPT v'+str(pre_v) if increment_meta.get('used_except') else ''}"
                    f"{' WHERE '+increment_meta['where_clause'] if increment_meta.get('where_clause') else ''}]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]prod-increment view build failed "
                    f"({type(e).__name__}: {e}); falling back to full-table compare[/]"
                )
                pinned_orig = (
                    f"{orig} VERSION AS OF {post_v}" if post_v is not None else orig
                )
                pinned_label = pinned_orig
                mode = "full_rewrite_fallback"
        else:
            pinned_orig = (
                f"{orig} VERSION AS OF {post_v}" if post_v is not None else orig
            )
            pinned_label = pinned_orig

        # Safety gate on LLM self_authorizing: only auto-exclude columns that
        # ALSO match the existing run-stamp name + TIMESTAMP/DATE type heuristic.
        # Prevents an over-classified A_by_definition (e.g., "filtered by
        # current_date") from silently widening the ignore set onto natural-key
        # dimension columns and collapsing the join.
        nd = nd_analysis.get(orig, {})
        ctypes = col_types_by_orig.get(orig, {})
        llm_safe = nd.get("self_authorizing_columns") or []
        gated_safe: list[str] = []
        gated_rejected: list[str] = []
        for c in llm_safe:
            ct = ctypes.get(c, "")
            if looks_run_stamp(c) and any(s in ct for s in ("TIMESTAMP", "DATE")):
                gated_safe.append(c)
            else:
                gated_rejected.append(c)
        if gated_rejected:
            console.print(
                f"    [dim]LLM self_authorizing not auto-excluded (failed "
                f"name+type gate, informational only): {gated_rejected}[/]"
            )
        effective_ignore = sorted(set(explicit_ignore) | set(gated_safe))
        probe_required = list(nd.get("probe_required_columns") or [])
        if gated_safe:
            console.print(
                f"    [dim]auto-ignored (self-authorizing, gated): {gated_safe}[/]"
            )
        if probe_required:
            console.print(
                f"    [yellow]probe-required nondeterminism (NOT excluded — "
                f"needs determinism probe / human sign-off): "
                f"{probe_required}[/]"
            )
        console.print(f"[cyan]→ diff_tables({pinned_label}, {sb})[/]")
        diff = diff_tables(
            table_a=pinned_orig, table_b=sb, timeout_seconds=1800,
            ignore_columns=effective_ignore or None,
        )
        diff["effective_ignore_columns"] = effective_ignore
        diff["nondeterminism_probe_required"] = probe_required
        diff["mode"] = mode
        if increment_meta:
            diff["increment"] = increment_meta
        diff["self_authorizing_rejected"] = gated_rejected
        # Empirical tie-break corroboration: for every probe-required column
        # with a declared deterministic sibling, check whether the
        # ORDER BY key actually matches on the diverging rows. Runs against
        # the underlying tables regardless of whether the columns are
        # currently excluded — answers "would excluding this be safe?"
        # independent of whether you already did.
        if probe_required:
            diff["tie_break_corroboration"] = _value_preservation_check(
                table_a=pinned_orig, table_b=sb,
                natural_key=diff.get("natural_key") or [],
                probe_required=probe_required,
                columns_detail=(nd.get("columns") or {}),
                console=console,
            )
        per_table[orig] = diff
        if diff["verdict"] == "REAL_DIFFERENCE":
            all_eq = False
        console.print(
            f"    verdict: {diff['verdict']} | "
            f"identical {diff['buckets']['identical']:,} | "
            f"extras {diff['buckets']['extra_in_b']:,} | "
            f"missing {diff['buckets']['missing_from_b']:,} | "
            f"drifted {diff['buckets']['same_key_drifted_metric']:,}"
        )
    return {
        "per_table": per_table, "all_eq": all_eq,
        "nondeterminism": nd_analysis,
    }


def _tier1_from_equivalence(eq: dict[str, Any]) -> dict[str, Any]:
    """Build the uniform tier1 score dict from `_equivalence_and_nd` output.
    Shape is a superset accepted by BOTH renderers."""
    per_table = eq["per_table"]
    verdicts = [v["verdict"] for v in per_table.values()]
    agg = _aggregate_verdict(verdicts) if verdicts else "UNKNOWN"
    return {
        "tier": 1,
        "passed": eq["all_eq"] and bool(per_table),
        "equivalence_verdict": agg,
        "per_table": {
            k: {"verdict": v["verdict"], "buckets": v["buckets"],
                "drift_concentration": v.get("drift_concentration", {})}
            for k, v in per_table.items()
        },
        "details": {"per_table": per_table},
    }


def _score_proposal(
    *, history: HistoryBaseline, clone: TaskCloneResult,
    optimized_succeeded: bool, optimized_ms: int, agent_result,
    proposal_dir: Path, console: Console,
) -> dict[str, Any]:
    """Score the proposal — T1 (canonical diff_tables + LLM nondeterminism)
    + T3. Tier 2 disabled for propose (no fixture-defined required_tools).

    Tier-1 routes through the shared `_equivalence_and_nd` helper so propose
    / propose-resume / propose-finalize stay consistent. The legacy
    hash-based path was retired: a bit-sensitive full-table hash cannot
    ignore columns, so it false-failed on float reorder AND nondeterminism.
    """
    eq = _equivalence_and_nd(
        write_orig=clone.write_targets_original,
        write_sb=clone.write_targets_sandbox,
        prod_boundary_versions=clone.prod_boundary_versions or {},
        explicit_ignore=[],  # fresh run: no human-confirmed list yet
        proposal_dir=proposal_dir,
        console=console,
        optimized_succeeded=optimized_succeeded,
        write_target_modes=clone.write_target_modes or {},
        prod_pre_versions=clone.prod_pre_versions or {},
        partition_columns_by_target=clone.partition_columns_by_target or {},
    )
    tier1 = _tier1_from_equivalence(eq)
    tier1["output_equivalence"] = tier1["equivalence_verdict"] in (
        "IDENTICAL", "FLOAT_REORDER_ONLY"
    )
    tier1["passed"] = tier1["passed"] and tier1["output_equivalence"]
    tier1["job_completion"] = optimized_succeeded

    # Tier 3 — improvement vs baseline median.
    if optimized_succeeded and history.median_duration_ms > 0:
        delta_pct = (history.median_duration_ms - optimized_ms) / history.median_duration_ms * 100
    else:
        delta_pct = 0.0
    tier3 = {
        "tier": 3,
        "passed": optimized_succeeded and delta_pct > 0,
        "baseline_median_ms": history.median_duration_ms,
        "optimized_ms": optimized_ms,
        "runtime_improvement_pct": round(delta_pct, 2),
    }

    return {
        "tier1": tier1,
        "tier3": tier3,
        "nondeterminism": eq["nondeterminism"],
        "tier1_full_detail": eq["per_table"],
    }


def _render_nd_section(summary: dict[str, Any]) -> str:
    """Shared LLM-nondeterminism markdown block — used by BOTH the propose
    and finalize/resume renderers so all three commands report it."""
    nd_all = summary.get("nondeterminism") or {}
    full_detail = summary.get("tier1_full_detail") or {}
    if not nd_all:
        return ""
    out = "\n## Nondeterministic output columns (LLM analysis of original query)\n\n"
    for orig, nd in nd_all.items():
        cols = nd.get("columns") or {}
        flagged = nd.get("nondeterministic_columns") or []
        if not flagged:
            out += (
                f"`{orig}`: no nondeterministic output columns detected — "
                f"all values are pure functions of the pinned inputs.\n\n"
            )
            continue
        eff = (full_detail.get(orig, {}) or {}).get(
            "effective_ignore_columns", []
        )
        out += (
            f"`{orig}` — model `{nd.get('model', '?')}`:\n\n"
            "| Column | Class | Authorization | Excluded from diff? | Rationale |\n"
            "|---|---|---|---|---|\n"
        )
        for c in flagged:
            e = cols.get(c, {})
            excluded = "✅ yes" if c in eff else "❌ no (needs probe)"
            rat = (e.get("rationale") or "").replace("|", "\\|")
            out += (
                f"| `{c}` | {e.get('class')} | "
                f"{e.get('authorization')} | {excluded} | {rat} |\n"
            )
        probe = (full_detail.get(orig, {}) or {}).get(
            "nondeterminism_probe_required", []
        )
        if probe:
            out += (
                f"\n> ⚠️ **{len(probe)} column(s) flagged probe-required** "
                f"({', '.join(f'`{c}`' for c in probe)}): data-derived "
                "nondeterminism (e.g. untied argmax). NOT auto-excluded — "
                "structurally indistinguishable from a real bug. The tie-break "
                "corroboration table below shows the empirical evidence.\n"
            )
        # Empirical tie-break corroboration (value-preservation check).
        tbc = (full_detail.get(orig, {}) or {}).get("tie_break_corroboration") or []
        if tbc:
            out += (
                "\n### Tie-break corroboration (empirical)\n\n"
                "For each probe-required column with a declared ORDER BY "
                "sibling, we joined prod-side vs sandbox-side on the stable "
                "natural key and inspected whether differing carried "
                "attributes are accompanied by matching sibling values "
                "(tie-break signature) or differing siblings (potential "
                "real bug).\n\n"
                "| Column | ORDER BY sibling | Rows where col differs | "
                "Sibling matches | Sibling differs | Verdict |\n"
                "|---|---|---|---|---|---|\n"
            )
            for t in tbc:
                v = t["verdict"]
                vmark = {
                    "TIE_BREAK_CONFIRMED": "✅ TIE_BREAK_CONFIRMED",
                    "DETERMINISTIC_EMPIRICALLY": "ℹ️ DETERMINISTIC_EMPIRICALLY",
                    "POSSIBLE_REAL_BUG": "❌ POSSIBLE_REAL_BUG",
                    "DIFFERS_NO_SIBLING_TO_CHECK": "⚠️ DIFFERS_NO_SIBLING_TO_CHECK",
                }.get(v, v)
                sib = t.get("sibling")
                sib_cell = f"`{sib}`" if sib else "—"
                sm = f"{t['sibling_SAME']:,}" if sib else "—"
                sd = f"{t['sibling_DIFF']:,}" if sib else "—"
                out += (
                    f"| `{t['column']}` | {sib_cell} | "
                    f"{t['diff_total']:,} | {sm} | {sd} | {vmark} |\n"
                )
            out += "\n"
            for t in tbc:
                out += f"- **`{t['column']}`**: {t['note']}\n"
            out += "\n"
            # Loud warning if any column flagged POSSIBLE_REAL_BUG.
            if any(t["verdict"] == "POSSIBLE_REAL_BUG" for t in tbc):
                bad = [t["column"] for t in tbc
                       if t["verdict"] == "POSSIBLE_REAL_BUG"]
                out += (
                    f"\n> ❌ **WARNING**: {', '.join(f'`{c}`' for c in bad)} "
                    "show differing ORDER BY siblings on some rows — this is "
                    "NOT pure tie-break and may indicate the optimization "
                    "broke the dedup algebra. Investigate before shipping.\n\n"
                )
        out += "\n"
    return out


def _render_proposal_md(
    *, summary: dict[str, Any], clone: TaskCloneResult, history: HistoryBaseline
) -> str:
    """Render the human-readable proposal document."""
    agent = summary.get("agent", {})
    scores = summary.get("scores", {})
    t1 = scores.get("tier1", {})
    t3 = scores.get("tier3", {})

    # Build the notebook diff (sandbox-pre-agent vs sandbox-post-agent).
    # The post-agent source is what's currently at the workspace path.
    try:
        from ..tools.databricks import get_notebook_source
        post = get_notebook_source(clone.sandbox_notebook_path)["content"]
    except Exception:
        post = "(could not fetch post-agent notebook source)"

    import difflib
    diff = "".join(difflib.unified_diff(
        clone.sandbox_notebook_source.splitlines(keepends=True),
        post.splitlines(keepends=True),
        fromfile="original_(write-targets-remapped)",
        tofile="agent_optimized",
        n=3,
    ))

    runtime_pct = t3.get("runtime_improvement_pct", 0)
    equiv_verdict = t1.get("equivalence_verdict", "UNKNOWN")

    # Headline status maps verdict → human label
    if t1.get("passed") and t3.get("passed"):
        verdict = {
            "IDENTICAL": "✅ PASS (output byte-identical)",
            "FLOAT_REORDER_ONLY": "✅ PASS (output equivalent — only machine-epsilon float drift)",
        }.get(equiv_verdict, "✅ PASS")
    else:
        verdict = "⚠️ NEEDS REVIEW"

    # Output-equivalence label for TL;DR
    equiv_label = {
        "IDENTICAL": "YES (byte-identical)",
        "FLOAT_REORDER_ONLY": "YES (machine-epsilon float drift only — semantically equivalent)",
        "REAL_DIFFERENCE": "NO (real semantic divergence)",
        "UNKNOWN": "UNKNOWN",
    }.get(equiv_verdict, equiv_verdict)

    # Per-table boundary section explanation
    per_table_count = len(t1.get('details', {}).get('per_table', {}))
    if equiv_verdict == "IDENTICAL":
        boundary_summary = f"All {per_table_count} table(s) byte-identical to prod."
    elif equiv_verdict == "FLOAT_REORDER_ONLY":
        boundary_summary = (
            f"All {per_table_count} table(s) match prod at machine-epsilon precision. "
            f"Hashes differ because doubles got summed in a different order — "
            f"row counts and integer/decimal columns match exactly; total revenue / "
            f"spend / counts agree to ~10⁻¹³ relative. Safe to apply."
        )
    else:
        boundary_summary = "SOME MISMATCHED — see Tier 1 details below."

    return f"""# Proposal: optimize `{summary['task_key']}` in job {summary['prod_job_id']}

**Status**: {verdict}
**Run id**: `{summary['run_id']}`

## TL;DR

- Baseline (median of last {history.samples} prod runs): **{history.median_duration_ms/1000:.0f} s**
- Optimized (sandbox clone): **{summary.get('optimized_run', {}).get('duration_ms', 0)/1000:.0f} s**
- **Runtime delta**: {runtime_pct:+.1f}%
- Output equivalence: **{equiv_label}** ({per_table_count} boundary table(s) checked)
- Sandbox job for proof: `{summary.get('sandbox_job_id')}`

## Diagnosis

{agent.get('final_text') or '(agent did not produce a final summary)'}

## Boundary contract

| Original (prod) | Sandbox (clone writes here) |
|---|---|
{chr(10).join(f"| `{orig}` | `{sb}` |" for orig, sb in zip(clone.write_targets_original, clone.write_targets_sandbox))}

{boundary_summary}
{_render_nd_section(summary)}
## Diff: what the agent changed

```diff
{diff or '(no diff — agent may not have modified the notebook)'}
```

## Approval checklist

To apply this proposal to prod:
1. Review the diff above.
2. Verify Tier 1 equivalence numbers in `proposal.json`.
3. Open a PR against `Pocket-Fm/de_databricks` modifying `{clone.original_notebook_path}`.
4. After merge, the next run of prod job {summary['prod_job_id']} picks up the change via git_source.
5. Monitor task `{summary['task_key']}` runtime; expected median ~{summary.get('optimized_run', {}).get('duration_ms', 0)/1000:.0f}s.

## Full agent trace + scores

- Tool calls: {agent.get('tool_calls', 0)} across {agent.get('iterations_used', 0)} LLM turns
- Sandbox violations: {agent.get('sandbox_violations', 0)}
- See `trace.jsonl` and `proposal.json` for raw data.
"""


def resume(
    run_id: str,
    *,
    console: Console | None = None,
) -> dict[str, Any]:
    """Resume a propose run that was interrupted (Ctrl-C, crash, LLM timeout).

    Loads the persisted agent message history + baseline + clone metadata,
    re-instantiates the RunContext + tool guards, and re-enters the agent
    loop. The agent continues from where it stopped — including any pending
    tool calls. After it terminates, the harness runs the optimized job (if
    needed) and emits proposal.md.

    Requires:
      - clone.json, baseline.json, messages.json all present in the proposal dir
      - The sandbox job + workspace notebook still exist on Databricks (i.e.
        the original run wasn't cleaned up)
    """
    console = console or Console()
    proposal_dir = PROPOSALS_ROOT / run_id

    for required in ("clone.json", "messages.json"):
        if not (proposal_dir / required).exists():
            raise FileNotFoundError(
                f"missing {required} in {proposal_dir} — cannot resume. "
                f"(Use `propose-finalize` instead if you just want a report.)"
            )
    clone_meta = json.loads((proposal_dir / "clone.json").read_text())
    prior_messages = json.loads((proposal_dir / "messages.json").read_text())

    # Baseline: try cached baseline.json, fall back to re-pulling
    baseline_path = proposal_dir / "baseline.json"
    if baseline_path.exists():
        baseline_dict = json.loads(baseline_path.read_text())
        baseline_seconds = baseline_dict["median_duration_ms"] / 1000
        write_targets_original = baseline_dict["output_table_fqns"]
        console.print(
            f"[dim]using cached baseline: median {baseline_seconds:.0f}s "
            f"({len(prior_messages)} prior messages loaded)[/]"
        )
    else:
        console.print("[yellow]no baseline.json — re-pulling from prod history[/]")
        history = build_history_baseline(
            prod_job_id=clone_meta["prod_job_id"],
            task_key=clone_meta["task_key"],
            write_targets=clone_meta["write_targets_original"],
        )
        baseline_dict = asdict(history)
        baseline_path.write_text(json.dumps(baseline_dict, indent=2, default=str))
        baseline_seconds = history.median_duration_ms / 1000
        write_targets_original = clone_meta["write_targets_original"]

    # Reconstruct context from clone.json (do NOT re-run setup)
    run_schema = clone_meta["run_schema"]
    ctx = _make_propose_run_context(run_id=run_id, run_schema=run_schema)

    console.print(
        f"[bold cyan]helios propose-resume[/] run_id=[yellow]{run_id}[/] "
        f"task=[yellow]{clone_meta['task_key']}[/]"
    )
    console.print(
        f"  prod_job (frozen):    {clone_meta['prod_job_id']}\n"
        f"  sandbox job:          {clone_meta['sandbox_job_id']}\n"
        f"  messages restored:    {len(prior_messages)}"
    )

    live_trace_path = proposal_dir / "trace.live.jsonl"
    message_log_path = proposal_dir / "messages.json"

    sandbox_output_fqn = (clone_meta["write_targets_sandbox"] or [""])[0]
    agent_result = run_agent(
        ctx,
        int(clone_meta["sandbox_job_id"]),
        sandbox_output_fqn,
        console=console,
        frozen_job_ids=frozenset({int(clone_meta["prod_job_id"])}),
        live_trace_path=live_trace_path,
        message_log_path=message_log_path,
        prior_messages=prior_messages,
    )

    summary: dict[str, Any] = {
        "mode": "propose-resume",
        "prod_job_id": clone_meta["prod_job_id"],
        "task_key": clone_meta["task_key"],
        "run_id": run_id,
        "resumed_at": int(time.time()),
        "sandbox_job_id": int(clone_meta["sandbox_job_id"]),
        "agent": {
            "final_job_id": agent_result.final_job_id,
            "iterations_used": agent_result.iterations_used,
            "tool_calls": len(agent_result.trace),
            "sandbox_violations": len(agent_result.sandbox_violations),
            "final_text": agent_result.final_text,
            "failed": agent_result.failed,
            "failure_reason": agent_result.failure_reason,
        },
        "baseline": baseline_dict,
    }
    _write_trace(proposal_dir, agent_result)

    if agent_result.failed:
        console.print(f"[red]agent failed on resume: {agent_result.failure_reason}[/]")
        summary["optimized_run"] = {"skipped": "agent_failed"}
    else:
        from ..tools.databricks import run_job_now, wait_for_job_run
        console.print(
            f"[cyan]→ triggering optimized sandbox job {agent_result.final_job_id} "
            f"for clean measurement[/]"
        )
        rn = run_job_now(agent_result.final_job_id)
        timeout_s = max(3600, int(baseline_seconds * 2))
        opt = wait_for_job_run(int(rn["run_id"]), timeout_seconds=timeout_s, poll_interval_seconds=20)
        succeeded = (not opt.get("timed_out")) and opt.get("result_state") == "SUCCESS"
        opt_ms = int(opt.get("execution_duration_ms") or 0)
        summary["optimized_run"] = {
            "run_id": opt.get("run_id"),
            "succeeded": succeeded,
            "duration_ms": opt_ms,
        }
        # Aliases the markdown renderer expects (same shape as finalize)
        summary["scored_run_id"] = opt.get("run_id")
        summary["scored_run_exec_ms"] = opt_ms
        # Score via shared equivalence+nondeterminism helper + Tier 3
        _sc = _score_resume(
            baseline_seconds=baseline_seconds,
            optimized_ms=opt_ms,
            write_orig=write_targets_original,
            write_sb=clone_meta["write_targets_sandbox"],
            optimized_succeeded=succeeded,
            proposal_dir=proposal_dir,
            console=console,
            prod_boundary_versions=clone_meta.get("prod_boundary_versions") or {},
            explicit_ignore=list(clone_meta.get("equivalence_ignore_columns") or []),
            write_target_modes=clone_meta.get("write_target_modes") or {},
            prod_pre_versions=clone_meta.get("prod_pre_versions") or {},
            partition_columns_by_target=clone_meta.get("partition_columns_by_target") or {},
        )
        summary["scores"] = {"tier1": _sc["tier1"], "tier3": _sc["tier3"]}
        summary["nondeterminism"] = _sc["nondeterminism"]
        summary["tier1_full_detail"] = _sc["tier1_full_detail"]

    (proposal_dir / "proposal.json").write_text(json.dumps(summary, indent=2, default=str))
    md = _render_finalize_md(summary=summary, clone_meta=clone_meta, proposal_dir=proposal_dir)
    (proposal_dir / "proposal.md").write_text(md)
    console.print(
        f"\n[bold green]done.[/] proposal at {proposal_dir / 'proposal.md'}"
    )
    return summary


def _score_resume(
    *,
    baseline_seconds: float,
    optimized_ms: int,
    write_orig: list[str],
    write_sb: list[str],
    optimized_succeeded: bool,
    proposal_dir: Path,
    console: Console,
    prod_boundary_versions: dict[str, int] | None = None,
    explicit_ignore: list[str] | None = None,
    write_target_modes: dict[str, str] | None = None,
    prod_pre_versions: dict[str, int] | None = None,
    partition_columns_by_target: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Score a resume run via the shared equivalence+nondeterminism helper —
    identical Tier-1 semantics to propose / propose-finalize."""
    eq = _equivalence_and_nd(
        write_orig=write_orig,
        write_sb=write_sb,
        prod_boundary_versions=prod_boundary_versions or {},
        explicit_ignore=list(explicit_ignore or []),
        proposal_dir=proposal_dir,
        console=console,
        write_target_modes=write_target_modes or {},
        prod_pre_versions=prod_pre_versions or {},
        partition_columns_by_target=partition_columns_by_target or {},
        optimized_succeeded=optimized_succeeded,
    )
    tier1 = _tier1_from_equivalence(eq)
    runtime_pct = 0.0
    base_ms = int(baseline_seconds * 1000)
    if optimized_succeeded and optimized_ms > 0 and base_ms > 0:
        runtime_pct = (base_ms - optimized_ms) / base_ms * 100
    return {
        "tier1": tier1,
        "tier3": {
            "passed": runtime_pct >= 15.0,
            "runtime_improvement_pct": round(runtime_pct, 2),
            "baseline_duration_ms": base_ms,
            "optimized_duration_ms": optimized_ms,
        },
        "nondeterminism": eq["nondeterminism"],
        "tier1_full_detail": eq["per_table"],
    }


def finalize(
    run_id: str,
    *,
    console: Console | None = None,
) -> dict[str, Any]:
    """Generate proposal.md for an in-progress / interrupted / agent-stuck
    propose run. Reads `clone.json` to find the boundary tables + sandbox job,
    picks the latest SUCCESS sandbox run, scores against current prod, emits
    proposal.md.

    Useful when the agent ran out of LLM budget mid-iteration, or you want to
    grab the current best state without waiting for it to converge.
    """
    from ..tools.databricks import diff_tables, list_job_runs, get_job_run, get_notebook_source
    console = console or Console()

    proposal_dir = PROPOSALS_ROOT / run_id
    if not (proposal_dir / "clone.json").exists():
        raise FileNotFoundError(
            f"no clone.json in {proposal_dir}; this run either never started cleanly "
            f"or pre-dates the clone-persistence change. Cannot finalize."
        )
    clone_meta = json.loads((proposal_dir / "clone.json").read_text())
    sandbox_job_id = int(clone_meta["sandbox_job_id"])
    prod_job_id = int(clone_meta["prod_job_id"])
    task_key = clone_meta["task_key"]
    write_orig = clone_meta["write_targets_original"]
    write_sb   = clone_meta["write_targets_sandbox"]

    console.print(
        f"[bold cyan]helios propose-finalize[/] run_id=[yellow]{run_id}[/] "
        f"task=[yellow]{task_key}[/]"
    )

    # 1. Find the latest SUCCESS sandbox run
    runs = list_job_runs(job_id=sandbox_job_id, limit=10)["runs"]
    successes = [r for r in runs if r.get("result_state") == "SUCCESS"]
    if not successes:
        raise RuntimeError(
            f"no SUCCESS runs found on sandbox job {sandbox_job_id}. The agent never "
            f"produced a working optimized version."
        )
    scored = max(successes, key=lambda r: r["start_time"])
    detail = get_job_run(scored["run_id"])
    scored_task = next(
        (t for t in detail.get("tasks") or [] if t.get("task_key") == task_key),
        None,
    )
    scored_exec_ms = (scored_task or {}).get("execution_duration") or 0
    console.print(
        f"  scoring against run [yellow]{scored['run_id']}[/] "
        f"(exec {scored_exec_ms/1000:.0f}s)"
    )

    # 2. Re-derive history baseline. If proposal.json exists with cached baseline,
    # reuse it; otherwise re-pull from prod history.
    summary: dict[str, Any] = {
        "mode": "propose-finalize",
        "prod_job_id": prod_job_id,
        "task_key": task_key,
        "run_id": run_id,
        "finalized_at": int(time.time()),
        "scored_run_id": scored["run_id"],
        "scored_run_exec_ms": scored_exec_ms,
    }
    existing_proposal = proposal_dir / "proposal.json"
    if existing_proposal.exists():
        prev = json.loads(existing_proposal.read_text())
        baseline = prev.get("baseline")
    else:
        baseline = None
    if baseline:
        console.print(f"  using cached baseline (median {baseline['median_duration_ms']/1000:.0f}s)")
    else:
        console.print("  no cached baseline — re-pulling from prod history...")
        baseline_obj = build_history_baseline(
            prod_job_id=prod_job_id, task_key=task_key, write_targets=write_orig,
        )
        baseline = asdict(baseline_obj)
        console.print(f"  baseline median {baseline['median_duration_ms']/1000:.0f}s")
    summary["baseline"] = baseline

    # 2b + 3. Tier 1 via the shared equivalence+nondeterminism helper.
    eq = _equivalence_and_nd(
        write_orig=write_orig,
        write_sb=write_sb,
        prod_boundary_versions=clone_meta.get("prod_boundary_versions") or {},
        explicit_ignore=list(clone_meta.get("equivalence_ignore_columns") or []),
        proposal_dir=proposal_dir,
        console=console,
        optimized_succeeded=True,
        write_target_modes=clone_meta.get("write_target_modes") or {},
        prod_pre_versions=clone_meta.get("prod_pre_versions") or {},
        partition_columns_by_target=clone_meta.get("partition_columns_by_target") or {},
    )
    summary["nondeterminism"] = eq["nondeterminism"]
    per_table = eq["per_table"]
    all_eq = eq["all_eq"]

    # 4. Tier 3 perf
    runtime_pct = 0.0
    base_ms = baseline.get("median_duration_ms") or 0
    if scored_exec_ms > 0 and base_ms > 0:
        runtime_pct = (base_ms - scored_exec_ms) / base_ms * 100
    summary["scores"] = {
        "tier1": {
            "passed": all_eq,
            "per_table": {k: {"verdict": v["verdict"],
                              "buckets": v["buckets"],
                              "drift_concentration": v.get("drift_concentration", {})}
                          for k, v in per_table.items()},
        },
        "tier3": {
            "passed": runtime_pct >= 15.0,
            "runtime_improvement_pct": round(runtime_pct, 2),
            "baseline_duration_ms": base_ms,
            "optimized_duration_ms": scored_exec_ms,
        },
    }
    summary["tier1_full_detail"] = per_table  # full diff for the report

    # 5. Write proposal.json
    (proposal_dir / "proposal.json").write_text(json.dumps(summary, indent=2, default=str))

    # 6. Render proposal.md
    md = _render_finalize_md(summary=summary, clone_meta=clone_meta, proposal_dir=proposal_dir)
    (proposal_dir / "proposal.md").write_text(md)

    # 7. Print verdict
    t1, t3 = summary["scores"]["tier1"], summary["scores"]["tier3"]
    verdict = "[bold green]PASS[/]" if (t1["passed"] and t3["passed"]) else "[bold yellow]NEEDS REVIEW[/]"
    console.print(f"\n  Overall: {verdict}")
    console.print(f"  Tier 1 (equivalence): {'PASS' if t1['passed'] else 'FAIL'}")
    console.print(f"  Tier 3 (perf):        {'PASS' if t3['passed'] else 'FAIL'}  "
                  f"({t3['runtime_improvement_pct']:+.1f}% vs {base_ms/1000:.0f}s baseline)")
    console.print(f"  proposal: {proposal_dir / 'proposal.md'}")
    return summary


def _render_finalize_md(
    *, summary: dict[str, Any], clone_meta: dict[str, Any], proposal_dir: Path,
) -> str:
    """Render a markdown proposal from a finalize() summary."""
    import difflib
    from ..tools.databricks import get_notebook_source

    t1 = summary["scores"]["tier1"]
    t3 = summary["scores"]["tier3"]
    verdict = "✅ PASS" if (t1["passed"] and t3["passed"]) else "⚠️ NEEDS REVIEW"

    # Build per-table verdict table (mode = full_rewrite or incremental)
    full_detail = summary.get("tier1_full_detail") or {}
    per_table_md = []
    increment_blocks: list[str] = []
    for orig, info in t1["per_table"].items():
        b = info["buckets"]
        det = full_detail.get(orig, {}) or {}
        mode = det.get("mode") or "full_rewrite"
        per_table_md.append(
            f"| `{orig}` | {mode} | {info['verdict']} | "
            f"{b.get('identical', 0):,} | {b.get('extra_in_b', 0):,} | "
            f"{b.get('missing_from_b', 0):,} | "
            f"{b.get('same_key_drifted_metric', 0):,} |"
        )
        inc = det.get("increment")
        if inc:
            pre = inc.get('pre_version')
            post = inc.get('post_version')
            wh  = inc.get('where_clause')
            ex  = inc.get('used_except')
            increment_blocks.append(
                f"- `{orig}`: compared **prod increment** (v{post}"
                + (f" EXCEPT v{pre}" if ex else "")
                + (f" WHERE `{wh}`" if wh else "")
                + ") vs sandbox. Prod's accumulated history is NOT in scope; "
                "only the rows this task's commit added/modified."
            )
    increment_section = (
        "\n### Incremental scope (this task is `INSERT INTO` / `MERGE INTO`)\n\n"
        + "\n".join(increment_blocks) + "\n"
    ) if increment_blocks else ""

    # Notebook diff (sandbox-pre-agent vs whatever's at the workspace path now)
    diff_section = ""
    pre_path = proposal_dir / "notebook_sandbox_pre_agent.txt"
    if pre_path.exists():
        try:
            current = get_notebook_source(clone_meta["sandbox_notebook_path"])["content"]
            pre = pre_path.read_text()
            diff = "".join(difflib.unified_diff(
                pre.splitlines(keepends=True), current.splitlines(keepends=True),
                fromfile="pre_agent", tofile="agent_optimized", n=3,
            ))
            if diff:
                diff_section = f"```diff\n{diff[:6000]}\n```"
        except Exception as e:
            diff_section = f"(could not fetch notebook diff: {e})"

    # Concentration of drift
    drift_section = ""
    for orig, info in t1["per_table"].items():
        conc = info.get("drift_concentration") or {}
        for bucket, entries in conc.items():
            if entries and isinstance(entries[0], dict) and "dim_value" in entries[0]:
                drift_section += f"\n### {orig} — top `{bucket}` by first dimension\n\n"
                for e in entries[:8]:
                    drift_section += f"- `{e['dim_value']}`: {e['rows']:,} rows\n"

    nd_section = _render_nd_section(summary)

    return f"""# Proposal: optimize `{summary['task_key']}` in job {summary['prod_job_id']}

**Status**: {verdict}
**Run id**: `{summary['run_id']}`
**Scored against**: sandbox run `{summary.get('scored_run_id') or summary.get('optimized_run', {}).get('run_id') or 'unknown'}` (latest SUCCESS run)

## TL;DR

| Metric | Value |
|---|---|
| Baseline median runtime | {summary['baseline'].get('median_duration_ms', 0)/1000:.0f}s ({summary['baseline'].get('median_duration_ms', 0)/60000:.1f} min) |
| Optimized runtime | {(t3.get('optimized_duration_ms') or 0)/1000:.0f}s ({(t3.get('optimized_duration_ms') or 0)/60000:.1f} min) |
| **Runtime improvement** | **{t3.get('runtime_improvement_pct', 0):+.1f}%** {'✅' if t3.get('passed') else '❌'} |
| Tier 1 (row-level equivalence) | **{'PASS' if t1['passed'] else 'FAIL'}** |
| Tier 3 (runtime ≥15%) | **{'PASS' if t3['passed'] else 'FAIL'}** |

## Equivalence by boundary table

| Boundary table | Mode | Verdict | Identical | Extra in sandbox | Missing | Drifted-metric |
|---|---|---|---|---|---|---|
{chr(10).join(per_table_md)}

(Verdict types: **IDENTICAL** = byte-identical, **FLOAT_REORDER_ONLY** = only machine-epsilon
drift on DOUBLE columns, **REAL_DIFFERENCE** = real semantic divergence — agent's algebra is off.
Modes: **full_rewrite** = compared against the full pinned prod table; **incremental** = compared
against just the prod-side increment for this task's commit.)
{increment_section}{drift_section}
{nd_section}
## Agent's notebook changes

{diff_section}

## Sandbox proof artifacts

- Sandbox table(s): {', '.join(f'`{x}`' for x in clone_meta['write_targets_sandbox'])}
- Sandbox notebook: `{clone_meta['sandbox_notebook_path']}`
- Sandbox job: `{clone_meta['sandbox_job_id']}`
- Raw scores + full diff_tables output: `evals/proposals/{summary['run_id']}/proposal.json`

To clean up when done: `helios eval cleanup {summary['run_id']}`

## Approval checklist (if both tiers pass)

1. Review the diff above.
2. Open a PR against `Pocket-Fm/de_databricks` modifying `{clone_meta['original_notebook_path']}`.
3. After merge, the next run of prod job {summary['prod_job_id']} picks up the change via git_source.
4. Monitor task `{summary['task_key']}` runtime; expected median ~{t3['optimized_duration_ms']/1000:.0f}s.

## If Tier 1 = REAL_DIFFERENCE (do NOT apply)

The agent's optimized output is not byte-equivalent to prod. The `drift_concentration` section
above shows where the divergence concentrates — use it to localize the broken CTE.
Common patterns:
- All extras on the most recent install dates → cardinality fan-out in a join
- Integer columns drift (BIGINT/DECIMAL) → real algebra error, not float reorder
- DOUBLE columns drift but totals match → likely float reorder, often acceptable
"""


def _make_propose_run_context(*, run_id: str, run_schema: str) -> sandbox.RunContext:
    """Build a RunContext that doesn't depend on a Fixture object.
    Propose mode has no fixture — the inputs are the prod job_id + task_key."""
    workspace_dir = f"{sandbox.WORKSPACE_ROOT}/proposal_{run_id}"
    # Use a stub fixture-like object that has the .seed_schema attribute the
    # other helpers expect. We never call ensure_seed_schema in propose mode.
    class _StubFixture:
        id = "propose"
        seed_schema = "propose"
        version = 0
        scope = None        # propose mode handles scope explicitly, not via fixture
        notebooks_dir = None
        tool_call_budget = 30
        # `investigation` / `fix` fields are accessed by some scorers; not used in propose.
        investigation = None
        fix = None
    return sandbox.RunContext(
        run_id=run_id,
        fixture=_StubFixture(),  # type: ignore[arg-type]
        seed_catalog=sandbox.SEED_CATALOG,
        seed_schema="propose",
        run_catalog=sandbox.RUN_CATALOG,
        run_schema=run_schema,
        workspace_dir=workspace_dir,
    )


def _write_trace(proposal_dir: Path, agent_result) -> None:
    path = proposal_dir / "trace.jsonl"
    with path.open("w") as f:
        for entry in agent_result.trace:
            f.write(json.dumps(asdict(entry), default=str) + "\n")
