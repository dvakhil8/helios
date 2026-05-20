"""Invoke the Helios agent against a sandboxed job and capture its trace.

The runner does three things:
  1. Builds the agent prompt: tells the model what catalog/schema it may write
     to and what job_id it's optimizing. Nothing else — the agent must
     investigate using its tools, same as it would in prod.
  2. Hooks `on_tool_call` / `on_tool_result` to capture a structured trace.
  3. Application-level write-guard: intercepts mutating SQL and rejects any
     write that targets a catalog/schema outside the run sandbox. This is
     belt-and-suspenders alongside the Databricks-side grants on the eval SP.

Returns the final job_id the agent settled on (which may differ from the one
we handed it, if the agent called create_job to make a new one).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from ..agent import run_turn
from ..tools import call_tool as original_call_tool
from .sandbox import RunContext


@dataclass
class TraceEntry:
    timestamp: float
    tool: str
    args: dict[str, Any]
    result_preview: str
    blocked: bool = False
    block_reason: str | None = None


@dataclass
class AgentRunResult:
    final_job_id: int
    final_text: str | None
    trace: list[TraceEntry] = field(default_factory=list)
    sandbox_violations: list[TraceEntry] = field(default_factory=list)
    iterations_used: int = 0
    failed: bool = False
    failure_reason: str | None = None


# Tools the agent uses to write data. Their args go through the write-guard.
_SQL_TOOLS: frozenset[str] = frozenset({"execute_sql"})

# Mutating tools that operate on a specific job_id. When propose mode is active,
# any of these whose job_id is in the frozen-set is rejected before dispatch —
# the prod job must never be touched, no matter what the agent's reasoning says.
_JOB_MUTATION_TOOLS: frozenset[str] = frozenset({
    "run_job_now",
    "add_job_tasks",
})

# Catalogs the agent may write to. Reads are unrestricted; writes are not.
# Pattern: matches anything qualifying a write to a catalog other than these.
_WRITE_VERBS = (
    "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "REPLACE",
    "DROP", "TRUNCATE", "ALTER", "OPTIMIZE", "VACUUM", "ANALYZE",
)
_WRITE_VERB_RE = re.compile(
    r"\b(" + "|".join(_WRITE_VERBS) + r")\b", re.IGNORECASE
)
# Catalog references in Spark SQL: catalog.schema.table or `catalog`.`schema`.`table`.
_CATALOG_REF_RE = re.compile(r"`?([A-Za-z0-9_]+)`?\.`?[A-Za-z0-9_]+`?\.`?[A-Za-z0-9_]+`?")


def _is_write_sql(sql: str) -> bool:
    return bool(_WRITE_VERB_RE.search(sql))


def _referenced_catalogs(sql: str) -> set[str]:
    return {m.group(1).lower() for m in _CATALOG_REF_RE.finditer(sql)}


def _make_guarded_call_tool(ctx: RunContext, trace: list[TraceEntry],
                             violations: list[TraceEntry],
                             frozen_job_ids: frozenset[int] = frozenset(),
                             live_log: Any = None):
    """Wrap call_tool so:
      - SQL writes outside the sandbox catalog are rejected (existing guard).
      - Job-mutation tools targeting a frozen_job_id are rejected (new — propose
        mode passes the prod job_id(s) here so the agent can read them but can
        NEVER trigger / modify them, no matter what it reasons).
    """
    allowed_catalogs = {ctx.run_catalog.lower()}

    def call(name: str, **kwargs: Any) -> Any:
        # Frozen-job guard — applies to every mutating job tool unconditionally.
        if name in _JOB_MUTATION_TOOLS and kwargs.get("job_id") in frozen_job_ids:
            reason = (
                f"frozen job: {name} blocked against job_id={kwargs.get('job_id')}. "
                f"This job is read-only in propose mode."
            )
            entry = TraceEntry(
                timestamp=time.time(), tool=name, args=kwargs,
                result_preview="", blocked=True, block_reason=reason,
            )
            violations.append(entry)
            _live_write(live_log, "blocked",
                        tool=name, args=kwargs, reason=reason)
            return {"error": reason}
        if name in _SQL_TOOLS:
            sql = str(kwargs.get("sql", ""))
            if _is_write_sql(sql):
                refs = _referenced_catalogs(sql)
                # If the SQL references a fully-qualified table outside the
                # allowed catalogs, refuse. Statements with no fully-qualified
                # ref (relying on a `catalog`/`schema` arg) are caught by the
                # explicit `catalog` kwarg below.
                outside = refs - allowed_catalogs if refs else set()
                catalog_kwarg = (kwargs.get("catalog") or "").lower()
                if catalog_kwarg and catalog_kwarg not in allowed_catalogs:
                    outside.add(catalog_kwarg)
                if outside:
                    reason = (
                        f"sandbox: writes only allowed against {sorted(allowed_catalogs)}; "
                        f"this SQL touches {sorted(outside)}"
                    )
                    entry = TraceEntry(
                        timestamp=time.time(),
                        tool=name,
                        args=kwargs,
                        result_preview="",
                        blocked=True,
                        block_reason=reason,
                    )
                    violations.append(entry)
                    _live_write(live_log, "blocked",
                                tool=name, args=kwargs, reason=reason)
                    return {"error": reason}
        return original_call_tool(name, **kwargs)

    return call


_AGENT_INSTRUCTIONS = """\
You are optimizing a Databricks job. Investigate it, find the performance bug,
and apply a fix using the tools available.

INPUTS:
  job_id          = {job_id}
  output table    = {output_table_fqn}   (the job writes here; you may inspect or
                                          query it)
  run sandbox     = {run_catalog}.{run_schema}
                    (the ONLY place you may write; reads are unrestricted)
{scope_block}
GOAL:
  Apply changes so the job runs faster / cheaper while producing the SAME output
  table content. You may:
    - Modify the existing job (update task SQL / notebook / cluster spec).
    - Run OPTIMIZE / ZORDER / ANALYZE against tables in the run sandbox.
    - Create a new job (you'll need to set its tags helios_eval_run_id={run_id} so
      teardown can find it).
  You may NOT write to any catalog other than {run_catalog}.

WHEN DONE:
  Respond with a one-line summary of the diagnosis and the fix you applied, and
  the final job_id that should be used to measure outcome (yours OR the original
  if you mutated it in place). Format the last line exactly as:

    FINAL_JOB_ID=<integer>
"""


_SCOPE_BLOCK_TEMPLATE = """
SCOPE:
  IN-SCOPE tasks (you MAY modify these): {in_scope}
  OUT-OF-SCOPE tasks (you may READ them — call get_notebook_source on their
    notebook_paths — but you must NOT modify their specs):
      everything else in the job's task list

  Equivalence is checked ONLY on these output tables; do not change them:
    {output_tables}

  Runtime is measured ONLY on the in-scope tasks (max of their durations).
  Other tasks' specs must remain byte-identical between the original job and
  your final job (strict scope adherence — a Tier 1 sub-score will diff them).
"""


_FINAL_JOB_RE = re.compile(r"FINAL_JOB_ID\s*=\s*(\d+)", re.IGNORECASE)


def _live_write(live_log: Any, event: str, **fields: Any) -> None:
    """Append one JSONL event to the live trace file. No-op if no file handle.

    Values are kept as native JSON types when they serialize <=2 KB; bigger
    payloads (notebook source, query results) are stringified and truncated so
    one 5 MB upload doesn't bloat the live trace file.
    """
    if live_log is None:
        return
    bounded: dict[str, Any] = {}
    for k, v in fields.items():
        try:
            s = json.dumps(v, default=str)
        except (TypeError, ValueError):
            s = str(v)
            v = s
        if len(s) <= 2000:
            bounded[k] = v
        else:
            bounded[k] = s[:2000] + f"...[+{len(s)-2000}b]"
    record = {"ts": time.time(), "event": event, **bounded}
    try:
        live_log.write(json.dumps(record, default=str) + "\n")
        live_log.flush()
    except Exception:
        pass


def run_agent(
    ctx: RunContext,
    candidate_job_id: int,
    output_table_fqn: str,
    *,
    max_iters: int = 80,
    console: Console | None = None,
    frozen_job_ids: frozenset[int] = frozenset(),
    extra_instructions: str = "",
    live_trace_path: Any = None,
    message_log_path: Any = None,
    prior_messages: list[dict[str, Any]] | None = None,
) -> AgentRunResult:
    """Drive the agent for one fixture run.

    Returns trace + final job_id even when the agent hits max_iters or raises —
    the trace is the most valuable diagnostic and must survive failure.
    """
    console = console or Console(quiet=True)

    trace: list[TraceEntry] = []
    violations: list[TraceEntry] = []

    # Live trace file — opens at start, flushed after every event, closed at end.
    live_log = None
    if live_trace_path:
        from pathlib import Path
        Path(live_trace_path).parent.mkdir(parents=True, exist_ok=True)
        live_log = open(live_trace_path, "w", buffering=1)  # line-buffered
        _live_write(live_log, "agent_start",
                    candidate_job_id=candidate_job_id,
                    frozen_job_ids=list(frozen_job_ids),
                    max_iters=max_iters)

    guarded = _make_guarded_call_tool(
        ctx, trace, violations, frozen_job_ids=frozen_job_ids, live_log=live_log,
    )

    def on_text(text: str) -> None:
        # Stream the model's reasoning into trace.jsonl so the live tail shows
        # the "why" alongside the tool I/O. Cap per-event size — long
        # reasoning bursts shouldn't bloat the log line indefinitely.
        snippet = text if len(text) <= 4000 else text[:4000] + "...[truncated]"
        _live_write(live_log, "assistant_text", text=snippet)

    def on_tool_call(name: str, args: dict[str, Any]) -> None:
        trace.append(TraceEntry(
            timestamp=time.time(), tool=name, args=args, result_preview=""
        ))
        _live_write(live_log, "tool_call", tool=name, args=args)

    def on_tool_result(name: str, result: Any) -> None:
        if trace and trace[-1].tool == name and not trace[-1].result_preview:
            preview = json.dumps(result, default=str)
            trace[-1].result_preview = (
                preview if len(preview) <= 400 else preview[:400] + "...[truncated]"
            )
            _live_write(live_log, "tool_result", tool=name, preview=trace[-1].result_preview)

    fixture = ctx.fixture
    if fixture.scope is not None:
        from .sandbox import render_scope_tables
        rendered_outputs = render_scope_tables(ctx, fixture.scope.output_tables)
        scope_block = _SCOPE_BLOCK_TEMPLATE.format(
            in_scope=fixture.scope.in_scope_task_keys,
            output_tables=", ".join(rendered_outputs),
        )
    else:
        scope_block = ""

    instructions = _AGENT_INSTRUCTIONS.format(
        job_id=candidate_job_id,
        output_table_fqn=output_table_fqn,
        run_catalog=ctx.run_catalog,
        run_schema=ctx.run_schema,
        run_id=ctx.run_id,
        scope_block=scope_block,
    )
    if extra_instructions:
        instructions += "\n\n" + extra_instructions

    import helios.agent as _agent_mod
    original = _agent_mod.call_tool
    _agent_mod.call_tool = guarded

    if prior_messages is not None:
        # Resume path — reuse the persisted history. Don't rebuild prompts.
        messages = list(prior_messages)
        _live_write(live_log, "resume_started", prior_message_count=len(prior_messages))
    else:
        from ..agent import build_system_message
        messages = [
            {"role": "system", "content": build_system_message()},
            {"role": "user", "content": instructions},
        ]
    failed = False
    failure_reason: str | None = None
    final_messages = messages
    try:
        final_messages = run_turn(
            messages,
            yolo=True,
            max_iters=max_iters,
            console=console,
            on_text=on_text,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            message_log_path=message_log_path,
        )
    except Exception as e:
        failed = True
        failure_reason = f"{type(e).__name__}: {e}"
        final_messages = messages  # whatever we accumulated before the raise
    finally:
        _agent_mod.call_tool = original

    final_text = None
    for m in reversed(final_messages):
        if m.get("role") == "assistant" and m.get("content"):
            final_text = m["content"]
            break

    final_job_id = _derive_final_job_id(
        trace=trace,
        sentinel_text=final_text,
        fallback_job_id=candidate_job_id,
    )
    iter_count = sum(1 for m in final_messages if m.get("role") == "assistant")

    if live_log is not None:
        _live_write(live_log, "agent_end",
                    failed=failed, failure_reason=failure_reason,
                    iterations=iter_count, tool_calls=len(trace),
                    sandbox_violations=len(violations),
                    final_job_id=final_job_id,
                    final_text=(final_text or "")[:500])
        try:
            live_log.close()
        except Exception:
            pass

    return AgentRunResult(
        final_job_id=final_job_id,
        final_text=final_text,
        trace=trace,
        sandbox_violations=violations,
        iterations_used=iter_count,
        failed=failed,
        failure_reason=failure_reason,
    )


def _derive_final_job_id(
    *, trace: list[TraceEntry], sentinel_text: str | None, fallback_job_id: int
) -> int:
    """Decide which job_id to measure against.

    Priority:
      1. FINAL_JOB_ID=<n> sentinel in the agent's last message (explicit choice).
      2. Last successful create_job call in the trace (agent created a new job).
      3. The candidate job_id we handed the agent (it modified in place or did nothing).
    """
    if sentinel_text:
        match = _FINAL_JOB_RE.search(sentinel_text)
        if match:
            return int(match.group(1))
    for entry in reversed(trace):
        if entry.tool != "create_job" or entry.blocked:
            continue
        # result_preview is a JSON string; pull the job_id if present
        try:
            preview = json.loads(entry.result_preview)
        except (json.JSONDecodeError, ValueError):
            continue
        jid = preview.get("job_id") if isinstance(preview, dict) else None
        if jid is not None:
            return int(jid)
    return fallback_job_id
