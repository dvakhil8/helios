"""End-to-end orchestrator for one eval run.

Flow:
  1. Load fixture, generate RunContext, create scratch schema.
  2. Ensure seed schema is at the fixture's declared version (re-seed if not).
  3. Acquire baseline: from cache, or by running the orig job and capturing.
  4. Clone the suboptimal job as the candidate (writes go to scratch.opt).
  5. Hand the candidate to the agent under the write-guard.
  6. Trigger the final agent-state job; wait; collect runtime + output stats.
  7. Score (Tier 1 only for now); write trace + scores to evals/results/<run_id>/.
  8. Teardown — drop scratch schema, delete jobs, remove workspace folder.

Each step is wrapped in best-effort try/finally so teardown always runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console

from ..tools.databricks import get_job, run_job_now, wait_for_job_run
from . import baselines, sandbox
from .baselines import Baseline, _capture_task_durations
from .fixtures import Fixture, load
from .runner import run_agent
from .sandbox import render_scope_tables
from .scorers.correctness import score as score_tier1
from .scorers.diagnosis import score as score_tier2
from .scorers.outcome import score as score_tier3


RESULTS_ROOT: Path = Path(__file__).resolve().parents[3] / "evals" / "results"


def run(fixture_id: str, *, refresh_baseline: bool = False,
        keep_artifacts: bool = False,
        console: Console | None = None) -> dict[str, Any]:
    """Run the eval for one fixture. Returns the scores dict (also written to disk).

    Teardown behavior:
      - Success path:           teardown unless keep_artifacts=True
      - Agent-failure path:     ALWAYS keep artifacts so the user can inspect
                                (override by passing keep_artifacts=False
                                 explicitly, but that's almost never useful)
    """
    console = console or Console()
    fixture = load(fixture_id)
    ctx = sandbox.new_run_context(fixture)
    results_dir = RESULTS_ROOT / ctx.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold cyan]helios eval[/] fixture=[yellow]{fixture.id}[/] "
        f"run_id=[yellow]{ctx.run_id}[/] schema=[dim]{ctx.run_catalog}.{ctx.run_schema}[/]"
    )

    summary: dict[str, Any] = {
        "fixture_id": fixture.id,
        "fixture_version": fixture.version,
        "run_id": ctx.run_id,
        "started_at": int(time.time()),
    }

    try:
        sandbox.ensure_catalogs_exist(ctx.seed_catalog, ctx.run_catalog)
        sandbox.create_run_schema(ctx)
        sandbox.ensure_seed_schema(fixture)

        # Render scope output_tables to FQNs (using base placeholders) — these
        # are the tables Tier 1 will hash, and we pass them to baseline capture.
        scope_outputs: list[str] | None = None
        if fixture.scope is not None:
            scope_outputs = render_scope_tables(ctx, fixture.scope.output_tables)
            console.print(f"[dim]scope.output_tables: {scope_outputs}[/]")

        baseline = _acquire_baseline(
            fixture, ctx, refresh=refresh_baseline, console=console,
            scope_outputs=scope_outputs,
        )
        summary["baseline"] = asdict(baseline)

        candidate_output_table = "opt_output"
        candidate_output_fqn = f"{ctx.run_catalog}.{ctx.run_schema}.{candidate_output_table}"
        candidate_job_id = sandbox.create_job(
            ctx, role="candidate", output_table=candidate_output_table
        )
        summary["candidate_job_id"] = candidate_job_id

        # Snapshot the candidate's task specs BEFORE the agent touches them —
        # needed for the scope_adherence sub-score.
        original_task_specs: dict[str, dict[str, Any]] = {}
        if fixture.scope is not None:
            cand_spec = get_job(candidate_job_id)
            for t in (cand_spec.get("settings", {}).get("tasks") or []):
                key = t.get("task_key")
                if key:
                    original_task_specs[key] = t

        live_trace_path = results_dir / "trace.live.jsonl"
        console.print(
            f"[cyan]→ invoking agent on job_id={candidate_job_id}[/]\n"
            f"  [dim]live trace: tail -f {live_trace_path}[/]"
        )
        agent_result = run_agent(
            ctx, candidate_job_id, candidate_output_fqn, console=console,
            live_trace_path=live_trace_path,
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
        # Persist the trace immediately — even on agent failure, it's our
        # primary diagnostic for what the agent attempted.
        _write_trace(results_dir, agent_result)

        # Tier 2 scores even on agent failure — diagnosis from a partial trace
        # is still informative ("agent asked the right questions but timed out").
        tier2 = score_tier2(
            fixture=fixture,
            trace=agent_result.trace,
            agent_final_text=agent_result.final_text,
        )

        if agent_result.failed:
            console.print(
                f"[red]agent failed: {agent_result.failure_reason}[/]\n"
                f"[dim]trace at {results_dir / 'trace.jsonl'}[/]"
            )
            summary["optimized_run"] = {"skipped": "agent_failed"}
            summary["scores"] = {
                "tier1": {"tier": 1, "passed": False, "skipped": "agent_failed"},
                "tier2": tier2.to_dict(),
                "tier3": {"tier": 3, "passed": False, "skipped": "agent_failed"},
            }
        else:
            console.print(
                f"[cyan]→ triggering optimized job (id={agent_result.final_job_id})[/]"
            )
            opt_run_id, opt_succeeded, opt_duration_ms = _trigger_and_wait(
                agent_result.final_job_id
            )
            summary["optimized_run"] = {
                "run_id": opt_run_id,
                "succeeded": opt_succeeded,
                "duration_ms": opt_duration_ms,
            }

            # For scoped fixtures, collect per-task durations + final task
            # specs so Tier 1 (scope adherence) and Tier 3 (scoped runtime)
            # can run.
            optimized_task_durations: dict[str, int] = {}
            final_task_specs: dict[str, dict[str, Any]] = {}
            if fixture.scope is not None and opt_succeeded:
                optimized_task_durations = _capture_task_durations(opt_run_id)
                final_spec = get_job(agent_result.final_job_id)
                for t in (final_spec.get("settings", {}).get("tasks") or []):
                    key = t.get("task_key")
                    if key:
                        final_task_specs[key] = t

            tier1 = score_tier1(
                baseline=baseline,
                optimized_output_fqn=candidate_output_fqn,
                optimized_run_succeeded=opt_succeeded,
                agent_result=agent_result,
                scope_outputs=scope_outputs,
                original_task_specs=original_task_specs or None,
                final_task_specs=final_task_specs or None,
                in_scope_task_keys=(fixture.scope.in_scope_task_keys if fixture.scope else None),
            )
            tier3 = score_tier3(
                fixture=fixture,
                baseline=baseline,
                optimized_duration_ms=opt_duration_ms,
                optimized_run_succeeded=opt_succeeded,
                optimized_task_durations_ms=optimized_task_durations or None,
            )
            summary["scores"] = {
                "tier1": tier1.to_dict(),
                "tier2": tier2.to_dict(),
                "tier3": tier3.to_dict(),
            }
        _print_verdict(console, summary)

        (results_dir / "scores.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    finally:
        agent_failed = bool(summary.get("agent", {}).get("failed"))
        should_keep = keep_artifacts or agent_failed
        if should_keep:
            reason = "user requested --keep-artifacts" if keep_artifacts else "agent failed; keeping for inspection"
            console.print(f"[yellow]→ skipping teardown ({reason})[/]")
            console.print(f"[dim]  schema:        {ctx.run_catalog}.{ctx.run_schema}[/]")
            console.print(f"[dim]  workspace dir: {ctx.workspace_dir}[/]")
            console.print(f"[dim]  results dir:   {results_dir}[/]")
            console.print(
                f"[dim]  to clean up later:  "
                f"helios eval cleanup {ctx.run_id}[/]"
            )
        else:
            console.print(f"[dim]→ teardown[/]")
            sandbox.teardown(ctx)


def _acquire_baseline(
    fixture: Fixture, ctx: sandbox.RunContext, *,
    refresh: bool, console: Console,
    scope_outputs: list[str] | None = None,
) -> Baseline:
    cached = None if refresh else baselines.get(fixture)
    if cached is not None:
        console.print(
            f"[green]✓ baseline cached[/] duration={cached.duration_ms}ms rows={cached.output_row_count}"
        )
        return cached

    console.print("[yellow]→ no baseline cached; running orig job to capture[/]")
    orig_output_table = "orig_output"
    orig_output_fqn = f"{ctx.run_catalog}.{ctx.run_schema}.{orig_output_table}"
    orig_job_id = sandbox.create_job(ctx, role="orig", output_table=orig_output_table)
    rn = run_job_now(orig_job_id)
    # Scoped fixtures don't write to a single "primary" output table — pass None
    # so capture_from_run skips that hash and uses extra_output_tables instead.
    primary_for_capture = None if fixture.scope is not None else orig_output_fqn
    captured = baselines.capture_from_run(
        fixture, run_id=int(rn["run_id"]), output_table_fqn=primary_for_capture,
        extra_output_tables=scope_outputs,
    )
    baselines.put(captured)
    console.print(
        f"[green]✓ baseline captured[/] duration={captured.duration_ms}ms rows={captured.output_row_count}"
    )
    return captured


def _trigger_and_wait(job_id: int, timeout_s: int = 3600) -> tuple[int, bool, int]:
    """Trigger the job, wait for terminal state. Returns (run_id, succeeded, duration_ms)."""
    rn = run_job_now(job_id)
    run_id = int(rn["run_id"])
    r = wait_for_job_run(run_id, timeout_seconds=timeout_s, poll_interval_seconds=10)
    succeeded = (not r.get("timed_out")) and r.get("result_state") == "SUCCESS"
    return run_id, succeeded, int(r.get("execution_duration_ms") or 0)


def _write_trace(results_dir: Path, agent_result) -> None:
    """Persist the agent's tool-call trace as JSONL (one entry per line)."""
    trace_path = results_dir / "trace.jsonl"
    with trace_path.open("w") as f:
        for entry in agent_result.trace:
            f.write(json.dumps(asdict(entry), default=str) + "\n")


def _print_verdict(console: Console, summary: dict[str, Any]) -> None:
    scores = summary["scores"]
    console.print()

    def verdict(tier_dict: dict[str, Any]) -> str:
        if tier_dict.get("skipped"):
            return f"[yellow]SKIP[/] ({tier_dict['skipped']})"
        return "[bold green]PASS[/]" if tier_dict.get("passed") else "[bold red]FAIL[/]"

    t1 = scores.get("tier1", {})
    console.print(f"  Tier 1 (correctness)          {verdict(t1)}")
    if not t1.get("skipped"):
        console.print(f"    output_equivalence          {t1.get('output_equivalence')}")
        console.print(f"    sandbox_compliance          {t1.get('sandbox_compliance')}")
        console.print(f"    job_completion              {t1.get('job_completion')}")
        if "scope_adherence" in t1:
            console.print(f"    scope_adherence             {t1.get('scope_adherence')}")

    t2 = scores.get("tier2", {})
    console.print(f"  Tier 2 (diagnosis)            {verdict(t2)}")
    if not t2.get("skipped"):
        console.print(
            f"    investigation coverage      {t2.get('coverage_score', 0):.0%} "
            f"({t2['details']['coverage']['hit']}/{t2['details']['coverage']['required']})"
        )
        console.print(
            f"    diagnosis keywords          {t2.get('keyword_score', 0):.0%} "
            f"({len(t2['details']['keywords']['hit'])}/{len(t2['details']['keywords']['expected'])})"
        )

    t3 = scores.get("tier3", {})
    console.print(f"  Tier 3 (outcome)              {verdict(t3)}")
    if not t3.get("skipped"):
        delta = t3.get("runtime_improvement_pct", 0)
        threshold = t3.get("runtime_threshold_pct", 0)
        color = "green" if delta >= threshold else "red"
        console.print(
            f"    runtime improvement         [{color}]{delta:+.1f}%[/] "
            f"(threshold ≥{threshold}%)"
        )
        details = t3.get("details", {})
        b_ms = details.get("baseline_duration_ms", 0)
        o_ms = details.get("optimized_duration_ms", 0)
        console.print(f"    [dim]baseline {b_ms}ms → optimized {o_ms}ms[/]")
