"""Tier 3 — outcome quality. Did the agent's fix actually make the job faster?

For now this is runtime-only. DBU and bytes-scanned are intentionally deferred:
  - DBU lives in system.billing.usage which has hours-to-days delay; not usable
    inline in an eval run.
  - bytes scanned lives in system.query.history which only covers SQL warehouses,
    not job clusters.

Both can be added later as separate sub-scores; runtime is the dominant signal
and is available immediately from `wait_for_job_run`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..baselines import Baseline
from ..fixtures import Fixture


@dataclass
class Tier3Score:
    runtime_improvement_pct: float
    runtime_threshold_pct: float
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": 3,
            "passed": self.passed,
            "runtime_improvement_pct": round(self.runtime_improvement_pct, 2),
            "runtime_threshold_pct": self.runtime_threshold_pct,
            "details": self.details,
        }


def score(
    *,
    fixture: Fixture,
    baseline: Baseline,
    optimized_duration_ms: int,
    optimized_run_succeeded: bool,
    optimized_task_durations_ms: dict[str, int] | None = None,
) -> Tier3Score:
    """Score Tier 3.

    For scoped fixtures, pass `optimized_task_durations_ms` (from `get_job_run`
    on the optimized run). The comparator then uses max(in-scope tasks) on both
    sides instead of wall-clock — this isolates the win to the bottleneck task
    and ignores noise from unchanged upstream/downstream tasks.
    """
    threshold = fixture.fix.runtime_pct_min
    details: dict[str, Any] = {
        "baseline_duration_ms": baseline.duration_ms,
        "optimized_duration_ms": optimized_duration_ms,
    }

    if not optimized_run_succeeded:
        details["status"] = "skipped: optimized run did not succeed"
        return Tier3Score(
            runtime_improvement_pct=0.0,
            runtime_threshold_pct=threshold,
            passed=False,
            details=details,
        )

    # Pick the comparator: scoped (max of in-scope task durations) or wall-clock.
    baseline_ms: int
    optimized_ms: int
    if fixture.scope is not None and baseline.task_durations_ms and optimized_task_durations_ms:
        in_scope = fixture.scope.in_scope_task_keys
        b_durs = [baseline.task_durations_ms[k] for k in in_scope if k in baseline.task_durations_ms]
        o_durs = [optimized_task_durations_ms[k] for k in in_scope if k in optimized_task_durations_ms]
        if not b_durs or not o_durs:
            details["status"] = (
                f"scoped: missing per-task durations (baseline keys: {list(baseline.task_durations_ms)}, "
                f"optimized keys: {list(optimized_task_durations_ms)})"
            )
            return Tier3Score(
                runtime_improvement_pct=0.0,
                runtime_threshold_pct=threshold,
                passed=False,
                details=details,
            )
        baseline_ms = max(b_durs)
        optimized_ms = max(o_durs)
        details["scoped"] = True
        details["in_scope_task_keys"] = in_scope
        details["baseline_in_scope_max_ms"] = baseline_ms
        details["optimized_in_scope_max_ms"] = optimized_ms
        details["baseline_per_task_ms"] = baseline.task_durations_ms
        details["optimized_per_task_ms"] = optimized_task_durations_ms
    else:
        baseline_ms = baseline.duration_ms
        optimized_ms = optimized_duration_ms

    if baseline_ms <= 0:
        details["status"] = "skipped: baseline duration is 0"
        return Tier3Score(
            runtime_improvement_pct=0.0,
            runtime_threshold_pct=threshold,
            passed=False,
            details=details,
        )

    improvement_pct = (baseline_ms - optimized_ms) / baseline_ms * 100
    passed = improvement_pct >= threshold
    if improvement_pct < 0:
        details["status"] = f"REGRESSION: optimized is {-improvement_pct:.1f}% slower than baseline"
    elif passed:
        details["status"] = f"meets threshold (≥{threshold}%)"
    else:
        details["status"] = (
            f"below threshold: {improvement_pct:.1f}% improvement, "
            f"need ≥{threshold}%"
        )

    return Tier3Score(
        runtime_improvement_pct=improvement_pct,
        runtime_threshold_pct=threshold,
        passed=passed,
        details=details,
    )
