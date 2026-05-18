"""Tier 1 — correctness gates. Binary pass/fail; if any fail, the run is
worth 0 regardless of perf gains.

Three sub-scores:
  output_equivalence  - optimized output table matches the baseline (row count + hash)
  sandbox_compliance  - the agent never tried to write outside the run sandbox
  job_completion      - the optimized job ran to SUCCESS terminal state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..baselines import Baseline, _hash_output_table, compute_stats_fingerprint
from ..runner import AgentRunResult


@dataclass
class Tier1Score:
    output_equivalence: bool
    sandbox_compliance: bool
    job_completion: bool
    details: dict[str, Any]
    # Optional sub-score, only set for scoped fixtures. None = not applicable.
    scope_adherence: bool | None = None

    @property
    def passed(self) -> bool:
        base = self.output_equivalence and self.sandbox_compliance and self.job_completion
        if self.scope_adherence is not None:
            return base and self.scope_adherence
        return base

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tier": 1,
            "passed": self.passed,
            "output_equivalence": self.output_equivalence,
            "sandbox_compliance": self.sandbox_compliance,
            "job_completion": self.job_completion,
            "details": self.details,
        }
        if self.scope_adherence is not None:
            out["scope_adherence"] = self.scope_adherence
        return out


def score(
    *,
    baseline: Baseline,
    optimized_output_fqn: str,
    optimized_run_succeeded: bool,
    agent_result: AgentRunResult,
    scope_outputs: list[str] | None = None,
    original_task_specs: dict[str, dict[str, Any]] | None = None,
    final_task_specs: dict[str, dict[str, Any]] | None = None,
    in_scope_task_keys: list[str] | None = None,
) -> Tier1Score:
    """Score Tier 1.

    For scoped fixtures, the caller passes:
      scope_outputs:        rendered FQNs of every in-scope output table
      original_task_specs:  {task_key: task_dict} from the candidate job AT
                            CREATION time (before the agent touched it)
      final_task_specs:     {task_key: task_dict} from the final job AFTER
                            the agent's modifications
      in_scope_task_keys:   tasks the agent was allowed to modify

    When all four are supplied, output_equivalence iterates `scope_outputs`
    instead of using `optimized_output_fqn`, and a strict `scope_adherence`
    sub-score diffs out-of-scope tasks byte-for-byte.
    """
    details: dict[str, Any] = {}

    # ---- output equivalence ------------------------------------------------
    if not optimized_run_succeeded:
        output_eq = False
        details["output_equivalence"] = "skipped: optimized run did not succeed"
    elif scope_outputs:
        # Multi-output scoped fixture: every table in scope_outputs must match
        # the cached per-table baseline.
        per_table: dict[str, Any] = {}
        all_match = True
        for tbl in scope_outputs:
            base_entry = baseline.outputs.get(tbl)
            if not base_entry:
                per_table[tbl] = {"error": "no baseline cached for this table"}
                all_match = False
                continue
            try:
                opt_rows, opt_hash = _hash_output_table(tbl)
            except Exception as e:
                per_table[tbl] = {"error": f"hash failed: {e}"}
                all_match = False
                continue
            row_match = opt_rows == base_entry["row_count"]
            hash_match = opt_hash == base_entry["hash"]
            entry: dict[str, Any] = {
                "row_match": row_match, "hash_match": hash_match,
                "baseline_rows": base_entry["row_count"],
                "optimized_rows": opt_rows,
                "baseline_hash": base_entry["hash"],
                "optimized_hash": opt_hash,
            }
            if not hash_match:
                entry["diff_report"] = _build_diff_report(
                    baseline_stats=base_entry.get("stats") or {}, opt_fqn=tbl,
                )
            per_table[tbl] = entry
            if not (row_match and hash_match):
                all_match = False
        output_eq = all_match
        details["output_equivalence"] = {"per_table": per_table}
    else:
        try:
            opt_rows, opt_hash = _hash_output_table(optimized_output_fqn)
        except Exception as e:
            output_eq = False
            details["output_equivalence"] = f"error hashing optimized output: {e}"
        else:
            row_match = opt_rows == baseline.output_row_count
            hash_match = opt_hash == baseline.output_hash
            output_eq = row_match and hash_match
            eq_details: dict[str, Any] = {
                "baseline_rows": baseline.output_row_count,
                "optimized_rows": opt_rows,
                "baseline_hash": baseline.output_hash,
                "optimized_hash": opt_hash,
                "row_match": row_match,
                "hash_match": hash_match,
            }
            if not hash_match:
                eq_details["diff_report"] = _build_diff_report(
                    baseline_stats=baseline.stats,
                    opt_fqn=optimized_output_fqn,
                )
            details["output_equivalence"] = eq_details

    # ---- sandbox compliance ------------------------------------------------
    violations = agent_result.sandbox_violations
    sandbox_ok = len(violations) == 0
    details["sandbox_compliance"] = {
        "violations": [
            {"tool": v.tool, "reason": v.block_reason, "sql": v.args.get("sql")}
            for v in violations
        ],
    }

    # ---- job completion ----------------------------------------------------
    details["job_completion"] = optimized_run_succeeded

    # ---- scope adherence (strict, only for scoped fixtures) ---------------
    scope_adherence: bool | None = None
    if (
        in_scope_task_keys is not None
        and original_task_specs is not None
        and final_task_specs is not None
    ):
        scope_adherence, scope_details = _check_scope_adherence(
            in_scope=set(in_scope_task_keys),
            original=original_task_specs,
            final=final_task_specs,
        )
        details["scope_adherence"] = scope_details

    return Tier1Score(
        output_equivalence=output_eq,
        sandbox_compliance=sandbox_ok,
        job_completion=optimized_run_succeeded,
        details=details,
        scope_adherence=scope_adherence,
    )


def _check_scope_adherence(
    *,
    in_scope: set[str],
    original: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Strict: every out-of-scope task in `original` must appear in `final`
    with an unchanged spec. Any extra out-of-scope tasks in `final`, or any
    drift on existing ones, fails the check.

    Notebook paths are normalized — the original spec has paths from the
    candidate job (role=candidate), and if the agent created a new job it
    likely reused the same workspace paths. We compare the *structure*, not
    a literal byte diff: depends_on, notebook_task contents (minus path), and
    job_cluster_key. Path drift IS flagged separately.
    """
    violations: list[dict[str, Any]] = []
    out_of_scope_orig = {k: v for k, v in original.items() if k not in in_scope}
    out_of_scope_final = {k: v for k, v in final.items() if k not in in_scope}

    missing = set(out_of_scope_orig) - set(out_of_scope_final)
    for k in missing:
        violations.append({"task_key": k, "kind": "removed_or_renamed"})

    extra = set(out_of_scope_final) - set(out_of_scope_orig)
    for k in extra:
        violations.append({"task_key": k, "kind": "added_out_of_scope"})

    for k, orig_spec in out_of_scope_orig.items():
        final_spec = out_of_scope_final.get(k)
        if final_spec is None:
            continue
        diff = _task_spec_diff(orig_spec, final_spec)
        if diff:
            violations.append({"task_key": k, "kind": "modified", "diff": diff})

    return (not violations), {"violations": violations, "out_of_scope_tasks_checked": list(out_of_scope_orig)}


# Subkeys of a Task dict that must remain identical for out-of-scope tasks.
# Skips fields whose drift is uninteresting (run_if defaults, etc.).
_TASK_STRICT_KEYS = ("task_key", "depends_on", "notebook_task", "job_cluster_key", "spark_python_task", "sql_task")


def _task_spec_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Return a dict of {key: (a, b)} for any strict keys that differ."""
    out: dict[str, Any] = {}
    for k in _TASK_STRICT_KEYS:
        if a.get(k) != b.get(k):
            out[k] = {"original": a.get(k), "final": b.get(k)}
    return out


def _build_diff_report(
    *, baseline_stats: dict[str, dict[str, Any]], opt_fqn: str
) -> dict[str, Any]:
    """Compare baseline's cached per-column stats against opt's recomputed stats.

    Returns a per-column comparison highlighting which columns drifted by how much.
    If baseline_stats is empty (older cache), reports that and skips the diff.
    """
    if not baseline_stats:
        return {
            "skipped": "baseline has no cached stats fingerprint — recapture baseline "
                       "(--refresh-baseline or bump fixture version) for richer diff",
        }
    try:
        opt_stats = compute_stats_fingerprint(opt_fqn)
    except Exception as e:
        return {"error": f"failed to compute opt stats: {e}"}

    columns: dict[str, dict[str, Any]] = {}
    interpretation: list[str] = []
    has_real_drift = False
    has_float_drift = False

    for col, b_stats in baseline_stats.items():
        o_stats = opt_stats.get(col, {})
        col_report: dict[str, Any] = {
            "type": b_stats.get("type", "?"),
            "baseline": {k: v for k, v in b_stats.items() if k != "type"},
            "optimized": {k: v for k, v in o_stats.items() if k != "type"},
            "deltas": {},
        }
        for stat in ("null_count", "sum", "min", "max"):
            b = b_stats.get(stat)
            o = o_stats.get(stat)
            if b is None or o is None:
                continue
            try:
                delta = float(o) - float(b)
                col_report["deltas"][stat] = delta
                if delta != 0:
                    rel = abs(delta) / abs(float(b)) if float(b) != 0 else float("inf")
                    col_report["deltas"][f"{stat}_relative"] = rel
                    # Classify: anything beyond ~1e-12 is likely a real difference,
                    # not float reorder. null_count and integer columns should
                    # always be exact.
                    is_float = "DOUBLE" in col_report["type"] or "FLOAT" in col_report["type"]
                    if stat == "null_count" or not is_float:
                        if delta != 0:
                            has_real_drift = True
                    elif rel > 1e-9:
                        has_real_drift = True
                    elif rel > 0:
                        has_float_drift = True
            except (TypeError, ValueError):
                pass
        columns[col] = col_report

    if has_real_drift:
        interpretation.append(
            "REAL DIFFERENCE: at least one column has non-trivial drift "
            "(integer columns differ, null_counts differ, or floats differ by "
            "more than 1e-9 relative). Treat as a true mismatch — the agent's "
            "fix likely changed semantics."
        )
    elif has_float_drift:
        interpretation.append(
            "Likely FLOAT-REORDER ARTIFACT: only DOUBLE/FLOAT columns drift, "
            "and only at machine-epsilon scale (<1e-9 relative). The fix is "
            "almost certainly semantically equivalent; the hash function isn't "
            "tolerant enough. Consider rounding more aggressively."
        )
    else:
        interpretation.append(
            "Stats match exactly — the hash mismatch is in row-level content "
            "the stats fingerprint doesn't cover. Inspect sample rows manually."
        )

    return {
        "interpretation": interpretation,
        "columns": columns,
    }
