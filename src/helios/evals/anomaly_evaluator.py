"""Anomaly evaluator agent — independent advisory verdict on prod-vs-sandbox
equivalence.

Runs as a multi-turn Claude agent loop AFTER the deterministic diff_tables
pass, with a strict read-only tool surface (execute_sql_readonly only). Its
verdict is ADVISORY: it appears in proposal.html alongside the deterministic
verdict, but does NOT gate Tier-1/Tier-3 PASS criteria.

Built because the deterministic diff pipeline has accumulated edge cases
that no fixed rule set can fully cover (NULL-vs-0 misclassified as
FLOAT_REORDER_ONLY, stable-key cross-products inflating tie-break counts,
LLM column-classifier over-flagging dimensions, etc.). The evaluator agent
can investigate iteratively — run a count, see something suspicious, follow
up — like a human reviewer.

Failure mode: graceful skip. If the LLM call fails or the final message
isn't parseable JSON, log a warning and return None. The caller proceeds
with the deterministic verdict only.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from rich.console import Console


# Verdict tiers (order = severity).
_VERDICT_VALUES = ("SHIPPABLE", "SHIP_WITH_CAVEATS", "DO_NOT_SHIP")

# Anomaly categories the evaluator can assign. These mirror the
# real-world classes we've actually seen on prior runs.
_ANOMALY_CATEGORIES = {
    "BENIGN_FLOAT_REORDER",          # DOUBLE drift within rtol bounds
    "BENIGN_NULL_ZERO_EQUIVALENT",   # NULL vs 0.0 on a numeric column
    "KNOWN_NONDETERMINISTIC_TIE_BREAK",  # untied ROW_NUMBER carried attribute
    "KNOWN_RUN_STAMP",               # current_timestamp / last_refresh_time
    "REAL_ALGEBRA_DIVERGENCE",       # actual semantic difference
    "REAL_ALGEBRA_DIVERGENCE_WRITE_MODE",  # INSERT INTO → CREATE OR REPLACE / INSERT OVERWRITE
    "SCHEMA_DRIFT",                  # column add/drop/type change
    "UNKNOWN_NEEDS_INVESTIGATION",   # observed but unclassified
}


_EVALUATOR_SYSTEM_PROMPT = """\
You are the Helios anomaly evaluator. Your job is to independently review
whether a candidate optimization of a Databricks ETL query produces output
semantically equivalent to the prod baseline. You are advisory, not
authoritative — your verdict is shown to a human reviewer alongside the
deterministic diff_tables verdict, and the human decides.

You receive (in the next user message): the prod table reference (pinned
to a specific Delta version), the sandbox table reference, both notebook
SQL sources (original and the agent's optimized version), and the full
output of the deterministic diff_tables call. Treat that diff_tables
verdict as a HYPOTHESIS, not a fact — it has historically misclassified
edge cases (NULL-vs-0 as FLOAT_REORDER_ONLY; stable-key duplication
inflating tie-break counts).

# Tooling

You have ONE tool: `execute_sql_readonly`. It's a strict read-only wrapper
around Spark SQL — any DDL/DML verb is rejected pre-flight. Use it to
investigate: row counts, per-column SUM/MIN/MAX/null-count, NULL-vs-value
asymmetry, sampling specific rows, schema checks. You may issue up to ~15
queries; stay focused. Do NOT attempt mutation; the guard will reject and
waste an iteration.

When referencing the prod side in queries, use the FULLY QUALIFIED pinned
reference (e.g. `spice_catalog.prod.foo VERSION AS OF 91`). Do not query
the live/unpinned prod table — that's a different (likely newer) state
than what the sandbox was computed against.

# Known anomaly catalog (cite the category that matches what you find)

When you observe a divergence between prod and sandbox, classify it as
one of these. The classification drives the severity and the verdict.

1. **BENIGN_FLOAT_REORDER** — DOUBLE/FLOAT column drift where |a-b| is
   small relative to magnitude (< 1e-6 relative). IEEE-754 aggregation
   is non-associative; different parallelism gives bit-different sums.
   Per-row max_rel_diff < 1e-6 AND aggregate totals match within
   ~1e-13. Verdict: SHIPPABLE.
   Recognition: per-column diff query that reports MAX(|a-b|/max(|a|,|b|));
   if max_rel < 1e-6 and totals match, this is float reorder.

2. **BENIGN_NULL_ZERO_EQUIVALENT** — On a numeric column, prod has NULL
   while sandbox has 0.0 (or vice versa) on rows where the "no data" /
   "empty group" condition applies. Aggregate totals are bit-identical
   because both NULL and 0 contribute zero to SUM. Semantically
   equivalent for SUM/AVG/MAX/MIN/comparisons; NOT equivalent for
   COUNT(col) or `WHERE col IS NULL` filters. Verdict: SHIP_WITH_CAVEATS
   unless you can verify downstream consumers don't care about the
   distinction (you can't, so default to SHIP_WITH_CAVEATS).
   Recognition: SUM(CASE WHEN p.col IS NULL AND s.col IS NOT NULL THEN 1)
   and the converse, plus SUM(p.col) - SUM(s.col) should be 0 to within
   FP tolerance. Critically: this is the bug class that the deterministic
   diff_tables MISCLASSIFIES as FLOAT_REORDER_ONLY (because abs(NULL - x)
   evaluates to NULL, making max_abs_diff/max_rel_diff return NULL and
   the verdict logic treats NULL-or-0 as "no drift"). Always probe for
   this when the diff_tables verdict is FLOAT_REORDER_ONLY on a column
   with a non-zero rows_drifted count and max_abs_diff IS NULL.

3. **KNOWN_NONDETERMINISTIC_TIE_BREAK** — The original notebook contains
   `ROW_NUMBER() OVER (... ORDER BY x DESC)` (or RANK/DENSE_RANK/FIRST/
   LAST/MAX_BY/MIN_BY/ANY_VALUE) with no unique tie-breaker. Multiple
   rows tie on the ranking expression x; Spark picks one arbitrarily;
   the picked row's CARRIED attributes (non-key, non-ordering columns)
   vary across runs of the same query. The carried column differs on
   tied rows but the ranking value itself does not. Verdict: SHIPPABLE
   (prod itself produces these on re-run; no algebra change).
   Recognition: for the suspect column, join prod vs sandbox on a stable
   key (dimensions + the ranking column itself, EXCLUDING the carried
   suspect column), count rows where suspect_col differs AND ranking_col
   matches → this is the tie-break signature. If ranking_col also
   differs on the same rows, it's NOT pure tie-break — likely real bug.

4. **KNOWN_RUN_STAMP** — Column written via current_timestamp() / now() /
   getdate() / uuid() / monotonically_increasing_id(). Differs on every
   row by construction; semantically not part of data identity.
   Common names: last_refresh_time, loaded_at, batch_id, etl_timestamp,
   processed_at, ingested_at, created_at (when set to current_timestamp).
   Verdict: SHIPPABLE (these are auto-excluded by the deterministic
   pipeline already; the evaluator should just confirm).

5. **REAL_ALGEBRA_DIVERGENCE** — A semantic difference that the
   optimization introduced. Examples: a missing GROUP BY column, a
   wrong filter predicate, a JOIN with different cardinality, an
   aggregate computed at the wrong grain. On a column with a real
   algebra difference: SUM totals DIFFER beyond float-reorder tolerance,
   per-row max_rel_diff is large (>>1e-6), or the row count differs.
   Verdict: DO_NOT_SHIP.

6. **REAL_ALGEBRA_DIVERGENCE_WRITE_MODE** — The agent changed the
   write verb (e.g. original is `INSERT INTO`, optimized is
   `INSERT OVERWRITE` or `CREATE OR REPLACE TABLE AS SELECT`). For
   partitioned tables on Databricks with
   `spark.sql.sources.partitionOverwriteMode=DYNAMIC`, INSERT OVERWRITE
   may be effectively safe — it overwrites only the partitions in the
   SELECT. With STATIC, it wipes the entire table. Without verifying
   the conf, treat as DO_NOT_SHIP. The right rewrite preserves the
   original's write verb.
   Recognition: scan both notebook sources for write verbs against the
   target FQN. Compare. If they differ, this category fires regardless
   of what the data looks like.

7. **SCHEMA_DRIFT** — Columns added, removed, or type-changed between
   prod and sandbox. Verdict: DO_NOT_SHIP (downstream consumers
   probably depend on the exact schema).
   Recognition: DESCRIBE both tables, compare column lists + types.

8. **UNKNOWN_NEEDS_INVESTIGATION** — You observed an anomaly that
   doesn't cleanly fit the above. Describe what you measured and flag
   for human review. Verdict implications: SHIP_WITH_CAVEATS at minimum,
   DO_NOT_SHIP if the magnitude is large.

# Workflow

1. **Anchor**: row counts on both sides (prod pinned version, sandbox).
2. **Schema check**: DESCRIBE both tables, compare column lists/types.
3. **Read the existing diff_tables result** (in the user message). It
   tells you which columns the deterministic pipeline thinks differ
   and how. Use it as your starting point — but verify, don't trust.
4. **For each flagged column**, probe with targeted queries:
   - NULL asymmetry counts (always — this is the bug class diff_tables
     misses).
   - SUM totals comparison (for floats).
   - MAX(abs(a-b)) and MAX(abs(a-b)/max(|a|,|b|)) (for floats).
   - For probe-required nondeterminism: tie-break corroboration query.
5. **Scan the SQL sources** to confirm hypotheses (e.g., does the
   optimized notebook contain `COALESCE(..., 0)` that explains a
   NULL-zero asymmetry? Did the write verb change?).
6. **Classify** each anomaly into the catalog above.
7. **Verdict**:
   - SHIPPABLE if every anomaly is BENIGN_* or KNOWN_*.
   - SHIP_WITH_CAVEATS if at least one BENIGN_NULL_ZERO_EQUIVALENT or
     UNKNOWN exists (semantically equivalent for typical use but with
     specific edge cases that should be confirmed with the data
     owner).
   - DO_NOT_SHIP if any REAL_ALGEBRA_DIVERGENCE, WRITE_MODE change,
     or SCHEMA_DRIFT.

# Final output

Your LAST message (no tool_calls) MUST be a single JSON object — no
prose, no markdown fences. Schema:

{
  "verdict": "SHIPPABLE" | "SHIP_WITH_CAVEATS" | "DO_NOT_SHIP",
  "narrative": "<= 300 words plain English summary",
  "row_counts": {"prod": <int>, "sandbox": <int>, "match": <bool>},
  "anomalies": [
    {
      "column": "<column name>",
      "category": "<one of the catalog keys>",
      "rows_affected": <int>,
      "evidence": "<concrete numbers from your queries>",
      "explanation_in_sql": "<which construct in the SQL explains it, or '' if unclear>",
      "severity": "LOW" | "MEDIUM" | "HIGH",
      "downstream_risk": "<who would notice and how>"
    },
    ...
  ],
  "investigation_log": [
    {"query": "<truncated SQL>", "finding": "<one-line takeaway>"},
    ...
  ]
}

If you observe NO anomalies (everything matches), still return a JSON
object with anomalies=[] and verdict="SHIPPABLE". Be concise but
specific in evidence — cite actual numbers, not vague descriptors.
"""


def _build_user_message(
    *,
    prod_fqn: str,
    sandbox_fqn: str,
    original_sql: str,
    optimized_sql: str,
    diff_result: dict[str, Any],
    natural_key: list[str] | None = None,
    write_target_mode: str = "full_rewrite",
) -> str:
    """Assemble the initial user message — all the context the agent needs
    to start investigating."""
    # Trim the diff_result to the fields the agent actually needs (the full
    # one can be 50KB+ with drift_concentration entries).
    compact_diff = {
        "verdict": diff_result.get("verdict"),
        "buckets": diff_result.get("buckets"),
        "worst_float_rel_diff": diff_result.get("worst_float_rel_diff"),
        "natural_key": diff_result.get("natural_key") or natural_key or [],
        "metric_columns": diff_result.get("metric_columns") or [],
        "ignored_columns": diff_result.get("ignored_columns") or [],
        "auto_ignored_columns": diff_result.get("auto_ignored_columns") or [],
        "tolerance": diff_result.get("tolerance"),
        "metric_summary": diff_result.get("metric_summary") or {},
        "drift_profile": diff_result.get("drift_profile") or [],
        "tie_break_corroboration": diff_result.get("tie_break_corroboration") or [],
        "nondeterminism_probe_required": diff_result.get("nondeterminism_probe_required") or [],
        "self_authorizing_rejected": diff_result.get("self_authorizing_rejected") or [],
        "mode": diff_result.get("mode"),
        "increment": diff_result.get("increment"),
    }
    # Cap notebook sources at ~8KB each so the prompt stays bounded.
    def _trim(s: str, n: int = 8000) -> str:
        if len(s) <= n:
            return s
        return s[:n] + f"\n...[truncated, full length {len(s)} chars]"
    return (
        "INVESTIGATE these two Databricks tables for equivalence anomalies.\n\n"
        f"prod side  (pinned):  {prod_fqn}\n"
        f"sandbox side       :  {sandbox_fqn}\n"
        f"write target mode  :  {write_target_mode}\n\n"
        "## Deterministic diff_tables result (your starting hypothesis)\n\n"
        "```json\n"
        + json.dumps(compact_diff, indent=2, default=str)
        + "\n```\n\n"
        "## Original (canonical prod) notebook SQL\n\n"
        "```sql\n" + _trim(original_sql) + "\n```\n\n"
        "## Optimized (agent's rewrite) notebook SQL\n\n"
        "```sql\n" + _trim(optimized_sql) + "\n```\n\n"
        "Use `execute_sql_readonly` to investigate. Your final message MUST "
        "be the JSON verdict object — no prose around it, no markdown "
        "fences. Begin investigating now."
    )


def _parse_evaluator_json(text: str) -> dict[str, Any] | None:
    """Defensive JSON extraction from the final assistant message.
    Strips code fences and finds the outermost {...}. Returns None on
    unparseable input (graceful skip)."""
    if not text:
        return None
    t = text.strip()
    # Strip ```json … ``` fence if present
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                return None
        return None


def _normalize_verdict(parsed: dict[str, Any]) -> dict[str, Any]:
    """Coerce loose model output into the locked schema; defensive against
    minor format drift. Returns the normalized dict."""
    out = dict(parsed)
    v = str(out.get("verdict", "")).upper()
    if v not in _VERDICT_VALUES:
        # Default to needs-review when verdict is malformed.
        out["verdict"] = "SHIP_WITH_CAVEATS"
        out.setdefault("narrative", "(evaluator returned unrecognized verdict; defaulting to SHIP_WITH_CAVEATS for human review)")
    else:
        out["verdict"] = v
    out.setdefault("narrative", "")
    out.setdefault("row_counts", {})
    raw_anomalies = out.get("anomalies") or []
    fixed: list[dict[str, Any]] = []
    for a in raw_anomalies:
        if not isinstance(a, dict):
            continue
        cat = str(a.get("category", "")).upper()
        if cat not in _ANOMALY_CATEGORIES:
            cat = "UNKNOWN_NEEDS_INVESTIGATION"
        sev = str(a.get("severity", "")).upper()
        if sev not in ("LOW", "MEDIUM", "HIGH"):
            sev = "MEDIUM"
        try:
            rows_aff = int(a.get("rows_affected") or 0)
        except (TypeError, ValueError):
            rows_aff = 0
        fixed.append({
            "column": str(a.get("column", "")),
            "category": cat,
            "rows_affected": rows_aff,
            "evidence": str(a.get("evidence", "")),
            "explanation_in_sql": str(a.get("explanation_in_sql", "")),
            "severity": sev,
            "downstream_risk": str(a.get("downstream_risk", "")),
        })
    out["anomalies"] = fixed
    out.setdefault("investigation_log", [])
    return out


def run_anomaly_evaluator(
    *,
    prod_fqn: str,
    sandbox_fqn: str,
    original_sql: str,
    optimized_sql: str,
    diff_result: dict[str, Any],
    proposal_dir: Path,
    console: Console,
    write_target_mode: str = "full_rewrite",
    max_iters: int = 25,
    persist_path: Path | None = None,
) -> dict[str, Any] | None:
    """Run the evaluator agent for ONE (prod, sandbox) pair. Returns the
    structured verdict on success, or None on failure (graceful skip —
    caller proceeds without it).

    `persist_path`: if given, the verdict JSON is persisted to that exact
    path. If None (default), the caller is responsible for persistence
    (used by multi-target flows that write a combined file).
    """
    try:
        from ..agent import run_turn, _model  # local import to keep cold-import cheap

        user_msg = _build_user_message(
            prod_fqn=prod_fqn, sandbox_fqn=sandbox_fqn,
            original_sql=original_sql, optimized_sql=optimized_sql,
            diff_result=diff_result,
            natural_key=diff_result.get("natural_key"),
            write_target_mode=write_target_mode,
        )
        messages = [
            {"role": "system", "content": _EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        console.print(
            f"[cyan]→ anomaly evaluator agent starting "
            f"(model {_model()}, read-only, max_iters={max_iters})[/]"
        )
        t0 = time.time()
        final_messages = run_turn(
            messages,
            yolo=True,
            max_iters=max_iters,
            console=console,
            tools_filter=lambda n: n == "execute_sql_readonly",
            system_override=_EVALUATOR_SYSTEM_PROMPT,
        )
        elapsed = time.time() - t0

        # Pull the last assistant message with content (the verdict JSON).
        last_assistant = None
        for m in reversed(final_messages):
            if m.get("role") == "assistant" and (m.get("content") or "").strip():
                last_assistant = m
                break
        if last_assistant is None:
            console.print(
                "[yellow]evaluator: no final assistant message — "
                "graceful skip[/]"
            )
            return None
        parsed = _parse_evaluator_json(last_assistant.get("content", ""))
        if parsed is None:
            console.print(
                "[yellow]evaluator: final message was not valid JSON — "
                "graceful skip[/]"
            )
            return None
        result = _normalize_verdict(parsed)
        # Add provenance.
        result["model"] = _model()
        result["elapsed_seconds"] = round(elapsed, 1)
        result["max_iters"] = max_iters
        result["captured_at"] = int(time.time())

        if persist_path is not None:
            try:
                persist_path.write_text(
                    json.dumps(result, indent=2, default=str)
                )
            except Exception as e:
                console.print(
                    f"[yellow]evaluator: could not persist {persist_path.name} ({e})[/]"
                )

        # Summary line for the console.
        verdict = result["verdict"]
        n_anom = len(result.get("anomalies") or [])
        color = {"SHIPPABLE": "green", "SHIP_WITH_CAVEATS": "yellow", "DO_NOT_SHIP": "red"}.get(verdict, "yellow")
        console.print(
            f"  [{color}]evaluator verdict: {verdict}[/]  "
            f"({n_anom} anomalies, {elapsed:.0f}s)"
        )
        return result
    except Exception as e:
        console.print(
            f"[yellow]evaluator agent failed: {type(e).__name__}: {e}; "
            "proceeding with deterministic verdict only[/]"
        )
        return None
