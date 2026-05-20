"""LLM-driven detection of nondeterministic OUTPUT columns.

The equivalence gate (`diff_tables`) tells us *which* cells differ. It cannot
tell us whether a difference is *allowed*: a column produced by an untied
`ROW_NUMBER() ORDER BY x` (pick among ties is shuffle-dependent) differs
run-to-run even on identical inputs — the original prod query is not
self-consistent on it either. That ambiguity is not in the data; it is in the
query. So we read the *query* and classify each FINAL OUTPUT column.

This module ONLY produces the hypothesis + rationale. It deliberately does
NOT authorize excluding a probe-required column — that requires the
determinism probe (running the original query on identical pinned inputs).
The split is encoded in `authorization`:

  - self_authorizing : non-pure by language semantics (rand(), uuid(),
                        current_timestamp()-derived run-stamps, unseeded
                        sample). Rationale alone licenses exclusion.
  - probe_required   : data-derived (untied argmax, order-sensitive agg,
                        opaque UDF). Structurally indistinguishable from a
                        real bug → must be empirically proven before exclude.
  - already_handled  : float reduction reorder — absorbed by diff_tables'
                        rtol/atol tolerance gate.
"""

from __future__ import annotations

import json
import re
from typing import Any

VALID_CLASSES = {
    "A_by_definition",
    "B_untied_pick",
    "C_order_sensitive_agg",
    "D_unseeded_sampling",
    "E_float_reorder",
    "F_udf",
}
_CLASS_AUTHORIZATION = {
    "A_by_definition": "self_authorizing",
    "D_unseeded_sampling": "self_authorizing",
    "E_float_reorder": "already_handled",
    "B_untied_pick": "probe_required",
    "C_order_sensitive_agg": "probe_required",
    "F_udf": "probe_required",
}

_SYSTEM = """\
You are a Spark SQL determinism auditor. Given an ETL notebook's SQL and the
FINAL OUTPUT columns it writes, classify EVERY output column as DETERMINISTIC
or NONDETERMINISTIC, with a precise rationale that traces the column back
through the CTE/projection chain to the construct responsible.

GOVERNING DEFINITION
  A column is NONDETERMINISTIC iff, holding the query's inputs FIXED, its value
  is not a pure function of those inputs — i.e. re-running the identical query
  on the identical input snapshot can produce a different value for that cell.
  (Source tables are pinned, so clock/env variance is out of scope EXCEPT where
  a value is literally stamped from current_timestamp()/now() etc.)

NONDETERMINISM CATALOG (assign the single best class)
  A_by_definition       Non-pure by language semantics: rand/randn/random,
                         uuid, monotonically_increasing_id, spark_partition_id,
                         input_file_name, current_timestamp/now/localtimestamp,
                         current_date/current_user, shuffle(array); AND
                         run-stamp columns set to one of these (last_refresh_time,
                         loaded_at, batch_id, etl_timestamp, processed_at, ...).
  B_untied_pick         Data-derived arbitrary selection: ROW_NUMBER/RANK/
                         DENSE_RANK OVER(... ORDER BY <not a total order>) then
                         keep rn=1; FIRST/LAST/ANY_VALUE without total order;
                         MAX_BY/MIN_BY when the ranking value ties; GROUP BY
                         selecting a non-grouped non-aggregated column; a join
                         that fans out then dedups to one arbitrary match.
  C_order_sensitive_agg collect_list/collect_set/array_agg element order,
                         string-agg over unordered group, first/last aggregate,
                         approx_count_distinct, percentile_approx, mode().
  D_unseeded_sampling    LIMIT without a total ORDER BY, TABLESAMPLE,
                         sample()/randomSplit() without a fixed seed.
  E_float_reorder        SUM/AVG/STDDEV/VAR/CORR over DOUBLE/FLOAT (non-
                         associative FP). DECIMAL is EXACT — never class E.
  F_udf                  A Python/Scala/pandas UDF that could be non-pure
                         (calls random/time/env/external service).

LINEAGE & PROPAGATION (critical — do not pattern-match blindly)
  Trace each FINAL output column to its source expression. A nondeterministic
  construct only taints columns it actually flows into. For a `ROW_NUMBER()
  OVER (PARTITION BY ... ORDER BY ...)` with potential ties, exactly THREE
  kinds of columns flow out:

    1. PARTITION KEY columns of the ROW_NUMBER → DETERMINISTIC.
       Reason: every tied row shares the SAME partition-key values by
       definition (they were grouped together). Projecting the partition
       key yields the same value regardless of which tied row is picked.

    2. ORDER BY KEY columns (the ranking expression itself) → DETERMINISTIC
       on the picked row. Reason: "tied on x" means all tied rows have the
       same x value, so projecting x is invariant under the arbitrary pick.

    3. ALL OTHER ATTRIBUTES carried from the picked row → NONDETERMINISTIC
       (this is the only class B case). These are the carried-along columns
       like a label, description, foreign-key id, etc. that differ across
       the tied rows.

  Worked examples:
    - max(x)/min(x) over tied rows            -> DETERMINISTIC
    - SUM over ALL tied rows                  -> DETERMINISTIC (order-free)
    - in `ROW_NUMBER() OVER (PARTITION BY uid, show_id ORDER BY ts)`:
        uid, show_id (partition keys)          -> DETERMINISTIC
        ts (ORDER BY key)                      -> DETERMINISTIC on ties
        plan_id (carried attribute)            -> NONDETERMINISTIC (class B)

  DO NOT classify partition-key columns or the ORDER BY key column itself
  as B_untied_pick. They are NOT carried-along attributes; they are the
  ROW_NUMBER's own structural inputs and are invariant under tied picks.

  REQUIRED for every B_untied_pick column:
    - Populate `deterministic_sibling` with the name of the ORDER BY KEY
      column (e.g. "create_time" / "ts" / "playtime"). This is what an
      empirical tie-break check joins on to corroborate your hypothesis.
    - Only set `deterministic_sibling` to null when the ORDER BY clause is
      a complex expression that doesn't correspond to a single output column
      (e.g. ORDER BY CASE WHEN x IS NULL THEN 1 ELSE 0 END, y — the sibling
      is `y`, not the CASE expression). In that case, identify the dominant
      ranking column.
    - A column may NEVER be its own deterministic_sibling.

OUTPUT — STRICT JSON ONLY, no prose, no markdown fences. Schema:
{
  "columns": {
    "<output_col>": {
      "verdict": "DETERMINISTIC" | "NONDETERMINISTIC",
      "class": one of the catalog keys or null (null iff DETERMINISTIC),
      "rationale": "<=60 words tracing the column to the responsible construct",
      "deterministic_sibling": "<col name>" | null
    },
    ... EVERY output column must appear ...
  },
  "notes": "<global caveats, e.g. column not found in SQL, dynamic select *>"
}
Default to DETERMINISTIC unless you can name the specific construct that makes
it not a pure function of fixed inputs. Be conservative: a wrong
"DETERMINISTIC" is safe (gets caught by diff), a wrong "NONDETERMINISTIC" on a
probe_required column is NOT auto-excluded anyway — but still justify precisely.
"""


def _parse_json_block(text: str) -> dict[str, Any]:
    """Defensive: strip ``` fences, take the outermost {...}."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(t[i : j + 1])
        raise


def detect_nondeterministic_columns(
    notebook_sql: str,
    output_columns: list[str],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Classify every final output column as deterministic / nondeterministic.

    Pure analysis (one LLM call, no tools, no table reads). Returns the
    per-column verdict + rationale plus convenience roll-ups split by
    authorization class. Does NOT mutate anything or decide exclusions.
    """
    from ..agent import _client, _model

    mdl = model or _model()
    user = (
        "FINAL OUTPUT COLUMNS (classify each, by exact name):\n"
        + json.dumps(output_columns)
        + "\n\nNOTEBOOK SQL:\n-----\n"
        + notebook_sql
        + "\n-----\n"
    )
    resp = _client().chat.completions.create(
        model=mdl,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    parsed = _parse_json_block(raw)

    cols_in = parsed.get("columns") or {}
    columns: dict[str, Any] = {}
    for col in output_columns:
        entry = cols_in.get(col) or {}
        verdict = str(entry.get("verdict", "DETERMINISTIC")).upper()
        if verdict not in ("DETERMINISTIC", "NONDETERMINISTIC"):
            verdict = "DETERMINISTIC"
        cls = entry.get("class")
        if verdict == "DETERMINISTIC":
            cls = None
        elif cls not in VALID_CLASSES:
            # Nondeterministic but unclassifiable → safest is probe-required.
            cls = "F_udf"
        columns[col] = {
            "verdict": verdict,
            "class": cls,
            "authorization": _CLASS_AUTHORIZATION.get(cls) if cls else None,
            "rationale": str(entry.get("rationale", "")).strip(),
            "deterministic_sibling": entry.get("deterministic_sibling") or None,
        }

    nd = [c for c, e in columns.items() if e["verdict"] == "NONDETERMINISTIC"]
    return {
        "model": mdl,
        "columns": columns,
        "nondeterministic_columns": nd,
        "self_authorizing_columns": [
            c for c in nd if columns[c]["authorization"] == "self_authorizing"
        ],
        "already_handled_columns": [
            c for c in nd if columns[c]["authorization"] == "already_handled"
        ],
        "probe_required_columns": [
            c for c in nd if columns[c]["authorization"] == "probe_required"
        ],
        "notes": str(parsed.get("notes", "")).strip(),
        "unanalyzed_columns": [c for c in output_columns if c not in cols_in],
    }
