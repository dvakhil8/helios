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
  construct only taints columns it actually flows into:
    - max(x)/min(x) over tied rows           -> DETERMINISTIC (value is stable)
    - a non-key attribute carried from the
      row picked by an untied ROW_NUMBER     -> NONDETERMINISTIC (class B)
    - SUM over ALL tied rows                  -> DETERMINISTIC (order-free)
    - the ORDER BY key itself                 -> DETERMINISTIC
  For a class B argmax, also identify the sibling DETERMINISTIC column that is
  the ranking value (e.g. the max the pick was based on) in
  `deterministic_sibling` — it is the corroborating evidence.

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
