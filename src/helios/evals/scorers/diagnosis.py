"""Tier 2 — diagnosis quality. Cheap, trace-based, runs every PR.

Two sub-scores:
  coverage_score - did the agent call the investigation tools the fixture
                   declared as required? Matches by tool name, and for
                   execute_sql also checks `sql_contains` substring (case-
                   insensitive) so we can require specific probes like
                   "DESCRIBE DETAIL" or a GROUP BY skew check.
  keyword_score  - does the agent's final summary mention the diagnosis
                   keywords the fixture declared (e.g. 'skew', 'null',
                   'user_id')? Case-insensitive substring match.

Both are simple substring/name matches. A more sophisticated LLM judge can be
added later as a separate sub-score; for now this gives a stable, reproducible
signal that doesn't drift between runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..fixtures import Fixture
from ..runner import TraceEntry


_PASS_THRESHOLD = 0.66  # both sub-scores must clear this for overall pass


@dataclass
class Tier2Score:
    coverage_score: float
    keyword_score: float
    overall_score: float
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": 2,
            "passed": self.passed,
            "coverage_score": round(self.coverage_score, 2),
            "keyword_score": round(self.keyword_score, 2),
            "overall_score": round(self.overall_score, 2),
            "details": self.details,
        }


def score(
    *,
    fixture: Fixture,
    trace: list[TraceEntry],
    agent_final_text: str | None,
) -> Tier2Score:
    required = fixture.investigation.required_tools
    expected_keywords = fixture.investigation.expected_diagnosis_keywords

    # ---- investigation coverage -------------------------------------------
    coverage_hits: list[dict[str, Any]] = []
    coverage_misses: list[dict[str, Any]] = []
    for req in required:
        hit = _trace_has_match(trace, req)
        (coverage_hits if hit else coverage_misses).append(req)
    coverage_score = len(coverage_hits) / max(len(required), 1)

    # ---- keyword coverage --------------------------------------------------
    text_lower = (agent_final_text or "").lower()
    keyword_hits = [kw for kw in expected_keywords if kw in text_lower]
    keyword_misses = [kw for kw in expected_keywords if kw not in text_lower]
    keyword_score = len(keyword_hits) / max(len(expected_keywords), 1)

    overall = (coverage_score + keyword_score) / 2
    passed = coverage_score >= _PASS_THRESHOLD and keyword_score >= _PASS_THRESHOLD

    return Tier2Score(
        coverage_score=coverage_score,
        keyword_score=keyword_score,
        overall_score=overall,
        passed=passed,
        details={
            "coverage": {
                "required": len(required),
                "hit": len(coverage_hits),
                "misses": coverage_misses,
            },
            "keywords": {
                "expected": expected_keywords,
                "hit": keyword_hits,
                "misses": keyword_misses,
            },
        },
    )


def _trace_has_match(trace: list[TraceEntry], req: dict[str, str]) -> bool:
    """Check whether any trace entry satisfies the requirement.

    `req` is a dict like {"tool": "execute_sql", "sql_contains": "DESCRIBE DETAIL"}.
    For execute_sql, sql_contains is matched case-insensitively against the SQL.
    """
    tool_name = req.get("tool")
    sql_contains = (req.get("sql_contains") or "").lower()
    for entry in trace:
        if entry.tool != tool_name:
            continue
        if sql_contains:
            sql = str(entry.args.get("sql") or "").lower()
            if sql_contains not in sql:
                continue
        return True
    return False
