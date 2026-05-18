"""Fixture definitions — load and validate `evals/fixtures/<id>/fixture.yaml`.

Each fixture is a self-contained directory:

    evals/fixtures/<id>/
        fixture.yaml      - metadata, annotation, expected gains (this file)
        seed.sql          - DDL/DML to populate helios_eval_seed.<id>.*
        job.json          - Databricks job spec with {{placeholders}} for catalogs
        notebooks/        - notebook source files referenced by job.json

The yaml is the contract the eval scores against. Authors are responsible for
keeping `bug`, `investigation`, and `fix` accurate — they're the ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# yaml is only needed when actually loading a fixture from disk (eval mode).
# Propose mode doesn't load fixtures, so we import lazily inside load() to
# avoid an ImportError on `from .evals.propose import ...`.


# Repo-root-relative location of fixture definitions. Resolved from this file's
# location so it works regardless of cwd.
FIXTURES_ROOT: Path = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


@dataclass(frozen=True)
class BugAnnotation:
    """What the fixture says the agent should diagnose."""

    category: str
    root_cause: str


@dataclass(frozen=True)
class InvestigationAnnotation:
    """What investigation steps the agent should perform (Tier 2 scoring)."""

    required_tools: list[dict[str, str]] = field(default_factory=list)
    expected_diagnosis_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FixAnnotation:
    """What kind of fix is acceptable, and how much it should improve things."""

    acceptable_categories: list[str]
    runtime_pct_min: float = 0.0
    dbu_pct_min: float = 0.0
    bytes_scanned_pct_min: float = 0.0


@dataclass(frozen=True)
class Scope:
    """Defines the optimization boundary inside a multi-task fixture.

    When set, the agent may inspect any task but may only MODIFY tasks in
    `in_scope_task_keys`. Equivalence is checked only against `output_tables`.
    Tier 3 measures only in-scope task runtime.

    output_tables and upstream_tables may contain `{{placeholders}}` that the
    harness substitutes at run time.
    """

    in_scope_task_keys: list[str]
    output_tables: list[str]              # may contain {{run_catalog}} etc.
    upstream_tables: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Fixture:
    """A fixture loaded from disk. Immutable; use load() to construct."""

    id: str
    version: int
    description: str
    bug: BugAnnotation
    investigation: InvestigationAnnotation
    fix: FixAnnotation
    tool_call_budget: int
    root: Path  # the fixture directory
    scope: Scope | None = None  # None for single-task fixtures

    @property
    def seed_sql_path(self) -> Path:
        return self.root / "seed.sql"

    @property
    def job_template_path(self) -> Path:
        return self.root / "job.json"

    @property
    def notebooks_dir(self) -> Path:
        return self.root / "notebooks"

    @property
    def seed_schema(self) -> str:
        """Schema under helios_eval_seed that holds this fixture's source tables."""
        return self.id


def load(fixture_id: str, fixtures_root: Path | None = None) -> Fixture:
    """Load and validate fixture by id. Raises FileNotFoundError or ValueError on issues."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load fixtures. `pip install 'pyyaml>=6.0'` "
            "(or reinstall this package: `pip install -e .`)"
        ) from e

    root = (fixtures_root or FIXTURES_ROOT) / fixture_id
    yaml_path = root / "fixture.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"fixture.yaml not found: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: top-level must be a mapping")

    fid = raw.get("id")
    if fid != fixture_id:
        raise ValueError(
            f"{yaml_path}: id field {fid!r} doesn't match directory {fixture_id!r}"
        )

    for req in ("version", "description", "bug", "investigation", "fix"):
        if req not in raw:
            raise ValueError(f"{yaml_path}: missing required field {req!r}")

    bug = raw["bug"]
    investigation = raw["investigation"]
    fix = raw["fix"]
    budget = raw.get("budget", {})
    scope_raw = raw.get("scope")
    scope: Scope | None = None
    if scope_raw is not None:
        scope = Scope(
            in_scope_task_keys=list(scope_raw["in_scope_task_keys"]),
            output_tables=list(scope_raw["output_tables"]),
            upstream_tables=list(scope_raw.get("upstream_tables") or []),
        )

    if not root.joinpath("seed.sql").exists():
        raise FileNotFoundError(f"seed.sql not found for fixture {fixture_id}")
    if not root.joinpath("job.json").exists():
        raise FileNotFoundError(f"job.json not found for fixture {fixture_id}")

    return Fixture(
        id=fid,
        version=int(raw["version"]),
        description=str(raw["description"]).strip(),
        bug=BugAnnotation(
            category=str(bug["category"]),
            root_cause=str(bug["root_cause"]).strip(),
        ),
        investigation=InvestigationAnnotation(
            required_tools=list(investigation.get("required_tools") or []),
            expected_diagnosis_keywords=[
                str(k).lower() for k in (investigation.get("expected_diagnosis_keywords") or [])
            ],
        ),
        fix=FixAnnotation(
            acceptable_categories=list(fix["acceptable_categories"]),
            runtime_pct_min=float(fix.get("expected_improvement", {}).get("runtime_pct_min", 0)),
            dbu_pct_min=float(fix.get("expected_improvement", {}).get("dbu_pct_min", 0)),
            bytes_scanned_pct_min=float(
                fix.get("expected_improvement", {}).get("bytes_scanned_pct_min", 0)
            ),
        ),
        tool_call_budget=int(budget.get("tool_calls_reference", 15)),
        root=root,
        scope=scope,
    )


def load_job_template(fixture: Fixture) -> dict[str, Any]:
    """Parse the job.json template (with {{placeholders}} still unresolved)."""
    return json.loads(fixture.job_template_path.read_text())


def list_fixtures(fixtures_root: Path | None = None) -> list[str]:
    """List fixture ids by scanning the fixtures root for valid directories."""
    root = fixtures_root or FIXTURES_ROOT
    if not root.exists():
        return []
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and (d / "fixture.yaml").exists()
    )
