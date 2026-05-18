"""Scratch-schema and job sandbox for the eval harness.

Responsibilities:
  - Create / drop the per-run scratch schema under `helios_eval_runs`.
  - Ensure the fixture's seed schema exists under `helios_eval_seed` (run seed.sql
    if the seed version on disk has drifted from what's in the workspace).
  - Substitute {{placeholders}} in job.json + notebook source.
  - Upload notebooks to a workspace folder and create a Databricks job from
    the resolved spec. Returns the job_id.
  - Tear down: delete jobs tagged for this run, drop scratch schema.

Sandbox boundary: this module is the *only* place that writes to
`helios_eval_seed.*` (and only when re-seeding). The eval agent runs with
permissions allowing writes solely against `helios_eval_runs.<run_schema>`.

Naming conventions used throughout:
  seed catalog  = helios_eval_seed
  seed schema   = <fixture_id>           (one per fixture, stable)
  run catalog   = helios_eval_runs
  run schema    = run_<uuid_hex_12>      (one per eval invocation, ephemeral)
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..tools.databricks import execute_sql, upload_notebook as _upload_notebook_tool, workspace
from .fixtures import Fixture, load_job_template


SEED_CATALOG: str = os.environ.get("HELIOS_EVAL_SEED_CATALOG", "helios_eval_seed")
RUN_CATALOG: str = os.environ.get("HELIOS_EVAL_RUN_CATALOG", "helios_eval_runs")
WORKSPACE_ROOT: str = os.environ.get(
    "HELIOS_EVAL_WORKSPACE_ROOT", "/Shared/helios_eval"
)

# Identifiers we accept inside fixture configs; anything else is a typo or
# (worse) an injection attempt. Catalog/schema/table names get this check
# before they're spliced into SQL strings.
_SAFE_IDENT = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class RunContext:
    """Everything a fixture run needs to substitute its placeholders."""

    run_id: str
    fixture: Fixture
    seed_catalog: str
    seed_schema: str
    run_catalog: str
    run_schema: str
    workspace_dir: str  # parent folder for this run's notebooks

    def placeholders(
        self,
        *,
        role: str,
        output_table: str,
        notebook_path: str,
        notebook_paths: dict[str, str] | None = None,
    ) -> dict[str, str]:
        base: dict[str, str] = {
            "run_id": self.run_id,
            "seed_catalog": self.seed_catalog,
            "seed_schema": self.seed_schema,
            "run_catalog": self.run_catalog,
            "run_schema": self.run_schema,
            "output_table": output_table,
            "notebook_path": notebook_path,
            "role": role,
        }
        # Multi-task fixtures use per-task placeholders like {{notebook_<stem>}}
        # so each task can reference its own notebook.
        for stem, path in (notebook_paths or {}).items():
            base[f"notebook_{stem}"] = path
        return base

    def base_placeholders(self) -> dict[str, str]:
        """Subset of placeholders that don't depend on a role / notebook —
        usable for rendering things like scope.output_tables."""
        return {
            "run_id": self.run_id,
            "seed_catalog": self.seed_catalog,
            "seed_schema": self.seed_schema,
            "run_catalog": self.run_catalog,
            "run_schema": self.run_schema,
        }


def new_run_context(fixture: Fixture) -> RunContext:
    """Generate a fresh run context with a unique scratch schema name."""
    run_id = uuid.uuid4().hex[:12]
    for name in (SEED_CATALOG, RUN_CATALOG, fixture.seed_schema):
        if not _SAFE_IDENT.match(name):
            raise ValueError(f"unsafe identifier {name!r}")
    return RunContext(
        run_id=run_id,
        fixture=fixture,
        seed_catalog=SEED_CATALOG,
        seed_schema=fixture.seed_schema,
        run_catalog=RUN_CATALOG,
        run_schema=f"run_{run_id}",
        workspace_dir=f"{WORKSPACE_ROOT}/{run_id}",
    )


# ---- Placeholder substitution ------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")


def render_scope_tables(ctx: RunContext, templates: list[str]) -> list[str]:
    """Render the placeholders inside scope.output_tables / scope.upstream_tables.
    Uses only catalog/schema/run_id placeholders (no role / notebook context)."""
    values = ctx.base_placeholders()
    return [render(t, values) for t in templates]


def render(template: str, values: dict[str, str]) -> str:
    """Substitute {{key}} placeholders. Unknown keys raise — fail loud, not silent."""

    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"unknown placeholder {{{{ {key} }}}}; have: {sorted(values)}")
        return values[key]

    return _PLACEHOLDER_RE.sub(sub, template)


# ---- Scratch schema lifecycle ------------------------------------------------


def ensure_catalogs_exist(seed_catalog: str, run_catalog: str) -> None:
    """Verify both catalogs exist. Catalog creation is an admin-scope action and
    is outside the harness's job — operators must create them once via UI (using
    Default Storage) or with `CREATE CATALOG ... MANAGED LOCATION '...'`."""
    rows = execute_sql(
        "SELECT catalog_name FROM system.information_schema.catalogs "
        f"WHERE catalog_name IN ('{seed_catalog}', '{run_catalog}')"
    )["rows"]
    present = {r["catalog_name"] for r in rows}
    missing = [c for c in (seed_catalog, run_catalog) if c not in present]
    if missing:
        raise RuntimeError(
            f"required catalog(s) missing: {missing}. "
            f"Create them once in Databricks (UI → Catalog → Create catalog, using "
            f"Default Storage) or with: CREATE CATALOG <name> MANAGED LOCATION '<path>';"
        )


def create_run_schema(ctx: RunContext) -> None:
    """Create the per-run scratch schema. Idempotent. Assumes the run catalog
    already exists (see ensure_catalogs_exist)."""
    execute_sql(
        f"CREATE SCHEMA IF NOT EXISTS {ctx.run_catalog}.{ctx.run_schema} "
        f"COMMENT 'helios eval run {ctx.run_id} (fixture: {ctx.fixture.id})'"
    )


def drop_run_schema(ctx: RunContext) -> None:
    """Drop the scratch schema with all contents. Safe to call on missing schema."""
    execute_sql(f"DROP SCHEMA IF EXISTS {ctx.run_catalog}.{ctx.run_schema} CASCADE")


# ---- Seed schema (per-fixture, semi-stable) ----------------------------------


def ensure_seed_schema(fixture: Fixture, force: bool = False) -> None:
    """Run the fixture's seed.sql if needed.

    Skips work when a marker table `_seed_version` in the schema reports the
    same version as fixture.version. `force=True` always re-runs.
    """
    seed_schema_fqn = f"{SEED_CATALOG}.{fixture.seed_schema}"
    current_version = -1
    if not force:
        try:
            result = execute_sql(
                f"SELECT version FROM {seed_schema_fqn}._seed_version LIMIT 1"
            )
            if result["row_count"] == 1:
                current_version = int(result["rows"][0]["version"])
        except Exception:
            current_version = -1

    if current_version == fixture.version:
        return

    # Drop and recreate so seed.sql can assume a clean slate.
    execute_sql(f"DROP SCHEMA IF EXISTS {seed_schema_fqn} CASCADE")

    rendered = render(
        fixture.seed_sql_path.read_text(),
        {"seed_catalog": SEED_CATALOG, "seed_schema": fixture.seed_schema},
    )
    for stmt in _split_statements(rendered):
        execute_sql(stmt)

    execute_sql(
        f"CREATE TABLE {seed_schema_fqn}._seed_version AS SELECT {fixture.version} AS version"
    )


def _split_statements(sql: str) -> list[str]:
    """Naive splitter: split on ';' at top level, strip whitespace + comments-only lines."""
    out: list[str] = []
    for raw in sql.split(";"):
        # Strip line-comments but keep the rest.
        lines = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if stmt:
            out.append(stmt)
    return out


# ---- Notebook upload + job creation ------------------------------------------


def upload_fixture_notebook(
    ctx: RunContext, local_path: Path, *, role: str, output_table: str
) -> str:
    """Render placeholders in a notebook then upload via the tools.databricks
    upload_notebook helper. Returns the workspace path."""
    name = local_path.stem
    remote_path = f"{ctx.workspace_dir}/{role}_{name}"
    rendered = render(
        local_path.read_text(),
        ctx.placeholders(role=role, output_table=output_table, notebook_path=remote_path),
    )
    _upload_notebook_tool(workspace_path=remote_path, content=rendered, language="PYTHON")
    return remote_path


def create_job(ctx: RunContext, *, role: str, output_table: str) -> int:
    """Resolve the fixture's job.json template and create the job. Returns job_id.

    `role` is "orig" (baseline) or "candidate" (agent's working copy) — it's
    recorded in tags so teardown can find every job tied to this run.
    """
    if role not in {"orig", "candidate"}:
        raise ValueError(f"role must be 'orig' or 'candidate', got {role!r}")

    template = load_job_template(ctx.fixture)

    # Upload each notebook referenced by a task. The placeholder convention is
    # {{notebook_path}} for single-task fixtures (legacy) and
    # {{notebook_<stem>}} for multi-task fixtures — both shapes are supported
    # simultaneously by passing the full notebook_paths dict.
    notebook_paths: dict[str, str] = {}
    for nb in sorted(ctx.fixture.notebooks_dir.glob("*.py")):
        notebook_paths[nb.stem] = upload_fixture_notebook(
            ctx, nb, role=role, output_table=output_table
        )

    primary_notebook_path = next(iter(notebook_paths.values())) if notebook_paths else ""

    rendered = render(
        json.dumps(template),
        ctx.placeholders(
            role=role,
            output_table=output_table,
            notebook_path=primary_notebook_path,
            notebook_paths=notebook_paths,
        ),
    )
    settings: dict[str, Any] = json.loads(rendered)

    from ..tools.databricks import create_job as create_job_tool

    return int(create_job_tool(settings)["job_id"])


# ---- Teardown ---------------------------------------------------------------


def delete_run_jobs(ctx: RunContext) -> int:
    """Delete every job tagged with this run_id. Returns count deleted."""
    w = workspace()
    deleted = 0
    for job in w.jobs.list(limit=100):
        tags = (job.settings.tags or {}) if job.settings else {}
        if tags.get("helios_eval_run_id") == ctx.run_id:
            w.jobs.delete(job_id=job.job_id)
            deleted += 1
    return deleted


def teardown(ctx: RunContext) -> None:
    """Best-effort: drop jobs, drop scratch schema, delete workspace folder."""
    try:
        delete_run_jobs(ctx)
    except Exception:
        pass
    try:
        drop_run_schema(ctx)
    except Exception:
        pass
    try:
        workspace().workspace.delete(ctx.workspace_dir, recursive=True)
    except Exception:
        pass


def cleanup_by_run_id(run_id: str) -> dict[str, Any]:
    """Tear down everything tied to a run_id without needing the Fixture object.

    Useful for cleaning up artifacts left behind by keep-on-failure runs.
    All operations are best-effort and report what was deleted.
    """
    w = workspace()
    # Both eval-mode and propose-mode artifact naming conventions are tried.
    # eval-mode:    schema=run_<id>           workspace=<root>/<id>
    # propose-mode: schema=proposal_<id>      workspace=<root>/proposal_<id>
    schemas = [f"run_{run_id}", f"proposal_{run_id}"]
    workspace_dirs = [f"{WORKSPACE_ROOT}/{run_id}", f"{WORKSPACE_ROOT}/proposal_{run_id}"]

    deleted_jobs: list[int] = []
    for job in w.jobs.list(limit=100):
        tags = (job.settings.tags or {}) if job.settings else {}
        if tags.get("helios_eval_run_id") == run_id:
            try:
                w.jobs.delete(job_id=job.job_id)
                deleted_jobs.append(int(job.job_id))
            except Exception:
                pass

    schemas_dropped: list[str] = []
    for schema in schemas:
        try:
            execute_sql(f"DROP SCHEMA IF EXISTS {RUN_CATALOG}.{schema} CASCADE")
            schemas_dropped.append(schema)
        except Exception:
            pass

    dirs_deleted: list[str] = []
    for wd in workspace_dirs:
        try:
            w.workspace.delete(wd, recursive=True)
            dirs_deleted.append(wd)
        except Exception:
            pass

    return {
        "run_id": run_id,
        "jobs_deleted": deleted_jobs,
        "schemas_dropped": schemas_dropped,
        "workspace_dirs_deleted": dirs_deleted,
    }
