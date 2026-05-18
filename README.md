# Helios

LLM-driven CLI agent that optimizes Databricks ETL jobs. Point it at a production job and it produces a verified optimization proposal — a sandbox clone running the agent's rewrite, scored against the prod baseline for both correctness and speed.

```
$ helios propose 149618509299562 --task-key device_master_level_daily_part_2

  cloning task device_master_level_daily_part_2 from prod job 149618509299562
  pinned 12 source tables to prod task run start time
  baseline median: 10792 s
  invoking agent (frozen prod job 149618509299562)
  → triggering optimized sandbox job for clean measurement

  Tier 1 (output equivalence)   ✅ PASS (machine-epsilon float drift only)
  Tier 3 (perf)                 ✅ PASS  +92.3% vs 10792 s baseline

  proposal: evals/proposals/9908003684af/proposal.md
```

## What it does

Helios is an autonomous LLM agent loop with 20+ purpose-built tools over the Databricks API, GitHub API, your local filesystem, and a persistent memory store. It runs in two modes:

- **`helios eval`** — regression-test the agent against synthetic Databricks fixtures with deliberately-suboptimal jobs. Scores correctness (output equivalence), diagnosis quality (did it identify the right bottleneck), and outcome (did it actually run faster). Used to validate prompt / model changes.
- **`helios propose`** — point the agent at a real production job. It investigates via `EXPLAIN`, `DESCRIBE HISTORY`, plan inspection, and skew probes, then proposes a rewrite that's verified end-to-end in an isolated sandbox catalog. The prod job is **never** modified — the agent has hard-frozen mutation guards on the prod job_id.

A human reviews the resulting `proposal.md`, applies the diff via PR, and the next prod run picks it up.

## Why this exists

Most production optimization is bottlenecked by senior-engineer attention, not by ideas. Helios narrows that bottleneck: the agent does the investigation and produces a concrete, runnable proposal with full reasoning trace, equivalence proof, and per-task runtime measurements. The engineer reviews the diff instead of building it.

## Two modes side by side

| | `helios eval` | `helios propose` |
|---|---|---|
| Audience | Agent maintainers, CI | Data engineers who own prod jobs |
| Input | YAML fixture in `evals/fixtures/` | Real prod `job_id` + `task_key` |
| Source notebooks | Local files | Fetched from GitHub (`Pocket-Fm/de_databricks`) |
| Source data | Synthetic (10M-row seed) | Real prod tables (read-only) |
| Baseline | Fresh-run the orig job | Median of last 10 prod runs |
| Source version pinning | n/a | `TIMESTAMP AS OF` aligned to prod task start (Delta time-travel) |
| Equivalence | `diff_tables` over full sandbox output vs cached baseline | `diff_tables` over sandbox output vs **versioned** prod boundary |
| Frozen job_ids | None | The prod `job_id` — hard reject every mutation tool call against it |
| Teardown | Drops sandbox unless `--keep-artifacts` | Always keeps artifacts (sandbox job IS the proof) |
| Output | `evals/results/<run_id>/scores.json` (PASS/FAIL) | `evals/proposals/<run_id>/proposal.md` (human-readable) |
| When you run it | Every PR / nightly CI | On-demand, per job |

The same agent and harness internals power both. They differ only in how inputs are sourced and how outputs are reported.

## Setup

### Requirements

- Python 3.10+
- Databricks workspace with Unity Catalog
- An LLM provider that speaks the OpenAI chat-completions API (OpenAI, Anthropic via gateway, etc.)
- GitHub PAT (optional — only needed for `propose` mode if your prod notebooks live in a repo)

### One-time setup

```bash
git clone <repo-url>
cd helios
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and fill in:

```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
DATABRICKS_WAREHOUSE_ID=...
OPENAI_API_KEY=sk-...
OPENAI_MODEL=claude-opus-4-7    # or gpt-4o, etc.
GITHUB_TOKEN=ghp_...            # optional — needed for propose mode
```

### One-time Databricks catalog setup

Create two Unity Catalog catalogs (UI: Catalog → Create catalog → Default Storage):

- `helios_eval_seed` — holds synthetic seed data for eval fixtures
- `helios_eval_runs` — scratch catalog where sandbox schemas land per run

The harness creates per-run schemas under `helios_eval_runs.*` and tears them down (or keeps them for inspection); seed data under `helios_eval_seed.<fixture_id>.*` is built once per fixture version.

## Usage

### One-shot question

```bash
helios ask "list all jobs in the workspace"
```

### Interactive REPL

```bash
helios chat
```

### List all tools

```bash
helios tools                    # all 20+
```

### Eval (synthetic fixtures)

```bash
helios eval list                                    # available fixtures
helios eval run skew_join_orders                    # run end-to-end
helios eval run skew_join_orders --refresh-baseline # ignore cached baseline
helios eval cleanup <run_id>                        # drop sandbox artifacts
```

Watch the agent live in another shell:

```bash
tail -f evals/results/<run_id>/trace.live.jsonl
```

### Propose (production)

```bash
helios propose <prod_job_id> --task-key <task_key>
```

The harness:

1. Reads the prod job spec via `get_job`. Finds the task.
2. Pulls the notebook source from GitHub.
3. Determines the prod task's **last successful run start time** — the moment Delta snapshot isolation captured for source reads.
4. Rewrites the notebook: write targets remapped to sandbox, source reads pinned with `TIMESTAMP AS OF '<prod_task_start>'` (per source: `MIN(task_start, source.latest_commit)`).
5. Uploads the rewritten notebook + creates a single-task sandbox job.
6. Snapshots the prod boundary table (hash + per-column stats fingerprint).
7. Invokes the agent with hard tool guards: any mutation against `prod_job_id` is rejected before reaching Databricks.
8. Agent investigates (`explain_query`, `describe_detail`, skew probes), modifies the sandbox notebook, runs it, verifies via `diff_tables`.
9. Harness triggers a clean measurement run, scores T1 + T3, emits `proposal.md`.

If you Ctrl-C mid-run, resume from where you stopped:

```bash
helios propose-resume <run_id>      # continues the agent's reasoning loop
helios propose-finalize <run_id>    # just regenerate proposal.md from current state
```

## Layout

```
helios/
├── src/helios/
│   ├── agent.py            # the LLM loop — call_llm → tool_calls → repeat
│   ├── cli.py              # Typer CLI (ask, chat, eval, propose, propose-resume, propose-finalize)
│   ├── tools/
│   │   ├── databricks.py   # 16 tools: SQL exec, jobs, runs, notebooks, EXPLAIN, diff_tables, ...
│   │   ├── github.py       # 9 tools: file fetch, PRs, staged-commit workflow
│   │   ├── terminal.py     # 5 tools: read_file, list_dir, grep, write_file, run_shell
│   │   ├── memory.py       # 4 tools: persistent cross-session memory
│   │   └── agents.py       # 6 tools: spawn sub-agents (ephemeral / persistent)
│   └── evals/
│       ├── harness.py      # eval-mode orchestrator
│       ├── propose.py      # propose-mode orchestrator + resume + finalize
│       ├── sandbox.py      # scratch catalog/schema lifecycle, notebook upload
│       ├── baselines.py    # baseline cache (history-based for propose, fresh-run for eval)
│       ├── runner.py       # agent invocation with write-guard + frozen-job-id support
│       ├── fixtures.py     # YAML fixture loader
│       └── scorers/        # Tier 1 (correctness), Tier 2 (diagnosis), Tier 3 (outcome)
├── evals/
│   ├── fixtures/           # checked-in: synthetic test definitions
│   │   ├── skew_join_orders/
│   │   └── skew_join_scoped/
│   ├── results/            # gitignored: per-eval-run outputs
│   └── proposals/          # gitignored: per-propose-run outputs (contains prod data)
├── pyproject.toml
└── .env.example
```

## Safety model

`helios propose` is designed to be safe to run against real production:

- **Application-layer write guard.** Every `execute_sql` is parsed for write verbs (INSERT/UPDATE/DELETE/CREATE/...). Any write targeting a catalog outside `helios_eval_runs` is rejected before reaching Databricks.
- **Frozen job_ids.** The prod `job_id` is added to a hard frozen set. `run_job_now`, `add_job_tasks`, and any other mutation tool that takes a `job_id` rejects calls targeting it. No agent reasoning can override this.
- **Read-only on prod tables.** The agent freely reads prod sources (it needs to investigate) but cannot write to them. The sandbox notebook's write targets are remapped at clone time.
- **Source version pinning.** Sources are read at the version current when the prod task last started — upstream refreshes during the experiment don't cause spurious equivalence failures.
- **Never auto-applies a proposal.** The output is `proposal.md` — a human reviews and applies via PR.

For full org-grade safety, the harness should also be run as a dedicated Databricks service principal with grants limited to `SELECT` on the prod schemas it reads and `ALL PRIVILEGES` only on `helios_eval_runs.*`. The application-layer guard is fast early-detection; the SP grants are the real boundary. Setting up the SP is currently a manual workspace-admin step.

## Equivalence check

`diff_tables(prod, sandbox)` does a FULL OUTER JOIN on the auto-detected natural key (string/date/timestamp columns + integer columns whose names aren't count-like) and categorizes every row into one of:

- `IDENTICAL` — row + content match exactly
- `FLOAT_REORDER_ONLY` — row counts match, integer/decimal columns match, only DOUBLE/FLOAT columns drift at machine-epsilon scale (~10⁻¹³ relative). The agent's optimization is semantically equivalent; the hash just isn't tolerant of float reorder under aggregation.
- `REAL_DIFFERENCE` — actual semantic divergence (extras / missing rows, or non-float metric drift)

Tier 1 PASSES on `IDENTICAL` or `FLOAT_REORDER_ONLY` and FAILS on `REAL_DIFFERENCE`. The `diff_report` localizes drift by dimension so the engineer reviewing knows exactly where the agent's algebra differs from prod.

## Optimization priority order

The agent prompt structures optimizations as a 5-tier priority — try lower categories first, only escalate when needed:

1. **Cluster / Spark config** — AQE, broadcast threshold, shuffle partitions, off-heap memory (no algebra change)
2. **Spark hints** — `BROADCAST`, `REPARTITION`, `COALESCE`
3. **Caching** — `CACHE TABLE` for intermediates reused 3+ times
4. **Predicate pushdown / partition pruning** — push WHERE clauses into source scans
5. **Table maintenance** — `OPTIMIZE ZORDER BY`, `ANALYZE TABLE`, file compaction
6. **Algorithmic rewrite** — restructured CTEs, INNER↔LEFT JOIN swaps, aggregation reordering (high correctness risk; one change per iteration; revert on equivalence failure)

Categories 1–5 don't change *what* the query computes — only *how*. Category 6 changes the algebra and is the most common source of correctness failures.

## Development

```bash
# Run smoke tests
.venv/bin/python -m pytest tests/ -v

# Run an eval against the synthetic skew fixture
.venv/bin/helios eval run skew_join_orders

# Run lint
.venv/bin/ruff check src/
```

## Status

- ✓ Core agent loop, tool surface, sandbox isolation
- ✓ Synthetic eval framework (T1/T2/T3 scoring on 2 fixtures)
- ✓ Live propose mode against prod (verified producing 90%+ runtime wins with byte-equivalent output)
- ✓ Source-version pinning via Delta time-travel
- ✓ Resume / finalize / clean recovery from interruption
- ✓ Live trace streaming (`tail -f trace.live.jsonl`)
- ☐ Databricks-side service principal for production-grade safety boundary
- ☐ Multi-table proposal flow (currently single boundary table per task)
- ☐ Automated PR creation from a passing proposal
- ☐ DBU cost measurement (requires `system.billing.usage` join, hours-delayed)
