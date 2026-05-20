# Helios

**A verifiable-reward (RLVR) environment for ETL optimization.** Helios wraps a frozen-policy LLM agent in a closed environment where every action is grounded against ground-truth signals — byte-level output equivalence and measured runtime delta — and turns the resulting attempts into structured trajectories ready for offline RL.

Point it at a real production Databricks job; Helios clones the task into an isolated sandbox, pins all sources via Delta time-travel, runs the agent through an investigate-rewrite-verify loop, and produces a `proposal.md` (the proof) plus a full trajectory log (the training data).

```
$ helios propose 149618509299562 --task-key device_master_level_daily_part_2

  cloning task device_master_level_daily_part_2 from prod job 149618509299562
  source-alignment timestamp:  2026-05-19 05:16:04.520
  basis:                       prod task last SUCCESS run START time (matches Delta snapshot isolation)
  pinned 12 source tables to per-source MIN(task_start, source.latest_commit)
  baseline median: 10792 s (median of last 10 prod runs)
  → analyzing nondeterminism on 152 output columns of spice_catalog.prod.<table>
    auto-ignored (self-authorizing): ['last_refresh_time']
    probe-required nondeterminism (NOT excluded): ['top_playtime_show_till_specific_date']
  → invoking agent (frozen prod job 149618509299562, max_iters=80)
  → diff_tables(spice_catalog.prod.<table> VERSION AS OF 1141, sandbox)
    verdict: IDENTICAL | identical 150,661,984 | extras 0 | missing 0 | drifted 0

  Tier 1 (output equivalence)   ✅ PASS (IDENTICAL within rtol=1e-9, atol=1e-9)
  Tier 3 (perf)                 ✅ PASS  +92.3% vs 10792 s baseline

  proposal: evals/proposals/9908003684af/proposal.md
```

## Why frame this as RLVR

Most "AI for code" systems use *learned* reward models — preference models trained on human ratings. They're a stand-in when ground truth is unavailable. ETL optimization doesn't have that problem: there's a deterministic, ground-truth answer to "did the rewrite produce the same data?" and a measurable answer to "did it run faster?" That makes the reward channel **verifiable** — the property that makes RLVR (Reinforcement Learning with Verifiable Rewards) work the way it does for math and code reasoning.

Helios builds the environment half of RLVR end-to-end. The agent currently operates as a **frozen policy under in-context adaptation** — its weights aren't updated by the reward. But every run emits the complete (state, action, reward) trajectory needed to close that loop via offline RL / SFT / DPO on accumulated successes.

| RL concept | Helios |
|---|---|
| State | Sandbox notebook + sandbox tables + job-run state |
| Actions | Tool calls (`upload_notebook`, `run_job_now`, `explain_query`, `diff_tables`, …) |
| Observations | Tool results (EXPLAIN plans, job status, `diff_tables` verdict + `drift_profile`, …) |
| Reward (verifiable) | `diff_tables` verdict + runtime delta — both deterministic, quantitative, ground-truth |
| Episode | One `helios propose` invocation; transitions = each in-loop sandbox attempt |
| Termination | Pass (`IDENTICAL`/`FLOAT_REORDER_ONLY` + ≥15% faster), fail, or `max_iters` |
| Trajectory | `proposal.json` + `trace.jsonl` (tool I/O **and** reasoning text) + `messages.json` |

## Two modes

| | `helios eval` | `helios propose` |
|---|---|---|
| Audience | Agent maintainers, CI | Data engineers who own prod jobs |
| Input | YAML fixture in `evals/fixtures/` | Real prod `job_id` + `task_key` |
| Source notebooks | Local files | Fetched from GitHub (`Pocket-Fm/de_databricks`) |
| Source data | Synthetic (10M-row seed) | Real prod tables (read-only) |
| Baseline | Fresh-run the orig job | Median of last 10 prod runs |
| Source version pinning | n/a | `TIMESTAMP AS OF` aligned to prod task START time |
| Equivalence | `diff_tables` over sandbox vs cached baseline | `diff_tables` over sandbox vs `VERSION AS OF`-pinned prod boundary |
| Frozen job_ids | None | Prod `job_id` — every mutating tool rejects it |
| Tier 2 (diagnosis) | ✅ scored against fixture ground truth | disabled (no labeled bottleneck) |
| Output | `evals/results/<run_id>/scores.json` | `evals/proposals/<run_id>/proposal.md` |

Same agent loop and Tier-1 engine power both. They differ in how inputs are sourced and how outputs are reported.

## Setup

### Requirements

- Python 3.10+, [`uv`](https://docs.astral.sh/uv/) (or pip)
- Databricks workspace with Unity Catalog
- An LLM provider that speaks the OpenAI chat-completions API (OpenAI, Anthropic via gateway, etc.)
- GitHub PAT (optional — only needed for `propose` mode when prod notebooks live in a repo)

### One-time setup

```bash
git clone <repo-url>
cd helios
uv sync                       # reproducible install from uv.lock
# or: python -m venv .venv && pip install -e .
```

Copy `.env.example` to `.env` and fill in:

```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
DATABRICKS_WAREHOUSE_ID=...
OPENAI_API_KEY=sk-...
OPENAI_MODEL=claude-opus-4-7    # or any model that speaks chat-completions
GITHUB_TOKEN=ghp_...            # optional — needed for propose mode
```

### One-time Databricks catalog setup

Create two Unity Catalog catalogs (UI: Catalog → Create catalog → Default Storage):

- `helios_eval_seed` — synthetic seed data for eval fixtures
- `helios_eval_runs` — scratch catalog where sandbox schemas land per run

Per-run schemas are created and torn down automatically.

## Usage

### Eval (synthetic fixtures)

```bash
uv run helios eval list                              # available fixtures
uv run helios eval run skew_join_orders              # run end-to-end
uv run helios eval cleanup <run_id>                  # drop sandbox artifacts
```

### Propose (production)

```bash
uv run helios propose <prod_job_id> --task-key <task_key>
```

What the harness does:

1. Reads the prod job spec, finds the task, pulls the notebook from GitHub.
2. Determines the prod task's most recent **task-level** SUCCESS start time — accepts a finished task even inside a still-running parent run, so source alignment uses the freshest possible snapshot.
3. Rewrites the notebook: write targets remapped to a sandbox catalog; source reads pinned with `TIMESTAMP AS OF '<task_start>'` (per source: `MIN(task_start, source.latest_commit)` — Spark refuses TS-AS-OF past a table's latest commit).
4. Captures the prod boundary's current Delta version (`VERSION AS OF <n>`) and computes a row-count + hash snapshot.
5. Uploads the rewritten notebook + creates a single-task sandbox job.
6. **Runs the LLM nondeterminism detector** on the original notebook against the boundary table's schema (see below).
7. Invokes the agent with hard frozen-`job_id` guards. The agent investigates (`explain_query`, `describe_history`, plan inspection, `diff_tables` after each change), modifies the sandbox notebook, retries on failure — all within one episode.
8. Triggers a final clean measurement run, scores Tier 1 + Tier 3 via the canonical `diff_tables` against the version-pinned boundary, emits `proposal.md`.

Resume / re-finalize from any interruption (same agent state, same scoring engine):

```bash
uv run helios propose-resume <run_id>      # continues the agent's reasoning loop
uv run helios propose-finalize <run_id>    # rescore + regenerate proposal.md
```

Watch the trajectory live in another shell:

```bash
tail -f evals/proposals/<run_id>/trace.live.jsonl
```

Each line is one event — `assistant_text` (the model's reasoning between tool calls), `tool_call`, `tool_result`, `blocked` (write-guard rejection), `agent_start/end`. Reasoning is captured **on every turn**, not just final, so the trajectory shows *why* the agent picked each action.

## The reward channel — `diff_tables`

`diff_tables(prod_VERSION_AS_OF_N, sandbox)` is the canonical equivalence check and the verifiable reward source. FULL OUTER JOIN on the auto-detected natural key, every row categorized, every column profiled.

### Magnitude-aware float tolerance + type gate

IEEE-754 double aggregation is non-associative — a *correct* rewrite that changes join/shuffle order produces bit-different `SUM`s (~1e-13 relative). Fixed-decimal rounding is the wrong instrument because FP error is relative to magnitude, not absolute. `diff_tables` uses the `numpy.isclose` criterion:

```
float cells equal  ⟺  |a − b|  ≤  atol + rtol · max(|a|, |b|)         (default rtol=atol=1e-9)
```

**Type gate:** tolerance applies *only* to `DOUBLE`/`FLOAT`. `DECIMAL`/`INT`/string are compared **exactly** — Spark `DECIMAL` is order-stable, so any drift on those is real.

### Verdict

- `IDENTICAL` — sub-tolerance float drift was absorbed; tables are functionally equivalent.
- `FLOAT_REORDER_ONLY` — only DOUBLE columns drift beyond tolerance, with `worst_float_rel_diff` ≤ `reorder_rel_threshold` (1e-6 default) — large-magnitude reorder, still negligible. Pass.
- `REAL_DIFFERENCE` — structural drift, or `DECIMAL`/`INT`/string drift, or float drift beyond the reorder threshold. Fail.

### Drift profile (the why)

For every column that drifted, the output includes `rows_drifted`, `max_abs_diff`, `max_rel_diff` — so the verdict is *evidence-based*. `"max relative drift 3.1e-13 over 299 rows"` reads as obviously reorder; `"max relative drift 0.4"` reads as a real bug. Both `proposal.md` and `proposal.json` carry the full profile.

## Reward-signal hygiene — LLM nondeterminism detector

A reward channel is only as good as its ability to distinguish "wrong rewrite" from "the original query was nondeterministic." Untied `ROW_NUMBER() ORDER BY x` picks arbitrarily among tied rows; `current_timestamp()` stamps wall-clock; `collect_list` order varies. These columns differ run-to-run *in prod itself* — comparing the sandbox against prod on them produces a false `REAL_DIFFERENCE` that has nothing to do with the optimization.

Helios runs an **LLM-driven detector** (`src/helios/evals/nondeterminism.py`) on the **original canonical notebook**. For every final output column it returns a verdict + class + authorization + rationale + (for argmax) the deterministic sibling column:

| Class | Examples | Authorization |
|---|---|---|
| **A_by_definition** | `current_timestamp()`/`now()`, `rand()`, `uuid()`, `monotonically_increasing_id()`; run-stamps derived from these (`last_refresh_time`, `loaded_at`, `batch_id`) | **self-authorizing** |
| **B_untied_pick** | `ROW_NUMBER ORDER BY x` with ties; `FIRST/LAST/ANY_VALUE`; `MAX_BY(label, val)` on tied `val` | **probe-required** |
| **C_order_sensitive_agg** | `collect_list/set`, `approx_count_distinct`, `percentile_approx`, `mode()` | **probe-required** |
| **D_unseeded_sampling** | `LIMIT n` without total `ORDER BY`, `TABLESAMPLE`, `sample()` without seed | **self-authorizing** |
| **E_float_reorder** | `SUM/AVG/STDDEV` over DOUBLE | **already handled by tolerance** |
| **F_udf** | Python/Scala UDF that may call `random`/`time`/external service | **probe-required** |

The detector traces lineage through CTEs to the construct responsible: `max(x)` over tied rows is deterministic (the value is stable); a non-key attribute carried from the row picked by an untied `ROW_NUMBER` is not.

### Authorization rule (the safety invariant)

- **Self-authorizing** columns are auto-excluded from the diff (same safety class as the existing run-stamp name heuristic — non-pure by language semantics, structurally distinguishable from a real bug).
- **Probe-required** columns are **never auto-excluded** — they're data-derived and indistinguishable from a real bug. They're surfaced prominently in `proposal.md` with a ⚠️ callout, awaiting either a human sign-off (write the column into `equivalence_ignore_columns` in `clone.json` with rationale) or an automated determinism probe (planned).
- **Already-handled** is the float tolerance band — nothing to do.

The detector + verdict are recorded in `proposal.json` under `nondeterminism`, surfaced in `proposal.md`, and never silently mutate the reward signal.

## The agent's optimization priority

The agent prompt structures optimizations as a 6-tier priority — try lower categories first, escalate only when needed:

1. **Cluster / Spark config** — AQE, broadcast threshold, shuffle partitions, off-heap memory (no algebra change)
2. **Spark hints** — `/*+ BROADCAST(rel) */`, `/*+ REPARTITION(N, key) */`, `/*+ COALESCE(N) */`, **`/*+ RANGE_JOIN(rel, bin_size) */`** for interval/inequality joins (BNLJ → linear)
3. **Caching** — `CACHE TABLE` for intermediates reused 3+ times
4. **Predicate pushdown / partition pruning**
5. **Table maintenance** — `OPTIMIZE ZORDER BY`, `ANALYZE TABLE`, file compaction
6. **Algorithmic rewrite** — only when 1–5 are exhausted; one change at a time; mandatory `diff_tables` after each; revert on `REAL_DIFFERENCE`

`explain_query` parses the physical plan and auto-flags common issues into `combined_warnings`: `SortMergeJoin` where `BroadcastHashJoin` would work, heavy shuffling, **`BroadcastNestedLoopJoin`** with a recommended `RANGE_JOIN` hint (mirroring Databricks' own optimizer suggestions).

## Trajectory artifacts (the RLVR training data)

Every run writes a complete, machine-readable trajectory to `evals/proposals/<run_id>/`:

| File | Content |
|---|---|
| `clone.json` | Pinned sources + boundary versions + scope; the "initial state" of the episode |
| `notebook_original.txt` | Canonical (unmodified) prod notebook source |
| `notebook_sandbox_pre_agent.txt` | Write-target-remapped notebook the agent started from |
| `messages.json` | Full LLM conversation history (system + user + assistant + tool messages) — atomic whole-file write per turn, so resume-from-crash works |
| `trace.live.jsonl` | Streaming events: `assistant_text` (reasoning) + `tool_call` + `tool_result` + `blocked` |
| `baseline.json` | History-based baseline (median + samples + per-table snapshot) |
| `proposal.json` | Tier 1 verdict + drift profile + nondeterminism analysis + Tier 3 runtime delta |
| `proposal.md` | Human-readable proposal (what changed, why, equivalence proof, ND callouts) |

These are exactly the (state, action, reward) trajectories an offline RL pass would consume. The system is collecting them; closing the training loop is the next step (see below).

## Safety model

- **Application-layer write guard.** Every `execute_sql` is parsed; writes outside `helios_eval_runs.*` are rejected pre-flight.
- **Frozen `job_id`.** Prod `job_id` enters a hard frozen set; every mutating tool that takes a `job_id` rejects it. No agent reasoning can override.
- **Read-only on prod tables.** Investigation reads freely; writes are remapped to sandbox at clone time.
- **Source version pinning.** `TIMESTAMP AS OF` on every source; `VERSION AS OF` on the boundary — upstream refreshes during the experiment cannot cause spurious mismatches.
- **Prompt-level boundary guard.** The exact pinned `diff_tables` command is injected into the prompt verbatim — no `<placeholder>` text for the agent to hallucinate around. Live/unpinned prod comparisons and invented `*__boundary` tables are explicitly forbidden in HARD CONSTRAINTS.
- **Never auto-applies.** Output is `proposal.md` — a human reviews and applies via PR.

For full org-grade safety, the harness should also run as a dedicated Databricks service principal with grants limited to `SELECT` on prod schemas and `ALL PRIVILEGES` only on `helios_eval_runs.*`. The application layer is fast early-detection; SP grants are the real boundary.

## Layout

```
helios/
├── src/helios/
│   ├── agent.py                  # LLM loop — call_llm → tool_calls → repeat (reasoning streamed)
│   ├── cli.py                    # ask, chat, eval, propose, propose-resume, propose-finalize
│   ├── tools/
│   │   ├── databricks.py         # diff_tables (tolerance + drift profile), explain_query (RANGE_JOIN warn), ...
│   │   ├── github.py             # 9 tools — file fetch, PRs, staged commits
│   │   ├── terminal.py           # 5 tools
│   │   ├── memory.py             # 4 tools — persistent cross-session memory
│   │   └── agents.py             # 6 tools — sub-agents
│   └── evals/
│       ├── harness.py            # eval-mode orchestrator
│       ├── propose.py            # propose + resume + finalize; shared Tier-1 helper
│       ├── nondeterminism.py     # LLM detector — A–F catalog, lineage tracing
│       ├── sandbox.py            # scratch schema lifecycle, notebook upload
│       ├── baselines.py          # baseline cache + prod snapshot
│       ├── runner.py             # agent invocation with write-guard + frozen-job-id
│       ├── fixtures.py           # YAML fixture loader
│       └── scorers/              # Tier 1 (correctness), Tier 2 (diagnosis), Tier 3 (outcome)
├── evals/
│   ├── fixtures/                 # checked-in: synthetic test definitions
│   ├── results/                  # gitignored: per-eval-run outputs
│   └── proposals/                # gitignored: per-propose-run outputs (contains prod data)
├── pyproject.toml
├── uv.lock                       # pinned reproducible install
└── .env.example
```

## Status

- ✅ Core agent loop, tool surface, sandbox isolation
- ✅ Synthetic eval framework (Tier 1/2/3 scoring on labeled fixtures)
- ✅ Live propose mode against prod (verified producing 90%+ runtime wins with byte-equivalent output)
- ✅ Source-version pinning via Delta time-travel; aligns to *task-level* SUCCESS even inside a running parent run
- ✅ Magnitude-relative float tolerance + type gate + per-column drift profile
- ✅ LLM nondeterminism detector with self-authorizing / probe-required split
- ✅ Unified Tier-1 across `propose` / `propose-resume` / `propose-finalize` (no legacy hash-scorer drift)
- ✅ Reasoning text streamed into `trace.jsonl` alongside tool I/O (full RLVR trajectories)
- ✅ Resume / finalize / clean recovery from interruption
- ☐ **Determinism probe** — automated verification for `probe_required` columns (designed; not built)
- ☐ **Cross-run lesson memory** — per `(prod_job_id, task_key)` distillation feeding the next run's prompt
- ☐ **Offline RL training pass** on accumulated trajectories — the step that closes the RLVR loop
- ☐ **Multi-task DAG scope** — co-optimization across in-scope tasks with chained sandbox reads
- ☐ **DBU-cost reward** — Tier 3 currently uses runtime delta only; cost would catch "faster but bigger cluster" regressions
- ☐ **Post-deploy regression monitor** — re-run `diff_tables` against rolling prod self-consistency after merge
- ☐ **Automated PR creation** from a passing proposal
- ☐ **Databricks-side service principal** for production-grade safety boundary (currently application-layer + frozen-id only)

## What this isn't (yet)

Honest framing — what Helios *is* and *isn't* matters for what claims you can make:

- It **is** an RLVR *environment* — the reward channel is verifiable, the episode structure is RL-shaped, the trajectories are training-ready.
- It **isn't** an RL-trained agent. The LLM's weights are frozen; the policy doesn't improve from the reward today. The "learning" within a run is in-context (the model reasons over its own prior tool outputs); across runs there is no carry-over yet. Closing that loop — offline RL / SFT-on-winners / DPO on accumulated trajectories — is the step that would let Helios honestly be called RL.
- It **is not** "self-improving" in the formal sense — only via accumulated training data feeding a future training pass.

The substrate is built. The training loop is the next mile.
