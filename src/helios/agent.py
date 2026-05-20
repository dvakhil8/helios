"""LLM agent loop — OpenAI-compatible chat-completions + the unified tools REGISTRY.

The loop:
  1. Send (messages + tool schemas) to the model.
  2. If the model returns tool_calls, execute each (with confirmation prompt for
     mutating tools), append tool_call_id results, and continue.
  3. If the model returns final text, hand it back to the caller.

The agent is integration-agnostic — it reads tools from `helios.tools.REGISTRY`,
which already merges Databricks + GitHub (and anything we add later).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from rich.console import Console
from rich.prompt import Confirm
from rich.syntax import Syntax

from .tools import all_schemas, call_tool
from .tools.memory import load_index as _load_memory_index


# Tools that mutate external state — confirmed by the CLI before each call,
# unless --yolo is set. stage_file is in-memory only and intentionally NOT here.
# Memory writes are local-only and frequent; we exclude them from confirmation
# so the agent can learn without interrupting the user.
MUTATING_TOOLS: frozenset[str] = frozenset(
    {
        # Databricks
        "create_job",
        "run_job_now",
        "add_job_tasks",
        "upload_notebook",
        # GitHub
        "commit_staged",
        "create_branch",
        "create_pr",
        # Terminal — local filesystem / shell
        "write_file",
        "run_shell",
    }
)


SYSTEM_PROMPT = """\
You are Helios, a CLI agent that operates on a Databricks workspace, a GitHub \
organization, and the user's local terminal through tool calls.

Capabilities:
- Databricks: SQL execution, table health diagnostics, jobs (list/get/create/run/modify), \
job runs and outputs, job permissions.
- GitHub: token introspection, repo/file listing, file fetch, PR status, and a \
staged-commit workflow (stage_file -> commit_staged -> create_pr) plus create_branch.
- Terminal: read_file, list_dir, grep, write_file, run_shell. All paths are jailed under \
the directory the CLI was launched from.
- Adaptive memory: save_memory, update_memory, delete_memory, recall_memory. \
Persists across sessions under ~/.helios/memory/.
- Sub-agents: spawn_agent, list_agents, agent_status, agent_result, wait_for_agent, \
kill_agent. Spawn ephemeral (in-process, dies with the session) or persistent \
(detached subprocess, outlives it).

Guidance:
- Use tools instead of guessing — current data beats training knowledge.
- Be concise. Use tables when comparing items; short paragraphs otherwise. No filler preamble.
- When you need an ID/path you don't have, call a list/search tool first.
- Mutating tools (create_job, run_job_now, add_job_tasks, create_branch, commit_staged, \
create_pr, write_file, run_shell) require human confirmation — the CLI handles that. \
Briefly explain what you'll do BEFORE calling so the user can decide.
- If a tool errors, read the message — don't blindly retry. Adapt or report.

Adaptive memory — learn from the user across sessions:
- The current MEMORY.md index is injected at the top of this system message between \
<memory_index> tags. Treat each line as a pointer; call recall_memory(slug) to read the \
full body when one looks relevant to the current task.
- Save (save_memory) when you learn something that will be useful in FUTURE sessions:
  * type=user — role, expertise, preferences that should tailor your help.
  * type=feedback — corrections you received OR non-obvious approaches the user validated. \
Include a "Why:" line with the reason, and a "How to apply:" line.
  * type=project — ongoing work, initiatives, deadlines, incidents. Convert relative dates \
("next Thursday") to absolute (YYYY-MM-DD) before saving.
  * type=reference — pointers to external systems (Linear board, Slack channel, dashboard URL).
- Do NOT save: code patterns, file paths, architecture, git history, debugging fix recipes — \
those are derivable from the repo. Do not save ephemeral chat state.
- Update or delete a memory the moment you discover it is wrong or outdated. Do not write a \
new memory when an existing one can be updated — check the index first.
- These memory writes are silent (no user confirmation). Be deliberate: a saved memory shapes \
every future session, so save only what is durable and non-obvious.

Sub-agents — when to delegate, and (more importantly) when NOT to:

DON'T spawn a sub-agent for work you could just do yourself in this turn. \
"Call get_job_permissions for each of 967 jobs" is a LOOP, not a delegation \
opportunity — write the loop here. Spawning + polling adds latency, iteration \
cost, and a coordination layer with zero parallelism gain. The model frequently \
over-delegates ("this looks vaguely big, let me spawn") — push back on that \
impulse. Also DON'T spawn for: summarizing content already in your context, or \
any task where you'll immediately block waiting for the result (that's just \
inline work routed through a slower path).

DO spawn ONLY when at least one applies:
  - INDEPENDENT PARALLELISM: multiple units can run concurrently and you'll \
merge later (audit 5 separate workspaces at once, then synthesize).
  - OUTLIVE THE SESSION: hours-long unattended work the user will check \
tomorrow — use mode='persistent'.
  - CONTEXT ISOLATION: you want a sub-agent to tackle something with a fresh \
message history, free of this conversation's framing.

Mechanics when you do spawn:
- Briefly describe what you'll spawn BEFORE calling, so the user can redirect.
- Pick mode deliberately: ephemeral for minutes-scale work; persistent for \
hours-long / come-back-tomorrow work.
- Sub-agents are read-only by default. Set allow_mutations=true ONLY when the \
sub-task genuinely needs writes (sub-agent runs without confirmation).
- spawn_agent returns immediately. After spawning, return to the user with the \
agent_id; do NOT busy-poll agent_status in the same turn. Each agent_status \
call costs one iteration of your loop — repeated polling will exhaust max_iters \
before the worker finishes. If the user explicitly asks you to wait, use \
wait_for_agent (one blocking call covers minutes; if it times out, call again).
- Set max_iters for large-N tasks — rule of thumb ~5 × number of items \
(auditing 100 jobs → max_iters=500). Default 50 is fine for small tasks.
- Tasks MUST be self-contained: sub-agents see no conversation history. Include \
all IDs, paths, and context they need to act cold.
"""


_MEMORY_OPEN = "<memory_index>"
_MEMORY_CLOSE = "</memory_index>"


def build_system_message() -> str:
    """SYSTEM_PROMPT with the current MEMORY.md index injected at the top.

    Called both at session start and at the top of every `run_turn` so that
    memories saved mid-session are visible on the next turn.
    """
    index = _load_memory_index().strip()
    if not index:
        index = "(no memories saved yet)"
    return f"{_MEMORY_OPEN}\n{index}\n{_MEMORY_CLOSE}\n\n{SYSTEM_PROMPT}"


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    # SDK-level retries — silent, exponential backoff with jitter, respects
    # Retry-After. Defaults to 5 here (vs. SDK default 2). Override via env.
    max_retries = _env_int("HELIOS_LLM_MAX_RETRIES", 5)
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        max_retries=max_retries,
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Errors we'll retry on. Excludes auth/bad-request/not-found — those are
# configuration bugs that won't fix themselves with retries.
_RETRYABLE_LLM_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


def _call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    console: Console,
) -> Any:
    """LLM call with a visible outer retry, on top of the SDK's silent inner retries.

    The SDK already retries transient errors. This wrapper catches what the SDK
    gives up on, surfaces a yellow notice in the console (so an interactive user
    sees the pause instead of a frozen prompt), and retries with longer backoff.
    """
    outer_attempts = _env_int("HELIOS_LLM_OUTER_RETRIES", 2) + 1
    last_err: Exception | None = None
    for attempt in range(1, outer_attempts + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages, tools=tools)
        except _RETRYABLE_LLM_ERRORS as e:
            last_err = e
            if attempt >= outer_attempts:
                break
            backoff = min(2 ** attempt + random.random(), 30.0)
            console.print(
                f"[yellow]LLM API {type(e).__name__}: {e}. "
                f"Retrying in {backoff:.1f}s "
                f"(outer attempt {attempt + 1}/{outer_attempts})...[/]"
            )
            time.sleep(backoff)
    assert last_err is not None
    raise last_err


def _model() -> str:
    m = os.environ.get("OPENAI_MODEL")
    if not m:
        raise RuntimeError("OPENAI_MODEL is not set")
    return m


_DEFAULT_MAX_ITERS: int = 50


def _default_max_iters() -> int:
    return _env_int("HELIOS_MAX_ITERS", _DEFAULT_MAX_ITERS)


class AgentCancelled(RuntimeError):
    """Raised inside run_turn when a cancel_event fires — sub-agents catch this
    and mark themselves 'killed' rather than 'error'."""


def _to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style schemas (used by our REGISTRY) to OpenAI tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


def _truncate(s: str, limit: int = 50_000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated, {len(s)} chars total]"


def _confirm(tool_name: str, args: dict[str, Any], yolo: bool, console: Console) -> bool:
    if yolo or tool_name not in MUTATING_TOOLS:
        return True
    console.print(f"\n[bold yellow]MUTATING tool requested:[/] [cyan]{tool_name}[/]")
    console.print(
        Syntax(json.dumps(args, indent=2, default=str), "json", theme="ansi_dark", word_wrap=True)
    )
    return Confirm.ask("Run it?", default=False, console=console)


def run_turn(
    messages: list[dict[str, Any]],
    *,
    yolo: bool = False,
    max_iters: int | None = None,
    console: Console | None = None,
    on_text: Callable[[str], None] | None = None,
    on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
    on_tool_result: Callable[[str, Any], None] | None = None,
    tools_filter: Callable[[str], bool] | None = None,
    cancel_event: threading.Event | None = None,
    message_log_path: Any = None,
) -> list[dict[str, Any]]:
    """Run the agent until the model returns final text. Returns updated messages.

    `max_iters` caps the tool-call loop. None resolves to HELIOS_MAX_ITERS env
    (default 50). Sub-agents working on long task lists can pass higher values.

    `tools_filter`, if given, is called with each tool name; only tools where it
    returns True are exposed to the model. Used by sub-agents to hide mutating
    tools when `allow_mutations` is False.
    """
    if max_iters is None:
        max_iters = _default_max_iters()
    console = console or Console()
    client = _client()
    model = _model()
    schemas = all_schemas()
    if tools_filter is not None:
        schemas = [s for s in schemas if tools_filter(s["name"])]
    tools = _to_openai_tools(schemas)

    # Refresh the system message so memories saved in earlier turns appear here.
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": build_system_message()}

    # Persist the messages array after each iteration so a Ctrl-C / crash
    # leaves enough state on disk for `propose-resume` to continue. Whole-file
    # rewrite per iteration (not per append) — atomic, simple, ~tens of KB.
    def _persist_messages() -> None:
        if not message_log_path:
            return
        try:
            from pathlib import Path as _Path
            _Path(message_log_path).parent.mkdir(parents=True, exist_ok=True)
            _Path(message_log_path).write_text(json.dumps(messages, default=str))
        except Exception:
            pass

    _persist_messages()  # initial snapshot of system+user messages

    for _ in range(max_iters):
        if cancel_event is not None and cancel_event.is_set():
            raise AgentCancelled("agent cancelled before LLM call")
        resp = _call_llm(client, model, messages, tools, console)
        msg = resp.choices[0].message
        if cancel_event is not None and cancel_event.is_set():
            raise AgentCancelled("agent cancelled after LLM call, before tool dispatch")

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # Surface the model's reasoning text on EVERY turn that has any —
        # not just the final no-tool-calls turn. The text immediately before
        # tool calls is the model's "thinking aloud" (why it's about to call
        # this tool); losing it leaves traces with tool I/O only and no
        # rationale. Callers that previously assumed `on_text` = final-only
        # still get the final call as their last invocation.
        if msg.content and on_text:
            on_text(msg.content)

        if not msg.tool_calls:
            _persist_messages()
            return messages

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                result: Any = {"error": f"invalid JSON arguments from model: {e}"}
            else:
                if on_tool_call:
                    on_tool_call(name, args)
                if not _confirm(name, args, yolo, console):
                    result = {"error": "user declined to run this tool"}
                else:
                    try:
                        result = call_tool(name, **args)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}

            if on_tool_result:
                on_tool_result(name, result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _truncate(json.dumps(result, default=str)),
                }
            )
        _persist_messages()

    raise RuntimeError(f"agent exceeded max_iters={max_iters} without a final answer")
