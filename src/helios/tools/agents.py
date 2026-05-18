"""Sub-agent spawning — ephemeral (in-process thread) or persistent (detached subprocess).

Both modes expose the same status/result API; choice of mode is left to the calling
LLM via the `spawn_agent` tool's `mode` parameter.

Layout (persistent only):
    ~/.helios/agents/<agent_id>/
      task.txt           # the prompt
      title.txt          # short label
      status.json        # {agent_id, title, mode, status, started_at, completed_at?,
                         #  pid?, allow_mutations, error?}
      transcript.jsonl   # one event per line: {ts, type: text|tool_call|tool_result|error, data}
      result.md          # final assistant text (written on success)

Ephemeral agents live entirely in the _EPHEMERAL dict; daemon threads die when the
CLI exits.

Depth cap (HELIOS_AGENT_MAX_DEPTH, default 3) limits recursive spawning. Persistent
sub-agents inherit depth via the HELIOS_AGENT_DEPTH env var; ephemeral via
threading.local.
"""

from __future__ import annotations

import errno
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


_PERSIST_ROOT: Path = Path(
    os.environ.get("HELIOS_AGENTS_ROOT", os.path.expanduser("~/.helios/agents"))
).resolve()

_MAX_DEPTH: int = int(os.environ.get("HELIOS_AGENT_MAX_DEPTH", "3"))
_VALID_MODES: frozenset[str] = frozenset({"ephemeral", "persistent"})
_TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "error", "killed"})


# ---- depth tracking ---------------------------------------------------------

_thread_state = threading.local()


def _current_depth() -> int:
    """Depth of the current sub-agent call stack.

    Persistent sub-agents inherit via HELIOS_AGENT_DEPTH env var.
    Ephemeral sub-agents inherit via threading.local (set inside the worker thread).
    The main CLI thread is depth 0.
    """
    env = os.environ.get("HELIOS_AGENT_DEPTH")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    return getattr(_thread_state, "depth", 0)


# ---- helpers ----------------------------------------------------------------


def _new_agent_id() -> str:
    """Lexically-sortable, short, unique enough for our purposes."""
    return f"{int(time.time())}-{secrets.token_hex(3)}"


def _fmt_ts(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _state_dir(agent_id: str) -> Path:
    return _PERSIST_ROOT / agent_id


def _write_status(agent_id: str, status: dict[str, Any]) -> None:
    sd = _state_dir(agent_id)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "status.json").write_text(json.dumps(status, indent=2, default=str))


def _read_status(agent_id: str) -> dict[str, Any] | None:
    p = _state_dir(agent_id) / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _append_persistent_event(agent_id: str, event_type: str, data: Any) -> None:
    sd = _state_dir(agent_id)
    sd.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": time.time(), "type": event_type, "data": data}, default=str)
    with (sd / "transcript.jsonl").open("a") as f:
        f.write(line + "\n")


def _read_transcript(agent_id: str) -> list[dict[str, Any]]:
    p = _state_dir(agent_id) / "transcript.jsonl"
    if not p.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in p.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


# ---- ephemeral registry -----------------------------------------------------


class _EphemeralRecord:
    __slots__ = (
        "agent_id", "title", "task", "allow_mutations", "max_iters",
        "status", "started_at", "completed_at", "error", "result",
        "transcript", "thread", "lock", "cancel_event",
    )

    def __init__(
        self,
        agent_id: str,
        title: str,
        task: str,
        allow_mutations: bool,
        max_iters: int | None,
    ) -> None:
        self.agent_id = agent_id
        self.title = title
        self.task = task
        self.allow_mutations = allow_mutations
        self.max_iters = max_iters
        self.status: str = "starting"
        self.started_at: float = time.time()
        self.completed_at: float | None = None
        self.error: str | None = None
        self.result: str | None = None
        self.transcript: list[dict[str, Any]] = []
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()

    def append_event(self, event_type: str, data: Any) -> None:
        with self.lock:
            self.transcript.append({"ts": time.time(), "type": event_type, "data": data})

    def summary(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "title": self.title,
            "mode": "ephemeral",
            "status": self.status,
            "started_at": _fmt_ts(self.started_at),
            "completed_at": _fmt_ts(self.completed_at),
            "event_count": len(self.transcript),
            "allow_mutations": self.allow_mutations,
            "error": self.error,
        }


_EPHEMERAL: dict[str, _EphemeralRecord] = {}
_EPHEMERAL_LOCK = threading.Lock()


# ---- ephemeral worker -------------------------------------------------------


def _run_ephemeral(record: _EphemeralRecord, parent_depth: int) -> None:
    """Body of an ephemeral sub-agent — runs in a daemon thread."""
    # Lazy import to avoid the tools/__init__ → agent → tools cycle at module load.
    from ..agent import AgentCancelled, MUTATING_TOOLS, build_system_message, run_turn

    _thread_state.depth = parent_depth + 1
    record.status = "running"

    def on_text(t: str) -> None:
        record.append_event("text", t)
        record.result = t

    def on_tool_call(name: str, args: dict[str, Any]) -> None:
        record.append_event("tool_call", {"name": name, "args": args})

    def on_tool_result(name: str, result: Any) -> None:
        record.append_event("tool_result", {"name": name, "result": result})

    tools_filter: Callable[[str], bool] | None = None
    if not record.allow_mutations:
        tools_filter = lambda n: n not in MUTATING_TOOLS

    messages = [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content": record.task},
    ]

    try:
        run_turn(
            messages,
            yolo=True,  # no human at this keyboard; safety comes from tools_filter
            max_iters=record.max_iters,
            on_text=on_text,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            tools_filter=tools_filter,
            cancel_event=record.cancel_event,
        )
        record.status = "done"
    except AgentCancelled:
        record.status = "killed"
        record.append_event("cancelled", {"at": _fmt_ts(time.time())})
    except Exception as e:
        record.status = "error"
        record.error = f"{type(e).__name__}: {e}"
        record.append_event("error", {"traceback": traceback.format_exc()})
    finally:
        record.completed_at = time.time()


# ---- persistent worker (entered via `helios _agent-run AGENT_ID`) -----------


def _persistent_runner(agent_id: str) -> None:
    """Body of a persistent sub-agent — runs in the detached subprocess."""
    from ..agent import MUTATING_TOOLS, build_system_message, run_turn

    sd = _state_dir(agent_id)
    if not sd.exists():
        sys.exit(2)

    # Catch SIGTERM (from kill_agent) and write a clean 'killed' status before exit.
    def _on_sigterm(_signum: int, _frame: Any) -> None:
        st = _read_status(agent_id) or {"agent_id": agent_id}
        st.update({"status": "killed", "completed_at": _fmt_ts(time.time())})
        _write_status(agent_id, st)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_MODEL"):
        status = _read_status(agent_id) or {"agent_id": agent_id}
        status.update(
            {
                "status": "error",
                "completed_at": _fmt_ts(time.time()),
                "error": "OPENAI_API_KEY or OPENAI_MODEL not set in subprocess env",
            }
        )
        _write_status(agent_id, status)
        sys.exit(2)

    task = (sd / "task.txt").read_text()
    status = _read_status(agent_id) or {}
    allow_mutations = bool(status.get("allow_mutations", False))
    max_iters = status.get("max_iters")  # may be None

    status.update(
        {"status": "running", "started_at": _fmt_ts(time.time()), "pid": os.getpid()}
    )
    _write_status(agent_id, status)

    def on_text(t: str) -> None:
        _append_persistent_event(agent_id, "text", t)

    def on_tool_call(name: str, args: dict[str, Any]) -> None:
        _append_persistent_event(agent_id, "tool_call", {"name": name, "args": args})

    def on_tool_result(name: str, result: Any) -> None:
        _append_persistent_event(agent_id, "tool_result", {"name": name, "result": result})

    tools_filter: Callable[[str], bool] | None = None
    if not allow_mutations:
        tools_filter = lambda n: n not in MUTATING_TOOLS

    messages = [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content": task},
    ]

    try:
        result_msgs = run_turn(
            messages,
            yolo=True,
            max_iters=max_iters,
            on_text=on_text,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            tools_filter=tools_filter,
        )
        final_text = ""
        for m in reversed(result_msgs):
            if m.get("role") == "assistant" and m.get("content"):
                final_text = m["content"]
                break
        (sd / "result.md").write_text(final_text)
        status.update({"status": "done", "completed_at": _fmt_ts(time.time())})
        _write_status(agent_id, status)
    except Exception as e:
        _append_persistent_event(agent_id, "error", {"traceback": traceback.format_exc()})
        status.update(
            {
                "status": "error",
                "completed_at": _fmt_ts(time.time()),
                "error": f"{type(e).__name__}: {e}",
            }
        )
        _write_status(agent_id, status)
        sys.exit(1)


# ---- unified read helpers ---------------------------------------------------


def _ephemeral_summary(agent_id: str) -> dict[str, Any] | None:
    with _EPHEMERAL_LOCK:
        rec = _EPHEMERAL.get(agent_id)
    return rec.summary() if rec else None


def _persistent_summary(agent_id: str) -> dict[str, Any] | None:
    status = _read_status(agent_id)
    if status is None:
        return None
    return {
        "agent_id": agent_id,
        "title": status.get("title", ""),
        "mode": "persistent",
        "status": status.get("status", "unknown"),
        "started_at": status.get("started_at"),
        "completed_at": status.get("completed_at"),
        "event_count": sum(1 for _ in (_state_dir(agent_id) / "transcript.jsonl").open())
        if (_state_dir(agent_id) / "transcript.jsonl").exists()
        else 0,
        "allow_mutations": bool(status.get("allow_mutations", False)),
        "pid": status.get("pid"),
        "error": status.get("error"),
    }


# =============================================================================
# spawn_agent
# =============================================================================

SPAWN_AGENT_SCHEMA: dict[str, Any] = {
    "name": "spawn_agent",
    "description": (
        "Spawn a sub-agent that works on `task` autonomously in the background. "
        "Returns an agent_id immediately.\n\n"
        "DON'T USE spawn_agent FOR:\n"
        "  - Sequential tool calls you could just do in your own loop ('call X "
        "for each of N items'). Spawning + polling adds overhead and iteration "
        "cost with ZERO parallelism gain. Write the loop here, in this turn.\n"
        "  - Summarizing or restructuring content already in your context.\n"
        "  - Anything where you'll immediately block waiting on the result — "
        "that's inline work routed through a slower indirection.\n\n"
        "USE spawn_agent ONLY when at least one applies:\n"
        "  - INDEPENDENT PARALLELISM: multiple units run concurrently and you "
        "merge later (audit 5 separate workspaces at once, then synthesize).\n"
        "  - OUTLIVE THE SESSION: hours-long unattended work the user checks "
        "tomorrow — use mode='persistent'.\n"
        "  - CONTEXT ISOLATION: you want a fresh message history for a sub-task, "
        "free of this conversation's framing.\n\n"
        "After spawning, return to the user with the agent_id; do NOT busy-poll "
        "agent_status. Use wait_for_agent if the user explicitly asks you to "
        "wait (one tool call covers minutes of work). Otherwise fetch via "
        "agent_result when you next check in.\n\n"
        "Modes:\n"
        "  - 'ephemeral' (default): in-process. Dies when this CLI session ends. "
        "Minutes-scale work, fan-out searches, side investigations.\n"
        "  - 'persistent': detached subprocess. SURVIVES /quit and closing the "
        "terminal. State at ~/.helios/agents/<agent_id>/. Hours-long or "
        "come-back-tomorrow work only.\n\n"
        "Sub-agents have the full read-only tool surface plus memory. They CANNOT "
        "call mutating tools unless you set allow_mutations=true (defaults false, "
        "the safe choice — they run unattended and mutations cannot be confirmed).\n\n"
        "`max_iters` caps the sub-agent's tool-call loop. Default 50 is enough "
        "for ~10 tool calls and synthesis. SET HIGHER for large-N tasks — rule "
        "of thumb ~5 × number of items (50 jobs to audit → 250). Too low and "
        "the agent dies mid-task with 'exceeded max_iters'.\n\n"
        "`task` MUST be self-contained — the sub-agent has no memory of this "
        "conversation. Include any IDs, paths, or context it needs to act cold."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Full task for the sub-agent. Self-contained, no implicit context.",
            },
            "title": {
                "type": "string",
                "description": "Short label shown in list_agents (e.g. 'audit job permissions').",
            },
            "mode": {
                "type": "string",
                "enum": sorted(_VALID_MODES),
                "default": "ephemeral",
            },
            "allow_mutations": {
                "type": "boolean",
                "default": False,
                "description": (
                    "If true, the sub-agent can call mutating tools without "
                    "confirmation. Defaults false (read-only)."
                ),
            },
            "max_iters": {
                "type": "integer",
                "description": (
                    "Cap on the sub-agent's tool-call iterations. Omit for the "
                    "default (HELIOS_MAX_ITERS env, fallback 50). Raise for "
                    "large-N tasks: rule of thumb ~5 × number of items."
                ),
            },
        },
        "required": ["task", "title"],
    },
}


def spawn_agent(
    task: str,
    title: str,
    mode: str = "ephemeral",
    allow_mutations: bool = False,
    max_iters: int | None = None,
) -> dict[str, Any]:
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; must be one of {sorted(_VALID_MODES)}")
    depth = _current_depth()
    if depth >= _MAX_DEPTH:
        raise RuntimeError(
            f"refusing to spawn: agent depth {depth} >= max {_MAX_DEPTH}. "
            f"Set HELIOS_AGENT_MAX_DEPTH to raise the cap."
        )

    agent_id = _new_agent_id()

    if mode == "ephemeral":
        rec = _EphemeralRecord(agent_id, title, task, allow_mutations, max_iters)
        with _EPHEMERAL_LOCK:
            _EPHEMERAL[agent_id] = rec
        t = threading.Thread(
            target=_run_ephemeral,
            args=(rec, depth),
            daemon=True,
            name=f"helios-agent-{agent_id}",
        )
        rec.thread = t
        t.start()
        return {"agent_id": agent_id, "mode": "ephemeral", "status": "starting"}

    # persistent
    sd = _state_dir(agent_id)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "task.txt").write_text(task)
    (sd / "title.txt").write_text(title)
    _write_status(
        agent_id,
        {
            "agent_id": agent_id,
            "title": title,
            "mode": "persistent",
            "status": "starting",
            "started_at": _fmt_ts(time.time()),
            "allow_mutations": allow_mutations,
            "max_iters": max_iters,
        },
    )
    env = {**os.environ, "HELIOS_AGENT_DEPTH": str(depth + 1)}
    subprocess.Popen(
        [sys.executable, "-m", "helios.cli", "_agent-run", agent_id],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
        cwd=os.getcwd(),
    )
    return {
        "agent_id": agent_id,
        "mode": "persistent",
        "status": "starting",
        "state_dir": str(sd),
    }


# =============================================================================
# list_agents
# =============================================================================

LIST_AGENTS_SCHEMA: dict[str, Any] = {
    "name": "list_agents",
    "description": (
        "List sub-agents (ephemeral + persistent) with their status. Optional "
        "filter by status (starting/running/done/error) or mode."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["starting", "running", "done", "error"],
            },
            "mode": {"type": "string", "enum": sorted(_VALID_MODES)},
            "limit": {"type": "integer", "default": 50},
        },
    },
}


def list_agents(
    status: str | None = None,
    mode: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with _EPHEMERAL_LOCK:
        for rec in _EPHEMERAL.values():
            rows.append(rec.summary())
    if _PERSIST_ROOT.exists():
        for d in _PERSIST_ROOT.iterdir():
            if not d.is_dir():
                continue
            summary = _persistent_summary(d.name)
            if summary:
                rows.append(summary)
    if status:
        rows = [r for r in rows if r.get("status") == status]
    if mode:
        rows = [r for r in rows if r.get("mode") == mode]
    rows.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    truncated = len(rows) > limit
    return {
        "count": min(len(rows), limit),
        "total": len(rows),
        "truncated": truncated,
        "agents": rows[:limit],
    }


# =============================================================================
# agent_status
# =============================================================================

AGENT_STATUS_SCHEMA: dict[str, Any] = {
    "name": "agent_status",
    "description": (
        "Get the current status of one sub-agent. Includes counts and the last "
        "few transcript events for a quick read on what it's doing right now. "
        "Cheap — use this to poll without fetching the full transcript."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string"},
            "tail_events": {
                "type": "integer",
                "default": 5,
                "description": "Number of most-recent transcript events to include.",
            },
        },
        "required": ["agent_id"],
    },
}


def agent_status(agent_id: str, tail_events: int = 5) -> dict[str, Any]:
    summary = _ephemeral_summary(agent_id)
    transcript: list[dict[str, Any]]
    if summary is not None:
        with _EPHEMERAL_LOCK:
            rec = _EPHEMERAL.get(agent_id)
            transcript = list(rec.transcript) if rec else []
    else:
        summary = _persistent_summary(agent_id)
        if summary is None:
            raise KeyError(f"no agent with id {agent_id!r}")
        transcript = _read_transcript(agent_id)
    summary["tail"] = transcript[-tail_events:] if tail_events > 0 else []
    return summary


# =============================================================================
# agent_result
# =============================================================================

AGENT_RESULT_SCHEMA: dict[str, Any] = {
    "name": "agent_result",
    "description": (
        "Fetch the final result of a sub-agent. Errors if the agent is still "
        "running — call agent_status first. Set include_transcript=true to also "
        "return the full event log (can be large)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string"},
            "include_transcript": {"type": "boolean", "default": False},
        },
        "required": ["agent_id"],
    },
}


def agent_result(agent_id: str, include_transcript: bool = False) -> dict[str, Any]:
    # Ephemeral path
    with _EPHEMERAL_LOCK:
        rec = _EPHEMERAL.get(agent_id)
    if rec is not None:
        if rec.status not in _TERMINAL_STATUSES:
            raise RuntimeError(
                f"agent {agent_id!r} is still {rec.status} — call agent_status to poll"
            )
        out: dict[str, Any] = {
            "agent_id": agent_id,
            "status": rec.status,
            "result": rec.result,
            "error": rec.error,
        }
        if include_transcript:
            with _EPHEMERAL_LOCK:
                out["transcript"] = list(rec.transcript)
        return out

    # Persistent path
    status = _read_status(agent_id)
    if status is None:
        raise KeyError(f"no agent with id {agent_id!r}")
    if status.get("status") not in _TERMINAL_STATUSES:
        raise RuntimeError(
            f"agent {agent_id!r} is still {status.get('status')} — call agent_status to poll"
        )
    result_path = _state_dir(agent_id) / "result.md"
    result_text = result_path.read_text() if result_path.exists() else None
    out = {
        "agent_id": agent_id,
        "status": status.get("status"),
        "result": result_text,
        "error": status.get("error"),
    }
    if include_transcript:
        out["transcript"] = _read_transcript(agent_id)
    return out


# =============================================================================
# wait_for_agent
# =============================================================================

WAIT_FOR_AGENT_SCHEMA: dict[str, Any] = {
    "name": "wait_for_agent",
    "description": (
        "Block until the sub-agent reaches a terminal state (done/error) or "
        "`timeout` seconds elapse. One tool call covers minutes of background "
        "work, so prefer this over a polling loop of agent_status. If the agent "
        "finishes in time you get the same payload as agent_result; otherwise "
        "you get the latest summary with `timed_out: true` and can call again."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string"},
            "timeout": {
                "type": "integer",
                "default": 60,
                "description": "Seconds to wait. Capped at 300 (5 min).",
            },
        },
        "required": ["agent_id"],
    },
}


def wait_for_agent(agent_id: str, timeout: int = 60) -> dict[str, Any]:
    timeout = max(1, min(timeout, 300))
    deadline = time.time() + timeout
    poll = 0.5
    while True:
        summary = _ephemeral_summary(agent_id) or _persistent_summary(agent_id)
        if summary is None:
            raise KeyError(f"no agent with id {agent_id!r}")
        if summary["status"] in _TERMINAL_STATUSES:
            return agent_result(agent_id)
        if time.time() >= deadline:
            with _EPHEMERAL_LOCK:
                rec = _EPHEMERAL.get(agent_id)
            transcript = (
                list(rec.transcript) if rec is not None else _read_transcript(agent_id)
            )
            summary["tail"] = transcript[-5:]
            summary["timed_out"] = True
            return summary
        time.sleep(poll)
        poll = min(poll * 1.5, 5.0)  # gentle backoff


# =============================================================================
# kill_agent
# =============================================================================

KILL_AGENT_SCHEMA: dict[str, Any] = {
    "name": "kill_agent",
    "description": (
        "Terminate a sub-agent. Semantics differ by mode:\n"
        "  - ephemeral: cooperative cancel. Signals the worker thread to stop at "
        "the NEXT iteration of its loop. CANNOT interrupt an in-flight LLM HTTP "
        "call — the thread will exit only after the current call returns "
        "(typically <30s). No-op if the agent has already finished.\n"
        "  - persistent: sends SIGTERM to the subprocess. If still alive after "
        "`grace` seconds, escalates to SIGKILL. Status becomes 'killed'.\n\n"
        "Already-terminal agents (done/error/killed) are returned as-is. Returns "
        "the agent's current summary plus `kill_method` describing what was done."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agent_id": {"type": "string"},
            "grace": {
                "type": "integer",
                "default": 5,
                "description": (
                    "Seconds to wait for graceful termination. Persistent: escalates "
                    "to SIGKILL after this. Ephemeral: waits this long before returning."
                ),
            },
        },
        "required": ["agent_id"],
    },
}


def _wait_for_terminal(agent_id: str, grace: float) -> dict[str, Any] | None:
    """Poll until agent reaches a terminal status or grace elapses. Returns
    the most recent summary (terminal or not), or None if the agent vanished."""
    deadline = time.time() + grace
    poll = 0.2
    while True:
        summary = _ephemeral_summary(agent_id) or _persistent_summary(agent_id)
        if summary is None:
            return None
        if summary["status"] in _TERMINAL_STATUSES or time.time() >= deadline:
            return summary
        time.sleep(poll)
        poll = min(poll * 1.5, 1.0)


def _signal_pid(pid: int, sig: int) -> str | None:
    """Send `sig` to `pid`. Returns None on success, or a human-readable error."""
    try:
        os.kill(pid, sig)
        return None
    except ProcessLookupError:
        return "process not found (already exited)"
    except PermissionError:
        return "permission denied"
    except OSError as e:
        if e.errno == errno.ESRCH:
            return "process not found (already exited)"
        return f"OSError: {e}"


def kill_agent(agent_id: str, grace: int = 5) -> dict[str, Any]:
    grace = max(0, min(grace, 60))

    # Already-terminal short-circuit.
    summary = _ephemeral_summary(agent_id) or _persistent_summary(agent_id)
    if summary is None:
        raise KeyError(f"no agent with id {agent_id!r}")
    if summary["status"] in _TERMINAL_STATUSES:
        summary["kill_method"] = "noop_already_terminal"
        return summary

    # Ephemeral: cooperative cancel.
    with _EPHEMERAL_LOCK:
        rec = _EPHEMERAL.get(agent_id)
    if rec is not None:
        rec.cancel_event.set()
        final = _wait_for_terminal(agent_id, grace) or summary
        final["kill_method"] = "cooperative_cancel"
        if final["status"] not in _TERMINAL_STATUSES:
            final["note"] = (
                "thread did not exit within grace — likely blocked on an LLM "
                "HTTP call; it will stop on the next iteration boundary"
            )
        return final

    # Persistent: SIGTERM, escalate to SIGKILL.
    status = _read_status(agent_id)
    if status is None:
        raise KeyError(f"no agent with id {agent_id!r}")
    pid = status.get("pid")
    if not pid:
        # Spawned but never reached "running" — just mark killed.
        status.update({"status": "killed", "completed_at": _fmt_ts(time.time())})
        _write_status(agent_id, status)
        status["kill_method"] = "marked_killed_no_pid"
        return status

    term_err = _signal_pid(pid, signal.SIGTERM)

    # If the process is already gone, no point escalating — just mark killed
    # in our records and return.
    if term_err and "not found" in term_err:
        status.update({"status": "killed", "completed_at": _fmt_ts(time.time())})
        _write_status(agent_id, status)
        status["kill_method"] = "process_already_gone"
        status["sigterm_error"] = term_err
        return status

    method = "sigterm"
    final = _wait_for_terminal(agent_id, grace) or status
    if final["status"] not in _TERMINAL_STATUSES:
        # Subprocess didn't update status.json on its own — likely stuck. Escalate.
        kill_err = _signal_pid(pid, signal.SIGKILL)
        method = "sigkill_after_sigterm"
        time.sleep(0.5)
        status = _read_status(agent_id) or status
        if status.get("status") not in _TERMINAL_STATUSES:
            status.update({"status": "killed", "completed_at": _fmt_ts(time.time())})
            if kill_err:
                status["kill_error"] = kill_err
            _write_status(agent_id, status)
        final = status
    final["kill_method"] = method
    if term_err:
        final["sigterm_error"] = term_err
    return final


# =============================================================================
# Registry
# =============================================================================

Tool = tuple[dict[str, Any], Callable[..., Any]]

REGISTRY: dict[str, Tool] = {
    "spawn_agent": (SPAWN_AGENT_SCHEMA, spawn_agent),
    "list_agents": (LIST_AGENTS_SCHEMA, list_agents),
    "agent_status": (AGENT_STATUS_SCHEMA, agent_status),
    "agent_result": (AGENT_RESULT_SCHEMA, agent_result),
    "wait_for_agent": (WAIT_FOR_AGENT_SCHEMA, wait_for_agent),
    "kill_agent": (KILL_AGENT_SCHEMA, kill_agent),
}
