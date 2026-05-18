"""Helios CLI — LLM-driven agent over Databricks, GitHub, and terminal tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import MUTATING_TOOLS, build_system_message, run_turn
from .tools import INTEGRATIONS, REGISTRY, all_schemas


_HISTORY_PATH: Path = Path.home() / ".helios" / "history"
# prompt_toolkit handles ANSI/width correctly, so escapes don't need \001..\002 wrappers.
_CHAT_PROMPT_ANSI: ANSI = ANSI("\033[1;36m>\033[0m ")


_CHAT_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/tools", "list every tool with description"),
    ("/clear", "reset the conversation history"),
    ("/yolo", "toggle auto-approval of mutating tools"),
    ("/quit", "exit the REPL (also Ctrl-D)"),
)


class _SlashCompleter(Completer):
    """Live dropdown for slash commands — only fires when the line starts with '/'.

    Triggers automatically on every keystroke via complete_while_typing=True,
    so the user sees the menu the instant they type '/'. Non-slash input
    yields no completions (the agent prompt should feel like a free-text box).
    """

    def __init__(self, commands: tuple[tuple[str, str], ...]) -> None:
        self.commands = commands

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in self.commands:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )


def _make_chat_session() -> PromptSession[str]:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        history=FileHistory(str(_HISTORY_PATH)),
        completer=_SlashCompleter(_CHAT_COMMANDS),
        complete_while_typing=True,
    )


def _print_banner(yolo: bool) -> None:
    """Pretty startup card for `helios chat` — brand, env, tools, commands."""
    model = os.environ.get("OPENAI_MODEL", "—")
    total_tools = sum(len(reg) for reg in INTEGRATIONS.values())

    title = Text("☀  HELIOS  ☀", style="bold bright_yellow", justify="center")
    tagline = Text(
        "LLM-driven agent · Databricks · GitHub · Terminal · Memory",
        style="italic dim",
        justify="center",
    )

    env = Text(justify="center")
    env.append("model ", style="dim")
    env.append(model, style="white")
    env.append("     yolo ", style="dim")
    env.append("on" if yolo else "off", style="bold green" if yolo else "dim")
    env.append("     memory ", style="dim")
    env.append("~/.helios/memory/", style="white")

    tools_tbl = Table(
        title="Tools",
        title_style="bold cyan",
        box=None,
        show_header=False,
        pad_edge=False,
        padding=(0, 1),
    )
    tools_tbl.add_column(style="cyan", no_wrap=True)
    tools_tbl.add_column(style="bright_yellow", justify="right", no_wrap=True)
    for name, reg in INTEGRATIONS.items():
        tools_tbl.add_row(name, str(len(reg)))
    tools_tbl.add_row(Text("total", style="dim"), Text(str(total_tools), style="bold"))

    cmds_tbl = Table(
        title="Commands",
        title_style="bold cyan",
        box=None,
        show_header=False,
        pad_edge=False,
        padding=(0, 1),
    )
    cmds_tbl.add_column(style="bright_green", no_wrap=True)
    cmds_tbl.add_column(style="dim")
    for cmd, desc in _CHAT_COMMANDS:
        cmds_tbl.add_row(cmd, desc)

    body = Group(
        title,
        tagline,
        Text(),
        env,
        Text(),
        Columns([tools_tbl, cmds_tbl], padding=(0, 6), align="center", expand=True),
    )
    console.print(Panel(body, border_style="yellow", padding=(1, 2)))


app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Helios — LLM-driven agent for Databricks, GitHub, and your terminal. Speaks OpenAI-compatible APIs.",
)
console = Console()


def _short(v: Any, limit: int = 60) -> str:
    s = str(v)
    return s if len(s) <= limit else s[:limit] + "..."


def _make_callbacks() -> tuple:
    def on_tool_call(name: str, args: dict[str, Any]) -> None:
        arg_str = ", ".join(f"{k}={_short(v)!r}" for k, v in args.items())
        console.print(f"[dim]  → calling[/] [cyan]{name}[/][dim]({arg_str})[/]")

    def on_tool_result(name: str, result: Any) -> None:
        text = json.dumps(result, default=str)
        if len(text) > 400:
            text = text[:400] + f"...[+{len(text) - 400} chars]"
        console.print(f"[dim]  ← {name} →[/] [dim]{text}[/]")

    def on_text(s: str) -> None:
        console.print()
        console.print(Panel(Markdown(s), border_style="green", padding=(0, 1)))

    return on_text, on_tool_call, on_tool_result


def _load_dotenv(path: str = ".env") -> None:
    """Best-effort: load KEY=VALUE pairs from .env into os.environ.

    Skips comments, blank lines, and keys that are already set in the environment
    (existing env vars win — useful for `OPENAI_API_KEY=... helios ...` overrides).
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip a single matching pair of surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)


def _ensure_env() -> None:
    _load_dotenv()
    missing = [v for v in ("OPENAI_API_KEY", "OPENAI_MODEL") if not os.environ.get(v)]
    if missing:
        console.print(f"[red]Missing env vars: {', '.join(missing)}[/]")
        console.print("Set them in [bold].env[/] (see [bold].env.example[/]).")
        raise typer.Exit(code=2)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question or task for the agent."),
    yolo: bool = typer.Option(
        False, "--yolo", help="Auto-approve all mutating tool calls. Use with care."
    ),
) -> None:
    """One-shot: send a single question and print the answer."""
    _ensure_env()
    on_text, on_tc, on_tr = _make_callbacks()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_message()},
        {"role": "user", "content": question},
    ]
    run_turn(
        messages,
        yolo=yolo,
        console=console,
        on_text=on_text,
        on_tool_call=on_tc,
        on_tool_result=on_tr,
    )


@app.command()
def chat(
    yolo: bool = typer.Option(False, "--yolo", help="Auto-approve mutating tools for the session."),
) -> None:
    """Interactive REPL. Commands: /quit, /tools, /clear, /yolo."""
    _ensure_env()
    on_text, on_tc, on_tr = _make_callbacks()
    messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_message()}]
    _print_banner(yolo)
    session = _make_chat_session()

    while True:
        try:
            console.print()  # blank line above the prompt
            line = session.prompt(_CHAT_PROMPT_ANSI).strip()
        except EOFError:
            console.print()
            return
        except KeyboardInterrupt:
            # Match modern REPLs: Ctrl-C clears the line and re-prompts; Ctrl-D exits.
            continue

        if not line:
            continue
        if line in ("/quit", "/exit", ":q"):
            return
        if line == "/clear":
            messages = [{"role": "system", "content": build_system_message()}]
            console.print("[dim](history cleared)[/]")
            continue
        if line == "/yolo":
            yolo = not yolo
            console.print(f"[dim]yolo {'on' if yolo else 'off'}[/]")
            continue
        if line == "/tools":
            _print_tools()
            continue

        messages.append({"role": "user", "content": line})
        try:
            messages = run_turn(
                messages,
                yolo=yolo,
                console=console,
                on_text=on_text,
                on_tool_call=on_tc,
                on_tool_result=on_tr,
            )
        except Exception as e:
            console.print(f"[red]error:[/] {e}")


@app.command()
def tools() -> None:
    """List all available tools."""
    _print_tools()


eval_app = typer.Typer(help="Run eval fixtures against the Helios agent.")
app.add_typer(eval_app, name="eval")


@eval_app.command("list")
def eval_list() -> None:
    """List available fixtures."""
    from .evals.fixtures import list_fixtures

    ids = list_fixtures()
    if not ids:
        console.print("[dim]no fixtures found under evals/fixtures/[/]")
        return
    for fid in ids:
        console.print(f"  [cyan]{fid}[/]")


@eval_app.command("run")
def eval_run(
    fixture_id: str = typer.Argument(..., help="Fixture id (directory name under evals/fixtures/)."),
    refresh_baseline: bool = typer.Option(
        False, "--refresh-baseline", help="Re-run the orig job and overwrite the cached baseline."
    ),
    keep_artifacts: bool = typer.Option(
        False, "--keep-artifacts",
        help="Skip teardown so the scratch schema, jobs, and notebooks remain "
             "for inspection. Agent-failure runs always keep artifacts regardless.",
    ),
) -> None:
    """Run one fixture end-to-end. Writes trace + scores to evals/results/<run_id>/."""
    _ensure_env()
    from .evals.harness import run as harness_run

    harness_run(
        fixture_id,
        refresh_baseline=refresh_baseline,
        keep_artifacts=keep_artifacts,
        console=console,
    )


@app.command("propose-resume")
def propose_resume_cmd(
    run_id: str = typer.Argument(..., help="The propose run_id to resume."),
) -> None:
    """Resume an interrupted propose run from its last saved message state.

    Continues the agent's LLM loop where it stopped — message history,
    pending tool calls, baseline, and sandbox job context are all restored
    from disk. After the agent terminates, the harness triggers the final
    optimized run and emits proposal.md.

    Requires: clone.json + messages.json from the original run (auto-
    persisted by `propose` from the v3 harness onward).
    """
    _ensure_env()
    from .evals.propose import resume

    resume(run_id=run_id, console=console)


@app.command("propose-finalize")
def propose_finalize_cmd(
    run_id: str = typer.Argument(..., help="The propose run_id whose proposal.md you want to generate (or regenerate)."),
) -> None:
    """Generate proposal.md for an in-progress / interrupted propose run.

    Use this when the agent ran out of LLM budget mid-iteration, you cancelled
    the harness, or you want the latest score without waiting for the agent to
    converge. Picks the latest SUCCESS sandbox run for this run_id, scores it
    against current prod via `diff_tables`, and writes proposal.md.
    """
    _ensure_env()
    from .evals.propose import finalize

    finalize(run_id=run_id, console=console)


@app.command("propose")
def propose_cmd(
    prod_job_id: int = typer.Argument(..., help="The PROD job_id to derive the proposal from. NEVER modified."),
    task_key: str = typer.Option(..., "--task-key", "-t", help="task_key inside the prod job to scope optimization to."),
    samples: int = typer.Option(10, "--samples", help="How many recent successful task runs to pull for baseline median."),
) -> None:
    """Generate an optimization proposal for ONE task of a real prod job.

    Reads the prod job's spec + notebook source (from GitHub), clones the
    chosen task into a sandbox catalog, lets the agent optimize the clone,
    and writes a `proposal.md` document. The original prod job is NEVER
    modified — any agent call that targets it is hard-rejected.
    """
    _ensure_env()
    from .evals.propose import propose

    propose(prod_job_id=prod_job_id, task_key=task_key, console=console, samples_to_pull=samples)


@eval_app.command("cleanup")
def eval_cleanup(
    run_id: str = typer.Argument(..., help="Run id (the 12-char hex from `helios eval run` output)."),
) -> None:
    """Tear down artifacts (jobs, schema, workspace folder) for a given run_id."""
    _ensure_env()
    from .evals.sandbox import cleanup_by_run_id

    r = cleanup_by_run_id(run_id)
    console.print(f"  jobs deleted:            {r['jobs_deleted']}")
    console.print(f"  schemas dropped:         {r['schemas_dropped']}")
    console.print(f"  workspace dirs deleted:  {r['workspace_dirs_deleted']}")


@app.command(name="_agent-run", hidden=True)
def _agent_run(
    agent_id: str = typer.Argument(..., help="Internal: id of the persistent sub-agent."),
) -> None:
    """Internal: execute a persistent sub-agent. Spawned by spawn_agent, not for direct use."""
    _load_dotenv()  # parent's cwd was inherited via Popen(cwd=)
    from .tools.agents import _persistent_runner

    _persistent_runner(agent_id)


def _print_tools() -> None:
    table = Table(show_header=True, header_style="bold", show_lines=False, box=None, pad_edge=False)
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Kind", style="yellow", no_wrap=True)
    table.add_column("Description", overflow="fold")
    for s in all_schemas():
        kind = "WRITE" if s["name"] in MUTATING_TOOLS else "read"
        first_line = s["description"].splitlines()[0].rstrip(".")
        table.add_row(s["name"], kind, first_line)
    console.print(table)
    console.print(f"\n[dim]Total: {len(REGISTRY)} tools[/]")


if __name__ == "__main__":
    app()
