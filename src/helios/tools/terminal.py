"""Local terminal / filesystem tools.

Self-contained per project convention: this file owns its own path-jail config
and helpers. All paths the LLM passes resolve under a single root captured at
import time (defaults to `os.getcwd()`; override via `HELIOS_WORK_ROOT`).

Read-only tools: read_file, list_dir, grep.
Mutating tools (confirmed by the CLI): run_shell, write_file.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable


# ---- Path jail ---------------------------------------------------------------
# Captured once at import time — the agent process can't escape it later by
# chdir'ing. Override by setting HELIOS_WORK_ROOT before launching the CLI.

_ROOT: Path = Path(os.environ.get("HELIOS_WORK_ROOT", os.getcwd())).resolve()


def _resolve(path: str) -> Path:
    """Resolve `path` under `_ROOT`, refusing any escape via .. or symlinks."""
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    resolved = p.resolve()
    try:
        resolved.relative_to(_ROOT)
    except ValueError as e:
        raise PermissionError(
            f"path {resolved} is outside the allowed root {_ROOT}"
        ) from e
    return resolved


def _display(p: Path) -> str:
    """Render `p` relative to root when possible — otherwise absolute."""
    try:
        return str(p.relative_to(_ROOT)) or "."
    except ValueError:
        return str(p)


def _trunc(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated, {len(s)} chars total]"


# =============================================================================
# read_file
# =============================================================================

READ_FILE_SCHEMA: dict[str, Any] = {
    "name": "read_file",
    "description": (
        "Read a local file (decoded as UTF-8 when possible, else base64). "
        "Path is jailed under the CLI's launch directory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file (relative or absolute)."},
            "max_bytes": {
                "type": "integer",
                "default": 200_000,
                "description": "Maximum bytes to return. Content beyond this is truncated.",
            },
        },
        "required": ["path"],
    },
}


def read_file(path: str, max_bytes: int = 200_000) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"not a file: {_display(resolved)}")
    raw = resolved.read_bytes()
    truncated = False
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        truncated = True
    try:
        content = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = base64.b64encode(raw).decode("ascii")
        encoding = "base64"
    return {
        "path": _display(resolved),
        "size_bytes": resolved.stat().st_size,
        "encoding": encoding,
        "truncated": truncated,
        "content": content,
    }


# =============================================================================
# list_dir
# =============================================================================

LIST_DIR_SCHEMA: dict[str, Any] = {
    "name": "list_dir",
    "description": (
        "List files and subdirectories under a path. Set recursive=True to walk "
        "all subdirectories (capped by limit). Path is jailed under the CLI root."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": ".", "description": "Directory to list."},
            "recursive": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 200, "description": "Max entries to return."},
        },
    },
}


def list_dir(path: str = ".", recursive: bool = False, limit: int = 200) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.is_dir():
        raise NotADirectoryError(f"not a directory: {_display(resolved)}")
    iterator = resolved.rglob("*") if recursive else resolved.iterdir()
    entries: list[dict[str, Any]] = []
    truncated = False
    for p in iterator:
        try:
            stat = p.stat()
        except OSError:
            continue
        entries.append(
            {
                "path": _display(p),
                "type": "dir" if p.is_dir() else "file",
                "size": stat.st_size if p.is_file() else None,
            }
        )
        if len(entries) >= limit:
            truncated = True
            break
    return {
        "root": _display(resolved),
        "count": len(entries),
        "truncated": truncated,
        "entries": entries,
    }


# =============================================================================
# grep
# =============================================================================

GREP_SCHEMA: dict[str, Any] = {
    "name": "grep",
    "description": (
        "Search for a regex pattern across files under `path` (file or directory). "
        "Binary files are skipped. Returns matches with path:line:text. Capped by max_results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regex pattern."},
            "path": {
                "type": "string",
                "default": ".",
                "description": "File or directory to search.",
            },
            "max_results": {"type": "integer", "default": 100},
            "ignore_case": {"type": "boolean", "default": False},
        },
        "required": ["pattern"],
    },
}


def grep(
    pattern: str,
    path: str = ".",
    max_results: int = 100,
    ignore_case: bool = False,
) -> dict[str, Any]:
    resolved = _resolve(path)
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        raise ValueError(f"invalid regex: {e}")

    targets = [resolved] if resolved.is_file() else list(resolved.rglob("*"))
    matches: list[dict[str, Any]] = []
    for f in targets:
        if not f.is_file():
            continue
        try:
            with f.open("rb") as fh:
                head = fh.read(4096)
            if b"\0" in head:
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matches.append(
                    {"path": _display(f), "line": lineno, "text": line[:500]}
                )
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break
    return {
        "pattern": pattern,
        "root": _display(resolved),
        "match_count": len(matches),
        "truncated": len(matches) >= max_results,
        "matches": matches,
    }


# =============================================================================
# write_file  (MUTATING)
# =============================================================================

WRITE_FILE_SCHEMA: dict[str, Any] = {
    "name": "write_file",
    "description": (
        "Create or overwrite a local file with UTF-8 text content. Path is jailed under "
        "the CLI root. Creates parent directories by default. This MUTATES local files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write."},
            "content": {"type": "string", "description": "Full file content (UTF-8)."},
            "mkdirs": {
                "type": "boolean",
                "default": True,
                "description": "Create parent directories if missing.",
            },
        },
        "required": ["path", "content"],
    },
}


def write_file(path: str, content: str, mkdirs: bool = True) -> dict[str, Any]:
    resolved = _resolve(path)
    if mkdirs:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    existed = resolved.exists()
    resolved.write_text(content, encoding="utf-8")
    return {
        "path": _display(resolved),
        "bytes_written": len(content.encode("utf-8")),
        "overwrote_existing": existed,
    }


# =============================================================================
# run_shell  (MUTATING)
# =============================================================================

RUN_SHELL_SCHEMA: dict[str, Any] = {
    "name": "run_shell",
    "description": (
        "Execute a shell command (uses /bin/sh -c, so pipes/globs/&&/||/redirects work). "
        "Returns exit_code, stdout, stderr — each truncated to 50KB. Default timeout 60s, "
        "max 600s. cwd defaults to the CLI root and must resolve under it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "cwd": {
                "type": "string",
                "description": "Working directory (relative to CLI root or absolute under it).",
            },
            "timeout": {
                "type": "integer",
                "default": 60,
                "description": "Seconds before the process is killed. Capped at 600.",
            },
        },
        "required": ["command"],
    },
}


def run_shell(command: str, cwd: str | None = None, timeout: int = 60) -> dict[str, Any]:
    timeout = max(1, min(timeout, 600))
    cwd_path = _resolve(cwd) if cwd else _ROOT
    if not cwd_path.is_dir():
        raise NotADirectoryError(f"cwd is not a directory: {_display(cwd_path)}")
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd_path),
        )
        return {
            "command": command,
            "cwd": _display(cwd_path),
            "exit_code": completed.returncode,
            "timed_out": False,
            "stdout": _trunc(completed.stdout or "", 50_000),
            "stderr": _trunc(completed.stderr or "", 50_000),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "command": command,
            "cwd": _display(cwd_path),
            "exit_code": None,
            "timed_out": True,
            "timeout_seconds": timeout,
            "stdout": _trunc(e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), 50_000),
            "stderr": _trunc(e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""), 50_000),
        }


# =============================================================================
# Registry
# =============================================================================

Tool = tuple[dict[str, Any], Callable[..., Any]]

REGISTRY: dict[str, Tool] = {
    "read_file": (READ_FILE_SCHEMA, read_file),
    "list_dir": (LIST_DIR_SCHEMA, list_dir),
    "grep": (GREP_SCHEMA, grep),
    "write_file": (WRITE_FILE_SCHEMA, write_file),
    "run_shell": (RUN_SHELL_SCHEMA, run_shell),
}
