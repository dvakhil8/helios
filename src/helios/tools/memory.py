"""Adaptive memory tools — persistent learnings across Helios sessions.

Self-contained per project convention: this file owns the on-disk layout,
frontmatter parsing, index management, and the four tools the agent uses.

Layout (global, per-user):
    ~/.helios/memory/
      MEMORY.md              # auto-injected into the system prompt every turn
      <slug>.md              # one file per memory, YAML frontmatter + body

Override the root via HELIOS_MEMORY_ROOT (mostly for tests).

Tool surface:
  - save_memory    (MUTATING) — create a new memory file + index entry
  - update_memory  (MUTATING) — partial update of an existing memory
  - delete_memory  (MUTATING) — remove a memory file and its index entry
  - recall_memory  (read)     — return the full body of one memory by slug
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable


VALID_TYPES: frozenset[str] = frozenset({"user", "feedback", "project", "reference"})
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_RESERVED_SLUGS: frozenset[str] = frozenset({"memory", "index"})

_ROOT: Path = Path(
    os.environ.get("HELIOS_MEMORY_ROOT", os.path.expanduser("~/.helios/memory"))
).resolve()
_INDEX_PATH: Path = _ROOT / "MEMORY.md"


# ---- internal helpers --------------------------------------------------------


def _ensure_root() -> None:
    _ROOT.mkdir(parents=True, exist_ok=True)


def _check_slug(slug: str) -> str:
    slug = slug.strip().lower()
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"invalid slug {slug!r}: must match [a-z0-9][a-z0-9_-]{{0,63}}"
        )
    if slug in _RESERVED_SLUGS:
        raise ValueError(f"slug {slug!r} is reserved")
    return slug


def _memory_path(slug: str) -> Path:
    return _ROOT / f"{slug}.md"


def _render(name: str, description: str, type_: str, content: str) -> str:
    body = content.rstrip() + "\n"
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"type: {type_}\n"
        "---\n\n"
        f"{body}"
    )


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (frontmatter, body). Tolerates missing/malformed frontmatter."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    header = text[4:end]
    body = text[end + 4 :].lstrip("\n")
    fm: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm, body


def _read_index() -> str:
    if not _INDEX_PATH.exists():
        return ""
    return _INDEX_PATH.read_text(encoding="utf-8")


def _write_index(text: str) -> None:
    _ensure_root()
    if text and not text.endswith("\n"):
        text += "\n"
    _INDEX_PATH.write_text(text, encoding="utf-8")


def _index_line(slug: str, name: str, description: str) -> str:
    return f"- [{name}]({slug}.md) — {description}"


_INDEX_LINE_RE = re.compile(r"^- \[[^\]]+\]\(([a-z0-9_-]+)\.md\)")


def _index_replace(slug: str, new_line: str | None) -> None:
    """Insert/replace/remove the index entry for `slug`.

    new_line=None deletes the entry; otherwise upsert keeps existing ordering
    and appends new slugs at the end.
    """
    existing = _read_index()
    out_lines: list[str] = []
    found = False
    for line in existing.splitlines():
        m = _INDEX_LINE_RE.match(line)
        if m and m.group(1) == slug:
            found = True
            if new_line is not None:
                out_lines.append(new_line)
            continue
        out_lines.append(line)
    if not found and new_line is not None:
        out_lines.append(new_line)
    _write_index("\n".join(out_lines))


def load_index() -> str:
    """Public: return the current MEMORY.md text, or '' if no memories yet."""
    return _read_index()


# =============================================================================
# save_memory  (MUTATING)
# =============================================================================

SAVE_MEMORY_SCHEMA: dict[str, Any] = {
    "name": "save_memory",
    "description": (
        "Save a new persistent memory under ~/.helios/memory/<slug>.md and add a "
        "pointer line to MEMORY.md. Errors if `slug` already exists — use "
        "update_memory for changes. Pick the most appropriate `type`: user "
        "(profile/preferences), feedback (rules from corrections or validated "
        "approaches — include a 'Why:' line), project (ongoing work/initiatives, "
        "convert relative dates to YYYY-MM-DD), reference (pointers to external "
        "systems). Do NOT save code patterns, file paths, git history, or fixes "
        "— those are derivable from the repo."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": (
                    "Filename stem (no .md). Lowercase, [a-z0-9_-], max 64 chars. "
                    "Use a semantic prefix like 'user_', 'feedback_', 'project_'."
                ),
            },
            "name": {"type": "string", "description": "Short human title for the memory."},
            "description": {
                "type": "string",
                "description": "One-line hook (<150 chars) — decides relevance in future turns.",
            },
            "type": {
                "type": "string",
                "enum": sorted(VALID_TYPES),
                "description": "Memory category — see tool description for guidance.",
            },
            "content": {
                "type": "string",
                "description": "Full memory body in markdown. For feedback/project, lead with the rule/fact then 'Why:' and 'How to apply:' lines.",
            },
        },
        "required": ["slug", "name", "description", "type", "content"],
    },
}


def save_memory(
    slug: str, name: str, description: str, type: str, content: str
) -> dict[str, Any]:
    slug = _check_slug(slug)
    if type not in VALID_TYPES:
        raise ValueError(f"invalid type {type!r}; must be one of {sorted(VALID_TYPES)}")
    _ensure_root()
    path = _memory_path(slug)
    if path.exists():
        raise FileExistsError(
            f"memory {slug!r} already exists — use update_memory to change it"
        )
    path.write_text(_render(name, description, type, content), encoding="utf-8")
    _index_replace(slug, _index_line(slug, name, description))
    return {"slug": slug, "path": str(path), "indexed": True}


# =============================================================================
# update_memory  (MUTATING)
# =============================================================================

UPDATE_MEMORY_SCHEMA: dict[str, Any] = {
    "name": "update_memory",
    "description": (
        "Update an existing memory by slug. Any of name/description/type/content "
        "may be omitted to keep the current value. Refreshes the MEMORY.md index "
        "entry. Errors if the memory does not exist."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Existing memory slug."},
            "name": {"type": "string", "description": "New title (optional)."},
            "description": {"type": "string", "description": "New one-liner (optional)."},
            "type": {
                "type": "string",
                "enum": sorted(VALID_TYPES),
                "description": "New type (optional).",
            },
            "content": {"type": "string", "description": "New body (optional)."},
        },
        "required": ["slug"],
    },
}


def update_memory(
    slug: str,
    name: str | None = None,
    description: str | None = None,
    type: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    slug = _check_slug(slug)
    path = _memory_path(slug)
    if not path.exists():
        raise FileNotFoundError(f"no memory with slug {slug!r}")
    fm, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    new_name = name if name is not None else fm.get("name", slug)
    new_desc = description if description is not None else fm.get("description", "")
    new_type = type if type is not None else fm.get("type", "project")
    if new_type not in VALID_TYPES:
        raise ValueError(f"invalid type {new_type!r}; must be one of {sorted(VALID_TYPES)}")
    new_body = content if content is not None else body
    path.write_text(_render(new_name, new_desc, new_type, new_body), encoding="utf-8")
    _index_replace(slug, _index_line(slug, new_name, new_desc))
    return {
        "slug": slug,
        "path": str(path),
        "changed": [
            field
            for field, value in (
                ("name", name),
                ("description", description),
                ("type", type),
                ("content", content),
            )
            if value is not None
        ],
    }


# =============================================================================
# delete_memory  (MUTATING)
# =============================================================================

DELETE_MEMORY_SCHEMA: dict[str, Any] = {
    "name": "delete_memory",
    "description": (
        "Delete a memory file and remove its line from MEMORY.md. Use when a "
        "memory is wrong, outdated, or the user asks you to forget it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Slug of the memory to delete."},
        },
        "required": ["slug"],
    },
}


def delete_memory(slug: str) -> dict[str, Any]:
    slug = _check_slug(slug)
    path = _memory_path(slug)
    existed = path.exists()
    if existed:
        path.unlink()
    _index_replace(slug, None)
    return {"slug": slug, "deleted": existed}


# =============================================================================
# recall_memory  (read)
# =============================================================================

RECALL_MEMORY_SCHEMA: dict[str, Any] = {
    "name": "recall_memory",
    "description": (
        "Read the full body of a saved memory by slug. The MEMORY.md index is "
        "already in your system prompt — use it to pick which slug to recall."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Slug from the MEMORY.md index."},
        },
        "required": ["slug"],
    },
}


def recall_memory(slug: str) -> dict[str, Any]:
    slug = _check_slug(slug)
    path = _memory_path(slug)
    if not path.exists():
        raise FileNotFoundError(f"no memory with slug {slug!r}")
    text = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    return {
        "slug": slug,
        "name": fm.get("name", slug),
        "description": fm.get("description", ""),
        "type": fm.get("type", ""),
        "content": body,
    }


# =============================================================================
# Registry
# =============================================================================

Tool = tuple[dict[str, Any], Callable[..., Any]]

REGISTRY: dict[str, Tool] = {
    "save_memory": (SAVE_MEMORY_SCHEMA, save_memory),
    "update_memory": (UPDATE_MEMORY_SCHEMA, update_memory),
    "delete_memory": (DELETE_MEMORY_SCHEMA, delete_memory),
    "recall_memory": (RECALL_MEMORY_SCHEMA, recall_memory),
}
