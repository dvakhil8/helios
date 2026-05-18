"""Tool registry — merges per-integration REGISTRYs into a single dispatch table.

To add another integration: create `tools/<system>.py` exposing a `REGISTRY` dict
of {tool_name: (schema, handler)}, then import-merge it below.
"""

from __future__ import annotations

from typing import Any, Callable

from . import agents, databricks, github, memory, terminal

Tool = tuple[dict[str, Any], Callable[..., Any]]


def _merge(*registries: dict[str, Tool]) -> dict[str, Tool]:
    out: dict[str, Tool] = {}
    for r in registries:
        dups = sorted(set(out) & set(r))
        if dups:
            raise RuntimeError(f"Tool name collision across integrations: {dups}")
        out.update(r)
    return out


# Ordered for display purposes (banner, /tools listing). Insertion order is the
# preferred display order. Adding a new integration: append one entry here and
# the banner/registry pick it up automatically.
INTEGRATIONS: dict[str, dict[str, Tool]] = {
    "Databricks": databricks.REGISTRY,
    "GitHub": github.REGISTRY,
    "Terminal": terminal.REGISTRY,
    "Memory": memory.REGISTRY,
    "Sub-agents": agents.REGISTRY,
}

REGISTRY: dict[str, Tool] = _merge(*INTEGRATIONS.values())


def all_schemas() -> list[dict[str, Any]]:
    return [schema for schema, _ in REGISTRY.values()]


def call_tool(tool_name: str, /, **kwargs: Any) -> Any:
    """Dispatch a tool call. `tool_name` is positional-only so tools whose own
    schemas include a `name` parameter (e.g. save_memory) can still be invoked
    as call_tool('save_memory', name='...', ...)."""
    if tool_name not in REGISTRY:
        raise KeyError(f"Unknown tool: {tool_name}")
    _, fn = REGISTRY[tool_name]
    return fn(**kwargs)


__all__ = ["REGISTRY", "INTEGRATIONS", "all_schemas", "call_tool"]
