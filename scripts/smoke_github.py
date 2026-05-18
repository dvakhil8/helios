"""Smoke test for helios GitHub tools.

Exercises the 5 read-only tools end-to-end (chaining IDs/paths discovered from
earlier calls) and `stage_file`, which only mutates an in-memory dict, not GitHub.

Write tools (`commit_staged`, `create_branch`, `create_pr`) are NOT exercised —
they need an explicit sandbox target.

Required env var:
  GITHUB_TOKEN   (or GH_TOKEN)
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from helios.tools import call_tool


PASS: list[str] = []
FAIL: list[tuple[str, str]] = []
SKIP: list[tuple[str, str]] = []


def section(t: str) -> None:
    print(f"\n{'=' * 72}\n {t}\n{'=' * 72}")


def show(o: Any, limit: int = 2000) -> None:
    s = json.dumps(o, indent=2, default=str)
    print(s if len(s) <= limit else s[:limit] + f"\n... (truncated, {len(s)} chars)")


def call(label: str, name: str, **kw: Any) -> Any | None:
    section(f"{label}  ::  {name}({', '.join(f'{k}={v!r}' for k, v in kw.items())})")
    try:
        r = call_tool(name, **kw)
        show(r)
        PASS.append(label)
        return r
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        FAIL.append((label, f"{type(e).__name__}: {e}"))
        return None


def skip(label: str, reason: str) -> None:
    section(f"{label}  ::  SKIPPED — {reason}")
    SKIP.append((label, reason))


def main() -> int:
    if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")):
        print("Missing GITHUB_TOKEN (or GH_TOKEN)", file=sys.stderr)
        return 2

    # 1. check_token
    call("check_token", "check_token")

    # 2. list_repos
    repos_result = call("list_repos", "list_repos", limit=5)
    repos = (repos_result or {}).get("repos", []) if isinstance(repos_result, dict) else []
    if not repos:
        for t in ("list_files:root", "list_files:recursive", "get_file", "stage_file:new", "stage_file:update", "get_pr_status"):
            skip(t, "no repos accessible")
        return finish()

    repo = repos[0]["full_name"]
    branch = repos[0]["default_branch"]
    print(f"\n(using repo='{repo}' branch='{branch}' for downstream tests)")

    # 3. list_files at root (non-recursive)
    files_result = call("list_files:root", "list_files", repo=repo, limit=20)
    entries = (files_result or {}).get("entries", []) if isinstance(files_result, dict) else []

    # 4. list_files recursive — use small limit just to validate the code path
    call("list_files:recursive", "list_files", repo=repo, recursive=True, limit=25)

    # 5. get_file — pick a text-ish file in root
    file_entry = None
    for e in entries:
        if e["type"] == "file" and e["path"].lower().endswith(
            (".md", ".txt", ".py", ".yml", ".yaml", ".json", ".toml", ".cfg")
        ):
            file_entry = e
            break
    if not file_entry:
        file_entry = next((e for e in entries if e["type"] == "file"), None)
    if file_entry:
        call("get_file", "get_file", repo=repo, path=file_entry["path"], max_bytes=5000)
    else:
        skip("get_file", "no file in root of first repo")

    # 6. stage_file — safe (in-memory only). Stage twice to exercise both
    # "new file" and "update existing-staged" code paths.
    call(
        "stage_file:new",
        "stage_file",
        repo=repo,
        branch=branch,
        path="HELIOS_SMOKE_PROBE.md",
        content="# smoke test probe\nv1\n",
    )
    call(
        "stage_file:update",
        "stage_file",
        repo=repo,
        branch=branch,
        path="HELIOS_SMOKE_PROBE.md",
        content="# smoke test probe\nv2 — diff should show v1 -> v2\n",
    )

    # 7. get_pr_status — scan first few repos for an open PR
    from helios.tools.github import _client  # internal access for scanning

    g = _client()
    pr_found = False
    for r in repos:
        try:
            repo_obj = g.get_repo(r["full_name"])
            pulls = list(repo_obj.get_pulls(state="open"))
        except Exception as exc:
            print(f"  (skipping PR scan on {r['full_name']}: {exc})")
            continue
        if pulls:
            call(
                "get_pr_status",
                "get_pr_status",
                repo=r["full_name"],
                pr_number=pulls[0].number,
            )
            pr_found = True
            break
    if not pr_found:
        skip("get_pr_status", "no open PRs in the first 5 accessible repos")

    # 8. Write tools intentionally skipped
    for t in ("commit_staged", "create_branch", "create_pr"):
        skip(t, "write tool — requires explicit sandbox confirmation")

    return finish()


def finish() -> int:
    section("Summary")
    print(f"  PASS  ({len(PASS)})")
    for n in PASS:
        print(f"    + {n}")
    print(f"  FAIL  ({len(FAIL)})")
    for n, err in FAIL:
        print(f"    - {n}  ::  {err}")
    print(f"  SKIP  ({len(SKIP)})")
    for n, reason in SKIP:
        print(f"    . {n}  ::  {reason}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
