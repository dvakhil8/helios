"""All GitHub LLM tools — read/explore + staged-write workflow.

Each tool is a function + a `*_SCHEMA` dict in Anthropic tool-use format.
The REGISTRY at the bottom maps tool name -> (schema, handler); the package-level
`tools/__init__.py` merges this with other integrations' REGISTRYs.

Staging workflow:
    stage_file(repo, branch, path, content)   # any number of times
    stage_file(...)                            # ...
    commit_staged(repo, branch, message)       # atomic multi-file commit
"""

from __future__ import annotations

import base64
import difflib
import os
import threading
from functools import lru_cache
from typing import Any, Callable

import requests
from github import Auth, Github, GithubException
from github.InputGitTreeElement import InputGitTreeElement


# ---- Client / auth -----------------------------------------------------------
# Self-contained: this file owns its own client. PAT loaded from GITHUB_TOKEN
# (or GH_TOKEN). No shared "client.py" outside this file.

_TOKEN_ENV_VARS = ("GITHUB_TOKEN", "GH_TOKEN")


def _token_with_source() -> tuple[str, str]:
    for name in _TOKEN_ENV_VARS:
        v = os.environ.get(name)
        if v:
            return v, name
    raise RuntimeError(
        f"No GitHub token found. Set one of: {', '.join(_TOKEN_ENV_VARS)}"
    )


@lru_cache(maxsize=1)
def _client() -> Github:
    token, _ = _token_with_source()
    return Github(auth=Auth.Token(token), per_page=100)


# ---- In-memory staging area --------------------------------------------------
# Keyed by (repo_full_name, branch); maps path -> content bytes.

_STAGE: dict[tuple[str, str], dict[str, bytes]] = {}
_STAGE_LOCK = threading.Lock()


def _stage_key(repo: str, branch: str) -> tuple[str, str]:
    return (repo, branch)


# =============================================================================
# Read / Explore
# =============================================================================

CHECK_TOKEN_SCHEMA: dict[str, Any] = {
    "name": "check_token",
    "description": (
        "Verify the GitHub token works. Reports identity (login, name), the env "
        "var the token was loaded from, the granted token scopes, and current "
        "rate-limit headroom."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


def check_token() -> dict[str, Any]:
    token, source = _token_with_source()
    r = requests.get(
        "https://api.github.com/user",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"GitHub /user returned {r.status_code}: {r.text[:200]}"
        )
    body = r.json()
    scopes_header = r.headers.get("X-OAuth-Scopes", "") or ""
    return {
        "ok": True,
        "token_source": source,
        "login": body.get("login"),
        "name": body.get("name"),
        "user_id": body.get("id"),
        "account_type": body.get("type"),
        "scopes": [s.strip() for s in scopes_header.split(",") if s.strip()],
        "rate_limit_remaining": r.headers.get("X-RateLimit-Remaining"),
        "rate_limit_reset": r.headers.get("X-RateLimit-Reset"),
    }


LIST_REPOS_SCHEMA: dict[str, Any] = {
    "name": "list_repos",
    "description": (
        "List repositories the token can access. Optional case-insensitive substring "
        "filter on the full repo name (owner/name). Sorted by most recently pushed by default."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name_filter": {
                "type": "string",
                "description": "Case-insensitive substring filter on owner/name.",
            },
            "limit": {
                "type": "integer",
                "default": 100,
                "description": "Max repos to return.",
            },
            "visibility": {
                "type": "string",
                "enum": ["all", "public", "private"],
                "default": "all",
            },
            "sort": {
                "type": "string",
                "enum": ["created", "updated", "pushed", "full_name"],
                "default": "pushed",
            },
        },
    },
}


def list_repos(
    name_filter: str | None = None,
    limit: int = 100,
    visibility: str = "all",
    sort: str = "pushed",
) -> dict[str, Any]:
    g = _client()
    user = g.get_user()
    out: list[dict[str, Any]] = []
    f = name_filter.lower() if name_filter else None
    for repo in user.get_repos(visibility=visibility, sort=sort):
        if f and f not in repo.full_name.lower():
            continue
        out.append(
            {
                "full_name": repo.full_name,
                "private": repo.private,
                "default_branch": repo.default_branch,
                "pushed_at": str(repo.pushed_at) if repo.pushed_at else None,
                "description": repo.description,
                "html_url": repo.html_url,
            }
        )
        if len(out) >= limit:
            break
    return {"count": len(out), "repos": out}


LIST_FILES_SCHEMA: dict[str, Any] = {
    "name": "list_files",
    "description": (
        "List files and directories at a given path in a repo. Set recursive=True "
        "to walk all subdirectories (uses the Git Trees API for efficiency)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name form."},
            "path": {
                "type": "string",
                "default": "",
                "description": "Path within the repo. Empty for root.",
            },
            "ref": {
                "type": "string",
                "description": "Branch, tag, or SHA. Defaults to the repo's default branch.",
            },
            "recursive": {
                "type": "boolean",
                "default": False,
                "description": "Walk all subdirectories.",
            },
            "limit": {
                "type": "integer",
                "default": 500,
                "description": "Max entries to return.",
            },
        },
        "required": ["repo"],
    },
}


def _looks_like_sha(s: str) -> bool:
    return len(s) >= 7 and len(s) <= 40 and all(c in "0123456789abcdef" for c in s.lower())


def list_files(
    repo: str,
    path: str = "",
    ref: str | None = None,
    recursive: bool = False,
    limit: int = 500,
) -> dict[str, Any]:
    r = _client().get_repo(repo)
    ref = ref or r.default_branch

    if recursive:
        sha = ref if _looks_like_sha(ref) else r.get_branch(ref).commit.sha
        tree = r.get_git_tree(sha=sha, recursive=True)
        entries = [
            {"path": e.path, "type": e.type, "size": e.size, "sha": e.sha}
            for e in tree.tree
        ]
        if path:
            prefix = path.rstrip("/") + "/"
            entries = [e for e in entries if e["path"].startswith(prefix) or e["path"] == path]
        return {
            "repo": repo,
            "ref": ref,
            "path": path,
            "count": min(len(entries), limit),
            "truncated": bool(tree.truncated) or len(entries) > limit,
            "entries": entries[:limit],
        }

    contents = r.get_contents(path or "", ref=ref)
    if not isinstance(contents, list):
        contents = [contents]
    entries = [
        {"path": c.path, "type": c.type, "size": c.size, "sha": c.sha}
        for c in contents[:limit]
    ]
    return {
        "repo": repo,
        "ref": ref,
        "path": path,
        "count": len(entries),
        "truncated": len(contents) > limit,
        "entries": entries,
    }


GET_FILE_SCHEMA: dict[str, Any] = {
    "name": "get_file",
    "description": (
        "Retrieve a file's content (decoded as UTF-8 when possible) and current SHA. "
        "The SHA identifies the file's blob — record it if you plan to update later."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name form."},
            "path": {"type": "string", "description": "Path to the file within the repo."},
            "ref": {
                "type": "string",
                "description": "Branch, tag, or SHA. Defaults to the repo's default branch.",
            },
            "max_bytes": {
                "type": "integer",
                "default": 200_000,
                "description": "Max bytes to return. Content is truncated past this.",
            },
        },
        "required": ["repo", "path"],
    },
}


def get_file(
    repo: str,
    path: str,
    ref: str | None = None,
    max_bytes: int = 200_000,
) -> dict[str, Any]:
    r = _client().get_repo(repo)
    ref = ref or r.default_branch
    c = r.get_contents(path, ref=ref)
    if isinstance(c, list):
        raise ValueError(f"{path!r} is a directory, not a file")

    raw = c.decoded_content
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
        "repo": repo,
        "path": c.path,
        "sha": c.sha,
        "size": c.size,
        "ref": ref,
        "encoding": encoding,
        "truncated": truncated,
        "content": content,
    }


GET_PR_STATUS_SCHEMA: dict[str, Any] = {
    "name": "get_pr_status",
    "description": (
        "Get PR status: state, draft/merged flags, mergeability, head/base, labels, "
        "assignees, requested reviewers, and CI check runs on the head commit."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name form."},
            "pr_number": {"type": "integer", "description": "Pull request number."},
        },
        "required": ["repo", "pr_number"],
    },
}


def get_pr_status(repo: str, pr_number: int) -> dict[str, Any]:
    r = _client().get_repo(repo)
    pr = r.get_pull(pr_number)

    checks: list[Any] = []
    checks_error: str | None = None
    try:
        checks = list(r.get_commit(pr.head.sha).get_check_runs())
    except GithubException as e:
        # 403: token lacks `checks:read` (fine-grained) / `repo` (classic).
        # Don't fail the whole call — return PR data without checks.
        checks_error = f"checks unavailable: {e.status} {e.data.get('message', '')}".strip()

    return {
        "repo": repo,
        "number": pr.number,
        "title": pr.title,
        "state": pr.state,
        "draft": pr.draft,
        "merged": pr.merged,
        "mergeable": pr.mergeable,
        "mergeable_state": pr.mergeable_state,
        "head": {"ref": pr.head.ref, "sha": pr.head.sha},
        "base": {"ref": pr.base.ref, "sha": pr.base.sha},
        "html_url": pr.html_url,
        "labels": [lbl.name for lbl in pr.labels],
        "assignees": [u.login for u in pr.assignees],
        "requested_reviewers": [u.login for u in pr.requested_reviewers],
        "comments": pr.comments,
        "review_comments": pr.review_comments,
        "additions": pr.additions,
        "deletions": pr.deletions,
        "changed_files": pr.changed_files,
        "checks_error": checks_error,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "conclusion": c.conclusion,
                "html_url": c.html_url,
            }
            for c in checks
        ],
        "checks_summary": {
            "total": len(checks),
            "success": sum(1 for c in checks if c.conclusion == "success"),
            "failure": sum(1 for c in checks if c.conclusion == "failure"),
            "in_progress": sum(1 for c in checks if c.status == "in_progress"),
        },
    }


# =============================================================================
# Write / Mutate
# =============================================================================

STAGE_FILE_SCHEMA: dict[str, Any] = {
    "name": "stage_file",
    "description": (
        "Stage a file for an upcoming commit. Adds or replaces (path, content) "
        "in an in-memory staging area keyed by (repo, branch). Returns a unified "
        "diff against the current file on branch (empty if the file is new). "
        "Multiple stage_file calls accumulate, then commit_staged commits them "
        "atomically. Re-staging the same path overwrites the previous staged copy."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Repository in owner/name form."},
            "branch": {
                "type": "string",
                "description": "Target branch the eventual commit will land on.",
            },
            "path": {"type": "string", "description": "Path within the repo."},
            "content": {"type": "string", "description": "New UTF-8 file content."},
        },
        "required": ["repo", "branch", "path", "content"],
    },
}


def stage_file(repo: str, branch: str, path: str, content: str) -> dict[str, Any]:
    new_bytes = content.encode("utf-8")
    r = _client().get_repo(repo)

    current_text = ""
    current_sha: str | None = None
    try:
        c = r.get_contents(path, ref=branch)
        if not isinstance(c, list):
            current_sha = c.sha
            try:
                current_text = c.decoded_content.decode("utf-8")
            except UnicodeDecodeError:
                current_text = ""
    except GithubException as e:
        if e.status != 404:
            raise

    diff_lines = list(
        difflib.unified_diff(
            current_text.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    diff = "".join(diff_lines)
    if len(diff) > 10_000:
        diff = diff[:10_000] + "\n... (diff truncated)"

    key = _stage_key(repo, branch)
    with _STAGE_LOCK:
        _STAGE.setdefault(key, {})[path] = new_bytes
        staged_paths = sorted(_STAGE[key].keys())

    return {
        "repo": repo,
        "branch": branch,
        "path": path,
        "bytes": len(new_bytes),
        "current_sha": current_sha,
        "is_new_file": current_sha is None,
        "diff": diff,
        "staged_paths_in_session": staged_paths,
    }


COMMIT_STAGED_SCHEMA: dict[str, Any] = {
    "name": "commit_staged",
    "description": (
        "Commit all staged files for (repo, branch) as a single atomic commit. "
        "Uses the Git Data API: creates blobs, builds a tree on top of the branch's "
        "current commit, creates a commit, and fast-forwards the branch ref. "
        "Clears the staging area for (repo, branch) on success. Errors if nothing is staged."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "branch": {"type": "string"},
            "message": {"type": "string", "description": "Commit message."},
        },
        "required": ["repo", "branch", "message"],
    },
}


def commit_staged(repo: str, branch: str, message: str) -> dict[str, Any]:
    key = _stage_key(repo, branch)
    with _STAGE_LOCK:
        files = dict(_STAGE.get(key, {}))
    if not files:
        raise ValueError(f"No staged files for {repo}@{branch}")

    r = _client().get_repo(repo)
    branch_obj = r.get_branch(branch)
    parent_sha = branch_obj.commit.sha
    base_tree = r.get_git_tree(branch_obj.commit.commit.tree.sha)

    elements: list[InputGitTreeElement] = []
    for path, data in files.items():
        try:
            blob = r.create_git_blob(data.decode("utf-8"), "utf-8")
        except UnicodeDecodeError:
            blob = r.create_git_blob(base64.b64encode(data).decode("ascii"), "base64")
        elements.append(
            InputGitTreeElement(path=path, mode="100644", type="blob", sha=blob.sha)
        )

    new_tree = r.create_git_tree(elements, base_tree=base_tree)
    new_commit = r.create_git_commit(
        message=message,
        tree=new_tree,
        parents=[r.get_git_commit(parent_sha)],
    )
    ref = r.get_git_ref(f"heads/{branch}")
    ref.edit(sha=new_commit.sha)

    with _STAGE_LOCK:
        _STAGE.pop(key, None)

    return {
        "repo": repo,
        "branch": branch,
        "commit_sha": new_commit.sha,
        "commit_url": new_commit.html_url,
        "parent_sha": parent_sha,
        "files_committed": sorted(files.keys()),
        "file_count": len(files),
    }


CREATE_BRANCH_SCHEMA: dict[str, Any] = {
    "name": "create_branch",
    "description": (
        "Create a branch in a repo. Idempotent: if the branch already exists, "
        "returns existed=True with its current head SHA instead of erroring."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "branch": {"type": "string", "description": "New branch name."},
            "from_branch": {
                "type": "string",
                "description": "Source branch to fork from. Defaults to the repo's default branch.",
            },
        },
        "required": ["repo", "branch"],
    },
}


def create_branch(
    repo: str, branch: str, from_branch: str | None = None
) -> dict[str, Any]:
    r = _client().get_repo(repo)
    from_branch = from_branch or r.default_branch
    source_sha = r.get_branch(from_branch).commit.sha
    existed = False
    try:
        r.create_git_ref(ref=f"refs/heads/{branch}", sha=source_sha)
    except GithubException as e:
        if e.status == 422:
            existed = True
        else:
            raise
    head = r.get_branch(branch)
    return {
        "repo": repo,
        "branch": branch,
        "from_branch": from_branch,
        "existed": existed,
        "head_sha": head.commit.sha,
    }


CREATE_PR_SCHEMA: dict[str, Any] = {
    "name": "create_pr",
    "description": (
        "Open a pull request. Idempotent: if an open PR already exists with the same "
        "head→base, returns that PR instead of creating a new one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "head": {
                "type": "string",
                "description": "Source branch (in same repo). For forks pass 'owner:branch'.",
            },
            "base": {
                "type": "string",
                "description": "Target branch. Defaults to the repo's default branch.",
            },
            "title": {"type": "string", "description": "PR title."},
            "body": {"type": "string", "default": "", "description": "PR body markdown."},
            "draft": {"type": "boolean", "default": False},
        },
        "required": ["repo", "head", "title"],
    },
}


def create_pr(
    repo: str,
    head: str,
    title: str,
    base: str | None = None,
    body: str = "",
    draft: bool = False,
) -> dict[str, Any]:
    r = _client().get_repo(repo)
    base = base or r.default_branch
    # Normalize head for the search filter (PyGithub expects "owner:branch").
    head_filter = head if ":" in head else f"{r.owner.login}:{head}"
    existing = list(r.get_pulls(state="open", head=head_filter, base=base))
    existed = bool(existing)
    pr = existing[0] if existed else r.create_pull(
        title=title, body=body, head=head, base=base, draft=draft
    )
    return {
        "repo": repo,
        "number": pr.number,
        "title": pr.title,
        "state": pr.state,
        "draft": pr.draft,
        "html_url": pr.html_url,
        "head": pr.head.ref,
        "base": pr.base.ref,
        "existed": existed,
    }


# =============================================================================
# Registry
# =============================================================================

Tool = tuple[dict[str, Any], Callable[..., Any]]

REGISTRY: dict[str, Tool] = {
    "check_token": (CHECK_TOKEN_SCHEMA, check_token),
    "list_repos": (LIST_REPOS_SCHEMA, list_repos),
    "list_files": (LIST_FILES_SCHEMA, list_files),
    "get_file": (GET_FILE_SCHEMA, get_file),
    "get_pr_status": (GET_PR_STATUS_SCHEMA, get_pr_status),
    "stage_file": (STAGE_FILE_SCHEMA, stage_file),
    "commit_staged": (COMMIT_STAGED_SCHEMA, commit_staged),
    "create_branch": (CREATE_BRANCH_SCHEMA, create_branch),
    "create_pr": (CREATE_PR_SCHEMA, create_pr),
}
