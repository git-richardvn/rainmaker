"""Best-effort persistence to GitHub's Contents API, so the app's data (your
portfolio, trade journal, predictions, settings...) survives Render's
free-tier ephemeral filesystem — which is wiped not just on redeploys but on
every restart/spin-down too, since Render only preserves local disk with a
paid persistent Disk add-on that free web services can't attach. This module
is the free alternative: the JSON files in data/ get mirrored to this same
GitHub repo on every write, and pulled back down once at process startup —
free, permanent, and something Richard already owns.

Enabled by setting two environment variables in Render's dashboard:
  GITHUB_TOKEN — a personal access token with read/write access to this repo's contents
  GITHUB_REPO  — "owner/repo", e.g. "git-richardvn/rainmaker"
Optional:
  GITHUB_BRANCH     — defaults to "main"
  GITHUB_DATA_PATH  — defaults to "rainmaker-1/data" (where the repo's data/ folder lives)

If GITHUB_TOKEN/GITHUB_REPO aren't set, every function below is a silent
no-op and the app falls back to local-disk-only behavior — which is fine for
local development, but risky on Render's free tier for anything Richard
actually wants to keep (see the warning banner store.py/app.py surface when
this isn't configured).

This intentionally never touches credentials itself beyond reading these two
env vars — Claude does not create GitHub tokens or paste them into hosting
dashboards; that's a one-time step for Richard to do himself, documented in
deployment-status.md.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger("rainmaker.ghsync")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "").strip()
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main").strip()
GITHUB_DATA_PATH = os.environ.get("GITHUB_DATA_PATH", "rainmaker-1/data").strip().strip("/")

_sha_cache: dict[str, str] = {}


def enabled() -> bool:
    return bool(GITHUB_TOKEN and GITHUB_REPO)


def _url(filename: str) -> str:
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_DATA_PATH}/{filename}"


def _request(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "rainmaker-app",
    }
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=12) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pull(filename: str):
    """Fetch filename's parsed JSON content from the repo, or None if it
    doesn't exist there yet, sync isn't configured, or the call fails for
    any reason. Never raises — a GitHub hiccup should never block the app
    from starting or serving a request."""
    if not enabled():
        return None
    try:
        data = _request(_url(filename))
        _sha_cache[filename] = data["sha"]
        content = base64.b64decode(data["content"]).decode("utf-8")
        return json.loads(content)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        log.warning("GitHub pull failed for %s: %s", filename, e)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("GitHub pull failed for %s: %s", filename, e)
        return None


def push(filename: str, obj) -> bool:
    """Commit filename's new content to the repo. Best-effort: returns False
    and logs on any failure rather than raising, since the local file the
    caller already wrote is the source of truth for the current process —
    this is belt-and-suspenders for the NEXT restart, not a transaction the
    current request depends on. Retries once with a freshly-fetched sha if
    the cached one is stale (another writer, or a previous partial failure)."""
    if not enabled():
        return False
    content_b64 = base64.b64encode(json.dumps(obj, indent=2, default=str).encode("utf-8")).decode("ascii")
    for attempt in range(2):
        try:
            payload = {
                "message": f"rainmaker: update {filename}",
                "content": content_b64,
                "branch": GITHUB_BRANCH,
            }
            sha = _sha_cache.get(filename)
            if sha is None:
                try:
                    existing = _request(_url(filename))
                    sha = existing.get("sha")
                except urllib.error.HTTPError as e:
                    if e.code != 404:
                        raise
                    sha = None
            if sha:
                payload["sha"] = sha
            result = _request(_url(filename), method="PUT", payload=payload)
            if result and "content" in result:
                _sha_cache[filename] = result["content"]["sha"]
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("GitHub push failed for %s (attempt %d): %s", filename, attempt + 1, e)
            _sha_cache.pop(filename, None)  # force a fresh sha lookup before the retry
    return False
