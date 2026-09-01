"""Push notifications to Richard's iPhone via ntfy.sh — a free, no-account
push service. Install the free 'ntfy' app from the App Store and subscribe
to the topic set in config.json; this module just POSTs to it. If no topic
is configured, notifications are silently skipped (never an error)."""
from __future__ import annotations

import json
import logging
import os

import requests

log = logging.getLogger("rainmaker.notify")

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


def _config() -> dict:
    if not os.path.exists(_CONFIG_FILE):
        return {}
    try:
        with open(_CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def send(title: str, message: str, priority: str = "default"):
    cfg = _config()
    topic = cfg.get("ntfy_topic")
    if not topic:
        log.info("ntfy_topic not set in config.json — skipping push (%s)", title)
        return
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            timeout=10,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("ntfy push failed: %s", e)
