"""Tiny JSON-file store for portfolio positions and the alert log. No database
needed for a single-user personal app."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any

_LOCK = threading.Lock()
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_PORTFOLIO_FILE = os.path.join(_DATA_DIR, "portfolio.json")
_ALERTS_FILE = os.path.join(_DATA_DIR, "alerts.json")
_WATCHLIST_FILE = os.path.join(_DATA_DIR, "watchlist.json")


def _ensure():
    os.makedirs(_DATA_DIR, exist_ok=True)
    for f in (_PORTFOLIO_FILE, _ALERTS_FILE, _WATCHLIST_FILE):
        if not os.path.exists(f):
            with open(f, "w") as fh:
                json.dump([], fh)


def _read(path: str) -> list:
    _ensure()
    with _LOCK:
        with open(path) as fh:
            return json.load(fh)


def _write(path: str, data: list):
    _ensure()
    with _LOCK:
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)


# --- portfolio ---------------------------------------------------------

def list_positions() -> list[dict]:
    return _read(_PORTFOLIO_FILE)


def add_position(ticker: str, entry_price: float, qty: float, entry_date: str | None = None) -> dict:
    positions = _read(_PORTFOLIO_FILE)
    pos = {
        "id": str(uuid.uuid4())[:8],
        "ticker": ticker.upper().strip(),
        "entry_price": float(entry_price),
        "qty": float(qty),
        "entry_date": entry_date or datetime.now().strftime("%Y-%m-%d"),
    }
    positions.append(pos)
    _write(_PORTFOLIO_FILE, positions)
    return pos


def remove_position(position_id: str) -> bool:
    positions = _read(_PORTFOLIO_FILE)
    new_positions = [p for p in positions if p["id"] != position_id]
    changed = len(new_positions) != len(positions)
    if changed:
        _write(_PORTFOLIO_FILE, new_positions)
    return changed


def held_tickers() -> set[str]:
    return {p["ticker"] for p in list_positions()}


# --- watchlist (tickers with no position, just tracked) -----------------

def list_watchlist() -> list[str]:
    return _read(_WATCHLIST_FILE)


def add_watchlist(ticker: str):
    wl = _read(_WATCHLIST_FILE)
    ticker = ticker.upper().strip()
    if ticker not in wl:
        wl.append(ticker)
        _write(_WATCHLIST_FILE, wl)


def remove_watchlist(ticker: str):
    wl = _read(_WATCHLIST_FILE)
    ticker = ticker.upper().strip()
    if ticker in wl:
        wl.remove(ticker)
        _write(_WATCHLIST_FILE, wl)


# --- alert log -----------------------------------------------------------

def log_alert(entry: dict):
    alerts = _read(_ALERTS_FILE)
    entry = {"time": datetime.now().strftime("%H:%M"), "date": datetime.now().strftime("%Y-%m-%d"), **entry}
    alerts.append(entry)
    alerts = alerts[-100:]  # keep it bounded
    _write(_ALERTS_FILE, alerts)


def list_alerts(today_only: bool = True) -> list[dict]:
    alerts = _read(_ALERTS_FILE)
    if today_only:
        today = datetime.now().strftime("%Y-%m-%d")
        alerts = [a for a in alerts if a.get("date") == today]
    return list(reversed(alerts))
