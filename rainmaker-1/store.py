"""Tiny JSON-file store for portfolio positions and the alert log. No database
needed for a single-user personal app."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any

_LOCK = threading.Lock()
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_PORTFOLIO_FILE = os.path.join(_DATA_DIR, "portfolio.json")
_ALERTS_FILE = os.path.join(_DATA_DIR, "alerts.json")
_WATCHLIST_FILE = os.path.join(_DATA_DIR, "watchlist.json")
_CLOSED_TRADES_FILE = os.path.join(_DATA_DIR, "closed_trades.json")
_SETTINGS_FILE = os.path.join(_DATA_DIR, "settings.json")

_DEFAULT_SETTINGS = {
    "account_size": None,      # VND — needed for the position-size calculator; None until Richard sets it
    "risk_per_trade_pct": 1.5,  # % of account risked per trade, per KB 4.6 (adjustable)
}


def _ensure():
    os.makedirs(_DATA_DIR, exist_ok=True)
    for f in (_PORTFOLIO_FILE, _ALERTS_FILE, _WATCHLIST_FILE, _CLOSED_TRADES_FILE):
        if not os.path.exists(f):
            with open(f, "w") as fh:
                json.dump([], fh)
    if not os.path.exists(_SETTINGS_FILE):
        with open(_SETTINGS_FILE, "w") as fh:
            json.dump(_DEFAULT_SETTINGS, fh)


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


def _read_obj(path: str, default: dict) -> dict:
    _ensure()
    with _LOCK:
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            return dict(default)


def _write_obj(path: str, data: dict):
    _ensure()
    with _LOCK:
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)


# --- account settings (position sizing) ---------------------------------

def get_settings() -> dict:
    s = dict(_DEFAULT_SETTINGS)
    s.update(_read_obj(_SETTINGS_FILE, _DEFAULT_SETTINGS))
    return s


def save_settings(account_size: float | None = None, risk_per_trade_pct: float | None = None) -> dict:
    s = get_settings()
    if account_size is not None:
        s["account_size"] = float(account_size)
    if risk_per_trade_pct is not None:
        s["risk_per_trade_pct"] = float(risk_per_trade_pct)
    _write_obj(_SETTINGS_FILE, s)
    return s


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


def close_position(position_id: str, exit_price: float, exit_date: str | None = None) -> dict | None:
    """Closes a position and logs it to the trade journal (closed_trades.json) —
    this is what the circuit breaker (below) and any future win-rate reporting
    read from. Returns the logged trade record, or None if the position wasn't found."""
    positions = _read(_PORTFOLIO_FILE)
    pos = next((p for p in positions if p["id"] == position_id), None)
    if pos is None:
        return None
    exit_date = exit_date or datetime.now().strftime("%Y-%m-%d")
    realized_pct = round((float(exit_price) / pos["entry_price"] - 1) * 100, 2) if pos["entry_price"] else None
    try:
        held_days = (datetime.strptime(exit_date, "%Y-%m-%d") - datetime.strptime(str(pos["entry_date"])[:10], "%Y-%m-%d")).days
    except Exception:
        held_days = None
    trade = {
        "ticker": pos["ticker"], "entry_price": pos["entry_price"], "exit_price": float(exit_price),
        "qty": pos["qty"], "entry_date": pos["entry_date"], "exit_date": exit_date,
        "held_days": held_days, "realized_pct": realized_pct,
    }
    trades = _read(_CLOSED_TRADES_FILE)
    trades.append(trade)
    _write(_CLOSED_TRADES_FILE, trades)
    remove_position(position_id)
    return trade


def list_closed_trades(limit: int = 200) -> list[dict]:
    trades = _read(_CLOSED_TRADES_FILE)
    return list(reversed(trades))[:limit]


def trading_circuit_breaker() -> dict:
    """A greedy short-term trader's most important guardrail: stop opening NEW
    positions after a bad stretch, rather than trying to trade your way back to
    even. Two independent trips, either one pauses new buys:
      - 3 or more consecutive losing closed trades (most recent first), or
      - the last 7 days of closed trades are net-realized-negative overall
        (only evaluated once there are at least 2 closed trades in that window,
        so a single early loss doesn't look like 'a bad week').
    This never touches existing holdings or exit alerts — only whether the
    dashboard is allowed to show a fresh 'buy'."""
    trades = _read(_CLOSED_TRADES_FILE)
    if not trades:
        return {"paused": False, "reason": None}

    ordered = list(reversed(trades))  # most recent first
    streak = 0
    for t in ordered:
        if t.get("realized_pct") is not None and t["realized_pct"] < 0:
            streak += 1
        else:
            break
    if streak >= 3:
        return {"paused": True, "reason": f"{streak} losing trades in a row — pausing new buys until you close a winner or reset manually."}

    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    recent = [t for t in trades if t.get("exit_date", "") >= cutoff]
    if len(recent) >= 2:
        net = sum(t.get("realized_pct") or 0 for t in recent)
        if net < 0:
            return {"paused": True,
                    "reason": f"Net {round(net,1)}% across {len(recent)} closed trades in the last 7 days — "
                              "pausing new buys so a rough week doesn't turn into forced, fee-losing trades."}
    return {"paused": False, "reason": None}


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
