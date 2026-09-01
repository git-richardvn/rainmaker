"""Tiny JSON-file store for portfolio positions and the alert log. No database
needed for a single-user personal app.

Every write also mirrors to GitHub (via gh_sync) when GITHUB_TOKEN/GITHUB_REPO
are configured, and every cold start pulls the latest copy down before
serving anything — see gh_sync.py's docstring for why this exists: Render's
free tier does not durably persist local disk, and Richard is putting his
real portfolio and financial settings into this app now, so it needs to
actually remember them across restarts, not just within one running process."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Optional

import gh_sync

_LOCK = threading.Lock()
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_PORTFOLIO_FILE = os.path.join(_DATA_DIR, "portfolio.json")
_ALERTS_FILE = os.path.join(_DATA_DIR, "alerts.json")
_WATCHLIST_FILE = os.path.join(_DATA_DIR, "watchlist.json")
_CLOSED_TRADES_FILE = os.path.join(_DATA_DIR, "closed_trades.json")
_SETTINGS_FILE = os.path.join(_DATA_DIR, "settings.json")
_PREDICTIONS_FILE = os.path.join(_DATA_DIR, "predictions.json")
_EQUITY_FILE = os.path.join(_DATA_DIR, "equity_history.json")
_SHORTLIST_FILE = os.path.join(_DATA_DIR, "shortlist.json")

_LIST_FILES = (_PORTFOLIO_FILE, _ALERTS_FILE, _WATCHLIST_FILE, _CLOSED_TRADES_FILE,
               _PREDICTIONS_FILE, _EQUITY_FILE)

_DEFAULT_SETTINGS = {
    "account_size": None,           # VND — needed for the position-size calculator and the target tracker; None until Richard sets it
    "risk_per_trade_pct": 1.5,      # % of account risked per trade, per KB 4.6 (adjustable)
    "monthly_target_pct": 15.0,     # Richard's stated goal (KB 1.2) — the 2-week target is derived from this, not a separate number
    "accuracy_target_pct": 80.0,    # Richard's stated bar for the app's own prediction correct-rate (KB Section 14)
    "prediction_eval_days": 14,     # how long a "buy" call gets to play out before it's graded right/wrong
    "review_mode": False,           # true = accuracy fell below target; confluence bar is temporarily raised
    "review_mode_reason": None,
}

_GH_SYNCED = False


def _sync_from_github_once():
    """Runs once per process, before the very first local read: if GitHub
    sync is configured, pull every tracked file down and overwrite whatever
    is (or isn't) on local disk with it. This is what makes a fresh Render
    instance — after a redeploy, a restart, or a free-tier spin-down — come
    back up already knowing Richard's portfolio, instead of starting empty."""
    global _GH_SYNCED
    if _GH_SYNCED:
        return
    _GH_SYNCED = True
    if not gh_sync.enabled():
        return
    os.makedirs(_DATA_DIR, exist_ok=True)
    for path in _LIST_FILES:
        remote = gh_sync.pull(os.path.basename(path))
        if remote is not None:
            with open(path, "w") as fh:
                json.dump(remote, fh, indent=2, default=str)
    remote_settings = gh_sync.pull(os.path.basename(_SETTINGS_FILE))
    if remote_settings is not None:
        with open(_SETTINGS_FILE, "w") as fh:
            json.dump(remote_settings, fh, indent=2, default=str)
    remote_shortlist = gh_sync.pull(os.path.basename(_SHORTLIST_FILE))
    if remote_shortlist is not None:
        with open(_SHORTLIST_FILE, "w") as fh:
            json.dump(remote_shortlist, fh, indent=2, default=str)


def _ensure():
    _sync_from_github_once()
    os.makedirs(_DATA_DIR, exist_ok=True)
    for f in _LIST_FILES:
        if not os.path.exists(f):
            with open(f, "w") as fh:
                json.dump([], fh)
    if not os.path.exists(_SETTINGS_FILE):
        with open(_SETTINGS_FILE, "w") as fh:
            json.dump(_DEFAULT_SETTINGS, fh)
    if not os.path.exists(_SHORTLIST_FILE):
        with open(_SHORTLIST_FILE, "w") as fh:
            json.dump({"generated_at": None, "cards": []}, fh)


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
    _push_to_github_async(path, data)


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
    _push_to_github_async(path, data)


def _push_to_github_async(path: str, data) -> None:
    """Fires the GitHub commit in a background thread instead of blocking the
    request that triggered the write. This matters: a dashboard load can
    write several files in one go (an equity snapshot, several new
    predictions), and each GitHub commit is its own network round trip that
    can occasionally stall or hit a transient failure — doing them
    synchronously was pushing some requests past Render's own proxy timeout,
    which showed up to Richard as the whole app silently hanging on
    'Loading...' forever. The local file is already written and is the
    source of truth for the running process either way; this thread is
    strictly belt-and-suspenders for the next restart."""
    if not gh_sync.enabled():
        return
    threading.Thread(target=gh_sync.push, args=(os.path.basename(path), data), daemon=True).start()


def gh_sync_enabled() -> bool:
    return gh_sync.enabled()


# --- account settings (position sizing, targets, accuracy bar) ---------

def get_settings() -> dict:
    s = dict(_DEFAULT_SETTINGS)
    s.update(_read_obj(_SETTINGS_FILE, _DEFAULT_SETTINGS))
    return s


def save_settings(account_size: float | None = None, risk_per_trade_pct: float | None = None,
                   monthly_target_pct: float | None = None, accuracy_target_pct: float | None = None,
                   prediction_eval_days: int | None = None) -> dict:
    s = get_settings()
    if account_size is not None:
        s["account_size"] = float(account_size)
    if risk_per_trade_pct is not None:
        s["risk_per_trade_pct"] = float(risk_per_trade_pct)
    if monthly_target_pct is not None:
        s["monthly_target_pct"] = float(monthly_target_pct)
    if accuracy_target_pct is not None:
        s["accuracy_target_pct"] = float(accuracy_target_pct)
    if prediction_eval_days is not None:
        s["prediction_eval_days"] = int(prediction_eval_days)
    _write_obj(_SETTINGS_FILE, s)
    return s


def get_review_mode() -> dict:
    s = get_settings()
    return {"active": bool(s.get("review_mode")), "reason": s.get("review_mode_reason")}


def set_review_mode(active: bool, reason: str | None = None) -> dict:
    s = get_settings()
    s["review_mode"] = bool(active)
    s["review_mode_reason"] = reason
    _write_obj(_SETTINGS_FILE, s)
    return get_review_mode()


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
    this is what the circuit breaker and the equity/target tracker (below)
    both read from. Returns the logged trade record, or None if the position
    wasn't found."""
    positions = _read(_PORTFOLIO_FILE)
    pos = next((p for p in positions if p["id"] == position_id), None)
    if pos is None:
        return None
    exit_date = exit_date or datetime.now().strftime("%Y-%m-%d")
    realized_pct = round((float(exit_price) / pos["entry_price"] - 1) * 100, 2) if pos["entry_price"] else None
    realized_vnd = round((float(exit_price) - pos["entry_price"]) * pos["qty"], 0)
    try:
        held_days = (datetime.strptime(exit_date, "%Y-%m-%d") - datetime.strptime(str(pos["entry_date"])[:10], "%Y-%m-%d")).days
    except Exception:
        held_days = None
    trade = {
        "ticker": pos["ticker"], "entry_price": pos["entry_price"], "exit_price": float(exit_price),
        "qty": pos["qty"], "entry_date": pos["entry_date"], "exit_date": exit_date,
        "held_days": held_days, "realized_pct": realized_pct, "realized_vnd": realized_vnd,
    }
    trades = _read(_CLOSED_TRADES_FILE)
    trades.append(trade)
    _write(_CLOSED_TRADES_FILE, trades)
    remove_position(position_id)
    return trade


def list_closed_trades(limit: int = 200) -> list[dict]:
    trades = _read(_CLOSED_TRADES_FILE)
    return list(reversed(trades))[:limit]


def cumulative_realized_pnl_vnd() -> float:
    """All-time realized P/L in VND across every closed trade — the precise
    half of the equity/target tracker's inputs (the other half, unrealized
    P/L on open positions, is computed by the caller since it needs live
    prices that store.py doesn't fetch)."""
    trades = _read(_CLOSED_TRADES_FILE)
    return float(sum(t.get("realized_vnd", 0) or 0 for t in trades))


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


# --- whole-market shortlist (twice-daily full re-assessment) ------------
#
# The full-market scan is expensive (a whole-market fetch + liquidity
# ranking + a deep per-ticker read on a capped candidate pool) and only
# meant to run twice a day, not on every request. Persisting the result
# means a Render restart between those two runs still has something to
# show — the last real scan — instead of an empty "Recommended stocks"
# list until the next scheduled run finally fires.

def save_shortlist(cards: list[dict], generated_at: str) -> dict:
    data = {"generated_at": generated_at, "cards": cards}
    _write_obj(_SHORTLIST_FILE, data)
    return data


def load_shortlist() -> dict:
    return _read_obj(_SHORTLIST_FILE, {"generated_at": None, "cards": []})


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


# --- 2-week profit target tracker (Richard's 15%/month mandate) ---------
#
# Periods are simple, deterministic calendar half-months (1st-15th, 16th to
# month-end) rather than a rolling window anchored to some stored "start
# date" — no extra state to lose, and "first half of the month" / "second
# half" is easy for Richard to reason about on the dashboard. The 15%/month
# target is split into two compounding ~2-week legs: (1+monthly)^0.5 - 1,
# not a flat half — the two periods actually need to compound to 15%, not
# just add to it.

def get_period_bounds(today: date | None = None) -> tuple[str, str, str]:
    today = today or date.today()
    if today.day <= 15:
        start = today.replace(day=1)
        end = today.replace(day=15)
    else:
        start = today.replace(day=16)
        if today.month == 12:
            end = today.replace(day=31)
        else:
            end = today.replace(month=today.month + 1, day=1) - timedelta(days=1)
    label = f"{today.strftime('%b')} {start.day}–{end.day}"
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), label


def record_equity_point(realized_pnl_vnd: float, unrealized_pnl_vnd: float, on_date: str | None = None) -> dict:
    """One point per calendar day (upserted, not appended, so repeated
    dashboard loads on the same day don't pile up): today's all-time
    cumulative P/L, split into realized (from the trade journal, exact) and
    unrealized (from live prices on open positions, supplied by the caller).
    This is the data get_period_progress() below reads to know how much was
    actually made or lost inside the current 2-week window."""
    on_date = on_date or datetime.now().strftime("%Y-%m-%d")
    total = round(realized_pnl_vnd + unrealized_pnl_vnd, 0)
    history = _read(_EQUITY_FILE)
    history = [h for h in history if h["date"] != on_date]
    history.append({"date": on_date, "realized_pnl_vnd": round(realized_pnl_vnd, 0),
                     "unrealized_pnl_vnd": round(unrealized_pnl_vnd, 0), "total_pnl_vnd": total})
    history.sort(key=lambda h: h["date"])
    history = history[-400:]  # bounded — well over a year of daily points
    _write(_EQUITY_FILE, history)
    return history[-1]


def get_period_progress() -> dict:
    settings = get_settings()
    account_size = settings.get("account_size")
    monthly_target_pct = settings.get("monthly_target_pct", 15.0)
    target_pct = round(((1 + monthly_target_pct / 100) ** 0.5 - 1) * 100, 2)

    start_s, end_s, label = get_period_bounds()
    start_d = date.fromisoformat(start_s)
    end_d = date.fromisoformat(end_s)
    today = date.today()
    days_total = (end_d - start_d).days + 1
    days_elapsed = min(max((today - start_d).days + 1, 0), days_total)
    days_remaining = max(days_total - days_elapsed, 0)

    history = _read(_EQUITY_FILE)
    before = [h for h in history if h["date"] < start_s]
    in_period = [h for h in history if start_s <= h["date"] <= end_s]
    baseline = before[-1]["total_pnl_vnd"] if before else (in_period[0]["total_pnl_vnd"] if in_period else 0)
    latest = in_period[-1]["total_pnl_vnd"] if in_period else baseline

    pnl_vnd_period = round(latest - baseline, 0)
    actual_pct = round((pnl_vnd_period / account_size) * 100, 2) if account_size else None

    return {
        "period_label": label, "period_start": start_s, "period_end": end_s,
        "days_total": days_total, "days_elapsed": days_elapsed, "days_remaining": days_remaining,
        "target_pct": target_pct, "monthly_target_pct": monthly_target_pct,
        "account_size": account_size, "pnl_vnd_period": pnl_vnd_period, "actual_pct": actual_pct,
        "gap_pct": round(target_pct - actual_pct, 2) if actual_pct is not None else None,
        "has_data": bool(history),
    }


# --- prediction tracking & correct-rate (KB Section 14 / Richard's ask) --
#
# Every "buy" call the scanner makes is a directional claim — this logs it,
# then grades it against what the market actually did after the review
# window. Only "buy" calls are logged: "watch"/"avoid" aren't directional
# claims and can't be honestly scored right or wrong. Grading compares the
# price at the eval date against the entry/stop/target the call was made
# with, exactly the way the backtest engine already scores a trade — so the
# app is held to the same yardstick live that it uses to test itself
# historically.

def log_prediction(ticker: str, entry: float, stop: float, target: float,
                    basis_tags: Optional[list] = None, eval_days: Optional[int] = None) -> Optional[dict]:
    preds = _read(_PREDICTIONS_FILE)
    if any(p["ticker"] == ticker and not p["evaluated"] for p in preds):
        return None  # already have an outstanding call on this ticker — don't spam duplicates every refresh
    settings = get_settings()
    eval_days = eval_days if eval_days is not None else settings.get("prediction_eval_days", 14)
    made = datetime.now().strftime("%Y-%m-%d")
    eval_date = (datetime.now() + timedelta(days=eval_days)).strftime("%Y-%m-%d")
    rec = {
        "id": str(uuid.uuid4())[:8], "ticker": ticker, "date_made": made, "eval_date": eval_date,
        "entry": entry, "stop": stop, "target": target, "basis_tags": basis_tags or [],
        "evaluated": False, "correct": None, "outcome": None, "eval_price": None,
    }
    preds.append(rec)
    _write(_PREDICTIONS_FILE, preds)
    return rec


def list_predictions(due_only: bool = False) -> list[dict]:
    preds = _read(_PREDICTIONS_FILE)
    if due_only:
        today = datetime.now().strftime("%Y-%m-%d")
        preds = [p for p in preds if not p["evaluated"] and p["eval_date"] <= today]
    return list(reversed(preds))


def grade_prediction(pred_id: str, eval_price: float) -> Optional[dict]:
    preds = _read(_PREDICTIONS_FILE)
    p = next((p for p in preds if p["id"] == pred_id), None)
    if p is None:
        return None
    hit_target = p.get("target") is not None and eval_price >= p["target"]
    hit_stop = p.get("stop") is not None and eval_price <= p["stop"]
    if hit_target and not hit_stop:
        correct, outcome = True, "hit target"
    elif hit_stop and not hit_target:
        correct, outcome = False, "hit stop"
    else:
        # neither level was clearly crossed by the eval date (or, rarely, both were,
        # which the endpoint price alone can't disambiguate) — grade by plain direction
        correct, outcome = eval_price > p["entry"], "neither level hit by review date — graded by direction"
    p["evaluated"] = True
    p["correct"] = bool(correct)
    p["outcome"] = outcome
    p["eval_price"] = eval_price
    _write(_PREDICTIONS_FILE, preds)
    return p


def prediction_accuracy(min_sample: int = 5) -> dict:
    """Overall and per-ticker correct rate across graded calls — the honest,
    measured reflection of whether the app's calls actually work, per
    Richard's ask. needs_review only trips once there's a big enough sample
    (min_sample) for the number to mean something; a couple of early misses
    on a brand-new tracker shouldn't read as a crisis."""
    preds = [p for p in _read(_PREDICTIONS_FILE) if p["evaluated"]]
    overall_evaluated = len(preds)
    overall_correct = sum(1 for p in preds if p["correct"])
    overall_pct = round(100 * overall_correct / overall_evaluated, 1) if overall_evaluated else None

    by_ticker: dict[str, dict] = {}
    for p in preds:
        t = p["ticker"]
        by_ticker.setdefault(t, {"evaluated": 0, "correct": 0})
        by_ticker[t]["evaluated"] += 1
        by_ticker[t]["correct"] += 1 if p["correct"] else 0
    for v in by_ticker.values():
        v["accuracy_pct"] = round(100 * v["correct"] / v["evaluated"], 1)

    settings = get_settings()
    accuracy_target = settings.get("accuracy_target_pct", 80.0)
    needs_review = overall_evaluated >= min_sample and overall_pct is not None and overall_pct < accuracy_target

    return {
        "overall_evaluated": overall_evaluated, "overall_correct": overall_correct,
        "overall_accuracy_pct": overall_pct, "accuracy_target_pct": accuracy_target,
        "min_sample": min_sample, "needs_review": needs_review,
        "by_ticker": by_ticker,
    }
