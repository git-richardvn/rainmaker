"""
Rainmaker — VN stock recommendation + watchlist tracker

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
Then open http://<this-computer's-LAN-IP>:8000 on your iPhone (same WiFi).

Scope (as of the latest revision): Richard asked to drop the personal
portfolio / financial-planning half of the app (position tracking, account
sizing, the 2-week profit target, the loss circuit breaker) and keep only
two things — a scanner that recommends potential stocks, and a personal
watchlist of tickers he wants to keep an eye on. This file reflects that.

Speed: the free vnstock data source throttles at 12 calls/60s, so a full
scan of the ticker universe can legitimately take over a minute. Rather than
running that scan inline on every page load (which is what made the app
feel slow, and made popular "double refresh" retries compound into even
slower ones), a background loop (see the bottom of this file) recomputes the
scan on a timer and caches the result. GET /api/dashboard just serves that
cache — normal page loads are near-instant. Only a manual "Refresh" or the
very first request after a cold boot waits on a real scan.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import data_source as ds
import engine
import gh_sync
import store
import notify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rainmaker.app")

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

app = FastAPI(title="Rainmaker")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _build_card(rec, reading) -> dict:
    return {
        "ticker": rec.ticker,
        "action": rec.action,
        "action_label": rec.action_label,
        "why": rec.why,
        "heads_up": rec.heads_up,
        "confidence": rec.confidence,
        "price": rec.price,
        "change_pct": round(rec.change_pct, 2),
        "entry": rec.entry,
        "stop": rec.stop,
        "target": rec.target,
        "reward_risk": rec.reward_risk,
        "basis_tags": rec.basis_tags,
        "bars": reading.bars,
        "conviction": rec.conviction,
        "max_hold_days": rec.max_hold_days,
    }


def analyze_ticker(ticker: str, strict: bool = False) -> Optional[dict]:
    """Full read for the recommendation scanner. Returns None when the
    engine doesn't see anything worth surfacing — the scan never pads its
    list with a weak setup just to hit a count."""
    try:
        df = ds.get_history(ticker)
    except Exception as e:  # noqa: BLE001
        return {
            "ticker": ticker, "error": True,
            "message": f"Couldn't fetch price data for {ticker} right now.",
            "detail": str(e),
        }
    reading = engine.build_reading(ticker, df)
    rec = engine.recommend(reading, strict=strict)
    if rec is None:
        return None
    return _build_card(rec, reading)


def analyze_watchlist_ticker(ticker: str) -> dict:
    """Always returns a card for a ticker Richard chose to track himself,
    even when the engine wouldn't call it a buy right now — this is for
    keeping an eye on something, not just a signal feed, so it uses the same
    honest fallback read the chart view uses (engine.explain_reading)."""
    try:
        df = ds.get_history(ticker)
    except Exception as e:  # noqa: BLE001
        return {
            "ticker": ticker, "error": True,
            "message": f"Couldn't fetch price data for {ticker} right now.",
            "detail": str(e),
        }
    reading = engine.build_reading(ticker, df)
    rec = engine.explain_reading(reading)
    return _build_card(rec, reading)


def market_status() -> dict:
    now = datetime.now(VN_TZ)
    if now.weekday() >= 5:
        return {"open": False, "label": "Market closed — weekend"}
    t = now.hour * 60 + now.minute
    def m(h, mi):
        return h * 60 + mi
    if m(9, 0) <= t < m(11, 30):
        return {"open": True, "label": "Morning session", "closes_in_min": m(11, 30) - t}
    if m(11, 30) <= t < m(13, 0):
        return {"open": False, "label": "Lunch break — afternoon session at 13:00"}
    if m(13, 0) <= t < m(14, 45):
        return {"open": True, "label": "Afternoon session", "closes_in_min": m(14, 45) - t}
    if m(14, 45) <= t < m(15, 0):
        return {"open": True, "label": "Closing auction (ATC)", "closes_in_min": m(15, 0) - t}
    return {"open": False, "label": "Market closed"}


def _run_prediction_maintenance() -> dict:
    """Grades any 'buy' predictions whose review window has passed, then
    checks the resulting correct-rate against the stated 80% bar. If it's
    fallen below that (with enough graded calls for the number to mean
    something) and the app isn't already in review mode, this mechanically
    tightens the confluence requirement for new buys from 2 signals to 3
    until accuracy recovers — logged as its own alert. This is not a
    retrain (a rule-based engine has no such thing); it's the same
    discipline lever already used for high-conviction calls, applied
    automatically."""
    for p in store.list_predictions(due_only=True):
        try:
            df = ds.get_history(p["ticker"], days=90)
            future = df[df["time"] >= pd.to_datetime(p["eval_date"])]
            eval_price = float(future["close"].iloc[0]) if len(future) else float(df["close"].iloc[-1])
            store.grade_prediction(p["id"], eval_price)
        except Exception:  # noqa: BLE001
            continue  # try again next cycle rather than let one bad fetch break the whole scan

    acc = store.prediction_accuracy()
    review = store.get_review_mode()
    if acc["needs_review"] and not review["active"]:
        reason = (f"Prediction accuracy fell to {acc['overall_accuracy_pct']}% across {acc['overall_evaluated']} "
                  f"graded calls — below the {acc['accuracy_target_pct']}% bar. Requiring 3 confirming signals "
                  "instead of 2 on new buys until accuracy recovers.")
        store.set_review_mode(True, reason)
        store.log_alert({"label": "Accuracy review triggered", "summary": reason})
    elif review["active"] and acc["overall_accuracy_pct"] is not None and acc["overall_accuracy_pct"] >= acc["accuracy_target_pct"]:
        store.set_review_mode(False, None)
        store.log_alert({"label": "Accuracy recovered", "summary":
                          f"Accuracy back to {acc['overall_accuracy_pct']}% — confluence bar returned to normal."})
    return acc


def build_dashboard() -> dict:
    """One full scan: grades due predictions, re-reads every ticker on
    Richard's own watchlist (always shown, whatever the signal), then scans
    the wider universe for fresh recommendations (only the ones actually
    worth surfacing). This is the slow, throttled part of the app — see the
    module docstring for why it runs on a background timer instead of
    inline per request."""
    accuracy = _run_prediction_maintenance()
    review = store.get_review_mode()

    my_tickers = store.list_watchlist()
    my_watchlist = [analyze_watchlist_ticker(t) for t in my_tickers]

    already_scanned = set(my_tickers)
    recommended = []
    for t in ds.DEFAULT_UNIVERSE:
        if t in already_scanned:
            continue
        card = analyze_ticker(t, strict=review["active"])
        if card and not card.get("error"):
            recommended.append(card)

    rank_order = {"buy": 0, "watch": 1}
    recommended.sort(key=lambda c: (rank_order.get(c["action"], 9), c["confidence"] != "high"))
    recommended = recommended[:10]  # at least/around 10 recommended tickers at a time

    # Log a directional prediction for every fresh "buy" call, so its outcome
    # can be graded once its review window passes (store.log_prediction skips
    # tickers that already have an outstanding, ungraded call).
    for c in recommended:
        if c["action"] == "buy" and c.get("entry") and c.get("stop") and c.get("target"):
            store.log_prediction(c["ticker"], c["entry"], c["stop"], c["target"], c.get("basis_tags", []))

    return {
        "generated_at": datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M"),
        "market_status": market_status(),
        "gh_sync_enabled": gh_sync.enabled(),
        "my_watchlist": my_watchlist,
        "recommended": recommended,
        "alerts": store.list_alerts(today_only=True),
        "prediction_accuracy": accuracy,
        "review_mode": review,
    }


# ---------------------------------------------------------------------------
# Shared cache — one scan feeds every reader, computed on a background timer
# ---------------------------------------------------------------------------

_cache_cond = threading.Condition()
_cache_state = {"building": False, "data": None, "error": None, "ts": 0.0}


def _refresh_cache() -> dict:
    """Runs a fresh scan and updates the shared cache. Never runs two scans
    at once — an overlapping caller (a manual refresh while the scheduled
    scan is already running, or two people loading the page at once) just
    waits for the in-flight scan's result instead of starting a redundant
    one that would compete for the same rate-limit budget."""
    with _cache_cond:
        if _cache_state["building"]:
            _cache_cond.wait(timeout=280)
            if _cache_state["data"] is not None and not _cache_state["building"]:
                return _cache_state["data"]
            if _cache_state["error"] is not None:
                raise RuntimeError(_cache_state["error"])
        _cache_state["building"] = True
        _cache_state["error"] = None

    try:
        data = build_dashboard()
        with _cache_cond:
            _cache_state["data"] = data
            _cache_state["ts"] = time.time()
        return data
    except Exception as e:  # noqa: BLE001
        log.exception("scan failed")
        with _cache_cond:
            _cache_state["error"] = str(e)
        raise
    finally:
        with _cache_cond:
            _cache_state["building"] = False
            _cache_cond.notify_all()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class WatchTicker(BaseModel):
    ticker: str


@app.get("/api/dashboard")
def get_dashboard():
    """Serves the last background-computed scan instantly — see the module
    docstring. Only blocks on a real scan when the cache is still empty
    (a cold boot that the background loop hasn't finished warming up yet)."""
    if _cache_state["data"] is not None:
        return _cache_state["data"]
    try:
        return _refresh_cache()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh")
def refresh_now():
    """Manual override — forces a fresh scan right now (rather than waiting
    for the next scheduled one) and sends the same alert a scheduled cycle
    would. Can take up to ~1-2 minutes since it's a real throttled scan; the
    shared cache lock means clicking it twice doesn't start two scans."""
    try:
        return run_alert_cycle(manual=True)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
def get_predictions():
    """Every buy call the scanner has made, whether it's been graded yet,
    and the resulting correct-rate — overall and per-ticker — against the
    stated 80% bar. This is the honest, measured answer to 'is this app's
    advice actually good', not a guessed number."""
    return {
        "predictions": store.list_predictions(),
        "accuracy": store.prediction_accuracy(),
        "review_mode": store.get_review_mode(),
    }


@app.post("/api/watchlist")
def post_watch(w: WatchTicker):
    store.add_watchlist(w.ticker)
    return {"ok": True}


@app.delete("/api/watchlist/{ticker}")
def delete_watch(ticker: str):
    store.remove_watchlist(ticker)
    return {"ok": True}


@app.get("/api/chart/{ticker}")
def get_chart(ticker: str):
    """Everything needed to draw a professional-style chart for one ticker:
    candles, moving averages, a trendline when the data supports one,
    liquidity/support-resistance levels, and the same honest read (why,
    entry/stop/target) the cards use — never hidden here even for tickers
    Rainmaker wouldn't suggest buying, since this view is for studying a
    ticker, not just a buy signal."""
    ticker = ticker.upper().strip()
    try:
        df = ds.get_history(ticker)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Couldn't fetch chart data for {ticker}: {e}")

    reading = engine.build_reading(ticker, df)
    rec = engine.explain_reading(reading)
    chart = engine.build_chart_payload(df, reading)

    return {
        "ticker": ticker,
        "price": reading.price,
        "change_pct": round(reading.change_pct, 2),
        "bars": reading.bars,
        "action": rec.action,
        "action_label": rec.action_label,
        "why": rec.why,
        "heads_up": rec.heads_up,
        "confidence": rec.confidence,
        "entry": rec.entry,
        "stop": rec.stop,
        "target": rec.target,
        "reward_risk": rec.reward_risk,
        "conviction": rec.conviction,
        "max_hold_days": rec.max_hold_days,
        **chart,
    }


@app.get("/api/backtest/{ticker}")
def get_backtest(ticker: str):
    """Runs the engine's current rules against this ticker's available price
    history — a measured, honest substitute for a guessed win rate. See
    engine.backtest's docstring for the simplifications this walk-forward
    simulation makes."""
    ticker = ticker.upper().strip()
    try:
        df = ds.get_history(ticker, days=1500)  # ask for as much history as the source will give
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Couldn't fetch history for {ticker}: {e}")
    return engine.backtest(df, ticker=ticker)


@app.get("/api/backup")
def get_backup():
    """Download your watchlist as JSON. On free hosting, local storage can
    reset on a redeploy or restart — GitHub sync (see gh_sync.py) already
    covers this automatically when configured; this is a manual fallback."""
    return {"watchlist": store.list_watchlist()}


class BackupData(BaseModel):
    watchlist: list = []


@app.post("/api/restore")
def post_restore(data: BackupData):
    store._write(store._WATCHLIST_FILE, data.watchlist)
    return {"ok": True, "watchlist_restored": len(data.watchlist)}


@app.get("/healthz")
def healthz():
    """Used by uptime pingers (e.g. cron-job.org) to keep a free instance awake."""
    return {"ok": True, "time": datetime.now(VN_TZ).isoformat()}


def run_alert_cycle(manual: bool = False) -> dict:
    dash = _refresh_cache()
    label = "Manual check" if manual else "Scheduled check"

    new_ideas = [c for c in dash["recommended"] if c["action"] == "buy"]
    if new_ideas:
        lines = [f"{c['ticker']} (new): {c['action_label']} — {c['why']}" for c in new_ideas]
        store.log_alert({"label": label, "summary": " / ".join(f"{c['ticker']}: {c['action_label']}" for c in new_ideas)})
        notify.send(title="Rainmaker", message="\n".join(lines[:5]))
    else:
        store.log_alert({"label": label, "summary": "No new buy ideas right now."})
    return dash


# ---------------------------------------------------------------------------
# Background scheduler — keeps the scan cache warm and fires the daily alert
# ---------------------------------------------------------------------------

_ALERT_TIMES = [(10, 30), (14, 30)]
_fired_today: set[str] = set()
_CACHE_REFRESH_INTERVAL_SEC = 15 * 60  # how often to re-scan while the market is open
_last_cache_refresh = {"ts": 0.0}


def _scheduler_loop():
    while True:
        try:
            now = datetime.now(VN_TZ)
            key_today = now.strftime("%Y-%m-%d")
            for h, m in _ALERT_TIMES:
                key = f"{key_today} {h:02d}:{m:02d}"
                if now.hour == h and now.minute == m and key not in _fired_today:
                    _fired_today.add(key)
                    if now.weekday() < 5:
                        log.info("Running scheduled alert cycle %s", key)
                        run_alert_cycle(manual=False)
            if len(_fired_today) > 20:
                _fired_today.clear()

            # Keep the recommendation cache warm so page loads never block on
            # a live scan: refresh once at cold boot, then periodically while
            # the market is open (prices don't move while it's closed, so
            # there's no reason to keep re-hitting the rate-limited source).
            ms = market_status()
            due = (time.time() - _last_cache_refresh["ts"]) >= _CACHE_REFRESH_INTERVAL_SEC
            if _cache_state["data"] is None or (ms["open"] and due):
                _last_cache_refresh["ts"] = time.time()
                log.info("Refreshing recommendation cache")
                try:
                    _refresh_cache()
                except Exception:  # noqa: BLE001
                    pass  # already logged inside _refresh_cache; keep the loop alive
        except Exception:  # noqa: BLE001
            log.exception("scheduler tick failed")
        time.sleep(30)


@app.on_event("startup")
def start_scheduler():
    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()
    log.info("Rainmaker started. Recommendation cache refreshes every %ds while the market is open; "
              "daily alert scheduled for 10:30 and 14:30 Vietnam time.", _CACHE_REFRESH_INTERVAL_SEC)


# --- static frontend -------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
