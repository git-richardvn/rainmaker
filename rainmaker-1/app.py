"""
Rainmaker — personal VN stock dashboard, v0.1

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
Then open http://<this-computer's-LAN-IP>:8000 on your iPhone (same WiFi).

This is a working first version, not the finished product described in the
design doc — see README.md for exactly what's implemented vs. still to come.
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
# Core analysis — shared by the dashboard endpoint and the alert scheduler
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str, held: Optional[dict] = None, circuit_breaker: Optional[dict] = None,
                    strict: bool = False) -> Optional[dict]:
    try:
        df = ds.get_history(ticker)
    except Exception as e:  # noqa: BLE001
        return {
            "ticker": ticker, "error": True,
            "message": f"Couldn't fetch price data for {ticker} right now.",
            "detail": str(e),
        }
    # Foreign net-flow, insider trades, upcoming events, and news are all
    # "nice to have" extra calls — only spend the request budget on them for
    # tickers actually held, not the whole watchlist scan.
    foreign_net = ds.get_foreign_net_today(ticker) if held else None
    reading = engine.build_reading(ticker, df, foreign_net_today=foreign_net)
    rec = engine.recommend(reading, held=held, strict=strict)
    if rec is None:
        return None

    # --- circuit breaker: pause NEW buys after a bad stretch (never touches exits/holds) ---
    if not held and rec.action == "buy" and circuit_breaker and circuit_breaker.get("paused"):
        rec.action, rec.action_label = "watch", "Buys paused — recent losses"
        rec.why = circuit_breaker["reason"]
        rec.heads_up = None
        rec.basis_tags.append("circuit-breaker-paused")

    # --- position-size suggestion for a live buy (greedy-but-disciplined sizing, KB 4.6/13) ---
    suggested_qty = None
    risk_amount = None
    if not held and rec.action == "buy" and rec.entry and rec.stop:
        settings = store.get_settings()
        if settings.get("account_size"):
            per_share_risk = rec.entry - rec.stop
            if per_share_risk > 0:
                risk_amount = round(settings["account_size"] * settings["risk_per_trade_pct"] / 100, 0)
                suggested_qty = int(risk_amount // per_share_risk)

    out = {
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
        "held_days": rec.held_days,
        "exit_alert": rec.exit_alert,
        "exit_reason": rec.exit_reason,
        "suggested_qty": suggested_qty,
        "risk_amount": risk_amount,
    }
    if held:
        out["insider"] = ds.get_insider_trading(ticker)
        out["upcoming_event"] = ds.get_upcoming_events(ticker)
        out["news"] = ds.get_recent_news(ticker)
    return out


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


def portfolio_vs_market() -> dict:
    positions = store.list_positions()
    if not positions:
        return {"portfolio_pct": None, "market_pct": None, "since": None,
                "message": "Add a position to see how you're doing against the market."}
    total_cost = 0.0
    total_value = 0.0
    earliest = min(p["entry_date"] for p in positions)
    for p in positions:
        try:
            df = ds.get_history(p["ticker"])
            current = float(df["close"].iloc[-1])
        except Exception:
            current = p["entry_price"]  # degrade gracefully rather than crash the whole dashboard
        total_cost += p["entry_price"] * p["qty"]
        total_value += current * p["qty"]
    portfolio_pct = round((total_value / total_cost - 1) * 100, 2) if total_cost else None
    market_pct = ds.get_vnindex_return_pct(earliest)
    return {"portfolio_pct": portfolio_pct, "market_pct": market_pct, "since": earliest}


def _run_prediction_maintenance() -> dict:
    """Grades any 'buy' predictions whose review window has passed (KB Section
    14), then checks the resulting correct-rate against Richard's stated 80%
    bar. If it's fallen below that (with enough graded calls for the number
    to mean something) and the app isn't already in review mode, this
    mechanically tightens the confluence requirement for new buys from 2
    signals to 3 until accuracy recovers — logged as its own alert so
    Richard can see exactly when and why. This is not a retrain (a
    rule-based engine has no such thing); it's the same discipline lever
    already used for high-conviction calls, applied automatically."""
    for p in store.list_predictions(due_only=True):
        try:
            df = ds.get_history(p["ticker"], days=90)
            future = df[df["time"] >= pd.to_datetime(p["eval_date"])]
            eval_price = float(future["close"].iloc[0]) if len(future) else float(df["close"].iloc[-1])
            store.grade_prediction(p["id"], eval_price)
        except Exception:  # noqa: BLE001
            continue  # try again next cycle rather than let one bad fetch break the whole dashboard

    acc = store.prediction_accuracy()
    review = store.get_review_mode()
    if acc["needs_review"] and not review["active"]:
        reason = (f"Prediction accuracy fell to {acc['overall_accuracy_pct']}% across {acc['overall_evaluated']} "
                  f"graded calls — below your {acc['accuracy_target_pct']}% bar. Requiring 3 confirming signals "
                  "instead of 2 on new buys until accuracy recovers.")
        store.set_review_mode(True, reason)
        store.log_alert({"label": "Accuracy review triggered", "summary": reason})
    elif review["active"] and acc["overall_accuracy_pct"] is not None and acc["overall_accuracy_pct"] >= acc["accuracy_target_pct"]:
        store.set_review_mode(False, None)
        store.log_alert({"label": "Accuracy recovered", "summary":
                          f"Accuracy back to {acc['overall_accuracy_pct']}% — confluence bar returned to normal."})
    return acc


def build_dashboard() -> dict:
    positions = store.list_positions()
    held_map = {p["ticker"]: p for p in positions}
    circuit_breaker = store.trading_circuit_breaker()
    accuracy = _run_prediction_maintenance()
    review = store.get_review_mode()

    portfolio_cards = []
    for p in positions:
        card = analyze_ticker(p["ticker"], held={"entry_price": p["entry_price"], "qty": p["qty"], "entry_date": p.get("entry_date")})
        if card is None:
            continue
        card["position_id"] = p["id"]
        card["qty"] = p["qty"]
        card["entry_price"] = p["entry_price"]
        card["entry_date"] = p["entry_date"]
        card["unrealized_pl_pct"] = (
            round((card["price"] / p["entry_price"] - 1) * 100, 2)
            if not card.get("error") else None
        )
        portfolio_cards.append(card)

    watch_candidates = list(store.list_watchlist())
    for t in ds.DEFAULT_UNIVERSE:
        if t not in held_map and t not in watch_candidates:
            watch_candidates.append(t)

    watchlist_cards = []
    for t in watch_candidates:
        if t in held_map:
            continue
        card = analyze_ticker(t, held=None, circuit_breaker=circuit_breaker, strict=review["active"])
        if card and not card.get("error"):
            watchlist_cards.append(card)

    rank_order = {"buy": 0, "watch": 1}
    watchlist_cards.sort(key=lambda c: (rank_order.get(c["action"], 9), c["confidence"] != "high"))
    watchlist_cards = watchlist_cards[:10]  # Richard asked for at least 10 recommended tickers at a time

    # Log a directional prediction for every fresh "buy" call, so its outcome
    # can be graded once its review window passes (store.log_prediction skips
    # tickers that already have an outstanding, ungraded call).
    for c in watchlist_cards:
        if c["action"] == "buy" and c.get("entry") and c.get("stop") and c.get("target"):
            store.log_prediction(c["ticker"], c["entry"], c["stop"], c["target"], c.get("basis_tags", []))

    # Equity snapshot for the 2-week target tracker: realized P/L is exact
    # (from the trade journal), unrealized is live prices on open positions.
    unrealized = sum((c["price"] - c["entry_price"]) * c["qty"] for c in portfolio_cards if not c.get("error"))
    store.record_equity_point(store.cumulative_realized_pnl_vnd(), unrealized)
    period_progress = store.get_period_progress()
    buy_candidates = [c for c in watchlist_cards if c["action"] == "buy"]
    strategic_plan = engine.build_strategic_plan(period_progress, buy_candidates, circuit_breaker)

    return {
        "generated_at": datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M"),
        "market_status": market_status(),
        "benchmark": portfolio_vs_market(),
        "circuit_breaker": circuit_breaker,
        "settings": store.get_settings(),
        "gh_sync_enabled": gh_sync.enabled(),
        "portfolio": portfolio_cards,
        "watchlist": watchlist_cards,
        "alerts": store.list_alerts(today_only=True),
        "period_progress": period_progress,
        "strategic_plan": strategic_plan,
        "prediction_accuracy": accuracy,
        "review_mode": review,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class NewPosition(BaseModel):
    ticker: str
    entry_price: float
    qty: float
    entry_date: Optional[str] = None


class WatchTicker(BaseModel):
    ticker: str


class SettingsUpdate(BaseModel):
    account_size: Optional[float] = None
    risk_per_trade_pct: Optional[float] = None
    monthly_target_pct: Optional[float] = None
    accuracy_target_pct: Optional[float] = None
    prediction_eval_days: Optional[int] = None


class ClosePosition(BaseModel):
    exit_price: float
    exit_date: Optional[str] = None


@app.get("/api/dashboard")
def get_dashboard():
    try:
        return build_dashboard()
    except Exception as e:  # noqa: BLE001
        log.exception("dashboard build failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio")
def post_position(pos: NewPosition):
    return store.add_position(pos.ticker, pos.entry_price, pos.qty, pos.entry_date)


@app.delete("/api/portfolio/{position_id}")
def delete_position(position_id: str):
    """Removes a position WITHOUT logging a trade — for correcting a mistaken
    entry, not for closing a real trade. Use /close for an actual exit, since
    that's what feeds the trade journal and the circuit breaker."""
    ok = store.remove_position(position_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Position not found")
    return {"ok": True}


@app.post("/api/portfolio/{position_id}/close")
def close_position(position_id: str, body: ClosePosition):
    """Closes a real trade at the given exit price and logs it to the trade
    journal — this is what powers the circuit breaker (store.trading_circuit_breaker)
    and any future realized win-rate reporting (KB Section 13)."""
    trade = store.close_position(position_id, body.exit_price, body.exit_date)
    if trade is None:
        raise HTTPException(status_code=404, detail="Position not found")
    return trade


@app.get("/api/trades")
def get_trades():
    return {"trades": store.list_closed_trades(), "circuit_breaker": store.trading_circuit_breaker()}


@app.get("/api/settings")
def get_settings():
    return store.get_settings()


@app.post("/api/settings")
def post_settings(s: SettingsUpdate):
    return store.save_settings(s.account_size, s.risk_per_trade_pct, s.monthly_target_pct,
                                s.accuracy_target_pct, s.prediction_eval_days)


@app.get("/api/target")
def get_target():
    """The 2-week profit-target tracker on its own, without re-running the
    full (throttled, slow) watchlist scan — /api/dashboard already returns
    the same period_progress/strategic_plan alongside live buy candidates on
    every load; this endpoint is for checking the number by itself."""
    period = store.get_period_progress()
    circuit_breaker = store.trading_circuit_breaker()
    plan = engine.build_strategic_plan(period, [], circuit_breaker)
    return {"period_progress": period, "strategic_plan": plan}


@app.get("/api/predictions")
def get_predictions():
    """Every buy call the app has made, whether it's been graded yet, and the
    resulting correct-rate — overall and per-ticker — against Richard's
    stated 80% bar (KB Section 14). This is the honest, measured answer to
    'is this app actually working', not a guessed number."""
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
    entry/stop/target) the dashboard cards use — never hidden here even for
    tickers Rainmaker wouldn't suggest buying, since this is for Richard to
    study and discuss, not just a buy signal."""
    ticker = ticker.upper().strip()
    try:
        df = ds.get_history(ticker)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Couldn't fetch chart data for {ticker}: {e}")

    positions = store.list_positions()
    held_pos = next((p for p in positions if p["ticker"] == ticker), None)
    held = {"entry_price": held_pos["entry_price"], "qty": held_pos["qty"], "entry_date": held_pos.get("entry_date")} if held_pos else None
    foreign_net = ds.get_foreign_net_today(ticker) if held else None

    reading = engine.build_reading(ticker, df, foreign_net_today=foreign_net)
    rec = engine.explain_reading(reading, held=held)
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
        "held_days": rec.held_days,
        "exit_alert": rec.exit_alert,
        "exit_reason": rec.exit_reason,
        "held": held,
        **chart,
    }


@app.get("/api/backtest/{ticker}")
def get_backtest(ticker: str):
    """Runs the engine's current rules (KB Section 13's answer to 'does this
    actually work') against this ticker's available price history — a
    measured, honest substitute for a guessed win rate. See engine.backtest's
    docstring for the simplifications this walk-forward simulation makes."""
    ticker = ticker.upper().strip()
    try:
        df = ds.get_history(ticker, days=1500)  # ask for as much history as the source will give
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Couldn't fetch history for {ticker}: {e}")
    return engine.backtest(df, ticker=ticker)


@app.get("/api/backup")
def get_backup():
    """Download everything as one JSON file. On free hosting, local storage
    resets on redeploy — export this occasionally, and use /api/restore to
    bring it back after one."""
    return {
        "positions": store.list_positions(),
        "watchlist": store.list_watchlist(),
    }


class BackupData(BaseModel):
    positions: list = []
    watchlist: list = []


@app.post("/api/restore")
def post_restore(data: BackupData):
    store._write(store._PORTFOLIO_FILE, data.positions)
    store._write(store._WATCHLIST_FILE, data.watchlist)
    return {"ok": True, "positions_restored": len(data.positions)}


@app.get("/healthz")
def healthz():
    """Used by uptime pingers (e.g. cron-job.org) to keep a free instance awake."""
    return {"ok": True, "time": datetime.now(VN_TZ).isoformat()}


@app.post("/api/refresh")
def refresh_now():
    """Manual trigger — same thing the 10:30/14:30 scheduler does, runnable anytime for testing."""
    return run_alert_cycle(manual=True)


def run_alert_cycle(manual: bool = False) -> dict:
    dash = build_dashboard()
    label = "Manual check" if manual else "Scheduled check"

    # --- exit alerts: unambiguous "act now" cases (stop hit / target hit / time-stop) ---
    # Sent as their own urgent push, separate from routine sell/trim/new-idea notes,
    # per Richard's explicit ask: "automatically send alert to me when I need to exit
    # any ticker." These never get buried in a longer list.
    exit_cards = [c for c in dash["portfolio"] if c.get("exit_alert")]
    if exit_cards:
        exit_lines = [f"🚨 {c['ticker']}: {c['action_label']} — {c.get('exit_reason') or c['why']}"
                      for c in exit_cards]
        notify.send(
            title="Rainmaker — EXIT ALERT",
            message="\n".join(exit_lines[:5]),
            priority="urgent",
        )
        store.log_alert({"label": f"{label} — EXIT ALERT",
                          "summary": " / ".join(f"{c['ticker']}: {c['action_label']}" for c in exit_cards)})

    # --- routine notes: everything else worth a look, but not act-now urgent ---
    needs_attention = [c for c in dash["portfolio"] if c["action"] in ("sell", "trim") and not c.get("exit_alert")]
    new_ideas = [c for c in dash["watchlist"] if c["action"] == "buy"]

    lines = []
    for c in needs_attention:
        lines.append(f"{c['ticker']}: {c['action_label']} — {c['why']}")
    for c in new_ideas:
        lines.append(f"{c['ticker']} (new): {c['action_label']} — {c['why']}")

    if lines:
        store.log_alert({"label": label, "summary": " / ".join(f"{c['ticker']}: {c['action_label']}"
                                                                  for c in needs_attention + new_ideas)})
        notify.send(
            title="Rainmaker",
            message="\n".join(lines[:5]),
        )
    elif not exit_cards:
        store.log_alert({"label": label, "summary": "No action needed — everything holding steady."})
    return dash


# ---------------------------------------------------------------------------
# Background scheduler — fires the two daily alerts even with no one watching
# ---------------------------------------------------------------------------

_ALERT_TIMES = [(10, 30), (14, 30)]
_fired_today: set[str] = set()


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
        except Exception:  # noqa: BLE001
            log.exception("scheduler tick failed")
        time.sleep(30)


@app.on_event("startup")
def start_scheduler():
    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()
    log.info("Rainmaker started. Daily alerts scheduled for 10:30 and 14:30 Vietnam time.")


# --- static frontend -------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
