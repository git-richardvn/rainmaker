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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import data_source as ds
import engine
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

def analyze_ticker(ticker: str, held: Optional[dict] = None) -> Optional[dict]:
    try:
        df = ds.get_history(ticker)
    except Exception as e:  # noqa: BLE001
        return {
            "ticker": ticker, "error": True,
            "message": f"Couldn't fetch price data for {ticker} right now.",
            "detail": str(e),
        }
    # Foreign net-flow is a "nice to have" extra call — only spend the request
    # budget on it for tickers actually held, not the whole watchlist scan.
    foreign_net = ds.get_foreign_net_today(ticker) if held else None
    reading = engine.build_reading(ticker, df, foreign_net_today=foreign_net)
    rec = engine.recommend(reading, held=held)
    if rec is None:
        return None
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
    }


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


def build_dashboard() -> dict:
    positions = store.list_positions()
    held_map = {p["ticker"]: p for p in positions}

    portfolio_cards = []
    for p in positions:
        card = analyze_ticker(p["ticker"], held={"entry_price": p["entry_price"], "qty": p["qty"]})
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
        card = analyze_ticker(t, held=None)
        if card and not card.get("error"):
            watchlist_cards.append(card)

    rank_order = {"buy": 0, "watch": 1}
    watchlist_cards.sort(key=lambda c: (rank_order.get(c["action"], 9), c["confidence"] != "high"))
    watchlist_cards = watchlist_cards[:6]

    return {
        "generated_at": datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M"),
        "market_status": market_status(),
        "benchmark": portfolio_vs_market(),
        "portfolio": portfolio_cards,
        "watchlist": watchlist_cards,
        "alerts": store.list_alerts(today_only=True),
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
    ok = store.remove_position(position_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Position not found")
    return {"ok": True}


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
    held = {"entry_price": held_pos["entry_price"], "qty": held_pos["qty"]} if held_pos else None
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
        "held": held,
        **chart,
    }


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
    needs_attention = [c for c in dash["portfolio"] if c["action"] in ("sell", "trim")]
    new_ideas = [c for c in dash["watchlist"] if c["action"] == "buy"]

    lines = []
    for c in needs_attention:
        lines.append(f"{c['ticker']}: {c['action_label']} — {c['why']}")
    for c in new_ideas:
        lines.append(f"{c['ticker']} (new): {c['action_label']} — {c['why']}")

    label = "Manual check" if manual else "Scheduled check"
    if lines:
        store.log_alert({"label": label, "summary": " / ".join(f"{c['ticker']}: {c['action_label']}"
                                                                  for c in needs_attention + new_ideas)})
        notify.send(
            title="Rainmaker",
            message="\n".join(lines[:5]) or "Check your portfolio.",
        )
    else:
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
