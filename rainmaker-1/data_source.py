"""
Thin wrapper around vnstock so the rest of the app never touches the
library directly. Everything here is defensive: vnstock's free sources can
be flaky or rename columns between versions, and this app's own rule (no
basis mistakes) means it's better to say "data not available" than to
silently fabricate a number.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

log = logging.getLogger("rainmaker.data")

_HISTORY_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_FOREIGN_CACHE: dict[str, tuple[float, Optional[float]]] = {}
_CACHE_TTL_SECONDS = 300  # don't hammer the free source; a 5-minute-old bar is fine for this use case

# vnstock's free/guest tier caps requests at 20/minute and kills the whole
# process with SystemExit when that's exceeded. Stay well under that with a
# simple sliding-window throttle shared by every call this app makes.
_RATE_LOCK = threading.Lock()
_RATE_WINDOW_SECONDS = 60
_RATE_MAX_CALLS = 12
_call_times: deque[float] = deque()


def _throttle() -> None:
    while True:
        wait = 0.0
        with _RATE_LOCK:
            now = time.time()
            while _call_times and now - _call_times[0] > _RATE_WINDOW_SECONDS:
                _call_times.popleft()
            if len(_call_times) < _RATE_MAX_CALLS:
                _call_times.append(now)
                return
            wait = _RATE_WINDOW_SECONDS - (now - _call_times[0]) + 0.5
        log.info("Throttling vnstock calls to respect free-tier rate limit, sleeping %.1fs", wait)
        time.sleep(wait)

VNINDEX_SYMBOL = "VNINDEX"

DEFAULT_UNIVERSE = [
    "VNM", "HPG", "FPT", "VIC", "MWG", "SSI", "VCB", "MSN", "VHM", "GAS",
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        if c in ("time", "tradingdate", "date"):
            rename[c] = "time"
    df = df.rename(columns=rename)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"vnstock history is missing expected column '{col}' — got {list(df.columns)}")
    return df.reset_index(drop=True)


def get_history(symbol: str, days: int = 260) -> pd.DataFrame:
    """Daily OHLCV, ascending by date. Raises on failure — callers must handle it
    and show 'data not available', never guess."""
    now = time.time()
    cached = _HISTORY_CACHE.get(symbol)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    from vnstock.api.quote import Quote

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=int(days * 1.6) + 10)).strftime("%Y-%m-%d")

    last_err = None
    for source in ("VCI", "TCBS"):
        try:
            _throttle()
            q = Quote(symbol=symbol, source=source)
            df = q.history(start=start, end=end, interval="1D")
            if df is None or len(df) == 0:
                raise ValueError("empty response")
            df = _normalize(df)
            if symbol != VNINDEX_SYMBOL:
                # vnstock reports individual-stock OHLC in thousands of VND
                # (e.g. 73 means 73,000₫) — convert to actual VND here, once,
                # so every downstream number (P/L, stop/target, display) is
                # in real currency and never silently off by 1000x. The
                # index itself is already in points, not currency — leave it.
                for col in ("open", "high", "low", "close"):
                    df[col] = df[col] * 1000
            _HISTORY_CACHE[symbol] = (now, df)
            return df
        except (Exception, SystemExit) as e:  # noqa: BLE001 — vnstock's rate limiter uses sys.exit()
            last_err = e
            log.warning("history fetch failed for %s via %s: %s", symbol, source, e)
            continue
    raise RuntimeError(f"Could not fetch price history for {symbol}: {last_err}")


def get_foreign_net_today(symbol: str) -> Optional[float]:
    """Best-effort net foreign buy/sell value for today. Returns None (never a
    guessed number) if the source doesn't expose it or the call fails."""
    now = time.time()
    cached = _FOREIGN_CACHE.get(symbol)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        from vnstock.api.trading import Trading
        _throttle()
        t = Trading(source="VCI")
        board = t.price_board([symbol])
        if board is None or len(board) == 0:
            _FOREIGN_CACHE[symbol] = (now, None)
            return None
        board.columns = [str(c).lower() for c in board.columns]
        buy_cols = [c for c in board.columns if "foreign" in c and "buy" in c]
        sell_cols = [c for c in board.columns if "foreign" in c and "sell" in c]
        if not buy_cols or not sell_cols:
            _FOREIGN_CACHE[symbol] = (now, None)
            return None
        row = board.iloc[0]
        net = float(row[buy_cols[0]]) - float(row[sell_cols[0]])
        _FOREIGN_CACHE[symbol] = (now, net)
        return net
    except (Exception, SystemExit) as e:  # noqa: BLE001 — vnstock's rate limiter uses sys.exit()
        log.warning("foreign flow fetch failed for %s: %s", symbol, e)
        _FOREIGN_CACHE[symbol] = (now, None)
        return None


def get_vnindex_return_pct(since: str) -> Optional[float]:
    """% return of the VN-Index from `since` (YYYY-MM-DD) to the latest close."""
    try:
        df = get_history(VNINDEX_SYMBOL, days=400)
        df = df[df["time"] >= pd.to_datetime(since)]
        if len(df) < 2:
            return None
        start_price = float(df["close"].iloc[0])
        end_price = float(df["close"].iloc[-1])
        if start_price <= 0:
            return None
        return round((end_price / start_price - 1) * 100, 2)
    except Exception as e:  # noqa: BLE001
        log.warning("VN-Index fetch failed: %s", e)
        return None
