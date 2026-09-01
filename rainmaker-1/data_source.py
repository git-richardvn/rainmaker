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

# A broader, still-liquid VN30/VN100-style universe for the daily scanner —
# hardcoded rather than fetched live (vnstock's group-membership endpoint is
# an extra, less reliable network hop for something that changes rarely; this
# list should be refreshed by hand every few months rather than trusted to
# always match the official VN30 basket exactly). Bigger than v0.1's 10-ticker
# list per Richard's "scan more than a fixed watchlist" ask — the throttle in
# _throttle() still protects the free vnstock rate limit either way.
DEFAULT_UNIVERSE = [
    "VNM", "HPG", "FPT", "VIC", "MWG", "SSI", "VCB", "MSN", "VHM", "GAS",
    "VIB", "VPB", "TCB", "MBB", "CTG", "BID", "ACB", "STB", "TPB", "SHB",
    "VRE", "PLX", "POW", "GVR", "SAB", "VJC", "PNJ", "DGC", "NKG", "HSG",
]

_INSIDER_CACHE: dict[str, tuple[float, Optional[str]]] = {}
_EVENT_CACHE: dict[str, tuple[float, Optional[str]]] = {}
_NEWS_CACHE: dict[str, tuple[float, Optional[str]]] = {}
_COMPANY_EXTRA_TTL_SECONDS = 3600  # these change slowly — an hour-old read is fine, and it's a big rate-limit saving


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
    cache_key = (symbol, days)  # keyed by days too — a 260-day dashboard read must never satisfy a 1500-day backtest request
    cached = _HISTORY_CACHE.get(cache_key)
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
            _HISTORY_CACHE[cache_key] = (now, df)
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


def _first_col(df: pd.DataFrame, *name_fragments: str) -> Optional[str]:
    """Finds the first column whose lowercased name contains any of the given
    fragments. vnstock's less-central endpoints (insider/events/news) aren't as
    stable in column naming as the core price history, so every reader below
    goes through this instead of a hardcoded column name."""
    for c in df.columns:
        lc = str(c).lower()
        if any(f in lc for f in name_fragments):
            return c
    return None


def get_insider_trading(symbol: str, lookback_days: int = 30) -> Optional[str]:
    """Most recent disclosed insider/major-shareholder transaction within the
    lookback window, as a short plain-English line — or None if the source
    doesn't have one, the call fails, or nothing recent exists. Best-effort
    only: this is a real VN-market signal per KB Section 8, but vnstock's
    coverage of it is not guaranteed, so it's never treated as a basis for a
    buy/sell call on its own, only as extra context on an already-held ticker."""
    now = time.time()
    cached = _INSIDER_CACHE.get(symbol)
    if cached and now - cached[0] < _COMPANY_EXTRA_TTL_SECONDS:
        return cached[1]
    try:
        from vnstock.api.company import Company
        _throttle()
        df = Company(symbol=symbol, source="VCI").insider_trading()
        if df is None or len(df) == 0:
            _INSIDER_CACHE[symbol] = (now, None)
            return None
        date_col = _first_col(df, "date", "time")
        who_col = _first_col(df, "name", "holder", "person")
        action_col = _first_col(df, "action", "type", "transaction")
        qty_col = _first_col(df, "quantity", "volume", "shares")
        if date_col is None:
            _INSIDER_CACHE[symbol] = (now, None)
            return None
        df = df.copy()
        df["_d"] = pd.to_datetime(df[date_col], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        recent = df[df["_d"] >= cutoff].sort_values("_d", ascending=False)
        if len(recent) == 0:
            _INSIDER_CACHE[symbol] = (now, None)
            return None
        row = recent.iloc[0]
        who = str(row[who_col]) if who_col else "An insider/major shareholder"
        action = str(row[action_col]) if action_col else "transacted"
        qty = f" ({row[qty_col]:,.0f} shares)" if qty_col and pd.notna(row[qty_col]) else ""
        line = f"{who} — {action}{qty} on {row['_d'].date()}"
        _INSIDER_CACHE[symbol] = (now, line)
        return line
    except (Exception, SystemExit) as e:  # noqa: BLE001
        log.info("insider trading fetch unavailable for %s: %s", symbol, e)
        _INSIDER_CACHE[symbol] = (now, None)
        return None


def get_upcoming_events(symbol: str, lookahead_days: int = 45) -> Optional[str]:
    """Nearest upcoming disclosed corporate event (AGM, dividend, rights issue,
    etc.) within the lookahead window — a short line, or None. Best-effort,
    same caveats as get_insider_trading above."""
    now = time.time()
    cached = _EVENT_CACHE.get(symbol)
    if cached and now - cached[0] < _COMPANY_EXTRA_TTL_SECONDS:
        return cached[1]
    try:
        from vnstock.api.company import Company
        _throttle()
        df = Company(symbol=symbol, source="VCI").events()
        if df is None or len(df) == 0:
            _EVENT_CACHE[symbol] = (now, None)
            return None
        date_col = _first_col(df, "date", "time")
        title_col = _first_col(df, "title", "event", "name", "content")
        if date_col is None:
            _EVENT_CACHE[symbol] = (now, None)
            return None
        df = df.copy()
        df["_d"] = pd.to_datetime(df[date_col], errors="coerce")
        now_ts = pd.Timestamp.now()
        upcoming = df[(df["_d"] >= now_ts) & (df["_d"] <= now_ts + pd.Timedelta(days=lookahead_days))]
        upcoming = upcoming.sort_values("_d")
        if len(upcoming) == 0:
            _EVENT_CACHE[symbol] = (now, None)
            return None
        row = upcoming.iloc[0]
        title = str(row[title_col]) if title_col else "Corporate event"
        line = f"{title} on {row['_d'].date()}"
        _EVENT_CACHE[symbol] = (now, line)
        return line
    except (Exception, SystemExit) as e:  # noqa: BLE001
        log.info("events fetch unavailable for %s: %s", symbol, e)
        _EVENT_CACHE[symbol] = (now, None)
        return None


def get_recent_news(symbol: str, lookback_days: int = 5) -> Optional[str]:
    """Most recent headline within the lookback window — a short line, or
    None. This is a headline flag, not sentiment analysis: it tells Richard
    something was published, not whether it's good or bad, matching KB
    Section 8's rule that news is context attached to a ticker, never a
    signal generated on its own."""
    now = time.time()
    cached = _NEWS_CACHE.get(symbol)
    if cached and now - cached[0] < _COMPANY_EXTRA_TTL_SECONDS:
        return cached[1]
    try:
        from vnstock.api.company import Company
        _throttle()
        df = Company(symbol=symbol, source="VCI").news()
        if df is None or len(df) == 0:
            _NEWS_CACHE[symbol] = (now, None)
            return None
        date_col = _first_col(df, "date", "time", "published")
        title_col = _first_col(df, "title", "headline", "content")
        if date_col is None or title_col is None:
            _NEWS_CACHE[symbol] = (now, None)
            return None
        df = df.copy()
        df["_d"] = pd.to_datetime(df[date_col], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        recent = df[df["_d"] >= cutoff].sort_values("_d", ascending=False)
        if len(recent) == 0:
            _NEWS_CACHE[symbol] = (now, None)
            return None
        row = recent.iloc[0]
        line = f"{str(row[title_col])[:140]} ({row['_d'].date()})"
        _NEWS_CACHE[symbol] = (now, line)
        return line
    except (Exception, SystemExit) as e:  # noqa: BLE001
        log.info("news fetch unavailable for %s: %s", symbol, e)
        _NEWS_CACHE[symbol] = (now, None)
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
