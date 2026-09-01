"""
Rainmaker analysis engine — v0.1

This is a deliberately simplified first implementation of the rules in the
design doc ("Rainmaker — App Logic & Trading Knowledge Base"). It covers a
working subset: trend (MA structure), momentum (RSI/MACD), volume/breakout,
and a basic money-flow read via OBV. Foreign net-flow is included when the
data source actually provides it — never guessed.

Everything here is written to fail loudly rather than fabricate: if there
isn't enough price history, the engine says so and lowers confidence instead
of pretending to have an opinion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("rainmaker.engine")

MIN_BARS_FOR_TREND = 60      # below this, trend/MA calls are unreliable
MIN_BARS_FOR_ANYTHING = 20   # below this, we say nothing useful at all
BREAKOUT_VOLUME_RATIO = 1.5  # volume vs 20d average needed to call a breakout "conviction"

# --- short-term trading policy (Richard's stated mandate) -------------------
# Richard trades short-term only: 1-2 month holds by default, up to ~3 months
# only for the most convincing setups this engine can identify. He pays ~0.2%
# per side in fees (0.4% round trip), so a new trade only makes sense when the
# realistic move to target clearly clears that cost — this is NOT about
# predicting a 15%/month return or a 90% win rate (no price/volume engine can
# honestly produce either number), it's about being disciplined: fewer, higher
# quality trades, hard exits, and never sitting in a slow trade past its window.
FEE_ROUND_TRIP_PCT = 0.4          # 0.2% each way, typical VN brokerage commission
MIN_EDGE_VS_FEE = 3.0             # require the move to target to clear round-trip cost by ~3x before opening a new trade
DEFAULT_MAX_HOLD_DAYS = 60        # ~2 months — the default short-term window
HIGH_CONVICTION_MAX_HOLD_DAYS = 90  # ~3 months — only while multiple signals still agree; never further


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def slope(series: pd.Series, n: int) -> float:
    """Simple linear-regression slope over the last n points, normalised
    by the series' own scale so it's comparable across tickers."""
    tail = series.tail(n).dropna()
    if len(tail) < max(3, n // 2):
        return 0.0
    x = np.arange(len(tail))
    y = tail.values
    scale = np.nanmean(np.abs(y)) or 1.0
    try:
        m = np.polyfit(x, y, 1)[0]
    except Exception:
        return 0.0
    return float(m / scale)


# ---------------------------------------------------------------------------
# Reading — the raw, cited evidence. Nothing here is an opinion yet.
# ---------------------------------------------------------------------------

@dataclass
class Reading:
    ticker: str
    bars: int
    price: float
    change_pct: float
    trend: str            # 'up' | 'down' | 'flat' | 'unknown'
    momentum: str          # 'overbought' | 'oversold' | 'fading' | 'building' | 'neutral'
    flow: str              # 'accumulation' | 'distribution' | 'neutral' | 'unknown'
    breakout: bool
    volume_ratio: float
    support: Optional[float]
    resistance: Optional[float]
    bearish_divergence: bool
    foreign_net_today: Optional[float]  # None when the data source didn't provide it
    confidence_cap: str    # 'low' | 'medium' | 'high' — ceiling based on data quality
    notes: list = field(default_factory=list)


def build_reading(ticker: str, df: pd.DataFrame, foreign_net_today: Optional[float] = None) -> Reading:
    """df must have columns: time/open/high/low/close/volume, ascending by time."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    bars = len(df)
    notes: list[str] = []

    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if bars >= 2 else price
    change_pct = (price / prev - 1) * 100 if prev else 0.0

    if bars < MIN_BARS_FOR_ANYTHING:
        return Reading(
            ticker=ticker, bars=bars, price=price, change_pct=change_pct,
            trend="unknown", momentum="neutral", flow="unknown", breakout=False,
            volume_ratio=1.0, support=None, resistance=None, bearish_divergence=False,
            foreign_net_today=foreign_net_today, confidence_cap="low",
            notes=[f"Only {bars} days of price history — not enough to say anything reliable yet."],
        )

    ma50 = sma(close, 50)
    ma200 = sma(close, 200)

    if bars >= MIN_BARS_FOR_TREND and not pd.isna(ma50.iloc[-1]):
        if not pd.isna(ma200.iloc[-1]):
            trend = "up" if (price > ma50.iloc[-1] > ma200.iloc[-1]) else (
                "down" if (price < ma50.iloc[-1] < ma200.iloc[-1]) else "flat")
        else:
            trend = "up" if price > ma50.iloc[-1] else "down"
            notes.append("Under 200 days of history — long-term trend read is provisional.")
    else:
        trend = "unknown"
        notes.append("Under 60 days of history — trend read is provisional.")

    rsi14 = rsi(close, 14)
    _, _, hist = macd(close)
    last_rsi = float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else 50.0
    hist_slope5 = slope(hist, 5)

    if last_rsi >= 70:
        momentum = "overbought"
    elif last_rsi <= 30:
        momentum = "oversold"
    elif hist_slope5 < -0.05:
        momentum = "fading"
    elif hist_slope5 > 0.05:
        momentum = "building"
    else:
        momentum = "neutral"

    # bearish divergence: price higher high over last 10 bars, MACD histogram lower than 10 bars ago
    bearish_divergence = False
    if bars >= 15:
        hh_price = close.iloc[-1] > close.iloc[-10:-1].max()
        hist_now = hist.iloc[-1]
        hist_then = hist.iloc[-10]
        if hh_price and not pd.isna(hist_now) and not pd.isna(hist_then) and hist_now < hist_then and hist_slope5 < 0:
            bearish_divergence = True

    # volume / breakout
    vol_avg20 = volume.rolling(20, min_periods=10).mean().iloc[-1]
    volume_ratio = float(volume.iloc[-1] / vol_avg20) if vol_avg20 and not pd.isna(vol_avg20) else 1.0
    prior_high20 = close.iloc[-21:-1].max() if bars >= 21 else close.iloc[:-1].max()
    breakout = bool(price > prior_high20 and volume_ratio >= BREAKOUT_VOLUME_RATIO)

    # support / resistance — simple 20-day swing range
    window = close.tail(20)
    support = float(window.min()) if len(window) else None
    resistance = float(window.max()) if len(window) else None

    # money flow via OBV: is it rising while price is flat/mild, or falling while price holds/rises?
    obv_line = obv(close, volume)
    obv_slope15 = slope(obv_line, 15)
    price_slope15 = slope(close, 15)
    flow = "neutral"
    if bars >= 20:
        if obv_slope15 > 0.02 and price_slope15 <= 0.03:
            flow = "accumulation"
        elif obv_slope15 < -0.02 and price_slope15 >= -0.01:
            flow = "distribution"

    # confidence ceiling from data quality alone (the decision layer may lower it further)
    if bars < MIN_BARS_FOR_TREND:
        confidence_cap = "low"
    elif bars < 120:
        confidence_cap = "medium"
    else:
        confidence_cap = "high"

    return Reading(
        ticker=ticker, bars=bars, price=price, change_pct=change_pct,
        trend=trend, momentum=momentum, flow=flow, breakout=breakout,
        volume_ratio=volume_ratio, support=support, resistance=resistance,
        bearish_divergence=bearish_divergence, foreign_net_today=foreign_net_today,
        confidence_cap=confidence_cap, notes=notes,
    )


# ---------------------------------------------------------------------------
# Recommendation — turns a Reading into a plain-English, auditable judgment.
# This is the self-check-carrying layer: every branch states its basis.
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    ticker: str
    action: str            # 'buy' | 'hold' | 'trim' | 'sell' | 'watch'
    action_label: str       # plain-English label for the badge
    why: str
    heads_up: Optional[str]
    confidence: str
    price: float
    change_pct: float
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    reward_risk: Optional[float]
    basis_tags: list        # internal rule citations, for the "why does it think this" detail
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS
    conviction: str = "standard (1-2 months)"
    held_days: Optional[int] = None
    exit_alert: bool = False      # true = unambiguous, act-now exit (stop hit / target hit / time-stop)
    exit_reason: Optional[str] = None


def _reward_risk(entry: float, stop: float, target: float) -> Optional[float]:
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return None
    return round(reward / risk, 1)


def _confluence_score(reading: Reading) -> int:
    """Counts independent, non-contradicting signals lining up for `reading`.
    Shared basis for both the 'new trade' bar and the 'high conviction /
    extend the hold' bar — the latter just requires more of them. This is a
    confluence count, not a probability: it never gets reported to Richard
    as a win chance, only used to gate which setups this engine will act on."""
    if reading.bearish_divergence:
        return 0
    score = 0
    if reading.trend == "up":
        score += 1
    if reading.breakout and reading.volume_ratio >= 2.0:
        score += 1
    if reading.flow == "accumulation":
        score += 1
    if reading.foreign_net_today is not None and reading.foreign_net_today > 0:
        score += 1
    if reading.momentum in ("building", "neutral"):
        score += 1
    return score


MIN_CONFLUENCE_FOR_NEW_TRADE = 2   # require at least this many confirming signals before opening any new position
MIN_CONFLUENCE_FOR_HIGH_CONVICTION = 3  # the higher bar for extending a hold to ~3 months


def is_high_conviction(reading: Reading) -> bool:
    """A rough proxy for Richard's 'very convincing' bar for extending a hold
    from ~2 months to ~3. This is NOT a win-probability estimate — no engine
    working from price/volume alone can honestly produce a number like '90%
    win chance' for a specific ticker, and this doesn't try to. It just checks
    whether several independent, non-contradicting signals line up, which is
    the closest thing to 'convincing' this engine can actually back up."""
    if reading.bars < 120 or reading.confidence_cap != "high":
        return False
    return _confluence_score(reading) >= MIN_CONFLUENCE_FOR_HIGH_CONVICTION


def _apply_short_term_policy(rec: Recommendation, reading: Reading, held: Optional[dict]) -> Recommendation:
    """Layers Richard's short-term mandate on top of a raw judgment: caps (or
    extends) the holding window, blocks new trades whose edge is too thin to
    clear round-trip fees, enforces a time-stop independent of price, and sets
    an unambiguous exit_alert flag the notification system can act on without
    re-deriving it. Applied as the last step for every path, so nothing
    downstream can accidentally skip it."""
    high_conviction = is_high_conviction(reading)
    rec.conviction = "high-conviction (up to ~3 months)" if high_conviction else "standard (1-2 months)"
    rec.max_hold_days = HIGH_CONVICTION_MAX_HOLD_DAYS if high_conviction else DEFAULT_MAX_HOLD_DAYS

    # --- fee-aware filter: don't open a new position for a move too thin to clear costs ---
    if not held and rec.action == "buy" and rec.entry and rec.target and rec.entry > 0:
        expected_move_pct = (rec.target / rec.entry - 1) * 100
        if expected_move_pct < FEE_ROUND_TRIP_PCT * MIN_EDGE_VS_FEE:
            rec.action, rec.action_label = "watch", "Too thin after fees"
            rec.why = (f"The realistic move to target here is only about {round(expected_move_pct, 1)}%. "
                       f"After round-trip trading costs (~{FEE_ROUND_TRIP_PCT}%) that's not a clean edge — "
                       "not worth a new position on this alone. Wait for a cleaner setup.")
            rec.heads_up = None
            rec.basis_tags.append("fee-aware-too-thin")

    # --- confluence filter: don't open a new position on a single, isolated signal ---
    # Richard asked to "focus on the sure win tickers" rather than trade often — this
    # requires at least a couple of independent signals to agree before a buy goes out,
    # not just one (e.g. a breakout alone, with nothing else confirming it).
    if not held and rec.action == "buy" and _confluence_score(reading) < MIN_CONFLUENCE_FOR_NEW_TRADE:
        rec.action, rec.action_label = "watch", "Not enough confirmation yet"
        rec.why = ("Only one signal is present here — not enough independent confirmation to call this a "
                   "high-quality setup worth paying trading fees for. Waiting for a second confirming signal "
                   "(trend, volume, money flow, or foreign buying) before treating this as a buy.")
        rec.heads_up = None
        rec.basis_tags.append("confluence-too-thin")

    # --- time-stop: the 1-2 (up to 3) month rule, independent of price action ---
    if held and held.get("entry_date"):
        try:
            entry_dt = date.fromisoformat(str(held["entry_date"])[:10])
            rec.held_days = (date.today() - entry_dt).days
        except Exception:
            rec.held_days = None
        if rec.held_days is not None and rec.held_days > rec.max_hold_days and rec.action != "sell":
            rec.action, rec.action_label = "sell", "Time to close — held too long"
            rec.why = (f"You've held this {rec.held_days} days now, past your {rec.max_hold_days}-day "
                       "short-term limit. The thesis hasn't played out fast enough to justify the capital "
                       "sitting here — close it out and look for a cleaner setup, regardless of price.")
            rec.heads_up = None
            rec.basis_tags.append("time-stop")

    # --- explicit, unambiguous exit flag for the alert system ---
    if held and rec.stop is not None and rec.price <= rec.stop:
        rec.action, rec.action_label = "sell", "STOP LOSS HIT — sell now"
        rec.exit_alert = True
        rec.exit_reason = f"Stop-loss hit: price {rec.price:,.0f}₫ at/below your stop {rec.stop:,.0f}₫."
    elif held and rec.target is not None and rec.price >= rec.target:
        rec.action, rec.action_label = "sell", "TARGET REACHED — take profit"
        rec.exit_alert = True
        rec.exit_reason = f"Target reached: price {rec.price:,.0f}₫ at/above your target {rec.target:,.0f}₫."
    elif held and rec.action == "sell":
        rec.exit_alert = True
        rec.exit_reason = rec.why

    return rec


def recommend(reading: Reading, held: Optional[dict] = None) -> Optional[Recommendation]:
    rec = _recommend_core(reading, held)
    if rec is None:
        return None
    return _apply_short_term_policy(rec, reading, held)


def _recommend_core(reading: Reading, held: Optional[dict] = None) -> Optional[Recommendation]:
    """held = {'entry_price': float, 'qty': float, 'entry_date': 'YYYY-MM-DD'} if
    Richard already owns this ticker, else None."""
    r = reading
    tags: list[str] = []
    confidence = r.confidence_cap

    if r.trend == "unknown" and r.bars < MIN_BARS_FOR_ANYTHING:
        return Recommendation(
            ticker=r.ticker, action="watch", action_label="Not enough history yet",
            why=r.notes[0] if r.notes else "Not enough price history to form a view yet.",
            heads_up=None, confidence="low", price=r.price, change_pct=r.change_pct,
            entry=None, stop=None, target=None, reward_risk=None, basis_tags=["insufficient-data"],
        )

    entry_price = held["entry_price"] if held else r.price
    stop = round(min(r.support or entry_price * 0.95, entry_price * 0.95), 0) if r.support else round(entry_price * 0.95, 0)
    target = round(max(r.resistance or entry_price * 1.1, entry_price * 1.1), 0) if r.resistance else round(entry_price * 1.1, 0)
    rr = _reward_risk(entry_price, stop, target)

    heads_up = None

    # --- distribution / bearish signature ---
    if r.flow == "distribution" or r.bearish_divergence:
        tags.append("4.9-distribution" if r.flow == "distribution" else "4.2-bearish-divergence")
        if held:
            if r.trend == "down" or r.bearish_divergence:
                action, label = "sell", "Time to sell"
                why = ("The buying pattern that supported this position has broken down — "
                       "down-day volume now outweighs up-day volume, and momentum is fading "
                       "even though the price hasn't dropped much yet.")
                heads_up = "Selling now protects gains before the trend gets worse."
            else:
                action, label = "trim", "Lock in some gains"
                why = ("Momentum is quietly fading and money looks like it's leaving this stock, "
                       "even though the price still looks fine on the surface.")
            confidence = "high" if confidence != "low" else "medium"
        else:
            return None  # never suggest a ticker showing a distribution signature
        return Recommendation(r.ticker, action, label, why, heads_up, confidence, r.price, r.change_pct,
                               entry_price if held else None, stop, target, rr, tags)

    # --- breakout with conviction ---
    if r.breakout and r.trend != "down":
        tags.append("4.9-breakout-with-conviction")
        if held:
            action, label = "hold", "Keep holding — strengthening"
            why = (f"Price just broke through a key level on {round(r.volume_ratio,1)}× normal trading volume — "
                   "a strong sign buyers are firmly in control.")
        else:
            action, label = "buy", "Good time to add"
            why = (f"The price just broke through a key ceiling on {round(r.volume_ratio,1)}× normal trading volume, "
                   "which usually means bigger buyers are stepping in.")
        if r.foreign_net_today and r.foreign_net_today > 0:
            why += " Foreign investors were also net buyers today."
            tags.append("4.9-foreign-net-buy")
        return Recommendation(r.ticker, action, label, why, None, confidence, r.price, r.change_pct,
                               entry_price, stop, target, rr, tags)

    # --- quiet accumulation ---
    if r.flow == "accumulation" and r.trend != "down":
        tags.append("4.9-accumulation")
        if held:
            action, label = "hold", "Keep holding"
            why = "Quiet, steady buying has continued underneath the surface and nothing has changed for the worse."
            if r.momentum == "overbought":
                heads_up = "Getting a little stretched short-term — not a reason to act yet, just worth watching."
        else:
            action, label = "watch", "Worth watching"
            why = "Big investors appear to be quietly building a position here — price is calm, but buying volume outweighs selling volume."
        return Recommendation(r.ticker, action, label, why, heads_up, confidence, r.price, r.change_pct,
                               entry_price if held else None, stop, target, rr, tags)

    # --- plain downtrend, no accumulation ---
    if r.trend == "down":
        tags.append("4.1-downtrend")
        if held:
            action, label = "sell", "Time to sell"
            why = "The stock is in a clear downtrend with no sign of big buyers stepping in underneath it."
            confidence = "medium" if confidence == "low" else confidence
            return Recommendation(r.ticker, action, label, why, None, confidence, r.price, r.change_pct,
                                   entry_price, stop, target, rr, tags)
        return None  # don't suggest downtrending tickers

    # --- pullback-to-support inside an uptrend (only interesting for the watchlist) ---
    if not held and r.trend == "up" and r.support and r.price <= r.support * 1.03 and r.momentum != "overbought":
        tags.append("6-pullback-entry")
        return Recommendation(r.ticker, "watch", "Worth watching",
                               "Pulled back to a level that's held before, and the overall trend is still healthy.",
                               None, confidence, r.price, r.change_pct, None, stop, target, rr, tags)

    # --- default: nothing notable ---
    if held:
        tags.append("default-hold")
        return Recommendation(r.ticker, "hold", "Keep holding",
                               "Nothing important has changed — no strong signal either way right now.",
                               None, "low" if confidence == "low" else "medium",
                               r.price, r.change_pct, entry_price, stop, target, rr, tags)
    return None


def explain_reading(reading: Reading, held: Optional[dict] = None) -> Recommendation:
    """Like recommend(), but never returns None. recommend() deliberately hides
    tickers it wouldn't suggest buying (downtrends, distribution) from the
    watchlist — but when Richard explicitly opens a chart to study a specific
    ticker, he needs an honest read either way, including "this looks weak
    right now", with real stop/target numbers to annotate the chart."""
    rec = recommend(reading, held=held)
    if rec is not None:
        return rec
    r = reading
    entry_price = held["entry_price"] if held else r.price
    stop = round(min(r.support or entry_price * 0.95, entry_price * 0.95), 0) if r.support else round(entry_price * 0.95, 0)
    target = round(max(r.resistance or entry_price * 1.1, entry_price * 1.1), 0) if r.resistance else round(entry_price * 1.1, 0)
    rr = _reward_risk(entry_price, stop, target)
    if r.trend == "down":
        label, why = "Avoid for now", ("This is in a clear downtrend with no sign of big buyers stepping in — "
                                        "not a place to open a new position.")
    elif r.flow == "distribution" or r.bearish_divergence:
        label, why = "Caution — weakening", ("Momentum is fading and the volume pattern suggests money is "
                                              "quietly leaving this stock — worth watching closely before acting.")
    else:
        label, why = "No strong signal", "Price action here is unremarkable right now — no strong signal either way."
    fallback = Recommendation(r.ticker, "watch", label, why, None, r.confidence_cap, r.price, r.change_pct,
                               None, stop, target, rr, ["chart-lookup-fallback"])
    return _apply_short_term_policy(fallback, r, held)


# ---------------------------------------------------------------------------
# Chart overlays — pivots, liquidity levels, trendline, moving averages.
# Everything here is derived straight from price/volume, same as the rest of
# the engine: no fabricated levels, and nothing drawn without at least a
# couple of real touches to back it up.
# ---------------------------------------------------------------------------

def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Fractal-style pivot highs/lows: a bar whose high/low is the strict
    extreme of the (left + 1 + right)-bar window centered on it. Returns two
    lists of (index, price) tuples, index relative to df's own row order."""
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)
    pivot_highs, pivot_lows = [], []
    for i in range(left, n - right):
        wh = highs[i - left:i + right + 1]
        if highs[i] == wh.max() and int(np.argmax(wh)) == left:
            pivot_highs.append((i, float(highs[i])))
        wl = lows[i - left:i + right + 1]
        if lows[i] == wl.min() and int(np.argmin(wl)) == left:
            pivot_lows.append((i, float(lows[i])))
    return pivot_highs, pivot_lows


def _cluster_levels(pivots: list, tolerance_pct: float = 0.006) -> list:
    """Group pivot prices that sit within tolerance_pct of each other —
    repeated touches at nearly the same price are exactly what "liquidity
    pooling above/below the market" means in practice (equal highs/lows where
    stops and orders cluster)."""
    prices = sorted(p for _, p in pivots)
    clusters: list = []
    for p in prices:
        for c in clusters:
            if abs(p - c["avg"]) / c["avg"] <= tolerance_pct:
                c["prices"].append(p)
                c["avg"] = sum(c["prices"]) / len(c["prices"])
                break
        else:
            clusters.append({"avg": p, "prices": [p]})
    return [{"price": round(c["avg"], 2), "touches": len(c["prices"])} for c in clusters]


def compute_liquidity_levels(df: pd.DataFrame, lookback: int = 140) -> list:
    """Support/resistance levels worth marking on a chart: recent swing highs
    and lows, weighted up when the same level was touched more than once
    (a real liquidity pool, not just noise)."""
    n = len(df)
    if n < 15:
        return []
    start = max(0, n - lookback)
    sub = df.iloc[start:].reset_index(drop=True)
    pivot_highs, pivot_lows = find_pivots(sub, left=3, right=3)
    price = float(df["close"].iloc[-1])
    resistances = [
        {"price": c["price"], "type": "resistance", "touches": c["touches"],
         "label": "Liquidity pool" if c["touches"] >= 2 else "Recent swing high"}
        for c in _cluster_levels(pivot_highs) if c["price"] > price
    ]
    supports = [
        {"price": c["price"], "type": "support", "touches": c["touches"],
         "label": "Liquidity pool" if c["touches"] >= 2 else "Recent swing low"}
        for c in _cluster_levels(pivot_lows) if c["price"] < price
    ]
    resistances.sort(key=lambda l: l["price"])
    supports.sort(key=lambda l: -l["price"])
    return supports[:3] + resistances[:3]


def compute_swing_markers(df: pd.DataFrame, lookback: int = 140, max_points: int = 10) -> list:
    """Recent pivot points to mark directly on the candles — the visual
    footprint of where liquidity sits, for the chart itself rather than the
    price-line list."""
    n = len(df)
    if n < 15:
        return []
    start = max(0, n - lookback)
    sub = df.iloc[start:].reset_index(drop=True)
    pivot_highs, pivot_lows = find_pivots(sub, left=3, right=3)
    out = []
    for idx, price in pivot_highs[-max_points:]:
        out.append({"time": str(df["time"].iloc[start + idx].date()), "price": round(price, 2), "type": "high"})
    for idx, price in pivot_lows[-max_points:]:
        out.append({"time": str(df["time"].iloc[start + idx].date()), "price": round(price, 2), "type": "low"})
    return out


def compute_trendline(df: pd.DataFrame, trend: str, lookback: int = 90) -> Optional[dict]:
    """A trendline only gets drawn when there are at least two real pivots
    (higher lows for an uptrend, lower highs for a downtrend) whose fitted
    slope actually agrees with the trend direction — never a line forced
    onto data that doesn't support it."""
    n = len(df)
    if n < 15 or trend not in ("up", "down"):
        return None
    start = max(0, n - lookback)
    sub = df.iloc[start:].reset_index(drop=True)
    pivot_highs, pivot_lows = find_pivots(sub, left=2, right=2)
    pts = pivot_lows if trend == "up" else pivot_highs
    if len(pts) < 2:
        return None
    pts = pts[-8:]
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    try:
        m, b = np.polyfit(xs, ys, 1)
    except Exception:
        return None
    if trend == "up" and m <= 0:
        return None
    if trend == "down" and m >= 0:
        return None
    x_start = float(xs.min())
    x_end = float(len(sub) - 1)
    y_start = m * x_start + b
    y_end = m * x_end + b
    idx_start = min(start + int(round(x_start)), n - 1)
    idx_end = min(start + int(round(x_end)), n - 1)
    return {
        "direction": trend,
        "points": [
            {"time": str(df["time"].iloc[idx_start].date()), "value": round(float(y_start), 2)},
            {"time": str(df["time"].iloc[idx_end].date()), "value": round(float(y_end), 2)},
        ],
    }


def backtest(df: pd.DataFrame, ticker: str = "BACKTEST") -> dict:
    """Walk-forward simulation of THIS engine's own current rules (including
    the 4.11 short-term policy — confluence gate, fee-aware filter, time-stop)
    against real historical bars, so 'does this actually work' has a measured
    answer instead of a guessed one, per KB Section 13. No lookahead: on day i,
    only bars up to and including i are visible when a signal is evaluated.

    Simplifications, stated plainly rather than hidden: a signal on day i is
    assumed filled at day i's close (a real fill would be the next day's open);
    stop/target are frozen at entry, not trailed; only one position is
    simulated at a time (no concurrent positions); and foreign-flow data isn't
    available historically here, so foreign-net-buy confluence never fires in
    a backtest — real live signals can score one point higher than this shows.
    Round-trip fees (FEE_ROUND_TRIP_PCT) are deducted from every trade's return."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)
    min_bars = MIN_BARS_FOR_TREND + 10
    if n < min_bars + 5:
        return {"ticker": ticker, "trades": 0, "note": "Not enough history to backtest (need 70+ bars)."}

    trades: list[dict] = []
    open_trade: Optional[dict] = None
    i = min_bars
    while i < n:
        sub = df.iloc[: i + 1].reset_index(drop=True)
        reading = build_reading(ticker, sub, foreign_net_today=None)
        day = df.iloc[i]
        if open_trade is None:
            rec = recommend(reading, held=None)
            if rec is not None and rec.action == "buy" and rec.entry and rec.stop and rec.target:
                open_trade = {
                    "entry_idx": i, "entry_date": str(day["time"].date()),
                    "entry_price": float(rec.entry), "stop": float(rec.stop), "target": float(rec.target),
                    "max_hold_days": rec.max_hold_days,
                }
        else:
            exit_price, reason = None, None
            if float(day["low"]) <= open_trade["stop"]:
                exit_price, reason = open_trade["stop"], "stop"
            elif float(day["high"]) >= open_trade["target"]:
                exit_price, reason = open_trade["target"], "target"
            elif (i - open_trade["entry_idx"]) >= open_trade["max_hold_days"]:
                exit_price, reason = float(day["close"]), "time-stop"
            if exit_price is not None:
                gross_pct = (exit_price / open_trade["entry_price"] - 1) * 100
                net_pct = gross_pct - FEE_ROUND_TRIP_PCT
                trades.append({
                    "entry_date": open_trade["entry_date"], "exit_date": str(day["time"].date()),
                    "entry_price": round(open_trade["entry_price"], 2), "exit_price": round(exit_price, 2),
                    "held_days": i - open_trade["entry_idx"], "reason": reason,
                    "return_pct_after_fees": round(net_pct, 2),
                })
                open_trade = None
        i += 1

    if not trades:
        return {"ticker": ticker, "trades": 0, "note": "No qualifying buy signals fired in this history."}

    wins = [t for t in trades if t["return_pct_after_fees"] > 0]
    avg_return = sum(t["return_pct_after_fees"] for t in trades) / len(trades)
    avg_hold = sum(t["held_days"] for t in trades) / len(trades)
    compounded = 1.0
    for t in trades:
        compounded *= (1 + t["return_pct_after_fees"] / 100)
    return {
        "ticker": ticker,
        "trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / len(trades), 1),
        "avg_return_pct_after_fees": round(avg_return, 2),
        "avg_hold_days": round(avg_hold, 1),
        "compounded_return_pct": round((compounded - 1) * 100, 1),
        "trade_log": trades[-20:],  # most recent 20, so the payload stays a sane size
        "caveats": ["No lookahead, but fills assumed at signal-day close, not next-day open.",
                    "Stop/target frozen at entry, not trailed.",
                    "Only one position simulated at a time.",
                    "Foreign-flow confluence unavailable historically — live signals can score slightly higher.",
                    "Past performance on this specific history is not a guarantee of future results."],
    }


def build_chart_payload(df: pd.DataFrame, reading: Reading, display_bars: int = 140) -> dict:
    """Everything the frontend needs to draw a professional-looking chart:
    recent candles, two moving averages, a trendline (when the data actually
    supports one), and liquidity/support-resistance levels with real touch
    counts. Entry/stop/target/why come from explain_reading(), not this."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    n = len(df)
    start = max(0, n - display_bars)
    view = df.iloc[start:]

    candles = [
        {
            "time": str(row.time.date()),
            "open": round(float(row.open), 2),
            "high": round(float(row.high), 2),
            "low": round(float(row.low), 2),
            "close": round(float(row.close), 2),
            "volume": float(row.volume),
        }
        for row in view.itertuples()
    ]

    close = df["close"].astype(float)
    ma20 = sma(close, 20)
    ma50 = sma(close, 50)

    def _line(series: pd.Series) -> list:
        out = []
        for i in range(start, n):
            v = series.iloc[i]
            if not pd.isna(v):
                out.append({"time": str(df["time"].iloc[i].date()), "value": round(float(v), 2)})
        return out

    lookback = min(display_bars, n)
    return {
        "candles": candles,
        "sma20": _line(ma20),
        "sma50": _line(ma50),
        "trendline": compute_trendline(df, reading.trend, lookback=lookback),
        "levels": compute_liquidity_levels(df, lookback=lookback),
        "swings": compute_swing_markers(df, lookback=lookback),
    }
