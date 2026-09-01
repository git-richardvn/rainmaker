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
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("rainmaker.engine")

MIN_BARS_FOR_TREND = 60      # below this, trend/MA calls are unreliable
MIN_BARS_FOR_ANYTHING = 20   # below this, we say nothing useful at all
BREAKOUT_VOLUME_RATIO = 1.5  # volume vs 20d average needed to call a breakout "conviction"


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


def _reward_risk(entry: float, stop: float, target: float) -> Optional[float]:
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return None
    return round(reward / risk, 1)


def recommend(reading: Reading, held: Optional[dict] = None) -> Optional[Recommendation]:
    """held = {'entry_price': float, 'qty': float} if Richard already owns this ticker, else None."""
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
