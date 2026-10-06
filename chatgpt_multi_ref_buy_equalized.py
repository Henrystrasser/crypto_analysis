#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Reference BUY Backtest + Refcoin-Equalizer (feste Faktoren)
=================================================================

Idee
----
Die Refcoins (Default BTC / SOL / ETH / XRP / BNB) sollen etwa gleich oft
"abstimmen". Dafür bekommt jeder Refcoin EINMAL einen Faktor, der seine
Bewegung multipliziert:

    Stimme UP   :  move_i * faktor_i >= +Schwelle
    Stimme DOWN :  move_i * faktor_i <= -Schwelle     (gleiche Faktoren)

    Beispiel: BTC +1.8%, Faktor 1.30 -> 2.34% >= 2.0% -> BTC stimmt mit.

Ruhige Coins (BTC) bekommen Faktor > 1 (verstärkt), volatile (SOL) < 1.
Effektive Schwelle je Refcoin = Schwelle / faktor_i. Jede Stimme bleibt
binär und zählt exakt 1; ein Event feuert, wenn das Quorum erreicht ist.

Eichung (einmal, deterministisch)
---------------------------------
- Eichfenster = die ersten CALIB_DAYS Tage ab dem effektiven Start
  (Refs UND Target haben Daten), je Event-Fenster.
- Für jeden Refcoin wird die Trefferquote P(|move_i| >= CALIB_THRESHOLD_PCT)
  bestimmt; Ziel-Quote r = Mittel dieser Quoten (Gesamtzahl der Stimmen
  bleibt also etwa gleich).
- Effektive Schwelle e_i = (1 - r)-Quantil von |move_i|  ->  jeder
  Refcoin erreicht im Eichfenster dieselbe Quote r.
- faktor_i = CALIB_THRESHOLD_PCT / e_i, auf 2 Nachkommastellen gerundet.
- Alternativ MANUAL_FACTORS setzen (dann keine Eichung; Test = ganzer
  effektiver Zeitraum).

Test (ohne Look-ahead)
----------------------
- Getestet wird NUR nach dem Eichfenster; BASE und EQ laufen über exakt
  denselben Zeitraum.
- Für JEDE Kombination EVENT_WINDOWS_MIN x THRESHOLDS_PCT x MIN_REFS x
  UP/DOWN: BASE = gleiche Schwelle für alle Refs, EQ = Bewegung x Faktor
  gegen dieselbe Schwelle. A/B-Tabelle je Kombination (alle Holds),
  danach Kompaktübersicht + Zählung, wie oft EQ BASE schlägt.
- Einstieg = Target-Close der Signal-Kerze, Exit = Close genau Hold später;
  je Hold non-overlapping (1 Slot, Cooldown = Hold), keine Trades über das
  Enddatum hinaus.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from laggard_common import (
    add_time_range_arguments,
    ms_to_berlin_str,
    resolve_event_window,
)


# ============================================================
# CONFIG
# ============================================================

REF_SYMBOLS: List[str] = ["BTC", "SOL", "ETH", "XRP", "BNB"]
TARGET_SYMBOL = "ENA"

# Event-Fenster (Minuten).
EVENT_WINDOWS_MIN = [60]

# Steigungs-/Fall-Schwellen; A/B-Vergleich für JEDE Schwelle.
THRESHOLDS_PCT = [1.0, 1.3,1.5,1.8, 2.0,2.3,2.6, 3.0]

# Quoren (mindestens so viele Refcoins müssen stimmen).
MIN_REFS = [1, 2, 3, 4, 5]

# Equalizer-Eichung: einmal am Anfang, auf den ersten CALIB_DAYS Tagen.
CALIB_DAYS = 180
CALIB_THRESHOLD_PCT = 2.0

# Manuelle Faktoren statt Eichung, z.B. {"BTC": 1.3, "SOL": 0.75, ...}.
# None = eichen.
MANUAL_FACTORS: Optional[Dict[str, float]] = None

# Holds.
HOLD_HORIZONS: Dict[str, int] = {
    "3h": 180,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
    "48h": 2880,
    "60h": 3600,
    "72h": 4320,
    "78h": 4680,
    "84h": 5040,
    "90h": 5400,
    "96h": 5760,
    "120h": 7200,
    "168h": 10080,
    "192h": 11520,
}

# Hold für die Kompaktübersicht / EQ-vs-BASE-Zählung.
SUMMARY_HOLD = "24h"

FROM_DATE: Optional[str] = "2024-01-01"
TO_DATE: Optional[str] = "2026-12-28"
LOOKBACK_DAYS = 400

KLINE_INTERVAL = "30m"

FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0

RUN_RANDOM = False
RANDOM_SEED = 46

SHOW_TRADES = False

BINANCE = "https://data-api.binance.vision"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "multi-ref-buy-equalized/2.0"})


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_symbol(raw: str) -> tuple[str, str]:
    s = (raw or "").strip().upper()

    if s.endswith("USDT") and len(s) > 4:
        base = s[:-4]
        return base, s

    if not s:
        raise ValueError("Leerer Coin.")

    return s, f"{s}USDT"


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-Reference BUY Backtest + Refcoin-Equalizer (feste Faktoren)"
    )
    p.add_argument(
        "--ref", "--refs", nargs="+", default=None, metavar="COIN",
        help=f"Referenzcoins (Default: {' '.join(REF_SYMBOLS)})",
    )
    p.add_argument(
        "--coin", "--target", dest="target", default=None, metavar="COIN",
        help=f"Ziel-Coin (Default: {TARGET_SYMBOL})",
    )
    p.add_argument(
        "--random", dest="run_random", action="store_true", default=None,
        help=f"RANDOM-Baseline berechnen (Default: RUN_RANDOM={RUN_RANDOM})",
    )
    p.add_argument(
        "--no-random", dest="run_random", action="store_false", default=None,
        help="RANDOM-Baseline überspringen (überschreibt RUN_RANDOM)",
    )
    p.add_argument(
        "--calib-days", type=int, default=None,
        help=f"Eichfenster in Tagen (Default: {CALIB_DAYS})",
    )
    add_time_range_arguments(p)
    return p.parse_args(argv)


def interval_to_minutes(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    if unit == "m":
        return n
    if unit == "h":
        return n * 60
    if unit == "d":
        return n * 1440
    raise ValueError(f"Unsupported interval: {interval}")


BAR_MIN = interval_to_minutes(KLINE_INTERVAL)
BAR_MS = BAR_MIN * 60_000

if any(w % BAR_MIN != 0 for w in EVENT_WINDOWS_MIN):
    raise SystemExit("Alle EVENT_WINDOWS_MIN müssen durch KLINE_INTERVAL teilbar sein.")

if any(h % BAR_MIN != 0 for h in HOLD_HORIZONS.values()):
    raise SystemExit("Alle HOLD_HORIZONS müssen durch KLINE_INTERVAL teilbar sein.")

if SUMMARY_HOLD not in HOLD_HORIZONS:
    raise SystemExit("SUMMARY_HOLD muss in HOLD_HORIZONS stehen.")

WINDOW_BARS = {w: w // BAR_MIN for w in EVENT_WINDOWS_MIN}
HOLD_BARS = {h: mins // BAR_MIN for h, mins in HOLD_HORIZONS.items()}
MAX_HOLD_MIN = max(HOLD_HORIZONS.values())
HORIZON_NAMES = list(HOLD_HORIZONS.keys())

BERLIN_TZ = ZoneInfo("Europe/Berlin")


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def sleep_polite(seconds: float = 0.15) -> None:
    time.sleep(seconds)


# ============================================================
# BINANCE
# ============================================================

def binance_usdt_symbols() -> set[str]:
    r = SESSION.get(f"{BINANCE}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    return {
        s["symbol"]
        for s in r.json()["symbols"]
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
    }


def resolve_pair(raw: str, usdt_set: set[str]) -> tuple[str, str]:
    base, pair = normalize_symbol(raw)
    aliases = {
        "RNDR": "RENDERUSDT",
        "RENDER": "RENDERUSDT",
        "MATIC": "MATICUSDT",
        "POL": "POLUSDT",
    }
    if base in aliases and aliases[base] in usdt_set:
        return base, aliases[base]
    if pair not in usdt_set:
        raise SystemExit(f"Pair {pair} nicht auf Binance Spot USDT.")
    return base, pair


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    rows: List[list] = []
    cursor = start_ms

    while cursor < end_ms:
        r = SESSION.get(
            f"{BINANCE}/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        if r.status_code == 429:
            sleep_polite(5)
            continue
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        next_cursor = data[-1][0] + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        sleep_polite()
        if len(data) < 1000:
            break

    cols = ["open_time", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(
        rows,
        columns=cols + ["close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"],
    )
    now_ms = int(time.time() * 1000)
    # Keine laufende Kerze.
    df = df[df["close_time"].astype(np.int64) < now_ms]
    df = df[cols].copy()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype(np.int64)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


# ============================================================
# RANGE
# ============================================================

def clamp_end_to_closed_candle(end: datetime) -> datetime:
    now_ms = int(time.time() * 1000)
    last_closed_open_ms = (now_ms // BAR_MS) * BAR_MS - BAR_MS
    last_closed = pd.Timestamp(last_closed_open_ms, unit="ms", tz="UTC")
    if end.tzinfo is not None:
        last_closed = last_closed.tz_convert(end.tzinfo)
    else:
        last_closed = last_closed.tz_localize(None)
    return min(end, last_closed.to_pydatetime())


def effective_range(start_ms: int, end_ms: int, *dfs: pd.DataFrame) -> Tuple[int, int]:
    """[start, end] begrenzt auf die tatsächlich vorhandenen Kerzen aller DataFrames."""
    lo, hi = int(start_ms), int(end_ms)
    for df in dfs:
        if df.empty:
            continue
        arr = df["open_time"].to_numpy(dtype=np.int64)
        lo = max(lo, int(arr.min()))
        inside = arr[arr <= int(end_ms)]
        if len(inside):
            hi = min(hi, int(inside.max()))
    return lo, hi


def berlin_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=BERLIN_TZ).strftime("%Y-%m-%d")


def block_range(from_ms: int, to_ms: int) -> str:
    """'from YYYY-MM-DD to YYYY-MM-DD' (Europe/Berlin) für Block-Header."""
    return f"from {berlin_date(from_ms)} to {berlin_date(to_ms)}"


# ============================================================
# REF-BEWEGUNGEN (vektorisiert)
# ============================================================

def build_move_matrix(
    ref_data: Dict[str, pd.DataFrame],
    refs: List[str],
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gemeinsame Kerzenzeiten aller Refs + Matrix der Window-Returns in %
    (Zeilen = Zeiten, Spalten = refs). Return je Ref über WINDOW_BARS Kerzen
    der eigenen Serie; Zeilen mit fehlendem Return werden verworfen.
    """
    common: Optional[np.ndarray] = None
    for ref in refs:
        t = ref_data[ref]["open_time"].to_numpy(dtype=np.int64)
        common = t if common is None else np.intersect1d(common, t)
    if common is None or len(common) == 0:
        return np.array([], dtype=np.int64), np.empty((0, len(refs)))

    cols = []
    for ref in refs:
        df = ref_data[ref]
        t = df["open_time"].to_numpy(dtype=np.int64)
        ret = (df["close"].pct_change(WINDOW_BARS[window]) * 100.0).to_numpy(dtype=np.float64)
        cols.append(ret[np.searchsorted(t, common)])
    moves = np.column_stack(cols)
    ok = ~np.isnan(moves).any(axis=1)
    return common[ok], moves[ok]


def votes_for(moves: np.ndarray, factors: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    """Stimme je Ref: move*faktor >= +Schwelle (UP) bzw. <= -Schwelle (DOWN)."""
    scaled = moves * factors[None, :]
    return scaled >= threshold if direction == "UP" else scaled <= -threshold


# ============================================================
# EICHUNG
# ============================================================

def calibrate_factors(
    moves: np.ndarray,
    threshold: float,
) -> Optional[np.ndarray]:
    """
    Faktor je Ref, sodass jeder Ref im Eichfenster gleich oft
    |move| * faktor >= threshold erreicht (Ziel-Quote = mittlere BASE-Quote).
    None, wenn keine Daten / keine Treffer.
    """
    if moves.size == 0:
        return None
    absm = np.abs(moves)
    rates = (absm >= threshold).mean(axis=0)
    r = float(rates.mean())
    if r <= 0.0:
        return None
    eff = np.quantile(absm, 1.0 - r, axis=0)
    if np.any(eff <= 0):
        return None
    return np.round(threshold / eff, 2)


# ============================================================
# TRADES (vektorisiert)
# ============================================================

def apply_costs(entry: float, exit_: float) -> Tuple[float, float]:
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    return entry * (1.0 + cost), exit_ * (1.0 - cost)


def trade_rois(
    tgt_times: np.ndarray,
    tgt_closes: np.ndarray,
    entry_times: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Einstieg = Close der Target-Kerze mit open_time == Signalzeit (keine
    Verschiebung bei Lücken), Exit = Close der Kerze genau Hold später;
    fehlt sie (z.B. nach dem Enddatum) -> NaN.
    """
    n = len(tgt_times)
    if n == 0 or len(entry_times) == 0:
        return np.array([], dtype=np.int64), {h: np.array([]) for h in HORIZON_NAMES}

    pos = np.searchsorted(tgt_times, entry_times)
    pc = np.minimum(pos, n - 1)
    ok = (pos < n) & (tgt_times[pc] == entry_times)
    ok &= tgt_closes[pc] > 0
    e = pc[ok]
    buy_ms = tgt_times[e]
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    buy = tgt_closes[e] * (1.0 + cost)

    rois: Dict[str, np.ndarray] = {}
    for h, bars in HOLD_BARS.items():
        xt = buy_ms + int(bars) * BAR_MS
        xp = np.searchsorted(tgt_times, xt)
        xpc = np.minimum(xp, n - 1)
        valid = (xp < n) & (tgt_times[xpc] == xt) & (tgt_closes[xpc] > 0)
        roi = np.full(len(buy_ms), np.nan)
        roi[valid] = (tgt_closes[xpc[valid]] * (1.0 - cost) / buy[valid] - 1.0) * 100.0
        rois[h] = roi
    return buy_ms, rois


def stats_from_values(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "pct_pos": None, "avg": None, "compound": None}
    n_pos = sum(v > 0 for v in vals)
    n_neg = sum(v < 0 for v in vals)
    equity = 1.0
    for v in vals:
        equity *= 1.0 + v / 100.0
    return {
        "n": len(vals),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pct_pos": 100.0 * n_pos / (n_pos + n_neg) if n_pos + n_neg else None,
        "avg": float(np.mean(vals)),
        "compound": (equity - 1.0) * 100.0,
    }


def summarize(buy_ms: np.ndarray, rois: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """
    Je Hold: 1 Slot, Cooldown = Hold (Trade bei exakt last_buy + hold
    erlaubt). Compound = Produkt der non-overlapping Trade-Returns.
    """
    result = {}
    for h in HORIZON_NAMES:
        roi = rois[h]
        valid = ~np.isnan(roi)
        b = buy_ms[valid]
        v = roi[valid]
        hold_ms = HOLD_HORIZONS[h] * 60_000
        kept: List[float] = []
        i = 0
        while i < len(b):
            kept.append(float(v[i]))
            i = int(np.searchsorted(b, b[i] + hold_ms, side="left"))
        result[h] = stats_from_values(kept)
    return result


def buy_and_hold(df: pd.DataFrame, start_ms: int, end_ms: int) -> Optional[dict]:
    sub = df[(df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)]
    if len(sub) < 2:
        return None
    entry = float(sub.iloc[0]["close"])
    exit_ = float(sub.iloc[-1]["close"])
    if entry <= 0:
        return None
    buy, sell = apply_costs(entry, exit_)
    return {
        "roi": (sell / buy - 1.0) * 100.0,
        # Anzeige als Kerzen-CLOSE.
        "buy_time": ms_to_berlin_str(int(sub.iloc[0]["open_time"]) + BAR_MS),
        "sell_time": ms_to_berlin_str(int(sub.iloc[-1]["open_time"]) + BAR_MS),
    }


# ============================================================
# RANDOM BASELINE
# ============================================================

def random_positions(
    times: np.ndarray,
    closes: np.ndarray,
    start_ms: int,
    end_ms: int,
    hold_bars: int,
    n_target: int,
    rng: np.random.Generator,
) -> List[int]:
    """Exakt n_target nicht-überlappende Zufallseinstiege (Abstand >= Hold)."""
    hold_ms = hold_bars * BAR_MS
    exit_exists = np.isin(times + hold_ms, times)
    candidates = [
        int(i)
        for i in np.flatnonzero(
            (times >= start_ms) & (times <= end_ms) & exit_exists & (closes > 0)
        )
    ]
    if not candidates or n_target <= 0:
        return []

    cand_arr = np.array(candidates, dtype=np.int64)
    t0 = int(times[cand_arr[0]])
    slot_of = {int((int(times[i]) - t0) // BAR_MS): int(i) for i in cand_arr}
    n_slots = max(slot_of) + 1
    free = n_slots - (n_target - 1) * hold_bars

    if free >= n_target:
        offsets = np.arange(n_target, dtype=np.int64) * hold_bars
        for _ in range(500):
            picks = np.sort(rng.choice(free, size=n_target, replace=False)) + offsets
            if all(int(k) in slot_of for k in picks):
                return [slot_of[int(k)] for k in picks]

    # Fallback (lückige Daten): greedy in zufälliger Reihenfolge.
    order = candidates.copy()
    rng.shuffle(order)
    accepted: List[int] = []
    for pos in order:
        t = int(times[pos])
        if all(abs(t - int(times[p])) >= hold_ms for p in accepted):
            accepted.append(pos)
        if len(accepted) >= n_target:
            break

    if len(accepted) < n_target:
        print(
            f"   Hinweis RANDOM: nur {len(accepted)}/{n_target} "
            f"nicht-überlappende Trades möglich."
        )
    return sorted(accepted, key=lambda p: int(times[p]))


def random_matched_stats(
    times: np.ndarray,
    closes: np.ndarray,
    target_n: Dict[str, int],
    start_ms: int,
    end_ms: int,
    rng: np.random.Generator,
) -> Dict[str, dict]:
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    result = {}
    for h in HORIZON_NAMES:
        positions = random_positions(
            times, closes, start_ms, end_ms, HOLD_BARS[h], target_n.get(h, 0), rng
        )
        vals = []
        for pos in positions:
            xp = int(np.searchsorted(times, int(times[pos]) + HOLD_BARS[h] * BAR_MS))
            entry, exit_ = float(closes[pos]), float(closes[xp])
            if entry > 0 and exit_ > 0:
                vals.append((exit_ * (1.0 - cost) / (entry * (1.0 + cost)) - 1.0) * 100.0)
        result[h] = stats_from_values(vals)
    return result


# ============================================================
# OUTPUT
# ============================================================

def fmt_pct(v: Optional[float]) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "n/a"
    return f"{v:+.2f}%"


def fmt_pos(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.0f}%"


def fmt_ratio(num: int, den: int) -> str:
    """num/den - 1 als Prozent; 'n/a' bei den == 0."""
    if not den:
        return "n/a"
    return f"{num / den - 1.0:+.2%}"


def quorum_label(q: int, total: int) -> str:
    return f"{q}/{total}"


def factor_line(refs: List[str], factors: np.ndarray) -> str:
    return " | ".join(f"{r} {f:.2f}" for r, f in zip(refs, factors))


def participation(votes: np.ndarray, mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Anteil der Events, in denen jeder Ref mitstimmt; Spread = max - min."""
    if not mask.any():
        return None, None
    part = votes[mask].mean(axis=0)
    return part, float(part.max() - part.min())


def fmt_part(refs: List[str], part: Optional[np.ndarray], spread: Optional[float]) -> str:
    if part is None:
        return "n/a"
    return (
        " ".join(f"{r} {p * 100:.0f}%" for r, p in zip(refs, part))
        + f" (Spread {spread * 100:.1f} pp)"
    )


def print_comparison(
    head: str,
    base_stats: Dict[str, dict],
    eq_stats: Dict[str, dict],
    bh: Optional[dict],
    rnd_base: Optional[Dict[str, dict]],
    rnd_eq: Optional[Dict[str, dict]],
) -> None:
    with_rnd = rnd_base is not None and rnd_eq is not None
    print()
    print("=" * 132)
    print(head)
    print("=" * 132)
    print(
        f"{'Hold':>7}{'Base n':>8}{'Base %pos':>10}{'Base Avg':>10}{'Base Cmp':>11}"
        f"{'Eq n':>7}{'Eq %pos':>9}{'Eq Avg':>10}{'Eq Cmp':>11}"
        f"{'Eq-Base':>11}{'Eq-vs B&H':>11}"
        + (f"{'Rnd(B) Cmp':>12}{'Rnd(E) Cmp':>12}" if with_rnd else "")
    )
    print("-" * 132)
    bh_roi = bh["roi"] if bh else None
    for h in HORIZON_NAMES:
        b = base_stats[h]
        e = eq_stats[h]
        delta = (
            e["compound"] - b["compound"]
            if e["compound"] is not None and b["compound"] is not None
            else None
        )
        eq_vs_bh = e["compound"] - bh_roi if e["compound"] is not None and bh_roi is not None else None
        rnd = ""
        if with_rnd:
            rnd = f"{fmt_pct(rnd_base[h]['compound']):>12}{fmt_pct(rnd_eq[h]['compound']):>12}"
        print(
            f"{h:>7}{b['n']:>8}{fmt_pos(b['pct_pos']):>10}{fmt_pct(b['avg']):>10}"
            f"{fmt_pct(b['compound']):>11}"
            f"{e['n']:>7}{fmt_pos(e['pct_pos']):>9}{fmt_pct(e['avg']):>10}"
            f"{fmt_pct(e['compound']):>11}"
            f"{fmt_pct(delta):>11}{fmt_pct(eq_vs_bh):>11}"
            + rnd
        )
    print("-" * 132)


def print_examples(label: str, times: np.ndarray, moves: np.ndarray, votes: np.ndarray,
                   mask: np.ndarray, refs: List[str], max_rows: int = 5) -> None:
    if not SHOW_TRADES:
        return
    idx = np.flatnonzero(mask)[:max_rows]
    if len(idx) == 0:
        return
    print(f"Beispiel-Events {label}:")
    for i in idx:
        mv = " ".join(f"{r}:{moves[i, k]:+.2f}%" for k, r in enumerate(refs))
        matched = ",".join(r for k, r in enumerate(refs) if votes[i, k])
        print(f"  {ms_to_berlin_str(int(times[i]) + BAR_MS)} | {mv} | matched={matched}")


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_cli_args(argv)

    calib_days = CALIB_DAYS if args.calib_days is None else args.calib_days
    if calib_days < 1:
        raise SystemExit("--calib-days muss >= 1 sein.")
    if any(x <= 0 for x in THRESHOLDS_PCT) or CALIB_THRESHOLD_PCT <= 0:
        raise SystemExit("Schwellen müssen > 0 sein.")

    refs_raw = list(dict.fromkeys(str(x).strip().upper() for x in (args.ref or REF_SYMBOLS)))
    target_raw = args.target or TARGET_SYMBOL
    run_random = RUN_RANDOM if args.run_random is None else args.run_random

    usdt_set = binance_usdt_symbols()
    refs: List[str] = []
    ref_pairs: Dict[str, str] = {}
    for raw in refs_raw:
        base, pair = resolve_pair(raw, usdt_set)
        if base in refs:
            continue
        refs.append(base)
        ref_pairs[base] = pair
    if not refs:
        raise SystemExit("Mindestens ein Referenzcoin nötig.")

    target_base, target_pair = resolve_pair(target_raw, usdt_set)
    if target_base in refs:
        raise SystemExit(f"Target {target_base} darf nicht gleichzeitig Referenz sein.")

    n_refs = len(refs)
    quorums = [q for q in MIN_REFS if 1 <= q <= n_refs]

    manual: Optional[np.ndarray] = None
    if MANUAL_FACTORS is not None:
        missing = [r for r in refs if r not in MANUAL_FACTORS]
        if missing:
            raise SystemExit(f"MANUAL_FACTORS fehlt für: {', '.join(missing)}")
        manual = np.array([float(MANUAL_FACTORS[r]) for r in refs])
        if np.any(manual <= 0):
            raise SystemExit("MANUAL_FACTORS müssen > 0 sein.")

    start, end, _ = resolve_event_window(
        from_s=args.from_date or FROM_DATE,
        to_s=args.to_date or TO_DATE,
        lookback_days=args.lookback,
        default_lookback=LOOKBACK_DAYS,
    )
    end = clamp_end_to_closed_candle(end)
    start_ms = utc_ms(start)
    end_ms = utc_ms(end)
    # Kein Nachladen über TO_DATE hinaus: kein Trade schließt nach dem Enddatum.
    fetch_end_ms = end_ms

    print("\n" + "=" * 118)
    print("MULTI-REFERENCE BUY BACKTEST + REFCOIN-EQUALIZER (feste Faktoren)")
    print("=" * 118)
    print(f"Refs:       {', '.join(refs)}")
    print(f"Target:     {target_base} ({target_pair})")
    print(f"Interval:   {KLINE_INTERVAL}")
    print(f"Zeitraum:   {start.strftime('%Y-%m-%d %H:%M %Z')} -> {end.strftime('%Y-%m-%d %H:%M %Z')}")
    print(
        f"Grid:       Fenster={EVENT_WINDOWS_MIN} | Schwellen={THRESHOLDS_PCT} | "
        f"Quoren={quorums} | UP/DOWN | {len(HORIZON_NAMES)} Holds"
    )
    if manual is None:
        print(
            f"Equalizer:  Faktor multipliziert die Bewegung (move x Faktor >= Schwelle); "
            f"einmal geeicht auf den ersten {calib_days} Tagen bei {CALIB_THRESHOLD_PCT:g}% "
            f"(gleiche Trefferquote je Ref)"
        )
    else:
        print(f"Equalizer:  MANUAL_FACTORS: {factor_line(refs, manual)} (keine Eichung)")
    print(f"RANDOM:     {'an' if run_random else 'aus'} (--random / --no-random, Config RUN_RANDOM)")
    print(f"Fee:        {FEE_BPS} bps/Seite | Slippage: {SLIPPAGE_BPS} bps")
    print("=" * 118)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    ref_data: Dict[str, pd.DataFrame] = {}
    print("\n1) Lade Referenzcoins:")
    for base in refs:
        pair = ref_pairs[base]
        print(f"   {base:>8} ({pair}) ...", end="", flush=True)
        df = fetch_klines(pair, KLINE_INTERVAL, start_ms, end_ms)
        df = df[df["open_time"] <= end_ms].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"\nKeine Kerzen für Referenz {pair}.")
        ref_data[base] = df
        print(f" {len(df)} Kerzen", flush=True)

    print(f"\n2) Lade Target {target_pair} ...", flush=True)
    target_df = fetch_klines(target_pair, KLINE_INTERVAL, start_ms, fetch_end_ms)
    # Target-Daten auf <= Enddatum begrenzen (Exits nie nach TO_DATE).
    target_df = target_df[target_df["open_time"] <= end_ms].reset_index(drop=True)
    if target_df.empty:
        raise SystemExit(f"Keine Kerzen für Target {target_pair}.")
    print(f"   {len(target_df)} Kerzen", flush=True)

    tgt_times = target_df["open_time"].to_numpy(dtype=np.int64)
    tgt_closes = target_df["close"].to_numpy(dtype=np.float64)

    eff_lo, eff_hi = effective_range(start_ms, end_ms, target_df, *ref_data.values())

    # Eich- und Testzeitraum (Kalendertage Europe/Berlin).
    if manual is None:
        calib_end = utc_ms(
            datetime.fromtimestamp(eff_lo / 1000.0, tz=BERLIN_TZ) + timedelta(days=calib_days)
        )
        test_lo = calib_end
    else:
        calib_end = eff_lo
        test_lo = eff_lo
    test_hi = eff_hi

    print(f"\n   Daten Refs + Target: {ms_to_berlin_str(eff_lo)} -> {ms_to_berlin_str(eff_hi)}")
    if manual is None:
        print(
            f"   Eichung:  {ms_to_berlin_str(eff_lo)} -> {ms_to_berlin_str(calib_end - 1)} "
            f"(NICHT getestet)"
        )
    if test_lo > test_hi:
        print(
            f"\n   !! Kein Testzeitraum: Nach den {calib_days} Eich-Tagen bleiben keine Daten. "
            f"Früheres --from oder kleineres --calib-days verwenden."
        )
        print("=" * 118)
        return
    test_period = block_range(test_lo, test_hi)
    print(f"   Test:     {ms_to_berlin_str(test_lo)} -> {ms_to_berlin_str(test_hi)} ({test_period})")

    bh = buy_and_hold(target_df, test_lo, test_hi)

    # --------------------------------------------------------
    # EICHUNG + A/B JE KOMBINATION
    # --------------------------------------------------------

    results: List[dict] = []

    for window in EVENT_WINDOWS_MIN:
        times, moves = build_move_matrix(ref_data, refs, window)

        if manual is None:
            cm = (times >= eff_lo) & (times < calib_end)
            factors = calibrate_factors(moves[cm], CALIB_THRESHOLD_PCT)
            print("\n" + "=" * 118)
            if factors is None:
                print(
                    f"FAKTOREN {window}m: Eichung nicht möglich (keine Daten/Treffer im "
                    f"Eichfenster) -> Fenster übersprungen"
                )
                print("=" * 118)
                continue
            print(
                f"FAKTOREN {window}m (geeicht {berlin_date(eff_lo)} bis "
                f"{berlin_date(calib_end - 1)}): {factor_line(refs, factors)}"
            )
            print("=" * 118)
            absm = np.abs(moves[cm])
            base_rate = (absm >= CALIB_THRESHOLD_PCT).mean(axis=0)
            eq_rate = (absm * factors[None, :] >= CALIB_THRESHOLD_PCT).mean(axis=0)
            print(f"   Eff. Schwelle bei {CALIB_THRESHOLD_PCT:g}%: "
                  + " | ".join(f"{r} {CALIB_THRESHOLD_PCT / f:.2f}%" for r, f in zip(refs, factors)))
            print("   Trefferquote |move| im Eichfenster  BASE: "
                  + " ".join(f"{r} {x * 100:.1f}%" for r, x in zip(refs, base_rate)))
            print("                                       EQ:   "
                  + " ".join(f"{r} {x * 100:.1f}%" for r, x in zip(refs, eq_rate)))
        else:
            factors = manual
            print("\n" + "=" * 118)
            print(f"FAKTOREN {window}m (manuell): {factor_line(refs, factors)}")
            print("=" * 118)

        tm = (times >= test_lo) & (times <= test_hi)
        t_test = times[tm]
        m_test = moves[tm]
        ones = np.ones(n_refs)

        print(f"\n3) A/B {window}m über {test_period} ...", flush=True)

        for threshold in THRESHOLDS_PCT:
            for direction in ("UP", "DOWN"):
                base_votes = votes_for(m_test, ones, threshold, direction)
                eq_votes = votes_for(m_test, factors, threshold, direction)
                base_cnt = base_votes.sum(axis=1)
                eq_cnt = eq_votes.sum(axis=1)

                for q in quorums:
                    key = (window, threshold, q, direction)
                    b_mask = base_cnt >= q
                    e_mask = eq_cnt >= q
                    nb, ne = int(b_mask.sum()), int(e_mask.sum())
                    sign = "+" if direction == "UP" else "-"
                    head = (
                        f"EQ vs BASE | {direction} | Event {window}m | {sign}{threshold:g}% | "
                        f"Quorum {quorum_label(q, n_refs)} | Target {target_pair} | {test_period}"
                    )

                    if nb == 0 and ne == 0:
                        results.append({"key": key, "skip": "keine Events (BASE und EQ)"})
                        print(f"\n{head}\n   -> übersprungen: keine Events (BASE und EQ)")
                        continue

                    b_buy, b_rois = trade_rois(tgt_times, tgt_closes, t_test[b_mask])
                    e_buy, e_rois = trade_rois(tgt_times, tgt_closes, t_test[e_mask])
                    base_stats = summarize(b_buy, b_rois)
                    eq_stats = summarize(e_buy, e_rois)

                    rnd_b = rnd_e = None
                    if run_random:
                        seed = (
                            RANDOM_SEED + window * 1000 + int(round(threshold * 100)) * 10
                            + q + (100_000 if direction == "DOWN" else 0)
                        )
                        rnd_b = random_matched_stats(
                            tgt_times, tgt_closes, {h: base_stats[h]["n"] for h in HORIZON_NAMES},
                            test_lo, test_hi, np.random.default_rng(seed + 500_000),
                        )
                        rnd_e = random_matched_stats(
                            tgt_times, tgt_closes, {h: eq_stats[h]["n"] for h in HORIZON_NAMES},
                            test_lo, test_hi, np.random.default_rng(seed + 600_000),
                        )

                    b_part, b_sp = participation(base_votes, b_mask)
                    e_part, e_sp = participation(eq_votes, e_mask)

                    print_comparison(head, base_stats, eq_stats, bh, rnd_b, rnd_e)
                    print(
                        "Eff. EQ-Schwellen: "
                        + " | ".join(f"{r} {threshold / f:.2f}%" for r, f in zip(refs, factors))
                    )
                    print(f"Raw Events: BASE={nb:,} | EQ={ne:,} | Diff={fmt_ratio(ne, nb)}")
                    print(f"Participation BASE: {fmt_part(refs, b_part, b_sp)}")
                    print(f"Participation EQ:   {fmt_part(refs, e_part, e_sp)}")
                    if run_random:
                        print(
                            "Rnd(B)/Rnd(E) = RANDOM-Einstiege im Testzeitraum mit exakt derselben "
                            "Trade-Anzahl wie BASE bzw. EQ (je Hold)."
                        )
                    if bh:
                        print(
                            f"B&H (Testzeitraum): {fmt_pct(bh['roi'])} | "
                            f"{bh['buy_time']} -> {bh['sell_time']}"
                        )
                    print_examples("BASE", t_test, m_test, base_votes, b_mask, refs)
                    print_examples("EQ", t_test, m_test, eq_votes, e_mask, refs)

                    results.append({
                        "key": key, "skip": None, "nb": nb, "ne": ne,
                        "b_sp": b_sp, "e_sp": e_sp,
                        "base": base_stats, "eq": eq_stats, "rnd_b": rnd_b, "rnd_e": rnd_e,
                    })

    # --------------------------------------------------------
    # KOMPAKTÜBERSICHT
    # --------------------------------------------------------

    sh = SUMMARY_HOLD

    def _best(stats: Dict[str, dict]) -> str:
        cands = [(stats[h]["compound"], h) for h in HORIZON_NAMES if stats[h]["compound"] is not None]
        if not cands:
            return "n/a"
        v, h = max(cands)
        return f"{h} {v:+.1f}%"

    def _sp(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v * 100:.1f}"

    print("\n" + "=" * 150)
    print(f"KOMPAKTÜBERSICHT EQ vs BASE | Hold {sh} | Test {test_period}")
    print("=" * 150)
    print(
        f"{'Event':>6} {'Thr':>5} {'Vote':>5} {'Dir':>5} {'Ev B':>6} {'Ev E':>6} "
        f"{'Sprd B':>7} {'Sprd E':>7} {'n B':>5} {'Base Cmp':>10} {'n E':>5} {'Eq Cmp':>10} "
        f"{'Eq-Base':>10} {'Best Base':>16} {'Best Eq':>16}"
        + (f" {'Rnd(B)':>10} {'Rnd(E)':>10}" if run_random else "")
    )
    print("-" * 150)

    win_sh = cmp_sh = 0
    win_all = cmp_all = 0
    for r in results:
        window, thr, q, direction = r["key"]
        lead = f"{window:>5}m {thr:>4.1f}% {quorum_label(q, n_refs):>5} {direction:>5}"
        if r["skip"]:
            print(f"{lead}   übersprungen: {r['skip']}")
            continue
        b, e = r["base"][sh], r["eq"][sh]
        delta = (
            e["compound"] - b["compound"]
            if e["compound"] is not None and b["compound"] is not None
            else None
        )
        if delta is not None:
            cmp_sh += 1
            win_sh += delta > 0
        for h in HORIZON_NAMES:
            bc, ec = r["base"][h]["compound"], r["eq"][h]["compound"]
            if bc is not None and ec is not None:
                cmp_all += 1
                win_all += ec > bc
        rnd = ""
        if run_random and r["rnd_b"] is not None:
            rnd = f" {fmt_pct(r['rnd_b'][sh]['compound']):>10} {fmt_pct(r['rnd_e'][sh]['compound']):>10}"
        print(
            f"{lead} {r['nb']:>6} {r['ne']:>6} {_sp(r['b_sp']):>7} {_sp(r['e_sp']):>7} "
            f"{b['n']:>5} {fmt_pct(b['compound']):>10} {e['n']:>5} {fmt_pct(e['compound']):>10} "
            f"{fmt_pct(delta):>10} {_best(r['base']):>16} {_best(r['eq']):>16}"
            + rnd
        )

    print("-" * 150)
    print(
        f"EQ schlägt BASE (Hold {sh}): {win_sh} von {cmp_sh} vergleichbaren Kombinationen"
        + (f" ({win_sh / cmp_sh:.0%})" if cmp_sh else "")
    )
    print(
        f"EQ schlägt BASE (alle Holds): {win_all} von {cmp_all} Kombination×Hold-Paaren"
        + (f" ({win_all / cmp_all:.0%})" if cmp_all else "")
    )
    if bh:
        print(f"B&H (Testzeitraum): {fmt_pct(bh['roi'])} | {bh['buy_time']} -> {bh['sell_time']}")
    print(
        "Sprd = Participation-Spread der Refs in den Events (max - min, pp; kleiner = "
        "gleichmäßiger). Best = Hold mit höchstem Compound. Compound = Produkt der "
        "non-overlapping Trade-Returns je Hold."
    )
    print("=" * 150)
    print("FERTIG")
    print("=" * 150)


if __name__ == "__main__":
    from datetime import datetime as _start_dt
    from zoneinfo import ZoneInfo as _StartZoneInfo
    print(f"Start: {_start_dt.now(_StartZoneInfo('Europe/Berlin')):%Y-%m-%d %H:%M:%S %Z}", flush=True)
    main()
