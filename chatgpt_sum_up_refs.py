#!/usr/bin/env python3
"""
multi_ref_strength.py

Multi-Reference Strength Backtest

Idee:
Mehrere Referenz-Coins werden gleichzeitig betrachtet.

Beispiel:
    BTC +2.0%
    ETH +1.4%
    SOL +0.8%

AVG:
    (2.0 + 1.4 + 0.8) / 3 = +1.40%

MEDIAN:
    +1.40%

Wenn AVG oder MEDIAN eine Event-Schwelle erreicht, wird der TARGET-Coin
gekauft.

UP:
    aggregierte Ref-Bewegung >= +Threshold

DOWN:
    aggregierte Ref-Bewegung <= -Threshold

Wichtig:
- UP und DOWN sind getrennte Event-Arme.
- Non-Overlap wird je Strategie/Horizont angewendet.
- BUY erfolgt sowohl bei UP als auch bei DOWN.
- AVG und MEDIAN werden separat ausgewertet.
- SUM wird nur diagnostisch ausgegeben.
- Compound ist absichtlich enthalten, weil ein häufiger kleiner Edge für
  einen Bot relevant sein kann.
- Buy&Hold ist der direkte Vergleich zum einfachen Halten des Zielcoins.

Keine Handelsempfehlung. Backtest-Skript.

Beispiel:
    python multi_ref_strength.py --coin VIRTUAL

Optional:
    python multi_ref_strength.py --coin VIRTUAL --refs BTC ETH SOL
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ============================================================
# KONFIGURATION
# ============================================================

REF_SYMBOLS = [
    "BTC",
    "XRP",
    "BNB",
    "ETH",
    "SOL"
]

TARGET_SYMBOL = "UNI"


# Event-Fenster.
# 60m = Bewegung über 60 Minuten usw.
EVENT_WINDOWS_MIN = [
    60,
    180,
    360,
]


# Event-Schwellen.
# Jede Kombination wird separat ausgewertet.
THRESHOLDS_PCT = [
    2.0,
    3.0,
    4.0
]


# Haltezeiten.
HOLD_HORIZONS: Dict[str, int] = {
    "3h": 180,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
    "48h": 2880,
    "72h": 4320,
    "96h": 5760,
    "120h": 7200,
    "192h": 11520,
}


# AVG und MEDIAN sind die eigentlichen Strategien.
STRENGTH_MODES = [
    "AVG",
    "MEDIAN",
]


# Zeitraum.
FROM_DATE: Optional[str] = "2026-01-01"
TO_DATE: Optional[str] = "2026-12-28"

# Wenn FROM_DATE = None:
LOOKBACK_DAYS = 400


# Binance Spot.
KLINE_INTERVAL = "1h"

# Kosten.
FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0


# Wenn True:
# Ein Event muss die Schwelle neu überschreiten.
#
# Beispiel:
#
# 12:00 +1.1%
# 13:00 +1.3%
# 14:00 +1.5%
#
# Bei True gibt es nicht drei Events.
# Es gibt nur das erste Überschreiten.
USE_THRESHOLD_CROSSING = True


# Seed wird momentan nur für eventuelle spätere Erweiterungen gebraucht.
RANDOM_SEED = 42


# Ausgabe.
SHOW_EVENTS = False


# Binance Data API.
BINANCE = "https://data-api.binance.vision"


SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "multi-ref-strength/1.0"
    }
)


# ============================================================
# DATENSTRUKTUREN
# ============================================================

@dataclass
class StrengthEvent:
    start_ms: int
    end_ms: int

    direction: str

    mode: str

    threshold_pct: float

    aggregate_move_pct: float

    average_move_pct: float

    median_move_pct: float

    sum_move_pct: float

    ref_count: int


# ============================================================
# SYMBOL-HILFSFUNKTIONEN
# ============================================================

def normalize_symbol(raw: str) -> Tuple[str, str]:
    """
    BTC -> BTC / BTCUSDT
    BTCUSDT -> BTC / BTCUSDT
    """
    s = (raw or "").strip().upper()

    if s.endswith("USDT") and len(s) > 4:
        base = s[:-4]
        pair = s
    else:
        base = s
        pair = f"{s}USDT"

    if not base:
        raise ValueError(f"Ungültiges Symbol: {raw!r}")

    return base, pair


def binance_usdt_symbols() -> set:
    r = SESSION.get(
        f"{BINANCE}/api/v3/exchangeInfo",
        timeout=30,
    )

    r.raise_for_status()

    return {
        s["symbol"]
        for s in r.json()["symbols"]
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
    }


def resolve_pair(raw: str, usdt_symbols: set) -> str:
    base, pair = normalize_symbol(raw)

    if pair not in usdt_symbols:
        raise SystemExit(
            f"{pair} ist kein handelbares Binance Spot USDT-Pair."
        )

    return pair


# ============================================================
# ZEIT
# ============================================================

def parse_datetime(value: str) -> datetime:
    """
    Unterstützt:
        2026-01-01
        2026-01-01 12:00
    """
    if len(value) == 10:
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        )

    return datetime.strptime(
        value,
        "%Y-%m-%d %H:%M",
    )


def datetime_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def resolve_period(
    from_date: Optional[str],
    to_date: Optional[str],
    lookback_days: int,
) -> Tuple[int, int]:

    if from_date:
        start = parse_datetime(from_date)

        if to_date:
            end = parse_datetime(to_date)
        else:
            end = datetime.now()

    else:
        end = datetime.now()

        start = end - pd.Timedelta(
            days=lookback_days
        )

    return (
        datetime_to_ms(start),
        datetime_to_ms(end),
    )


# ============================================================
# BINANCE KLINES
# ============================================================

def sleep_polite(seconds: float = 0.15) -> None:
    time.sleep(seconds)


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
            time.sleep(5)
            continue

        r.raise_for_status()

        data = r.json()

        if not data:
            break

        rows.extend(data)

        next_cursor = int(data[-1][0]) + 1

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        sleep_polite()

        if len(data) < 1000:
            break

    if not rows:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "trades",
            "tb_base",
            "tb_quote",
            "ignore",
        ],
    )

    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[col] = df[col].astype(float)

    df["open_time"] = df["open_time"].astype(np.int64)

    df = (
        df
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# INTERVALL
# ============================================================

def interval_to_minutes(interval: str) -> int:

    unit = interval[-1]
    number = int(interval[:-1])

    if unit == "m":
        return number

    if unit == "h":
        return number * 60

    if unit == "d":
        return number * 24 * 60

    raise ValueError(
        f"Unsupported interval: {interval}"
    )


BAR_MIN = interval_to_minutes(KLINE_INTERVAL)


# ============================================================
# EVENT-ERKENNUNG
# ============================================================

def calculate_ref_moves(
    ref_dfs: Dict[str, pd.DataFrame],
    window_bars: int,
) -> pd.DataFrame:

    series = []

    for symbol, df in ref_dfs.items():

        if df.empty:
            continue

        s = (
            df
            .set_index("open_time")["close"]
            .pct_change(window_bars)
            * 100.0
        )

        s.name = symbol

        series.append(s)

    if not series:
        return pd.DataFrame()

    moves = pd.concat(
        series,
        axis=1,
    )

    # Nur Zeitpunkte verwenden, an denen ALLE Refs vorhanden sind.
    moves = moves.dropna(
        how="any"
    )

    if moves.empty:
        return moves

    moves["AVG"] = moves[
        list(ref_dfs.keys())
    ].mean(axis=1)

    moves["MEDIAN"] = moves[
        list(ref_dfs.keys())
    ].median(axis=1)

    moves["SUM"] = moves[
        list(ref_dfs.keys())
    ].sum(axis=1)

    return moves


def detect_strength_events(
    moves: pd.DataFrame,
    ref_symbols: List[str],
    window_bars: int,
    threshold_pct: float,
    mode: str,
) -> List[StrengthEvent]:

    if moves.empty:
        return []

    if mode not in ("AVG", "MEDIAN"):
        raise ValueError(
            f"Ungültiger Mode: {mode}"
        )

    events: List[StrengthEvent] = []

    aggregate = moves[mode]

    times = moves.index.to_numpy()

    for i in range(len(moves)):

        value = aggregate.iloc[i]

        if pd.isna(value):
            continue

        value = float(value)

        previous = (
            float(aggregate.iloc[i - 1])
            if i > 0
            else np.nan
        )

        # ----------------------------
        # UP
        # ----------------------------

        if value >= threshold_pct:

            already_above = (
                pd.notna(previous)
                and previous >= threshold_pct
            )

            if USE_THRESHOLD_CROSSING and already_above:
                pass

            else:

                start_i = i - window_bars

                if start_i >= 0:

                    ref_values = [
                        float(moves.iloc[i][ref])
                        for ref in ref_symbols
                    ]

                    events.append(
                        StrengthEvent(
                            start_ms=int(
                                times[start_i]
                            ),
                            end_ms=int(times[i]),
                            direction="UP",
                            mode=mode,
                            threshold_pct=threshold_pct,
                            aggregate_move_pct=value,
                            average_move_pct=float(
                                moves.iloc[i]["AVG"]
                            ),
                            median_move_pct=float(
                                moves.iloc[i]["MEDIAN"]
                            ),
                            sum_move_pct=float(
                                moves.iloc[i]["SUM"]
                            ),
                            ref_count=len(ref_values),
                        )
                    )

        # ----------------------------
        # DOWN
        # ----------------------------

        if value <= -threshold_pct:

            already_below = (
                pd.notna(previous)
                and previous <= -threshold_pct
            )

            if USE_THRESHOLD_CROSSING and already_below:
                pass

            else:

                start_i = i - window_bars

                if start_i >= 0:

                    ref_values = [
                        float(moves.iloc[i][ref])
                        for ref in ref_symbols
                    ]

                    events.append(
                        StrengthEvent(
                            start_ms=int(
                                times[start_i]
                            ),
                            end_ms=int(times[i]),
                            direction="DOWN",
                            mode=mode,
                            threshold_pct=threshold_pct,
                            aggregate_move_pct=value,
                            average_move_pct=float(
                                moves.iloc[i]["AVG"]
                            ),
                            median_move_pct=float(
                                moves.iloc[i]["MEDIAN"]
                            ),
                            sum_move_pct=float(
                                moves.iloc[i]["SUM"]
                            ),
                            ref_count=len(ref_values),
                        )
                    )

    return events


# ============================================================
# KOSTEN
# ============================================================

def apply_costs(
    entry: float,
    exit_: float,
) -> Tuple[float, float]:

    cost = (
        FEE_BPS + SLIPPAGE_BPS
    ) / 10_000.0

    buy = entry * (1.0 + cost)

    sell = exit_ * (1.0 - cost)

    return buy, sell


# ============================================================
# ROI
# ============================================================

def roi_for_hold(
    df: pd.DataFrame,
    entry_pos: int,
    hold_bars: int,
) -> Optional[float]:

    exit_pos = entry_pos + hold_bars

    if exit_pos >= len(df):
        return None

    entry = float(
        df.iloc[entry_pos]["close"]
    )

    exit_ = float(
        df.iloc[exit_pos]["close"]
    )

    if entry <= 0:
        return None

    buy, sell = apply_costs(
        entry,
        exit_,
    )

    return (
        sell / buy - 1.0
    ) * 100.0


# ============================================================
# EVENT -> TRADES
# ============================================================

def find_entry_position(
    df: pd.DataFrame,
    time_ms: int,
) -> Optional[int]:

    positions = np.where(
        df["open_time"].to_numpy()
        >= time_ms
    )[0]

    if len(positions) == 0:
        return None

    return int(positions[0])


def simulate_events(
    target_df: pd.DataFrame,
    events: List[StrengthEvent],
) -> List[dict]:

    trades = []

    for event in events:

        entry_pos = find_entry_position(
            target_df,
            event.end_ms,
        )

        if entry_pos is None:
            continue

        entry_price = float(
            target_df.iloc[entry_pos]["close"]
        )

        if entry_price <= 0:
            continue

        row = {
            "event": event,
            "buy_time_ms": int(
                target_df.iloc[entry_pos]["open_time"]
            ),
            "entry_pos": entry_pos,
        }

        for name, hold_min in HOLD_HORIZONS.items():

            hold_bars = (
                hold_min // BAR_MIN
            )

            row[
                f"roi_{name}"
            ] = roi_for_hold(
                target_df,
                entry_pos,
                hold_bars,
            )

        trades.append(row)

    return trades


# ============================================================
# NON-OVERLAP
# ============================================================

def filter_non_overlapping(
    trades: List[dict],
    horizon: str,
) -> List[dict]:

    hold_ms = (
        HOLD_HORIZONS[horizon]
        * 60_000
    )

    ordered = sorted(
        trades,
        key=lambda x: x["buy_time_ms"],
    )

    result = []

    last_buy_ms = None

    for trade in ordered:

        roi = trade.get(
            f"roi_{horizon}"
        )

        if roi is None:
            continue

        buy_ms = int(
            trade["buy_time_ms"]
        )

        if (
            last_buy_ms is not None
            and buy_ms
            <= last_buy_ms + hold_ms
        ):
            continue

        result.append(trade)

        last_buy_ms = buy_ms

    return result


# ============================================================
# STATISTIK
# ============================================================

def calculate_stats(
    trades: List[dict],
    horizon: str,
) -> dict:

    col = f"roi_{horizon}"

    values = []

    for trade in trades:

        value = trade.get(col)

        if value is None:
            continue

        if np.isnan(value):
            continue

        values.append(
            float(value)
        )

    if not values:

        return {
            "n": 0,
            "positive": None,
            "avg": None,
            "compound": None,
            "sum": None,
        }

    positive = (
        sum(v > 0 for v in values)
        / len(values)
        * 100.0
    )

    average = float(
        np.mean(values)
    )

    additive = float(
        np.sum(values)
    )

    equity = 1.0

    for value in values:
        equity *= (
            1.0 + value / 100.0
        )

    compound = (
        equity - 1.0
    ) * 100.0

    return {
        "n": len(values),
        "positive": positive,
        "avg": average,
        "compound": compound,
        "sum": additive,
    }


# ============================================================
# BUY & HOLD
# ============================================================

def calculate_buy_hold(
    df: pd.DataFrame,
    start_ms: int,
    end_ms: int,
) -> Optional[float]:

    sub = df[
        (df["open_time"] >= start_ms)
        & (df["open_time"] <= end_ms)
    ]

    if len(sub) < 2:
        return None

    entry = float(
        sub.iloc[0]["close"]
    )

    exit_ = float(
        sub.iloc[-1]["close"]
    )

    if entry <= 0:
        return None

    buy, sell = apply_costs(
        entry,
        exit_,
    )

    return (
        sell / buy - 1.0
    ) * 100.0


# ============================================================
# FORMATIERUNG
# ============================================================

def fmt_pct(
    value: Optional[float],
) -> str:

    if value is None:
        return "n/a"

    if np.isnan(value):
        return "n/a"

    return f"{value:+.2f}%"


# ------------------------------------------------------------
# Block-Header: effektiver Zeitraum (Europe/Berlin)
# ------------------------------------------------------------

def _effective_range_ms(start_ms, end_ms, *time_arrays):
    """Effektiver Zeitraum: [start, end] begrenzt auf die tatsaechlich vorhandenen Kerzen
    (spaeteres Listing / frueheres Datenende)."""
    lo, hi = int(start_ms), int(end_ms)
    for ts in time_arrays:
        if ts is None or len(ts) == 0:
            continue
        arr = np.asarray(ts, dtype=np.int64)
        lo = max(lo, int(arr.min()))
        inside = arr[arr <= int(end_ms)]
        if len(inside):
            hi = min(hi, int(inside.max()))
    return lo, hi


def _block_range(from_ms, to_ms) -> str:
    """'from YYYY-MM-DD to YYYY-MM-DD' (Europe/Berlin) fuer Block-Header."""
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZoneInfo
    _tz = _ZoneInfo("Europe/Berlin")
    a = _dt.fromtimestamp(from_ms / 1000.0, tz=_tz).strftime("%Y-%m-%d")
    b = _dt.fromtimestamp(to_ms / 1000.0, tz=_tz).strftime("%Y-%m-%d")
    return f"from {a} to {b}"


def print_results(
    target: str,
    mode: str,
    window_min: int,
    threshold: float,
    events: List[StrengthEvent],
    trades: List[dict],
    bh: Optional[float],
    period: Optional[str] = None,
):

    print()
    print("=" * 110)

    print(
        f"{mode:<7} | "
        f"Event {window_min:>3}m | "
        f"{threshold:+.0f}% | "
        f"Target {target}"
        + (f" | {period}" if period else "")
    )

    print("=" * 110)

    print(
        f"{'Hold':>6} "
        f"{'n':>8} "
        f"{'%pos':>8} "
        f"{'Ø ROI':>11} "
        f"{'Compound':>12} "
        f"{'vs B&H':>11}"
    )

    print("-" * 110)

    for horizon in HOLD_HORIZONS:

        filtered = filter_non_overlapping(
            trades,
            horizon,
        )

        stats = calculate_stats(
            filtered,
            horizon,
        )

        if (
            bh is not None
            and stats["compound"] is not None
        ):
            vs_bh = (
                stats["compound"]
                - bh
            )
        else:
            vs_bh = None

        print(
            f"{horizon:>6} "
            f"{stats['n']:>8} "
            f"{fmt_pct(stats['positive']):>8} "
            f"{fmt_pct(stats['avg']):>11} "
            f"{fmt_pct(stats['compound']):>12} "
            f"{fmt_pct(vs_bh):>11}"
        )

    print()

    print(
        f"Events: {len(events)}"
    )

    print(
        f"Buy & Hold: {fmt_pct(bh)}"
    )

    if events:

        moves = [
            e.aggregate_move_pct
            for e in events
        ]

        sums = [
            e.sum_move_pct
            for e in events
        ]

        print(
            f"Ø Aggregate Event: "
            f"{np.mean(moves):+.2f}%"
        )

        print(
            f"Ø SUM der Refs: "
            f"{np.mean(sums):+.2f}%"
        )


# ============================================================
# SCORECARD
# ============================================================

def print_compact_summary(
    rows: List[dict],
):

    if not rows:
        return

    print()
    print()
    print("#" * 120)
    print("KOMPAKTE ÜBERSICHT")
    print("#" * 120)

    print(
        f"{'Mode':<8}"
        f"{'Event':>8}"
        f"{'Thr':>7}"
        f"{'Hold':>8}"
        f"{'n':>7}"
        f"{'Ø ROI':>11}"
        f"{'Compound':>12}"
        f"{'vs B&H':>11}"
    )

    print("-" * 120)

    for row in rows:

        print(
            f"{row['mode']:<8}"
            f"{row['window']:>8}m"
            f"{row['threshold']:>6.0f}%"
            f"{row['hold']:>8}"
            f"{row['n']:>7}"
            f"{fmt_pct(row['avg']):>11}"
            f"{fmt_pct(row['compound']):>12}"
            f"{fmt_pct(row['vs_bh']):>11}"
        )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Multi-Reference Strength Backtest"
        )
    )

    parser.add_argument(
        "--coin",
        "--target",
        dest="target",
        default=None,
        help=(
            f"Ziel-Coin "
            f"(Default: {TARGET_SYMBOL})"
        ),
    )

    parser.add_argument(
        "--refs",
        nargs="+",
        default=None,
        help=(
            "Referenz-Coins, z.B. "
            "BTC ETH SOL"
        ),
    )

    parser.add_argument(
        "--from",
        dest="from_date",
        default=None,
        help="Startdatum YYYY-MM-DD",
    )

    parser.add_argument(
        "--to",
        dest="to_date",
        default=None,
        help="Enddatum YYYY-MM-DD",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Lookback in Tagen",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    target_raw = (
        args.target
        or TARGET_SYMBOL
    )

    ref_raw = (
        args.refs
        if args.refs
        else REF_SYMBOLS
    )

    from_date = (
        args.from_date
        if args.from_date is not None
        else FROM_DATE
    )

    to_date = (
        args.to_date
        if args.to_date is not None
        else TO_DATE
    )

    lookback = (
        args.lookback
        if args.lookback is not None
        else LOOKBACK_DAYS
    )

    target_base, _ = normalize_symbol(
        target_raw
    )

    ref_bases = [
        normalize_symbol(x)[0]
        for x in ref_raw
    ]

    # --------------------------------------------------------
    # Zeitraum
    # --------------------------------------------------------

    start_ms, end_ms = resolve_period(
        from_date,
        to_date,
        lookback,
    )

    # Für Holds müssen wir über das Event-Ende hinaus laden.
    max_hold_min = max(
        HOLD_HORIZONS.values()
    )

    fetch_end_ms = (
        end_ms
        + max_hold_min * 60_000
    )

    # --------------------------------------------------------
    # Binance Pairs
    # --------------------------------------------------------

    print()
    print(
        "=== Multi-Reference Strength Backtest ==="
    )

    print()
    print(
        f"Target: {target_base}"
    )

    print(
        "Refs: "
        + ", ".join(ref_bases)
    )

    print(
        f"Intervall: {KLINE_INTERVAL}"
    )

    print(
        f"Event-Fenster: "
        f"{EVENT_WINDOWS_MIN}"
    )

    print(
        f"Thresholds: "
        f"{THRESHOLDS_PCT}"
    )

    print(
        f"Modes: "
        f"{', '.join(STRENGTH_MODES)}"
    )

    print()

    usdt_symbols = (
        binance_usdt_symbols()
    )

    target_pair = resolve_pair(
        target_raw,
        usdt_symbols,
    )

    ref_pairs = []

    for ref in ref_raw:

        pair = resolve_pair(
            ref,
            usdt_symbols,
        )

        if pair == target_pair:
            raise SystemExit(
                "Target darf nicht gleichzeitig "
                "Reference sein."
            )

        ref_pairs.append(pair)

    # Doppelte Refs entfernen.
    ref_pairs = list(
        dict.fromkeys(ref_pairs)
    )

    # --------------------------------------------------------
    # Daten laden
    # --------------------------------------------------------

    print(
        "Lade Referenzdaten..."
    )

    ref_dfs = {}

    for pair in ref_pairs:

        print(
            f"  {pair} ...",
            flush=True,
        )

        df = fetch_klines(
            pair,
            KLINE_INTERVAL,
            start_ms,
            fetch_end_ms,
        )

        if df.empty:
            raise SystemExit(
                f"Keine Daten für {pair}."
            )

        ref_dfs[pair] = df

        print(
            f"    {len(df)} Kerzen"
        )

    print()
    print(
        f"Lade Target {target_pair}..."
    )

    target_df = fetch_klines(
        target_pair,
        KLINE_INTERVAL,
        start_ms,
        fetch_end_ms,
    )

    if target_df.empty:
        raise SystemExit(
            f"Keine Daten für {target_pair}."
        )

    print(
        f"  {len(target_df)} Kerzen"
    )

    # --------------------------------------------------------
    # Buy & Hold
    # --------------------------------------------------------

    bh = calculate_buy_hold(
        target_df,
        start_ms,
        end_ms,
    )

    # Effektiver Zeitraum fuer die Block-Header. Events werden auf allen
    # geladenen Ref-Kerzen erkannt (bis fetch_end_ms), daher diese Grenze.
    block_period = _block_range(*_effective_range_ms(
        start_ms,
        fetch_end_ms,
        target_df["open_time"].to_numpy(),
        *[d["open_time"].to_numpy() for d in ref_dfs.values()],
    ))

    print(
        f"Buy & Hold: {fmt_pct(bh)}"
    )

    # --------------------------------------------------------
    # Events + Simulation
    # --------------------------------------------------------

    all_summary_rows = []

    for window_min in EVENT_WINDOWS_MIN:

        if window_min % BAR_MIN != 0:

            print(
                f"Überspringe {window_min}m: "
                f"nicht durch {BAR_MIN}m teilbar."
            )

            continue

        window_bars = (
            window_min // BAR_MIN
        )

        print()
        print(
            f"Berechne Events "
            f"{window_min}m..."
        )

        moves = calculate_ref_moves(
            ref_dfs,
            window_bars,
        )

        if moves.empty:
            print(
                "  Keine gemeinsamen Ref-Daten."
            )
            continue

        # ----------------------------------------------------
        # Diagnose: Beispiel der letzten gemeinsamen Bewegung
        # ----------------------------------------------------

        last = moves.iloc[-1]

        print(
            "  letzte Ref-Bewegungen: "
            + " | ".join(
                f"{ref}={last[ref]:+.2f}%"
                for ref in ref_pairs
            )
        )

        print(
            f"  AVG={last['AVG']:+.2f}% "
            f"MEDIAN={last['MEDIAN']:+.2f}% "
            f"SUM={last['SUM']:+.2f}%"
        )

        # ----------------------------------------------------
        # Modes
        # ----------------------------------------------------

        for mode in STRENGTH_MODES:

            for threshold in THRESHOLDS_PCT:

                events = (
                    detect_strength_events(
                        moves,
                        ref_pairs,
                        window_bars,
                        threshold,
                        mode,
                    )
                )

                print_results(
                    target=target_base,
                    mode=mode,
                    window_min=window_min,
                    threshold=threshold,
                    events=events,
                    trades=simulate_events(
                        target_df,
                        events,
                    ),
                    bh=bh,
                    period=block_period,
                )

                trades = simulate_events(
                    target_df,
                    events,
                )

                for horizon in HOLD_HORIZONS:

                    filtered = (
                        filter_non_overlapping(
                            trades,
                            horizon,
                        )
                    )

                    stats = (
                        calculate_stats(
                            filtered,
                            horizon,
                        )
                    )

                    if (
                        bh is not None
                        and stats["compound"] is not None
                    ):
                        vs_bh = (
                            stats["compound"]
                            - bh
                        )
                    else:
                        vs_bh = None

                    all_summary_rows.append(
                        {
                            "mode": mode,
                            "window": window_min,
                            "threshold": threshold,
                            "hold": horizon,
                            "n": stats["n"],
                            "avg": stats["avg"],
                            "compound": stats[
                                "compound"
                            ],
                            "vs_bh": vs_bh,
                        }
                    )

    # --------------------------------------------------------
    # Kompakte Übersicht
    # --------------------------------------------------------

    print_compact_summary(
        all_summary_rows
    )

    # --------------------------------------------------------
    # Ende
    # --------------------------------------------------------

    print()
    print(
        "Fertig."
    )

    print(
        "AVG/MEDIAN sind die eigentlichen "
        "Strength-Signale."
    )

    print(
        "SUM wurde nur diagnostisch "
        "ausgegeben."
    )

    print(
        "Compound = Wiederanlage nach "
        "jedem nicht überlappenden Trade."
    )

    print(
        "vs B&H = Compound - Buy&Hold."
    )


if __name__ == "__main__":
    main()