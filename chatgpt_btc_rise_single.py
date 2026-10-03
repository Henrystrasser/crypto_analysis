
#!/usr/bin/env python3
"""
01_single_ref.py

Bot-Research:
    Referenz-Coin macht ein definiertes UP/DOWN-Event
        ->
    Ziel-Coin kaufen
        ->
    für einen festen Zeitraum halten
        ->
    Kapital wieder freigeben

Beispiel:
    BTC +2% innerhalb 6h
        -> VIRTUAL kaufen
        -> 24h halten

Ziel:
    Wiederholbare Trading-Setups finden, bei denen nicht nur einzelne
    Trades gut aussehen, sondern auch viele Trades zusammen ein gutes
    Compound-Ergebnis erzeugen.

Wichtige Metriken:
    - Compound
    - Endkapital aus Startkapital
    - Anzahl Trades
    - Trades/Jahr
    - Ø ROI
    - Median ROI
    - Winrate
    - Max Drawdown
    - vs RANDOM
    - vs Buy&Hold

Jeder Hold-Horizont ist eine eigene Strategie.
Es gibt kein gemeinsames Portfolio über verschiedene Holds.

Beispiel:
    python 01_single_ref.py
    python 01_single_ref.py --ref BTC --coin VIRTUAL
    python 01_single_ref.py --ref ETH --coin AAVE

Abhängigkeiten:
    pip install pandas numpy requests
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

from laggard_common import (
    add_time_range_arguments,
    ms_to_berlin_str,
    resolve_event_window,
)


# ============================================================
# CONFIG
# ============================================================

REF_SYMBOL = "BTC"
TARGET_SYMBOL = "VIRTUAL"

FROM_DATE: Optional[str] = "2024-01-01"
TO_DATE: Optional[str] = "2026-09-30"
LOOKBACK_DAYS = 400

KLINE_INTERVAL = "1h"

# Mehrere Event-Varianten werden gleichzeitig getestet.
EVENT_WINDOWS_MIN = [180, 360, 720]       # 3h, 6h, 12h
EVENT_THRESHOLDS_PCT = [1.0, 2.0, 3.0]   # +1%, +2%, +3%

# Haltedauer
HOLD_HORIZONS: Dict[str, int] = {
    "3h": 180,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
    "48h": 2880,
    "72h": 4320,
    "96h": 5760,
    "120h": 7200,
}

# Kosten
FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0

# RANDOM-Baseline
RUN_RANDOM = True
RANDOM_SEED = 42

# Wenn None:
# RANDOM bekommt pro Setup genau so viele Trades wie die Event-Strategie.
N_RANDOM_TRADES: Optional[int] = None

# Startkapital für die Equity-Kurve.
START_CAPITAL = 10_000.0

# Nur Ergebnisse mit mindestens dieser Anzahl Trades anzeigen.
MIN_TRADES = 10

# Anzeige
SHOW_TOP_RESULTS = 30


# ============================================================
# API
# ============================================================

BINANCE = "https://data-api.binance.vision"

SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "single-ref-bot-research/1.0"}
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Event:
    start_ms: int
    end_ms: int
    move_pct: float
    direction: str


# ============================================================
# SYMBOL HELPERS
# ============================================================

def normalize_symbol(raw: str) -> Tuple[str, str]:
    """
    BTC -> ("BTC", "BTCUSDT")
    BTCUSDT -> ("BTC", "BTCUSDT")
    """
    s = (raw or "").strip().upper()

    if not s:
        raise ValueError("Leeres Symbol.")

    if s.endswith("USDT") and len(s) > 4:
        base = s[:-4]
        pair = s
    else:
        base = s
        pair = f"{s}USDT"

    return base, pair


def get_usdt_pairs() -> set[str]:
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


# ============================================================
# BINANCE DATA
# ============================================================

def sleep_polite(seconds: float = 0.2) -> None:
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

    return (
        df.drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


# ============================================================
# INTERVAL
# ============================================================

def interval_to_minutes(interval: str) -> int:

    unit = interval[-1]
    n = int(interval[:-1])

    if unit == "m":
        return n

    if unit == "h":
        return n * 60

    if unit == "d":
        return n * 24 * 60

    raise ValueError(
        f"Unsupported interval: {interval}"
    )


BAR_MIN = interval_to_minutes(KLINE_INTERVAL)


# ============================================================
# EVENT DETECTION
# ============================================================

def detect_events(
    ref_df: pd.DataFrame,
    window_min: int,
    threshold_pct: float,
) -> Tuple[List[Event], List[Event]]:

    if window_min % BAR_MIN != 0:
        raise ValueError(
            f"Event-Fenster {window_min}m muss durch "
            f"{BAR_MIN}m teilbar sein."
        )

    bars = window_min // BAR_MIN

    df = ref_df.copy()

    df["move_pct"] = (
        df["close"].pct_change(bars) * 100.0
    )

    ups: List[Event] = []
    downs: List[Event] = []

    for i in range(len(df)):

        ret = df.at[i, "move_pct"]

        if pd.isna(ret):
            continue

        start_i = i - bars

        if start_i < 0:
            continue

        prev = (
            df.at[i - 1, "move_pct"]
            if i > 0
            else np.nan
        )

        start_ms = int(df.at[start_i, "open_time"])
        end_ms = int(df.at[i, "open_time"])

        # UP
        if ret >= threshold_pct:

            # Nur der erste Treffer nach dem Überschreiten.
            if (
                pd.notna(prev)
                and prev >= threshold_pct
            ):
                pass
            else:
                ups.append(
                    Event(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        move_pct=float(ret),
                        direction="UP",
                    )
                )

        # DOWN
        if ret <= -threshold_pct:

            if (
                pd.notna(prev)
                and prev <= -threshold_pct
            ):
                pass
            else:
                downs.append(
                    Event(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        move_pct=float(ret),
                        direction="DOWN",
                    )
                )

    return ups, downs


# ============================================================
# COSTS
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
# TRADE CREATION
# ============================================================

def event_trades(
    target_df: pd.DataFrame,
    events: List[Event],
) -> List[dict]:

    if target_df.empty:
        return []

    times = target_df["open_time"].to_numpy()

    trades: List[dict] = []

    for event in events:

        # Erste Ziel-Coin-Kerze >= Event-Ende.
        pos = np.searchsorted(
            times,
            event.end_ms,
            side="left",
        )

        if pos >= len(target_df):
            continue

        entry_price = float(
            target_df.iloc[pos]["close"]
        )

        if entry_price <= 0:
            continue

        trades.append(
            {
                "entry_pos": int(pos),
                "buy_time_ms": int(
                    target_df.iloc[pos]["open_time"]
                ),
                "event_start_ms": event.start_ms,
                "event_end_ms": event.end_ms,
                "event_move_pct": event.move_pct,
                "direction": event.direction,
            }
        )

    return trades


def add_hold_returns(
    target_df: pd.DataFrame,
    trades: List[dict],
) -> None:

    for trade in trades:

        entry_pos = trade["entry_pos"]

        for name, hold_min in HOLD_HORIZONS.items():

            if hold_min % BAR_MIN != 0:
                raise ValueError(
                    f"Hold {name} ist nicht durch "
                    f"{BAR_MIN}m teilbar."
                )

            hold_bars = hold_min // BAR_MIN
            exit_pos = entry_pos + hold_bars

            if exit_pos >= len(target_df):

                trade[f"roi_{name}"] = None
                continue

            entry_raw = float(
                target_df.iloc[entry_pos]["close"]
            )

            exit_raw = float(
                target_df.iloc[exit_pos]["close"]
            )

            if entry_raw <= 0:
                trade[f"roi_{name}"] = None
                continue

            buy, sell = apply_costs(
                entry_raw,
                exit_raw,
            )

            trade[f"roi_{name}"] = (
                sell / buy - 1.0
            )


# ============================================================
# NON OVERLAP
# ============================================================

def filter_non_overlapping(
    trades: List[dict],
    horizon: str,
) -> List[dict]:

    hold_ms = (
        HOLD_HORIZONS[horizon]
        * 60_000
    )

    col = f"roi_{horizon}"

    ordered = sorted(
        trades,
        key=lambda x: x["buy_time_ms"],
    )

    kept: List[dict] = []

    last_buy: Optional[int] = None

    for trade in ordered:

        if trade.get(col) is None:
            continue

        buy_time = int(
            trade["buy_time_ms"]
        )

        if (
            last_buy is not None
            and buy_time <= last_buy + hold_ms
        ):
            continue

        kept.append(trade)

        last_buy = buy_time

    return kept


# ============================================================
# STATS
# ============================================================

def trade_stats(
    trades: List[dict],
    horizon: str,
) -> dict:

    col = f"roi_{horizon}"

    values = [
        float(t[col])
        for t in trades
        if t.get(col) is not None
    ]

    if not values:
        return {
            "n": 0,
            "avg": None,
            "median": None,
            "winrate": None,
            "compound": None,
            "end_capital": START_CAPITAL,
            "max_drawdown": None,
        }

    values_np = np.array(values)

    # --------------------------------------------------------
    # Sequential Compound
    # --------------------------------------------------------

    equity = START_CAPITAL

    peak = equity
    max_drawdown = 0.0

    for roi in values:

        equity *= 1.0 + roi

        peak = max(
            peak,
            equity,
        )

        drawdown = (
            equity / peak - 1.0
        )

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

    compound = (
        equity / START_CAPITAL - 1.0
    )

    wins = np.sum(
        values_np > 0
    )

    winrate = (
        wins / len(values_np)
    )

    return {
        "n": len(values),
        "avg": float(
            np.mean(values_np)
        ),
        "median": float(
            np.median(values_np)
        ),
        "winrate": float(winrate),
        "compound": float(compound),
        "end_capital": float(equity),
        "max_drawdown": float(
            max_drawdown
        ),
    }


# ============================================================
# RANDOM BASELINE
# ============================================================

def random_trades(
    target_df: pd.DataFrame,
    n: int,
    period_start_ms: int,
    period_end_ms: int,
    horizon: str,
    rng: np.random.Generator,
) -> List[dict]:

    if n <= 0:
        return []

    hold_bars = (
        HOLD_HORIZONS[horizon]
        // BAR_MIN
    )

    candidates: List[int] = []

    for i in range(len(target_df)):

        t = int(
            target_df.iloc[i]["open_time"]
        )

        if t < period_start_ms:
            continue

        if t > period_end_ms:
            continue

        if i + hold_bars >= len(target_df):
            continue

        candidates.append(i)

    if not candidates:
        return []

    rng.shuffle(candidates)

    selected: List[int] = []

    hold_ms = (
        HOLD_HORIZONS[horizon]
        * 60_000
    )

    for pos in candidates:

        t = int(
            target_df.iloc[pos]["open_time"]
        )

        if any(
            abs(
                t
                - int(
                    target_df.iloc[p]["open_time"]
                )
            )
            <= hold_ms
            for p in selected
        ):
            continue

        selected.append(pos)

        if len(selected) >= n:
            break

    trades = []

    for pos in sorted(selected):

        entry_raw = float(
            target_df.iloc[pos]["close"]
        )

        exit_pos = pos + hold_bars

        exit_raw = float(
            target_df.iloc[exit_pos]["close"]
        )

        buy, sell = apply_costs(
            entry_raw,
            exit_raw,
        )

        roi = sell / buy - 1.0

        trades.append(
            {
                f"roi_{horizon}": roi,
                "buy_time_ms": int(
                    target_df.iloc[pos]["open_time"]
                ),
            }
        )

    return trades


# ============================================================
# BUY & HOLD
# ============================================================

def buy_hold(
    target_df: pd.DataFrame,
    start_ms: int,
    end_ms: int,
) -> Optional[float]:

    df = target_df[
        (target_df["open_time"] >= start_ms)
        & (target_df["open_time"] <= end_ms)
    ]

    if len(df) < 2:
        return None

    entry = float(
        df.iloc[0]["close"]
    )

    exit_ = float(
        df.iloc[-1]["close"]
    )

    buy, sell = apply_costs(
        entry,
        exit_,
    )

    return sell / buy - 1.0


# ============================================================
# FORMAT
# ============================================================

def pct(value: Optional[float]) -> str:

    if value is None:
        return "   n/a"

    return f"{value * 100:+7.2f}%"


def number(value: Optional[float]) -> str:

    if value is None:
        return "   n/a"

    return f"{value:8.2f}"


# ============================================================
# CLI
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "Single Reference Coin -> Target Coin "
            "Bot Strategy Backtest"
        )
    )

    p.add_argument(
        "--ref",
        default=None,
        help=f"Referenz-Coin (Default: {REF_SYMBOL})",
    )

    p.add_argument(
        "--coin",
        default=None,
        help=f"Ziel-Coin (Default: {TARGET_SYMBOL})",
    )

    p.add_argument(
        "--no-random",
        action="store_true",
        help="Random-Baseline deaktivieren",
    )

    add_time_range_arguments(p)

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    ref_base, ref_pair = normalize_symbol(
        args.ref or REF_SYMBOL
    )

    target_base, target_pair = normalize_symbol(
        args.coin or TARGET_SYMBOL
    )

    if ref_pair == target_pair:
        raise SystemExit(
            "Referenz-Coin und Ziel-Coin dürfen nicht identisch sein."
        )

    run_random = (
        RUN_RANDOM
        and not args.no_random
    )

    start, end, window_label = resolve_event_window(
        from_s=args.from_date or FROM_DATE,
        to_s=args.to_date or TO_DATE,
        lookback_days=args.lookback,
        default_lookback=LOOKBACK_DAYS,
    )

    start_ms = int(
        start.timestamp() * 1000
    )

    end_ms = int(
        end.timestamp() * 1000
    )

    max_hold = max(
        HOLD_HORIZONS.values()
    )

    fetch_end = (
        end
        + pd.Timedelta(
            minutes=max_hold
        )
    )

    fetch_end_ms = int(
        fetch_end.timestamp() * 1000
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        f"SINGLE REF BOT TEST | "
        f"{ref_base} -> {target_base}"
    )
    print("=" * 110)

    print(
        f"Zeitraum: "
        f"{start.strftime('%Y-%m-%d')} -> "
        f"{end.strftime('%Y-%m-%d')}"
    )

    print(
        f"Event-Fenster: "
        f"{[f'{x}m' for x in EVENT_WINDOWS_MIN]}"
    )

    print(
        f"Thresholds: "
        f"{[f'{x:g}%' for x in EVENT_THRESHOLDS_PCT]}"
    )

    print(
        f"Holds: "
        f"{list(HOLD_HORIZONS.keys())}"
    )

    print(
        f"Kosten: "
        f"{FEE_BPS} bps Fee + "
        f"{SLIPPAGE_BPS} bps Slippage je Seite"
    )

    print(
        f"Startkapital: "
        f"{START_CAPITAL:,.2f}"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        f"Lade {ref_pair} ..."
    )

    ref_df = fetch_klines(
        ref_pair,
        KLINE_INTERVAL,
        start_ms,
        end_ms,
    )

    if ref_df.empty:
        raise SystemExit(
            f"Keine Daten für {ref_pair}."
        )

    print(
        f"  {len(ref_df):,} Ref-Kerzen"
    )

    print(
        f"Lade {target_pair} ..."
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
        f"  {len(target_df):,} Ziel-Kerzen"
    )

    bh = buy_hold(
        target_df,
        start_ms,
        end_ms,
    )

    print(
        f"Buy & Hold: {pct(bh)}"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results: List[dict] = []

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # EVENT LOOP
    # --------------------------------------------------------

    for window in EVENT_WINDOWS_MIN:

        for threshold in EVENT_THRESHOLDS_PCT:

            ups, downs = detect_events(
                ref_df,
                window,
                threshold,
            )

            for direction, events in [
                ("UP", ups),
                ("DOWN", downs),
            ]:

                if not events:
                    continue

                raw_trades = event_trades(
                    target_df,
                    events,
                )

                add_hold_returns(
                    target_df,
                    raw_trades,
                )

                for horizon in HOLD_HORIZONS:

                    filtered = (
                        filter_non_overlapping(
                            raw_trades,
                            horizon,
                        )
                    )

                    stats = trade_stats(
                        filtered,
                        horizon,
                    )

                    if (
                        stats["n"]
                        < MIN_TRADES
                    ):
                        continue

                    random_stats = None

                    if run_random:

                        n_random = (
                            N_RANDOM_TRADES
                            if N_RANDOM_TRADES
                            is not None
                            else stats["n"]
                        )

                        r_trades = random_trades(
                            target_df,
                            n_random,
                            start_ms,
                            end_ms,
                            horizon,
                            rng,
                        )

                        random_stats = trade_stats(
                            r_trades,
                            horizon,
                        )

                    event_compound = (
                        stats["compound"]
                    )

                    random_compound = (
                        None
                        if random_stats is None
                        else random_stats["compound"]
                    )

                    vs_random = (
                        None
                        if (
                            event_compound is None
                            or random_compound is None
                        )
                        else
                        event_compound
                        - random_compound
                    )

                    vs_bh = (
                        None
                        if (
                            event_compound is None
                            or bh is None
                        )
                        else
                        event_compound - bh
                    )

                    years = max(
                        (
                            end_ms - start_ms
                        )
                        / (
                            365.25
                            * 24
                            * 60
                            * 60
                            * 1000
                        ),
                        1e-9,
                    )

                    trades_per_year = (
                        stats["n"]
                        / years
                    )

                    results.append(
                        {
                            "window": window,
                            "threshold": threshold,
                            "direction": direction,
                            "horizon": horizon,
                            "n": stats["n"],
                            "trades_per_year":
                                trades_per_year,
                            "avg": stats["avg"],
                            "median": stats["median"],
                            "winrate":
                                stats["winrate"],
                            "compound":
                                event_compound,
                            "end_capital":
                                stats["end_capital"],
                            "drawdown":
                                stats["max_drawdown"],
                            "random_compound":
                                random_compound,
                            "vs_random":
                                vs_random,
                            "vs_bh":
                                vs_bh,
                        }
                    )

    if not results:

        raise SystemExit(
            "\nKeine Setups mit ausreichend Trades gefunden."
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    results.sort(
        key=lambda x: (
            x["window"],
            x["threshold"],
            x["direction"],
            HOLD_HORIZONS[x["horizon"]],
        )
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    print()
    print("=" * 140)
    print(
        f"TOP {min(SHOW_TOP_RESULTS, len(results))} "
        f"SETUPS — sortiert nach Compound"
    )
    print("=" * 140)

    header = (
        f"{'Event':<18}"
        f"{'Hold':>6}"
        f"{'N':>6}"
        f"{'Trades/Y':>10}"
        f"{'Ø':>10}"
        f"{'Med':>10}"
        f"{'Win':>8}"
        f"{'Cmp':>11}"
        f"{'Endkapital':>14}"
        f"{'DD':>10}"
        f"{'vsRnd':>11}"
        f"{'vsBH':>11}"
    )

    print(header)
    print("-" * len(header))

    for row in results[
        :SHOW_TOP_RESULTS
    ]:

        event_name = (
            f"{row['direction']} "
            f"{row['threshold']:g}%/"
            f"{row['window'] // 60:g}h"
        )

        print(
            f"{event_name:<18}"
            f"{row['horizon']:>6}"
            f"{row['n']:>6}"
            f"{row['trades_per_year']:>10.1f}"
            f"{pct(row['avg']):>10}"
            f"{pct(row['median']):>10}"
            f"{pct(row['winrate']):>8}"
            f"{pct(row['compound']):>11}"
            f"{row['end_capital']:>14,.0f}"
            f"{pct(row['drawdown']):>10}"
            f"{pct(row['vs_random']):>11}"
            f"{pct(row['vs_bh']):>11}"
        )

    # --------------------------------------------------------
    # BEST BY DIRECTION
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("BESTE SETUPS JE EVENT-RICHTUNG")
    print("=" * 110)

    for direction in [
        "UP",
        "DOWN",
    ]:

        subset = [
            r
            for r in results
            if r["direction"]
            == direction
        ]

        if not subset:
            continue

        best = max(
            subset,
            key=lambda x: x["compound"]
        )

        print(
            f"{direction:<5} "
            f"{best['threshold']:g}% / "
            f"{best['window'] // 60:g}h "
            f"-> Hold {best['horizon']} | "
            f"N={best['n']} | "
            f"Compound={pct(best['compound'])} | "
            f"DD={pct(best['drawdown'])} | "
            f"vsRandom={pct(best['vs_random'])} | "
            f"vsB&H={pct(best['vs_bh'])}"
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "Hinweise:"
    )
    print(
        "  Compound = sequenzielles Reinvestieren des gesamten Kapitals."
    )
    print(
        "  Jeder Hold ist eine eigene Strategie."
    )
    print(
        "  Non-Overlap verhindert gleichzeitig offene Trades "
        "innerhalb derselben Hold-Strategie."
    )
    print(
        "  vsRandom = Compound Event-Strategie minus RANDOM-Compound."
    )
    print(
        "  vsB&H = Compound Event-Strategie minus Buy&Hold."
    )
    print(
        "  DD = maximaler Drawdown der simulierten Equity-Kurve."
    )
    print(
        f"  Nur Setups mit mindestens {MIN_TRADES} Trades werden angezeigt."
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
