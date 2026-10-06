#!/usr/bin/env python3

"""
Multi-Reference Event -> Target-Coin Buy Backtest
mit direktem BASELINE vs. ANTI-CHASE Vergleich.

ANTI-CHASE:
    UP-Signal:
        Wenn TARGET in den letzten X Stunden bereits >= Y%
        gestiegen ist -> Event wird verworfen.

    DOWN-Signal:
        Wenn TARGET in den letzten X Stunden bereits <= -Y%
        gefallen ist -> Event wird verworfen.

Wichtig:
    - UP und DOWN sind beide BUY-Signale.
    - Target-Einstieg = exakter Close der Target-Kerze
      am Eventzeitpunkt.
    - Keine Vorwärtssuche.
    - Non-overlap separat je Hold.
    - Re-entry exakt am Hold-Ende erlaubt.
    - Anti-Chase wird VOR dem Non-overlap angewendet.
    - BASE und ANTI-CHASE werden direkt verglichen.

CLI:
    python3 chatgpt_multiref_antichase.py
    python3 chatgpt_multiref_antichase.py --coin ENA --ref BTC SOL ETH XRP BNB
    python3 chatgpt_multiref_antichase.py --from 2025-01-01 --to 2026-10-05
"""


from __future__ import annotations

import argparse
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

REF_SYMBOLS: List[str] = [
    "BTC",
    "SOL",
    "ETH",
    "XRP",
    "BNB",
]

TARGET_SYMBOL = "ENA"


# ============================================================
# EVENT SETTINGS
# ============================================================

EVENT_WINDOWS_MIN = [
    60,
    180,
]

THRESHOLDS_PCT = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]

MIN_REFS = [
    1,
    2,
    3,
    4,
    5,
]


# ============================================================
# HOLD SETTINGS
# ============================================================

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


# ============================================================
# DATE RANGE
# ============================================================

FROM_DATE: Optional[str] = "2024-01-01"
TO_DATE: Optional[str] = "2026-12-28"

LOOKBACK_DAYS = 400


# ============================================================
# KLINES
# ============================================================

KLINE_INTERVAL = "1h"


# ============================================================
# COSTS
# ============================================================

FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0


# ============================================================
# ANTI-CHASE TESTS
# ============================================================
#
# Beispiel:
#
#   3h / 2%
#
# Bei einem UP-Signal:
#   Target Return letzte 3h >= +2%
#   -> BLOCK
#
# Bei einem DOWN-Signal:
#   Target Return letzte 3h <= -2%
#   -> BLOCK
#
# Bewegung gegen die Signalrichtung wird NICHT blockiert.
#

ANTI_CHASE_TESTS = [
    {
        "name": "1h / 1%",
        "lookback_min": 60,
        "max_move_pct": 1.0,
    },
    {
        "name": "1h / 2%",
        "lookback_min": 60,
        "max_move_pct": 2.0,
    },
    {
        "name": "1h / 3%",
        "lookback_min": 60,
        "max_move_pct": 3.0,
    },
    {
        "name": "3h / 1%",
        "lookback_min": 180,
        "max_move_pct": 1.0,
    },
    {
        "name": "3h / 2%",
        "lookback_min": 180,
        "max_move_pct": 2.0,
    },
    {
        "name": "3h / 3%",
        "lookback_min": 180,
        "max_move_pct": 3.0,
    },
    {
        "name": "6h / 2%",
        "lookback_min": 360,
        "max_move_pct": 2.0,
    },
    {
        "name": "6h / 3%",
        "lookback_min": 360,
        "max_move_pct": 3.0,
    },
    {
        "name": "6h / 5%",
        "lookback_min": 360,
        "max_move_pct": 5.0,
    },
]


# ============================================================
# BINANCE
# ============================================================

BINANCE = "https://data-api.binance.vision"

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": "multi-ref-antichase/1.0"
    }
)


# ============================================================
# INTERVAL HELPERS
# ============================================================

def interval_to_minutes(
    interval: str,
) -> int:

    unit = interval[-1]
    n = int(interval[:-1])

    if unit == "m":
        return n

    if unit == "h":
        return n * 60

    if unit == "d":
        return n * 1440

    raise ValueError(
        f"Unsupported interval: {interval}"
    )


BAR_MIN = interval_to_minutes(
    KLINE_INTERVAL
)


WINDOW_BARS = {
    window: window // BAR_MIN
    for window in EVENT_WINDOWS_MIN
}


HOLD_BARS = {
    name: minutes // BAR_MIN
    for name, minutes in HOLD_HORIZONS.items()
}


HORIZON_NAMES = list(
    HOLD_HORIZONS.keys()
)


MAX_HOLD_MIN = max(
    HOLD_HORIZONS.values()
)


# ============================================================
# VALIDATION
# ============================================================

if any(
    window % BAR_MIN != 0
    for window in EVENT_WINDOWS_MIN
):
    raise SystemExit(
        "Alle EVENT_WINDOWS_MIN müssen "
        "durch KLINE_INTERVAL teilbar sein."
    )


if any(
    minutes % BAR_MIN != 0
    for minutes in HOLD_HORIZONS.values()
):
    raise SystemExit(
        "Alle HOLD_HORIZONS müssen "
        "durch KLINE_INTERVAL teilbar sein."
    )


if any(
    test["lookback_min"] % BAR_MIN != 0
    for test in ANTI_CHASE_TESTS
):
    raise SystemExit(
        "Alle Anti-Chase Lookbacks müssen "
        "durch KLINE_INTERVAL teilbar sein."
    )


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class MultiRefEvent:

    end_ms: int

    direction: str

    window_min: int

    threshold_pct: float

    min_refs: int

    ref_moves: Dict[str, float]

    matched_refs: List[str]

    @property
    def score(self) -> int:
        return len(self.matched_refs)


# ============================================================
# CLI
# ============================================================

def parse_cli_args(
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Multi-Reference UP/DOWN -> "
            "Target-Coin Buy Backtest "
            "mit Anti-Chase."
        )
    )

    parser.add_argument(
        "--ref",
        "--refs",
        nargs="+",
        default=None,
        metavar="COIN",
        help=(
            "Referenzcoins "
            f"(Default: {' '.join(REF_SYMBOLS)})"
        ),
    )

    parser.add_argument(
        "--coin",
        "--target",
        dest="target",
        default=None,
        metavar="COIN",
        help=(
            "Target-Coin "
            f"(Default: {TARGET_SYMBOL})"
        ),
    )

    add_time_range_arguments(
        parser
    )

    return parser.parse_args(
        argv
    )


# ============================================================
# SYMBOL HELPERS
# ============================================================

def normalize_symbol(
    raw: str,
) -> Tuple[str, str]:

    value = (
        raw
        or ""
    ).strip().upper()

    if value.endswith("USDT") and len(value) > 4:

        base = value[:-4]

        return base, value

    if not value:
        raise ValueError(
            "Leerer Coin."
        )

    return value, f"{value}USDT"


def binance_usdt_symbols() -> set[str]:

    response = SESSION.get(
        f"{BINANCE}/api/v3/exchangeInfo",
        timeout=30,
    )

    response.raise_for_status()

    return {
        item["symbol"]
        for item in response.json()["symbols"]
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
        )
    }


def resolve_pair(
    raw: str,
    usdt_set: set[str],
) -> Tuple[str, str]:

    base, pair = normalize_symbol(
        raw
    )

    aliases = {
        "RNDR": "RENDERUSDT",
        "RENDER": "RENDERUSDT",
        "MATIC": "MATICUSDT",
        "POL": "POLUSDT",
    }

    if (
        base in aliases
        and aliases[base] in usdt_set
    ):
        return (
            base,
            aliases[base],
        )

    if pair not in usdt_set:

        raise SystemExit(
            f"Pair {pair} nicht auf Binance "
            f"Spot USDT."
        )

    return (
        base,
        pair,
    )


# ============================================================
# TIME HELPERS
# ============================================================

def utc_ms(
    dt: datetime,
) -> int:

    return int(
        dt.timestamp() * 1000
    )


def clamp_end_to_closed_candle(
    end: datetime,
) -> datetime:
    """
    Wenn TO_DATE in der Zukunft liegt,
    nur bis zur letzten abgeschlossenen Kerze.
    """

    bar_ms = (
        int(BAR_MIN)
        * 60_000
    )

    now_ms = int(
        time.time() * 1000
    )

    last_closed_open_ms = (
        now_ms // bar_ms
    ) * bar_ms - bar_ms

    last_closed = pd.Timestamp(
        last_closed_open_ms,
        unit="ms",
        tz="UTC",
    )

    if end.tzinfo is not None:

        last_closed = (
            last_closed
            .tz_convert(
                end.tzinfo
            )
        )

    else:

        last_closed = (
            last_closed
            .tz_localize(None)
        )

    return min(
        end,
        last_closed.to_pydatetime(),
    )


# ============================================================
# DOWNLOAD
# ============================================================

def sleep_polite(
    seconds: float = 0.15,
) -> None:

    time.sleep(
        seconds
    )


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:

    rows: List[list] = []

    cursor = int(
        start_ms
    )

    max_retries = 8

    while cursor < end_ms:

        data = None

        for attempt in range(
            1,
            max_retries + 1,
        ):

            try:

                response = SESSION.get(
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

                if response.status_code == 429:

                    wait = min(
                        5 * attempt,
                        60,
                    )

                    print(
                        f"\n      HTTP 429 "
                        f"-> Retry in {wait}s ...",
                        end="",
                    )

                    sleep_polite(
                        wait
                    )

                    continue

                response.raise_for_status()

                data = response.json()

                break

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:

                if attempt >= max_retries:

                    raise RuntimeError(
                        f"{symbol}: Download nach "
                        f"{max_retries} Versuchen "
                        f"fehlgeschlagen: {exc}"
                    ) from exc

                wait = min(
                    2 ** (attempt - 1),
                    30,
                )

                print(
                    f"\n      Netzwerkfehler "
                    f"(Versuch {attempt}/"
                    f"{max_retries}) "
                    f"-> Retry in {wait}s ...",
                    end="",
                )

                sleep_polite(
                    wait
                )

        if data is None:

            raise RuntimeError(
                f"{symbol}: Download "
                f"fehlgeschlagen."
            )

        if not data:
            break

        rows.extend(
            data
        )

        next_cursor = (
            int(data[-1][0])
            + 1
        )

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        sleep_polite(
            0.15
        )

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

    # Nur abgeschlossene Kerzen
    now_ms = int(
        time.time() * 1000
    )

    df = df[
        df["close_time"]
        .astype(np.int64)
        < now_ms
    ]

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

    for col in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):

        df[col] = df[col].astype(
            float
        )

    df["open_time"] = (
        df["open_time"]
        .astype(np.int64)
    )

    return (
        df
        .drop_duplicates(
            "open_time"
        )
        .sort_values(
            "open_time"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# EXACT POSITION
# ============================================================

def exact_position(
    df: pd.DataFrame,
    t_ms: int,
) -> Optional[int]:
    """
    Exakte Kerzenzeit.

    KEINE Vorwärtssuche.
    """

    times = (
        df["open_time"]
        .to_numpy(
            dtype=np.int64
        )
    )

    position = int(
        np.searchsorted(
            times,
            int(t_ms),
        )
    )

    if (
        position < len(times)
        and int(times[position])
        == int(t_ms)
    ):

        return position

    return None


# ============================================================
# EVENT MOVES
# ============================================================

def build_reference_move_series(
    df: pd.DataFrame,
    windows: List[int],
) -> Dict[int, pd.Series]:

    result = {}

    for window_min in windows:

        bars = WINDOW_BARS[
            window_min
        ]

        result[window_min] = (
            df["close"]
            .pct_change(bars)
            * 100.0
        )

    return result


# ============================================================
# EVENT DETECTION
# ============================================================

def detect_multi_ref_events(
    ref_data: Dict[str, pd.DataFrame],
    windows: List[int],
    thresholds: List[float],
    min_refs_list: List[int],
    start_ms: int,
    end_ms: int,
) -> Dict[
    Tuple[int, float, int, str],
    List[MultiRefEvent],
]:

    # --------------------------------------------------------
    # Gemeinsame Kerzenzeiten
    # --------------------------------------------------------

    common_times: Optional[
        set[int]
    ] = None

    for df in ref_data.values():

        times = set(
            int(x)
            for x in df[
                "open_time"
            ].tolist()
        )

        if common_times is None:

            common_times = times

        else:

            common_times &= times

    if not common_times:
        return {}

    common_times = {
        t
        for t in common_times
        if (
            start_ms
            <= t
            <= end_ms
        )
    }

    # --------------------------------------------------------
    # Position maps
    # --------------------------------------------------------

    position_maps = {}

    for ref, df in ref_data.items():

        position_maps[ref] = {
            int(timestamp): index
            for index, timestamp
            in enumerate(
                df["open_time"].tolist()
            )
        }

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    moves_by_ref = {}

    for ref, df in ref_data.items():

        moves_by_ref[ref] = (
            build_reference_move_series(
                df,
                windows,
            )
        )

    # --------------------------------------------------------
    # Event dictionary initialisieren
    # --------------------------------------------------------

    events = {}

    for window in windows:

        for threshold in thresholds:

            for min_refs in min_refs_list:

                for direction in (
                    "UP",
                    "DOWN",
                ):

                    events[
                        (
                            window,
                            threshold,
                            min_refs,
                            direction,
                        )
                    ] = []

    # --------------------------------------------------------
    # Events erzeugen
    # --------------------------------------------------------

    for window in windows:

        for threshold in thresholds:

            for min_refs in min_refs_list:

                for direction in (
                    "UP",
                    "DOWN",
                ):

                    key = (
                        window,
                        threshold,
                        min_refs,
                        direction,
                    )

                    for timestamp in sorted(
                        common_times
                    ):

                        moves = {}

                        for ref in ref_data:

                            position = (
                                position_maps[
                                    ref
                                ].get(
                                    timestamp
                                )
                            )

                            if position is None:
                                continue

                            value = (
                                moves_by_ref[
                                    ref
                                ][window]
                                .iloc[position]
                            )

                            if pd.notna(
                                value
                            ):

                                moves[ref] = (
                                    float(value)
                                )

                        if (
                            len(moves)
                            != len(ref_data)
                        ):
                            continue

                        up_refs = [
                            ref
                            for ref, move
                            in moves.items()
                            if move >= threshold
                        ]

                        down_refs = [
                            ref
                            for ref, move
                            in moves.items()
                            if move <= -threshold
                        ]

                        if direction == "UP":

                            if (
                                len(up_refs)
                                >= min_refs
                            ):

                                events[key].append(
                                    MultiRefEvent(
                                        end_ms=timestamp,
                                        direction="UP",
                                        window_min=window,
                                        threshold_pct=threshold,
                                        min_refs=min_refs,
                                        ref_moves=moves,
                                        matched_refs=up_refs,
                                    )
                                )

                        else:

                            if (
                                len(down_refs)
                                >= min_refs
                            ):

                                events[key].append(
                                    MultiRefEvent(
                                        end_ms=timestamp,
                                        direction="DOWN",
                                        window_min=window,
                                        threshold_pct=threshold,
                                        min_refs=min_refs,
                                        ref_moves=moves,
                                        matched_refs=down_refs,
                                    )
                                )

    return events


# ============================================================
# TARGET MOVE
# ============================================================

def target_move_before_event(
    target_df: pd.DataFrame,
    event_ms: int,
    lookback_min: int,
) -> Optional[float]:
    """
    Target Return direkt vor dem Event.

    Beispiel:
        Event 12:00
        Lookback 3h

        -> Close 09:00 bis Close 12:00
    """

    current_pos = exact_position(
        target_df,
        event_ms,
    )

    if current_pos is None:
        return None

    lookback_ms = (
        lookback_min
        * 60
        * 1000
    )

    previous_ms = (
        int(event_ms)
        - lookback_ms
    )

    previous_pos = exact_position(
        target_df,
        previous_ms,
    )

    if previous_pos is None:
        return None

    current_price = float(
        target_df.iloc[
            current_pos
        ]["close"]
    )

    previous_price = float(
        target_df.iloc[
            previous_pos
        ]["close"]
    )

    if (
        current_price <= 0
        or previous_price <= 0
    ):
        return None

    return (
        current_price
        / previous_price
        - 1.0
    ) * 100.0


# ============================================================
# COSTS
# ============================================================

def apply_costs(
    entry: float,
    exit_: float,
) -> Tuple[float, float]:

    cost = (
        FEE_BPS
        + SLIPPAGE_BPS
    ) / 10_000.0

    buy_price = (
        entry
        * (1.0 + cost)
    )

    sell_price = (
        exit_
        * (1.0 - cost)
    )

    return (
        buy_price,
        sell_price,
    )


# ============================================================
# ROI
# ============================================================

def roi_for_hold(
    df: pd.DataFrame,
    entry_pos: int,
    hold_bars: int,
) -> Optional[float]:

    entry_ms = int(
        df[
            "open_time"
        ].iat[
            entry_pos
        ]
    )

    exit_ms = (
        entry_ms
        + (
            int(hold_bars)
            * int(BAR_MIN)
            * 60_000
        )
    )

    exit_pos = exact_position(
        df,
        exit_ms,
    )

    if exit_pos is None:
        return None

    entry_price = float(
        df.iloc[
            entry_pos
        ]["close"]
    )

    exit_price = float(
        df.iloc[
            exit_pos
        ]["close"]
    )

    if (
        entry_price <= 0
        or exit_price <= 0
    ):
        return None

    buy_price, sell_price = (
        apply_costs(
            entry_price,
            exit_price,
        )
    )

    return (
        sell_price
        / buy_price
        - 1.0
    ) * 100.0


# ============================================================
# SIMULATE RAW EVENTS
# ============================================================

def simulate_events(
    target_df: pd.DataFrame,
    events: List[MultiRefEvent],
    target_pair: str,
) -> List[dict]:

    trades = []

    for event in events:

        entry_pos = exact_position(
            target_df,
            event.end_ms,
        )

        if entry_pos is None:
            continue

        entry_time_ms = int(
            target_df.iloc[
                entry_pos
            ]["open_time"]
        )

        entry_price = float(
            target_df.iloc[
                entry_pos
            ]["close"]
        )

        if entry_price <= 0:
            continue

        row = {
            "coin": target_pair,

            "direction":
                event.direction,

            "window_min":
                event.window_min,

            "threshold_pct":
                event.threshold_pct,

            "min_refs":
                event.min_refs,

            "score":
                event.score,

            "matched_refs":
                ",".join(
                    event.matched_refs
                ),

            "event_time_ms":
                event.end_ms,

            "buy_time_ms":
                entry_time_ms,

            "buy_time":
                ms_to_berlin_str(
                    entry_time_ms
                    + int(BAR_MIN)
                    * 60_000
                ),

            "ref_moves":
                event.ref_moves,

            "buy_price":
                entry_price
                * (
                    1.0
                    + (
                        FEE_BPS
                        + SLIPPAGE_BPS
                    )
                    / 10_000.0
                ),
        }

        any_ok = False

        for (
            horizon_name,
            bars,
        ) in HOLD_BARS.items():

            roi = roi_for_hold(
                target_df,
                entry_pos,
                bars,
            )

            row[
                f"roi_{horizon_name}"
            ] = roi

            if roi is not None:
                any_ok = True

        if any_ok:
            trades.append(
                row
            )

    return trades


# ============================================================
# ANTI-CHASE FILTER
# ============================================================

def anti_chase_passes(
    target_df: pd.DataFrame,
    trade: dict,
    lookback_min: int,
    max_move_pct: float,
) -> Tuple[
    bool,
    Optional[float],
]:

    move_pct = target_move_before_event(
        target_df=target_df,
        event_ms=int(
            trade["event_time_ms"]
        ),
        lookback_min=lookback_min,
    )

    if move_pct is None:
        return (
            False,
            None,
        )

    direction = trade[
        "direction"
    ]

    # UP:
    # Target ist schon stark gestiegen
    if direction == "UP":

        if move_pct >= max_move_pct:

            return (
                False,
                move_pct,
            )

    # DOWN:
    # Target ist schon stark gefallen
    elif direction == "DOWN":

        if move_pct <= -max_move_pct:

            return (
                False,
                move_pct,
            )

    return (
        True,
        move_pct,
    )


def filter_anti_chase(
    target_df: pd.DataFrame,
    trades: List[dict],
    lookback_min: int,
    max_move_pct: float,
) -> Tuple[
    List[dict],
    dict,
]:

    kept = []

    blocked = 0
    missing_history = 0

    target_moves = []

    for trade in trades:

        passes, move_pct = (
            anti_chase_passes(
                target_df=target_df,
                trade=trade,
                lookback_min=lookback_min,
                max_move_pct=max_move_pct,
            )
        )

        if move_pct is None:

            missing_history += 1

            continue

        target_moves.append(
            move_pct
        )

        if not passes:

            blocked += 1

            continue

        new_trade = dict(
            trade
        )

        new_trade[
            "anti_chase_move_pct"
        ] = move_pct

        kept.append(
            new_trade
        )

    stats = {
        "raw":
            len(trades),

        "blocked":
            blocked,

        "kept":
            len(kept),

        "missing":
            missing_history,

        "removed_pct":
            (
                100.0
                * blocked
                / len(trades)
                if trades
                else 0.0
            ),

        "target_move_avg":
            (
                float(
                    np.mean(
                        target_moves
                    )
                )
                if target_moves
                else None
            ),
    }

    return (
        kept,
        stats,
    )


# ============================================================
# NON-OVERLAP
# ============================================================

def filter_non_overlapping(
    trades: List[dict],
    horizon: str,
) -> List[dict]:
    """
    Pro Hold ein Slot.

    Re-entry exakt am Hold-Ende erlaubt.
    """

    hold_ms = (
        HOLD_HORIZONS[
            horizon
        ]
        * 60_000
    )

    roi_column = (
        f"roi_{horizon}"
    )

    ordered = sorted(
        trades,
        key=lambda trade:
            int(
                trade[
                    "buy_time_ms"
                ]
            ),
    )

    kept = []

    last_buy_ms: Optional[int] = None

    for trade in ordered:

        roi = trade.get(
            roi_column
        )

        if roi is None:
            continue

        buy_ms = int(
            trade[
                "buy_time_ms"
            ]
        )

        if (
            last_buy_ms is not None
            and
            buy_ms
            <
            last_buy_ms
            + hold_ms
        ):
            continue

        kept.append(
            trade
        )

        last_buy_ms = buy_ms

    return kept


# ============================================================
# STATS
# ============================================================

def column_stats(
    trades: List[dict],
    column: str,
) -> dict:

    values = []

    for trade in trades:

        value = trade.get(
            column
        )

        if value is None:
            continue

        if (
            isinstance(
                value,
                (
                    float,
                    np.floating,
                ),
            )
            and np.isnan(value)
        ):
            continue

        values.append(
            float(value)
        )

    if not values:

        return {
            "n": 0,
            "n_pos": 0,
            "n_neg": 0,
            "pct_pos": None,
            "avg": None,
            "compound": None,
        }

    n_pos = sum(
        value > 0
        for value in values
    )

    n_neg = sum(
        value < 0
        for value in values
    )

    equity = 1.0

    for value in values:

        equity *= (
            1.0
            + value / 100.0
        )

    return {
        "n":
            len(values),

        "n_pos":
            n_pos,

        "n_neg":
            n_neg,

        "pct_pos":
            (
                100.0
                * n_pos
                / (n_pos + n_neg)
                if (
                    n_pos + n_neg
                )
                else None
            ),

        "avg":
            float(
                np.mean(values)
            ),

        "compound":
            (
                equity - 1.0
            )
            * 100.0,
    }


def summarize(
    trades: List[dict],
) -> Dict[str, dict]:

    result = {}

    for horizon in HORIZON_NAMES:

        filtered = (
            filter_non_overlapping(
                trades,
                horizon,
            )
        )

        result[horizon] = (
            column_stats(
                filtered,
                f"roi_{horizon}",
            )
        )

    return result


# ============================================================
# BUY & HOLD
# ============================================================

def buy_and_hold(
    df: pd.DataFrame,
    start_ms: int,
    end_ms: int,
) -> Optional[dict]:

    sub = df[
        (df["open_time"] >= start_ms)
        &
        (df["open_time"] <= end_ms)
    ]

    if len(sub) < 2:
        return None

    entry = float(
        sub.iloc[0]["close"]
    )

    exit_price = float(
        sub.iloc[-1]["close"]
    )

    if entry <= 0:
        return None

    buy_price, sell_price = (
        apply_costs(
            entry,
            exit_price,
        )
    )

    return {
        "roi":
            (
                sell_price
                / buy_price
                - 1.0
            )
            * 100.0,

        "buy_time":
            ms_to_berlin_str(
                int(
                    sub.iloc[0][
                        "open_time"
                    ]
                )
                + int(BAR_MIN)
                * 60_000
            ),

        "sell_time":
            ms_to_berlin_str(
                int(
                    sub.iloc[-1][
                        "open_time"
                    ]
                )
                + int(BAR_MIN)
                * 60_000
            ),
    }


# ============================================================
# FORMATTING
# ============================================================

def fmt_pct(
    value: Optional[float],
) -> str:

    if value is None:
        return "n/a"

    if (
        isinstance(
            value,
            (
                float,
                np.floating,
            ),
        )
        and np.isnan(value)
    ):
        return "n/a"

    return f"{value:+.2f}%"


def fmt_pos(
    value: Optional[float],
) -> str:

    if value is None:
        return "n/a"

    return f"{value:.0f}%"


def min_ref_label(
    min_refs: int,
    total_refs: int,
) -> str:

    if min_refs == 1:
        return (
            f"ANY (1/{total_refs})"
        )

    if min_refs == total_refs:
        return (
            f"ALL ({total_refs}/{total_refs})"
        )

    return (
        f"{min_refs}/{total_refs}"
    )


# ============================================================
# EFFECTIVE PERIOD
# ============================================================

def effective_range_ms(
    start_ms: int,
    end_ms: int,
    *time_arrays,
) -> Tuple[int, int]:

    low = int(start_ms)
    high = int(end_ms)

    for timestamps in time_arrays:

        if (
            timestamps is None
            or len(timestamps) == 0
        ):
            continue

        arr = np.asarray(
            timestamps,
            dtype=np.int64,
        )

        low = max(
            low,
            int(arr.min()),
        )

        inside = arr[
            arr <= int(end_ms)
        ]

        if len(inside):

            high = min(
                high,
                int(inside.max()),
            )

    return (
        low,
        high,
    )


def block_range(
    from_ms: int,
    to_ms: int,
) -> str:

    from zoneinfo import (
        ZoneInfo
    )

    timezone = ZoneInfo(
        "Europe/Berlin"
    )

    first = datetime.fromtimestamp(
        from_ms / 1000.0,
        tz=timezone,
    ).strftime(
        "%Y-%m-%d"
    )

    last = datetime.fromtimestamp(
        to_ms / 1000.0,
        tz=timezone,
    ).strftime(
        "%Y-%m-%d"
    )

    return (
        f"from {first} to {last}"
    )


# ============================================================
# DIRECT COMPARISON OUTPUT
# ============================================================

def print_direct_comparison(
    results: List[dict],
    total_refs: int,
) -> None:
    """
    Base und Anti-Chase direkt nebeneinander.
    """

    if not results:
        return

    ordered = sorted(
        results,
        key=lambda row: (
            row["test"],
            row["window"],
            row["threshold"],
            row["min_refs"],
            row["direction"],
            HOLD_HORIZONS[
                row["hold"]
            ],
        ),
    )

    current_test = None

    for row in ordered:

        if (
            row["test"]
            != current_test
        ):

            current_test = row[
                "test"
            ]

            print()
            print()
            print(
                "=" * 150
            )

            print(
                f"ANTI-CHASE "
                f"{current_test}"
            )

            print(
                "=" * 150
            )

            print(
                f"{'Event':>7} "
                f"{'Base':>7} "
                f"{'Refs':>10} "
                f"{'Dir':>6} "
                f"{'Hold':>7} "
                f"{'Base N':>8} "
                f"{'Base %':>8} "
                f"{'Base Ø':>10} "
                f"{'Base Cmp':>12} "
                f"{'AC N':>8} "
                f"{'AC %':>8} "
                f"{'AC Ø':>10} "
                f"{'AC Cmp':>12} "
                f"{'Δ Cmp':>11}"
            )

            print(
                "-" * 150
            )

        base = row[
            "base"
        ]

        anti = row[
            "anti_chase"
        ]

        print(
            f"{row['window']:>6}m "
            f"{row['threshold']:>6.1f}% "
            f"{min_ref_label(row['min_refs'], total_refs):>10} "
            f"{row['direction']:>6} "
            f"{row['hold']:>7} "
            f"{base['n']:>8} "
            f"{fmt_pos(base['pct_pos']):>8} "
            f"{fmt_pct(base['avg']):>10} "
            f"{fmt_pct(base['compound']):>12} "
            f"{anti['n']:>8} "
            f"{fmt_pos(anti['pct_pos']):>8} "
            f"{fmt_pct(anti['avg']):>10} "
            f"{fmt_pct(anti['compound']):>12} "
            f"{fmt_pct(row['delta']):>11}"
        )


# ============================================================
# FILTER OVERVIEW
# ============================================================

def print_filter_overview(
    results: List[dict],
) -> None:

    print()
    print()
    print(
        "=" * 100
    )

    print(
        "ANTI-CHASE FILTERSTÄRKE"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Test':>12} "
        f"{'Raw':>10} "
        f"{'Blocked':>10} "
        f"{'Kept':>10} "
        f"{'Removed':>10}"
    )

    print(
        "-" * 100
    )

    summary = {}

    for row in results:

        name = row[
            "test"
        ]

        if name not in summary:

            summary[name] = {
                "raw": 0,
                "blocked": 0,
                "kept": 0,
            }

        # Wichtig:
        # hier verwenden wir die Event-Filterstatistik
        # VOR dem Hold-spezifischen Non-overlap.
        filter_info = row[
            "filter"
        ]

        summary[name][
            "raw"
        ] += filter_info["raw"]

        summary[name][
            "blocked"
        ] += filter_info["blocked"]

        summary[name][
            "kept"
        ] += filter_info["kept"]

    # Da jedes Ergebnis für jeden Hold wiederholt wird,
    # normalisieren wir hier bewusst auf die eindeutigen
    # Event-Konfigurationen später nicht separat.
    #
    # Für die Darstellung nehmen wir die Durchschnittswerte
    # pro Testkonfiguration.

    seen = {}

    for row in results:

        key = (
            row["test"],
            row["window"],
            row["threshold"],
            row["min_refs"],
            row["direction"],
        )

        seen[key] = row[
            "filter"
        ]

    final_summary = {}

    for (
        key,
        info,
    ) in seen.items():

        name = key[0]

        if name not in final_summary:

            final_summary[name] = {
                "raw": 0,
                "blocked": 0,
                "kept": 0,
            }

        final_summary[name][
            "raw"
        ] += info["raw"]

        final_summary[name][
            "blocked"
        ] += info["blocked"]

        final_summary[name][
            "kept"
        ] += info["kept"]

    for test in ANTI_CHASE_TESTS:

        name = test[
            "name"
        ]

        info = final_summary.get(
            name,
            {
                "raw": 0,
                "blocked": 0,
                "kept": 0,
            },
        )

        raw = info[
            "raw"
        ]

        removed_pct = (
            100.0
            * info["blocked"]
            / raw
            if raw
            else 0.0
        )

        print(
            f"{name:>12} "
            f"{raw:>10} "
            f"{info['blocked']:>10} "
            f"{info['kept']:>10} "
            f"{removed_pct:>9.2f}%"
        )


# ============================================================
# BEST ANTI-CHASE RESULT
# ============================================================

def print_best_per_test(
    results: List[dict],
) -> None:

    print()
    print()
    print(
        "=" * 150
    )

    print(
        "BESTES ANTI-CHASE ERGEBNIS JE FILTER"
    )

    print(
        "=" * 150
    )

    for test in ANTI_CHASE_TESTS:

        candidates = [
            row
            for row in results
            if (
                row["test"]
                == test["name"]
                and
                row["anti_chase"][
                    "compound"
                ]
                is not None
            )
        ]

        if not candidates:
            continue

        best = max(
            candidates,
            key=lambda row:
                row["anti_chase"][
                    "compound"
                ],
        )

        print(
            f"{test['name']:>12} | "
            f"Event {best['window']}m | "
            f"Base {best['threshold']:.1f}% | "
            f"Refs {min_ref_label(best['min_refs'], 5)} | "
            f"{best['direction']} | "
            f"Hold {best['hold']} | "
            f"N {best['anti_chase']['n']} | "
            f"Win "
            f"{fmt_pos(best['anti_chase']['pct_pos'])} | "
            f"Ø "
            f"{fmt_pct(best['anti_chase']['avg'])} | "
            f"Compound "
            f"{fmt_pct(best['anti_chase']['compound'])} | "
            f"Δ "
            f"{fmt_pct(best['delta'])}"
        )


# ============================================================
# MAIN
# ============================================================

def main(
    argv: Optional[List[str]] = None,
) -> None:

    args = parse_cli_args(
        argv
    )

    refs_raw = (
        args.ref
        or REF_SYMBOLS
    )

    target_raw = (
        args.target
        or TARGET_SYMBOL
    )

    # --------------------------------------------------------
    # Unique references
    # --------------------------------------------------------

    refs_raw = list(
        dict.fromkeys(
            str(x)
            .strip()
            .upper()
            for x in refs_raw
        )
    )

    if len(refs_raw) < 1:

        raise SystemExit(
            "Mindestens ein Referenzcoin nötig."
        )

    # --------------------------------------------------------
    # Resolve pairs
    # --------------------------------------------------------

    usdt_set = (
        binance_usdt_symbols()
    )

    ref_bases = []
    ref_pairs = {}

    for raw in refs_raw:

        base, pair = resolve_pair(
            raw,
            usdt_set,
        )

        if base in ref_bases:
            continue

        ref_bases.append(
            base
        )

        ref_pairs[
            base
        ] = pair

    target_base, target_pair = (
        resolve_pair(
            target_raw,
            usdt_set,
        )
    )

    if target_base in ref_bases:

        raise SystemExit(
            f"Target {target_base} "
            f"darf nicht gleichzeitig "
            f"Referenz sein."
        )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    start, end, window_label = (
        resolve_event_window(
            from_s=(
                args.from_date
                or FROM_DATE
            ),
            to_s=(
                args.to_date
                or TO_DATE
            ),
            lookback_days=args.lookback,
            default_lookback=
                LOOKBACK_DAYS,
        )
    )

    end = (
        clamp_end_to_closed_candle(
            end
        )
    )

    start_ms = utc_ms(
        start
    )

    end_ms = utc_ms(
        end
    )

    fetch_end_ms = utc_ms(
        end
        + pd.Timedelta(
            minutes=MAX_HOLD_MIN
        )
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 118
    )

    print(
        "MULTI-REFERENCE "
        "BASELINE vs ANTI-CHASE"
    )

    print(
        "=" * 118
    )

    print(
        f"Refs:   "
        f"{', '.join(ref_bases)}"
    )

    print(
        f"Target: "
        f"{target_base} "
        f"({target_pair})"
    )

    print(
        "Events: "
        + ", ".join(
            f"{x}m"
            for x in EVENT_WINDOWS_MIN
        )
    )

    print(
        "Thresholds: "
        + ", ".join(
            f"{x:g}%"
            for x in THRESHOLDS_PCT
        )
    )

    print(
        "Min Refs: "
        + ", ".join(
            min_ref_label(
                x,
                len(ref_bases),
            )
            for x in MIN_REFS
        )
    )

    print(
        f"Zeitraum: "
        f"{start.strftime('%Y-%m-%d %H:%M %Z')}"
        f" -> "
        f"{end.strftime('%Y-%m-%d %H:%M %Z')}"
    )

    print(
        f"Interval: "
        f"{KLINE_INTERVAL} | "
        f"Fee: {FEE_BPS} bps/Seite | "
        f"Slippage: {SLIPPAGE_BPS} bps"
    )

    print(
        "UP und DOWN sind beide BUY-Signale."
    )

    print(
        "Anti-Chase wird VOR dem "
        "Non-overlap angewendet."
    )

    print(
        "=" * 118
    )

    # --------------------------------------------------------
    # Load refs
    # --------------------------------------------------------

    print(
        "\n1) Lade Referenzcoins:"
    )

    ref_data = {}

    for base in ref_bases:

        pair = ref_pairs[
            base
        ]

        print(
            f"   {base:>8} "
            f"({pair}) ...",
            end="",
            flush=True,
        )

        df = fetch_klines(
            pair,
            KLINE_INTERVAL,
            start_ms,
            end_ms,
        )

        if df.empty:

            raise SystemExit(
                f"\nKeine Kerzen für "
                f"Referenz {pair}."
            )

        ref_data[
            base
        ] = df

        print(
            f" {len(df)} Kerzen"
        )

    # --------------------------------------------------------
    # Load target
    # --------------------------------------------------------

    print(
        f"\n2) Lade Target "
        f"{target_pair} ..."
    )

    target_df = fetch_klines(
        target_pair,
        KLINE_INTERVAL,
        start_ms,
        fetch_end_ms,
    )

    if target_df.empty:

        raise SystemExit(
            f"Keine Kerzen für "
            f"Target {target_pair}."
        )

    print(
        f"   {len(target_df)} Kerzen"
    )

    # --------------------------------------------------------
    # Buy & Hold
    # --------------------------------------------------------

    bh = buy_and_hold(
        target_df,
        start_ms,
        end_ms,
    )

    # --------------------------------------------------------
    # Effective period
    # --------------------------------------------------------

    period_from, period_to = (
        effective_range_ms(
            start_ms,
            end_ms,
            target_df[
                "open_time"
            ].to_numpy(),
            *[
                df[
                    "open_time"
                ].to_numpy()
                for df in ref_data.values()
            ],
        )
    )

    period_label = block_range(
        period_from,
        period_to,
    )

    # --------------------------------------------------------
    # Detect events
    # --------------------------------------------------------

    print(
        "\n3) Erzeuge "
        "Multi-Reference Events ..."
    )

    all_events = (
        detect_multi_ref_events(
            ref_data=ref_data,
            windows=EVENT_WINDOWS_MIN,
            thresholds=THRESHOLDS_PCT,
            min_refs_list=MIN_REFS,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    )

    total_event_count = sum(
        len(event_list)
        for event_list
        in all_events.values()
    )

    print(
        f"   {total_event_count} "
        f"Event-Instanzen über "
        f"{len(all_events)} "
        f"Konfigurationen"
    )

    # --------------------------------------------------------
    # Build baseline cache
    # --------------------------------------------------------

    print(
        "\n4) Berechne Baseline ..."
    )

    result_cache = {}

    for window in EVENT_WINDOWS_MIN:

        for threshold in THRESHOLDS_PCT:

            for min_refs in MIN_REFS:

                for direction in (
                    "UP",
                    "DOWN",
                ):

                    key = (
                        window,
                        threshold,
                        min_refs,
                        direction,
                    )

                    events = (
                        all_events.get(
                            key,
                            [],
                        )
                    )

                    trades = (
                        simulate_events(
                            target_df=
                                target_df,
                            events=
                                events,
                            target_pair=
                                target_pair,
                        )
                    )

                    stats = (
                        summarize(
                            trades
                        )
                    )

                    result_cache[
                        key
                    ] = (
                        trades,
                        stats,
                    )

    # --------------------------------------------------------
    # Anti-Chase
    # --------------------------------------------------------

    print(
        "\n5) Berechne Anti-Chase ..."
    )

    comparison_results = []

    for (
        key,
        cache_value,
    ) in result_cache.items():

        (
            trades,
            base_stats,
        ) = cache_value

        (
            window,
            threshold,
            min_refs,
            direction,
        ) = key

        for test in ANTI_CHASE_TESTS:

            filtered_trades, filter_stats = (
                filter_anti_chase(
                    target_df=
                        target_df,
                    trades=
                        trades,
                    lookback_min=
                        test[
                            "lookback_min"
                        ],
                    max_move_pct=
                        test[
                            "max_move_pct"
                        ],
                )
            )

            anti_stats = summarize(
                filtered_trades
            )

            for horizon in HORIZON_NAMES:

                base = base_stats[
                    horizon
                ]

                anti = anti_stats[
                    horizon
                ]

                delta = None

                if (
                    base["compound"]
                    is not None
                    and
                    anti["compound"]
                    is not None
                ):

                    delta = (
                        anti["compound"]
                        - base["compound"]
                    )

                comparison_results.append(
                    {
                        "test":
                            test["name"],

                        "window":
                            window,

                        "threshold":
                            threshold,

                        "min_refs":
                            min_refs,

                        "direction":
                            direction,

                        "hold":
                            horizon,

                        "base":
                            base,

                        "anti_chase":
                            anti,

                        "delta":
                            delta,

                        "filter":
                            filter_stats,
                    }
                )

    # --------------------------------------------------------
    # Direct comparison
    # --------------------------------------------------------

    print(
        "\n6) BASELINE vs ANTI-CHASE\n"
    )

    print_direct_comparison(
        results=
            comparison_results,
        total_refs=
            len(ref_bases),
    )

    # --------------------------------------------------------
    # Filter overview
    # --------------------------------------------------------

    print_filter_overview(
        comparison_results
    )

    # --------------------------------------------------------
    # Best Anti-Chase
    # --------------------------------------------------------

    print_best_per_test(
        comparison_results
    )

    # --------------------------------------------------------
    # B&H
    # --------------------------------------------------------

    print()
    print()
    print(
        "=" * 100
    )

    print(
        "BUY & HOLD"
    )

    print(
        "=" * 100
    )

    if bh:

        print(
            f"{target_pair}: "
            f"{fmt_pct(bh['roi'])}"
        )

        print(
            f"{bh['buy_time']} "
            f"-> "
            f"{bh['sell_time']}"
        )

    else:

        print(
            "B&H nicht verfügbar."
        )

    # --------------------------------------------------------
    # End
    # --------------------------------------------------------

    print()
    print(
        "=" * 118
    )

    print(
        "Fertig."
    )

    print(
        "Base = unveränderte Multi-Ref-Strategie."
    )

    print(
        "AC = Base nach Anti-Chase-Filter."
    )

    print(
        "Δ Cmp = AC Compound minus "
        "Base Compound."
    )

    print(
        "Anti-Chase wird vor dem "
        "Hold-spezifischen Non-overlap angewendet."
    )

    print(
        "=" * 118
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()