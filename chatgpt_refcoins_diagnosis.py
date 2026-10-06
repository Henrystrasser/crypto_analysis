#!/usr/bin/env python3

"""
Reference Contribution Diagnostic

Ziel:
    Herausfinden, wie unterschiedlich BTC/SOL/ETH/XRP/BNB
    bei einem Multi-Ref 3/5-Signal tatsächlich beitragen.

Es werden NICHTS an der eigentlichen Strategie verändert.

Analysiert wird:

1. Wie häufig erreicht jeder Ref den Threshold überhaupt?
2. Wie häufig ist jeder Ref bei einem tatsächlichen 3/5-Event dabei?
3. Was passiert mit dem Target danach, wenn der Ref dabei ist?
4. Was passiert mit dem Target, wenn der Ref NICHT dabei ist?
5. Wie groß ist der Unterschied?

Dies ist die Vorbereitung für einen späteren Test
mit individuellen Thresholds pro Reference-Coin.

Beispiel später:
    BTC  >= 0.8%
    SOL  >= 1.2%
    ETH  >= 1.0%
    XRP  >= 1.1%
    BNB  >= 1.0%

Aber:
    DIESES SCRIPT setzt noch keine individuellen Thresholds.
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
# DIAGNOSTIC PARAMETER
# ============================================================
#
# Wir konzentrieren uns zunächst bewusst auf 3/5,
# weil genau dort die Frage nach den "üblichen Verdächtigen"
# interessant ist.
#

DIAGNOSTIC_MIN_REFS = 3


# Mehrere Event-Fenster
DIAGNOSTIC_EVENT_WINDOWS_MIN = [
    60,
    180,
]


# Mehrere Thresholds
DIAGNOSTIC_THRESHOLDS_PCT = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
]


# Für die Target-Performance-Diagnose
DIAGNOSTIC_HOLDS_HOURS = [
    3,
    6,
    12,
]


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
# BINANCE
# ============================================================

BINANCE = "https://data-api.binance.vision"

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": "multi-ref-diagnostic/1.0"
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
    for window in DIAGNOSTIC_EVENT_WINDOWS_MIN
}


HOLD_MINUTES = {
    f"{hours}h": hours * 60
    for hours in DIAGNOSTIC_HOLDS_HOURS
}


HOLD_BARS = {
    name: minutes // BAR_MIN
    for name, minutes in HOLD_MINUTES.items()
}


if any(
    window % BAR_MIN != 0
    for window in DIAGNOSTIC_EVENT_WINDOWS_MIN
):

    raise SystemExit(
        "Alle Diagnostic Event Windows müssen "
        "durch KLINE_INTERVAL teilbar sein."
    )


if any(
    hours * 60 % BAR_MIN != 0
    for hours in DIAGNOSTIC_HOLDS_HOURS
):

    raise SystemExit(
        "Alle Diagnostic Holds müssen "
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
        return len(
            self.matched_refs
        )


# ============================================================
# CLI
# ============================================================

def parse_cli_args(
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic für den Beitrag einzelner "
            "Reference-Coins bei Multi-Ref 3/5."
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
            f"Target-Coin "
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

        return (
            base,
            value,
        )

    if not value:

        raise ValueError(
            "Leerer Coin."
        )

    return (
        value,
        f"{value}USDT",
    )


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
            item.get("status")
            == "TRADING"
            and
            item.get("quoteAsset")
            == "USDT"
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
        and
        aliases[base] in usdt_set
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
                f"{symbol}: Download fehlgeschlagen."
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
    timestamp_ms: int,
) -> Optional[int]:

    times = (
        df["open_time"]
        .to_numpy(
            dtype=np.int64
        )
    )

    pos = int(
        np.searchsorted(
            times,
            int(timestamp_ms),
        )
    )

    if (
        pos < len(times)
        and int(times[pos])
        == int(timestamp_ms)
    ):

        return pos

    return None


# ============================================================
# REFERENCE MOVE SERIES
# ============================================================

def build_reference_move_series(
    df: pd.DataFrame,
) -> Dict[int, pd.Series]:

    result = {}

    for window_min in (
        DIAGNOSTIC_EVENT_WINDOWS_MIN
    ):

        bars = WINDOW_BARS[
            window_min
        ]

        result[
            window_min
        ] = (
            df["close"]
            .pct_change(bars)
            * 100.0
        )

    return result


# ============================================================
# COMMON TIMESTAMPS
# ============================================================

def get_common_timestamps(
    ref_data: Dict[str, pd.DataFrame],
    start_ms: int,
    end_ms: int,
) -> List[int]:

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
        return []

    return sorted(
        t
        for t in common_times
        if (
            start_ms
            <= t
            <= end_ms
        )
    )


# ============================================================
# BUILD DIAGNOSTIC EVENTS
# ============================================================

def build_diagnostic_events(
    ref_data: Dict[str, pd.DataFrame],
    window_min: int,
    threshold_pct: float,
    min_refs: int,
    common_times: List[int],
) -> List[MultiRefEvent]:

    # Timestamp -> position
    position_maps = {}

    for ref, df in ref_data.items():

        position_maps[ref] = {
            int(timestamp): index
            for index, timestamp
            in enumerate(
                df["open_time"].tolist()
            )
        }

    move_series = {}

    for ref, df in ref_data.items():

        move_series[ref] = (
            build_reference_move_series(
                df
            )
        )

    events = []

    bars = WINDOW_BARS[
        window_min
    ]

    for timestamp in common_times:

        moves = {}

        for ref in ref_data:

            pos = (
                position_maps[
                    ref
                ].get(
                    timestamp
                )
            )

            if pos is None:
                continue

            value = (
                move_series[
                    ref
                ][window_min]
                .iloc[pos]
            )

            if pd.notna(value):

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
            if move >= threshold_pct
        ]

        down_refs = [
            ref
            for ref, move
            in moves.items()
            if move <= -threshold_pct
        ]

        if len(up_refs) >= min_refs:

            events.append(
                MultiRefEvent(
                    end_ms=timestamp,
                    direction="UP",
                    window_min=window_min,
                    threshold_pct=threshold_pct,
                    min_refs=min_refs,
                    ref_moves=moves,
                    matched_refs=up_refs,
                )
            )

        if len(down_refs) >= min_refs:

            events.append(
                MultiRefEvent(
                    end_ms=timestamp,
                    direction="DOWN",
                    window_min=window_min,
                    threshold_pct=threshold_pct,
                    min_refs=min_refs,
                    ref_moves=moves,
                    matched_refs=down_refs,
                )
            )

    return events


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
# TARGET ROI
# ============================================================

def target_roi(
    target_df: pd.DataFrame,
    event_ms: int,
    hold_bars: int,
) -> Optional[float]:

    entry_pos = exact_position(
        target_df,
        event_ms,
    )

    if entry_pos is None:
        return None

    exit_ms = (
        int(event_ms)
        + (
            int(hold_bars)
            * int(BAR_MIN)
            * 60_000
        )
    )

    exit_pos = exact_position(
        target_df,
        exit_ms,
    )

    if exit_pos is None:
        return None

    entry_price = float(
        target_df.iloc[
            entry_pos
        ]["close"]
    )

    exit_price = float(
        target_df.iloc[
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
# REFERENCE CONTRIBUTION
# ============================================================

def reference_hit_stats(
    ref_data: Dict[str, pd.DataFrame],
    common_times: List[int],
    window_min: int,
    threshold_pct: float,
) -> Dict[str, dict]:

    position_maps = {}

    for ref, df in ref_data.items():

        position_maps[ref] = {
            int(timestamp): index
            for index, timestamp
            in enumerate(
                df["open_time"].tolist()
            )
        }

    result = {}

    for ref, df in ref_data.items():

        moves = (
            build_reference_move_series(
                df
            )[window_min]
        )

        valid_values = []

        for timestamp in common_times:

            pos = (
                position_maps[
                    ref
                ].get(
                    timestamp
                )
            )

            if pos is None:
                continue

            value = moves.iloc[
                pos
            ]

            if pd.notna(value):

                valid_values.append(
                    float(value)
                )

        arr = np.array(
            valid_values,
            dtype=float,
        )

        if len(arr) == 0:

            result[ref] = {
                "n_valid": 0,
                "up_hits": 0,
                "down_hits": 0,
                "up_rate": None,
                "down_rate": None,
                "abs_rate": None,
                "mean": None,
                "median": None,
            }

            continue

        up_hits = int(
            np.sum(
                arr >= threshold_pct
            )
        )

        down_hits = int(
            np.sum(
                arr <= -threshold_pct
            )
        )

        result[ref] = {
            "n_valid":
                len(arr),

            "up_hits":
                up_hits,

            "down_hits":
                down_hits,

            "up_rate":
                100.0
                * up_hits
                / len(arr),

            "down_rate":
                100.0
                * down_hits
                / len(arr),

            "abs_rate":
                100.0
                * (
                    up_hits
                    + down_hits
                )
                / len(arr),

            "mean":
                float(
                    np.mean(arr)
                ),

            "median":
                float(
                    np.median(arr)
                ),
        }

    return result


# ============================================================
# EVENT CONTRIBUTION STATS
# ============================================================

def contribution_stats(
    events: List[MultiRefEvent],
    target_df: pd.DataFrame,
) -> Dict[str, dict]:

    result = {}

    # --------------------------------------------------------
    # Erst Target-Returns einmal berechnen
    # --------------------------------------------------------

    prepared_events = []

    for event in events:

        roi_by_hold = {}

        for hold_name, hold_bars in (
            HOLD_BARS.items()
        ):

            roi_by_hold[
                hold_name
            ] = target_roi(
                target_df,
                event.end_ms,
                hold_bars,
            )

        prepared_events.append(
            (
                event,
                roi_by_hold,
            )
        )

    # --------------------------------------------------------
    # Für jeden Ref:
    # matched vs not matched
    # --------------------------------------------------------

    for ref in REF_SYMBOLS:

        matched_events = []
        absent_events = []

        for event, roi_by_hold in (
            prepared_events
        ):

            if ref in event.matched_refs:

                matched_events.append(
                    roi_by_hold
                )

            else:

                absent_events.append(
                    roi_by_hold
                )

        result[ref] = {
            "matched_n":
                len(matched_events),

            "absent_n":
                len(absent_events),

            "holds":
                {},
        }

        for hold_name in HOLD_BARS:

            matched_values = [
                x[hold_name]
                for x in matched_events
                if x[hold_name] is not None
            ]

            absent_values = [
                x[hold_name]
                for x in absent_events
                if x[hold_name] is not None
            ]

            matched_arr = np.array(
                matched_values,
                dtype=float,
            )

            absent_arr = np.array(
                absent_values,
                dtype=float,
            )

            result[ref][
                "holds"
            ][hold_name] = {
                "matched_n":
                    len(matched_arr),

                "matched_avg":
                    (
                        float(
                            np.mean(
                                matched_arr
                            )
                        )
                        if len(matched_arr)
                        else None
                    ),

                "matched_win":
                    (
                        100.0
                        * np.mean(
                            matched_arr > 0
                        )
                        if len(matched_arr)
                        else None
                    ),

                "absent_n":
                    len(absent_arr),

                "absent_avg":
                    (
                        float(
                            np.mean(
                                absent_arr
                            )
                        )
                        if len(absent_arr)
                        else None
                    ),

                "absent_win":
                    (
                        100.0
                        * np.mean(
                            absent_arr > 0
                        )
                        if len(absent_arr)
                        else None
                    ),
            }

            matched_avg = result[ref][
                "holds"
            ][hold_name][
                "matched_avg"
            ]

            absent_avg = result[ref][
                "holds"
            ][hold_name][
                "absent_avg"
            ]

            result[ref][
                "holds"
            ][hold_name][
                "avg_delta"
            ] = (
                matched_avg
                - absent_avg
                if (
                    matched_avg
                    is not None
                    and absent_avg
                    is not None
                )
                else None
            )

    return result


# ============================================================
# PRINT HELPERS
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

    return (
        f"{value:+.2f}%"
    )


def fmt_rate(
    value: Optional[float],
) -> str:

    if value is None:
        return "n/a"

    return (
        f"{value:.1f}%"
    )


# ============================================================
# PRINT GLOBAL REFERENCE HIT RATES
# ============================================================

def print_hit_rate_table(
    stats: Dict[str, dict],
    window_min: int,
    threshold_pct: float,
) -> None:

    print()
    print(
        "-" * 100
    )

    print(
        f"THRESHOLD-HIT-RATE | "
        f"Event {window_min}m | "
        f"Threshold ±{threshold_pct:g}%"
    )

    print(
        "-" * 100
    )

    print(
        f"{'Ref':>8} "
        f"{'Valid':>9} "
        f"{'UP hits':>10} "
        f"{'UP rate':>10} "
        f"{'DOWN hits':>12} "
        f"{'DOWN rate':>12} "
        f"{'ABS rate':>11}"
    )

    print(
        "-" * 100
    )

    for ref, s in stats.items():

        print(
            f"{ref:>8} "
            f"{s['n_valid']:>9} "
            f"{s['up_hits']:>10} "
            f"{fmt_rate(s['up_rate']):>10} "
            f"{s['down_hits']:>12} "
            f"{fmt_rate(s['down_rate']):>12} "
            f"{fmt_rate(s['abs_rate']):>11}"
        )


# ============================================================
# PRINT CONTRIBUTION
# ============================================================

def print_contribution_table(
    stats: Dict[str, dict],
    window_min: int,
    threshold_pct: float,
    direction: str,
) -> None:

    print()
    print(
        "-" * 125
    )

    print(
        f"REFERENCE CONTRIBUTION | "
        f"Event {window_min}m | "
        f"Threshold {threshold_pct:g}% | "
        f"{direction} | "
        f"3/{len(REF_SYMBOLS)}"
    )

    print(
        "-" * 125
    )

    print(
        f"{'Ref':>8} "
        f"{'Dabei':>8} "
        f"{'Anteil':>9} "
        f"{'3h Ø':>10} "
        f"{'3h Win':>9} "
        f"{'3h Ø ohne':>11} "
        f"{'Δ 3h':>10} "
        f"{'6h Ø':>10} "
        f"{'6h Win':>9} "
        f"{'Δ 6h':>10}"
    )

    print(
        "-" * 125
    )

    # Gesamtzahl der Events wurde vom Caller
    # bereits explizit eingetragen.
    actual_total = stats.get(
        "__total_events__",
        0,
    )

    for ref in REF_SYMBOLS:

        if ref not in stats:
            continue

        s = stats[ref]

        holds = s["holds"]

        h3 = holds.get(
            "3h",
            {},
        )

        h6 = holds.get(
            "6h",
            {},
        )

        share = (
            100.0
            * s["matched_n"]
            / actual_total
            if actual_total
            else None
        )

        print(
            f"{ref:>8} "
            f"{s['matched_n']:>8} "
            f"{fmt_rate(share):>9} "
            f"{fmt_pct(h3.get('matched_avg')):>10} "
            f"{fmt_rate(h3.get('matched_win')):>9} "
            f"{fmt_pct(h3.get('absent_avg')):>11} "
            f"{fmt_pct(h3.get('avg_delta')):>10} "
            f"{fmt_pct(h6.get('matched_avg')):>10} "
            f"{fmt_rate(h6.get('matched_win')):>9} "
            f"{fmt_pct(h6.get('avg_delta')):>10}"
        )


# ============================================================
# CONFIG DIAGNOSTIC
# ============================================================

def run_configuration_diagnostic(
    ref_data: Dict[str, pd.DataFrame],
    target_df: pd.DataFrame,
    common_times: List[int],
    window_min: int,
    threshold_pct: float,
) -> None:

    # --------------------------------------------------------
    # Hit rates des einzelnen Refs
    # --------------------------------------------------------

    hit_stats = reference_hit_stats(
        ref_data=ref_data,
        common_times=common_times,
        window_min=window_min,
        threshold_pct=threshold_pct,
    )

    print_hit_rate_table(
        stats=hit_stats,
        window_min=window_min,
        threshold_pct=threshold_pct,
    )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    events = build_diagnostic_events(
        ref_data=ref_data,
        window_min=window_min,
        threshold_pct=threshold_pct,
        min_refs=DIAGNOSTIC_MIN_REFS,
        common_times=common_times,
    )

    up_events = [
        event
        for event in events
        if event.direction == "UP"
    ]

    down_events = [
        event
        for event in events
        if event.direction == "DOWN"
    ]

    # --------------------------------------------------------
    # Allgemeine Event-Zahlen
    # --------------------------------------------------------

    print()
    print(
        "=" * 100
    )

    print(
        f"EVENT SUMMARY | "
        f"{window_min}m | "
        f"{threshold_pct:g}% | "
        f"{DIAGNOSTIC_MIN_REFS}/{len(REF_SYMBOLS)}"
    )

    print(
        "=" * 100
    )

    print(
        f"UP raw events   : "
        f"{len(up_events):,}"
    )

    print(
        f"DOWN raw events : "
        f"{len(down_events):,}"
    )

    print(
        f"Total raw events: "
        f"{len(events):,}"
    )

    # --------------------------------------------------------
    # UP contribution
    # --------------------------------------------------------

    if up_events:

        up_stats = contribution_stats(
            events=up_events,
            target_df=target_df,
        )

        up_stats["__total_events__"] = (
            len(up_events)
        )

        print_contribution_table(
            stats=up_stats,
            window_min=window_min,
            threshold_pct=threshold_pct,
            direction="UP",
        )

    # --------------------------------------------------------
    # DOWN contribution
    # --------------------------------------------------------

    if down_events:

        down_stats = contribution_stats(
            events=down_events,
            target_df=target_df,
        )

        down_stats["__total_events__"] = (
            len(down_events)
        )

        print_contribution_table(
            stats=down_stats,
            window_min=window_min,
            threshold_pct=threshold_pct,
            direction="DOWN",
        )


# ============================================================
# AGGREGATE REFERENCE RANKING
# ============================================================

def aggregate_reference_summary(
    all_summary_rows: List[dict],
) -> None:

    if not all_summary_rows:
        return

    print()
    print()
    print(
        "=" * 125
    )

    print(
        "AGGREGIERTE REFERENZ-ÜBERSICHT"
    )

    print(
        "=" * 125
    )

    grouped = {}

    for row in all_summary_rows:

        ref = row["ref"]

        if ref not in grouped:

            grouped[ref] = {
                "threshold_hit_rates": [],
                "event_shares": [],
                "delta_3h": [],
                "delta_6h": [],
            }

        if row["abs_rate"] is not None:
            grouped[ref][
                "threshold_hit_rates"
            ].append(
                row["abs_rate"]
            )

        if row["event_share"] is not None:
            grouped[ref][
                "event_shares"
            ].append(
                row["event_share"]
            )

        if row["delta_3h"] is not None:
            grouped[ref][
                "delta_3h"
            ].append(
                row["delta_3h"]
            )

        if row["delta_6h"] is not None:
            grouped[ref][
                "delta_6h"
            ].append(
                row["delta_6h"]
            )

    print(
        f"{'Ref':>8} "
        f"{'Ø HitRate':>12} "
        f"{'Ø EventShare':>14} "
        f"{'Ø Δ3h':>10} "
        f"{'Ø Δ6h':>10}"
    )

    print(
        "-" * 125
    )

    for ref in REF_SYMBOLS:

        g = grouped.get(
            ref,
            {},
        )

        hit_rates = g.get(
            "threshold_hit_rates",
            [],
        )

        shares = g.get(
            "event_shares",
            [],
        )

        delta_3h = g.get(
            "delta_3h",
            [],
        )

        delta_6h = g.get(
            "delta_6h",
            [],
        )

        print(
            f"{ref:>8} "
            f"{(
                f'{np.mean(hit_rates):.1f}%'
                if hit_rates
                else 'n/a'
            ):>12} "
            f"{(
                f'{np.mean(shares):.1f}%'
                if shares
                else 'n/a'
            ):>14} "
            f"{(
                f'{np.mean(delta_3h):+.2f}%'
                if delta_3h
                else 'n/a'
            ):>10} "
            f"{(
                f'{np.mean(delta_6h):+.2f}%'
                if delta_6h
                else 'n/a'
            ):>10}"
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

    refs_raw = list(
        dict.fromkeys(
            str(x)
            .strip()
            .upper()
            for x in refs_raw
        )
    )

    if len(refs_raw) < 2:

        raise SystemExit(
            "Mindestens zwei Referenzcoins nötig."
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

    max_hold_minutes = max(
        DIAGNOSTIC_HOLDS_HOURS
    ) * 60

    target_fetch_end_ms = utc_ms(
        end
        + pd.Timedelta(
            minutes=max_hold_minutes
        )
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print(
        "=" * 125
    )

    print(
        "MULTI-REFERENCE CONTRIBUTION DIAGNOSTIC"
    )

    print(
        "=" * 125
    )

    print(
        f"Refs:   {', '.join(ref_bases)}"
    )

    print(
        f"Target: {target_base} "
        f"({target_pair})"
    )

    print(
        f"3/{len(ref_bases)} Diagnostic"
    )

    print(
        "Event Windows: "
        + ", ".join(
            f"{x}m"
            for x in DIAGNOSTIC_EVENT_WINDOWS_MIN
        )
    )

    print(
        "Thresholds: "
        + ", ".join(
            f"{x:g}%"
            for x in DIAGNOSTIC_THRESHOLDS_PCT
        )
    )

    print(
        "Holds: "
        + ", ".join(
            f"{x}h"
            for x in DIAGNOSTIC_HOLDS_HOURS
        )
    )

    print(
        f"Zeitraum: "
        f"{start.strftime('%Y-%m-%d %H:%M %Z')}"
        f" -> "
        f"{end.strftime('%Y-%m-%d %H:%M %Z')}"
    )

    print(
        f"Interval: {KLINE_INTERVAL} | "
        f"Fee: {FEE_BPS} bps/Seite"
    )

    print(
        "=" * 125
    )

    # --------------------------------------------------------
    # Load references
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
        target_fetch_end_ms,
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
    # Common times
    # --------------------------------------------------------

    print(
        "\n3) Bestimme gemeinsame "
        "Reference-Timestamps ..."
    )

    common_times = get_common_timestamps(
        ref_data=ref_data,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    print(
        f"   {len(common_times):,} "
        f"gemeinsame Zeitpunkte"
    )

    if not common_times:

        raise SystemExit(
            "Keine gemeinsamen "
            "Reference-Zeitpunkte."
        )

    # --------------------------------------------------------
    # Run diagnostics
    # --------------------------------------------------------

    print(
        "\n4) Diagnostik ..."
    )

    all_summary_rows = []

    for window_min in (
        DIAGNOSTIC_EVENT_WINDOWS_MIN
    ):

        for threshold_pct in (
            DIAGNOSTIC_THRESHOLDS_PCT
        ):

            run_configuration_diagnostic(
                ref_data=ref_data,
                target_df=target_df,
                common_times=common_times,
                window_min=window_min,
                threshold_pct=threshold_pct,
            )

            # ----------------------------------------------
            # Summary rows für spätere Aggregation
            # ----------------------------------------------

            events = build_diagnostic_events(
                ref_data=ref_data,
                window_min=window_min,
                threshold_pct=threshold_pct,
                min_refs=DIAGNOSTIC_MIN_REFS,
                common_times=common_times,
            )

            by_direction = {
                "UP": [
                    event
                    for event in events
                    if event.direction == "UP"
                ],
                "DOWN": [
                    event
                    for event in events
                    if event.direction == "DOWN"
                ],
            }

            for direction, direction_events in (
                by_direction.items()
            ):

                if not direction_events:
                    continue

                stats = contribution_stats(
                    events=direction_events,
                    target_df=target_df,
                )

                total_events = len(
                    direction_events
                )

                hit_stats = reference_hit_stats(
                    ref_data=ref_data,
                    common_times=common_times,
                    window_min=window_min,
                    threshold_pct=threshold_pct,
                )

                for ref in ref_bases:

                    event_share = (
                        100.0
                        * stats[ref][
                            "matched_n"
                        ]
                        / total_events
                        if total_events
                        else None
                    )

                    all_summary_rows.append(
                        {
                            "ref":
                                ref,

                            "window":
                                window_min,

                            "threshold":
                                threshold_pct,

                            "direction":
                                direction,

                            "abs_rate":
                                hit_stats[ref][
                                    "abs_rate"
                                ],

                            "event_share":
                                event_share,

                            "delta_3h":
                                stats[ref][
                                    "holds"
                                ][
                                    "3h"
                                ][
                                    "avg_delta"
                                ],

                            "delta_6h":
                                stats[ref][
                                    "holds"
                                ][
                                    "6h"
                                ][
                                    "avg_delta"
                                ],
                        }
                    )

    # --------------------------------------------------------
    # Aggregate overview
    # --------------------------------------------------------

    aggregate_reference_summary(
        all_summary_rows
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print()
    print(
        "=" * 125
    )

    print(
        "FERTIG"
    )

    print(
        "=" * 125
    )

    print(
        "Interpretation:"
    )

    print(
        "  Threshold-Hit-Rate = "
        "Wie oft ein Ref den Threshold "
        "überhaupt erreicht."
    )

    print(
        "  Event-Anteil = "
        "Wie oft der Ref bei einem tatsächlichen "
        "3/5 Event dabei ist."
    )

    print(
        "  Δ3h / Δ6h = "
        "Target-Ø wenn Ref dabei ist "
        "minus Target-Ø wenn Ref nicht dabei ist."
    )

    print(
        "Je größer die Unterschiede der Hit-Rates, "
        "desto interessanter wird ein späterer "
        "Ref-spezifischer Threshold."
    )

    print(
        "=" * 125
    )


if __name__ == "__main__":
    main()