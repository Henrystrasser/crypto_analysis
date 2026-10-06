#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Reference BUY Backtest + Reference-Coin Equalizer
========================================================

Dieses Script basiert auf dem bisherigen Multi-Reference BUY Backtest.

Es macht zwei Dinge:

1. BASE
   Event mit dem jeweils konfigurierten Quorum und der identischen 2.00%-Schwelle für
   BTC / SOL / ETH / XRP / BNB.

2. EQUALIZED
   1/5, 2/5, 3/5, 4/5 und 5/5-Events mit individuell optimierten Schwellen je Refcoin.
   Ziel:
     - Event-Anzahl möglichst nahe an BASE halten
     - BTC/SOL/ETH/XRP/BNB innerhalb der Events möglichst
       gleich häufig abstimmen lassen
     - keine Gewichtung der Stimmen
     - jede Stimme zählt exakt 1

Der Equalizer wird für jedes Event-Fenster und jede Richtung separat
auf exakt denselben 30m-Daten berechnet, die auch der Backtest verwendet.

Wichtig:
- UP und DOWN werden separat equalisiert.
- 60m und 180m sind die primären Equalizer-Fenster.
- Die übrigen ursprünglichen Multi-Reference-Konfigurationen bleiben
  optional im normalen Scan erhalten.
- Der entscheidende A/B-Vergleich steht in der COMPARISON-Tabelle.

BASE:
    BTC >= 2%, SOL >= 2%, ETH >= 2%, XRP >= 2%, BNB >= 2%
    mindestens das konfigurierte Quorum

EQUALIZED-Beispiel:
    BTC >= 2.3%
    SOL >= 1.8%
    ETH >= 2.1%
    XRP >= 1.7%
    BNB >= 2.4%
    mindestens 3/5

Jede Coin-Stimme bleibt binär und gleichwertig.
"""

from __future__ import annotations

import argparse
import math
import time
import random
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

REF_SYMBOLS: List[str] = ["BTC", "SOL", "ETH", "XRP", "BNB"]
TARGET_SYMBOL = "PEPE"

# Ursprüngliche Testfenster.
EVENT_WINDOWS_MIN = [60, 120, 180]

# Ursprüngliche feste Schwellen für den normalen Scan.
THRESHOLDS_PCT = [1.0, 1.5, 2.0, 3.0]

# Ursprüngliche MIN_REFS.
MIN_REFS = [1, 2, 3, 4, 5]

# Der A/B-Hauptvergleich wird für alle Quoren 1/5 bis 5/5 ausgegeben.
EQUALIZER_QUORUMS = [1, 2, 3, 4, 5]

# Equalizer-Fenster.
EQUALIZER_WINDOWS_MIN = EVENT_WINDOWS_MIN.copy()

BASE_THRESHOLD_PCT = 2.0

# Threshold-Grid des Equalizers.
EQUALIZER_STEP_PCT = 0.10
EQUALIZER_MIN_PCT = 0.50
EQUALIZER_MAX_PCT = 6.00

# Suche.
EQUALIZER_RESTARTS = 5
EQUALIZER_STEPS = 50
EQUALIZER_RANDOM_MOVE_MIN = 1
EQUALIZER_RANDOM_MOVE_MAX = 4
EQUALIZER_RANDOM_SEED = 42

# Event-Anzahl darf idealerweise max. +/- 5% von BASE abweichen.
MAX_EVENT_COUNT_DEVIATION = 0.05

# 3/5 entspricht natürlicherweise 60%.
TARGET_PARTICIPATION = 0.60
TARGET_PARTICIPATION_WEIGHT = 0.20

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
SESSION.headers.update({"User-Agent": "multi-ref-buy-equalized/1.0"})


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
        description="Multi-Reference BUY Backtest + Refcoin Equalizer"
    )

    p.add_argument(
        "--ref",
        "--refs",
        nargs="+",
        default=None,
        metavar="COIN",
        help=f"Referenzcoins (Default: {' '.join(REF_SYMBOLS)})",
    )

    p.add_argument(
        "--coin",
        "--target",
        dest="target",
        default=None,
        metavar="COIN",
        help=f"Ziel-Coin (Default: {TARGET_SYMBOL})",
    )

    p.add_argument(
        "--no-random",
        dest="run_random",
        action="store_false",
        default=None,
        help="RANDOM-Baseline überspringen",
    )

    p.add_argument(
        "--equalizer-restarts",
        type=int,
        default=EQUALIZER_RESTARTS,
        help=f"Equalizer Restarts (Default: {EQUALIZER_RESTARTS})",
    )

    p.add_argument(
        "--equalizer-steps",
        type=int,
        default=EQUALIZER_STEPS,
        help=f"Equalizer Schritte je Restart (Default: {EQUALIZER_STEPS})",
    )

    p.add_argument(
        "--no-full-scan",
        dest="full_scan",
        action="store_false",
        default=True,
        help="Originalen Vollscan überspringen (nicht empfohlen, wenn vollständiger Vergleich gewünscht ist).",
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

if any(w % BAR_MIN != 0 for w in EVENT_WINDOWS_MIN):
    raise SystemExit(
        "Alle EVENT_WINDOWS_MIN müssen durch KLINE_INTERVAL teilbar sein."
    )

if any(h % BAR_MIN != 0 for h in HOLD_HORIZONS.values()):
    raise SystemExit(
        "Alle HOLD_HORIZONS müssen durch KLINE_INTERVAL teilbar sein."
    )

WINDOW_BARS = {
    w: w // BAR_MIN
    for w in EVENT_WINDOWS_MIN
}

HOLD_BARS = {
    h: mins // BAR_MIN
    for h, mins in HOLD_HORIZONS.items()
}

MAX_HOLD_MIN = max(HOLD_HORIZONS.values())
HORIZON_NAMES = list(HOLD_HORIZONS.keys())


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def sleep_polite(seconds: float = 0.15) -> None:
    time.sleep(seconds)


# ============================================================
# BINANCE
# ============================================================

def binance_usdt_symbols() -> set[str]:
    r = SESSION.get(
        f"{BINANCE}/api/v3/exchangeInfo",
        timeout=30,
    )
    r.raise_for_status()

    return {
        s["symbol"]
        for s in r.json()["symbols"]
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
        )
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
        raise SystemExit(
            f"Pair {pair} nicht auf Binance Spot USDT."
        )

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

    now_ms = int(time.time() * 1000)

    # Keine laufende Kerze.
    df = df[
        df["close_time"].astype(np.int64) < now_ms
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

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df["open_time"] = df["open_time"].astype(np.int64)

    return (
        df
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


# ============================================================
# EVENT MODEL
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


@dataclass
class EqualizedResult:
    thresholds: Dict[str, float]
    baseline_count: int
    equalized_count: int
    baseline_participation: Dict[str, float]
    equalized_participation: Dict[str, float]
    baseline_spread: float
    equalized_spread: float
    score: float


# ============================================================
# COMMON DATA / RETURNS
# ============================================================

def build_common_reference_times(
    ref_data: Dict[str, pd.DataFrame],
) -> np.ndarray:

    common: Optional[set[int]] = None

    for df in ref_data.values():
        times = set(
            int(x)
            for x in df["open_time"].tolist()
        )

        common = (
            times
            if common is None
            else common & times
        )

    if not common:
        return np.array([], dtype=np.int64)

    return np.array(
        sorted(common),
        dtype=np.int64,
    )


def build_return_arrays(
    ref_data: Dict[str, pd.DataFrame],
    windows: List[int],
) -> Dict[Tuple[int, str], Dict[int, float]]:

    result: Dict[
        Tuple[int, str],
        Dict[int, float]
    ] = {}

    for ref, df in ref_data.items():
        times = df["open_time"].to_numpy(
            dtype=np.int64
        )
        closes = df["close"].to_numpy(
            dtype=np.float64
        )

        time_to_pos = {
            int(t): i
            for i, t in enumerate(times)
        }

        for window in windows:
            bars = WINDOW_BARS[window]

            returns = (
                pd.Series(closes)
                .pct_change(bars)
                .to_numpy()
                * 100.0
            )

            result[(window, ref)] = {
                int(t): float(returns[i])
                for i, t in enumerate(times)
                if pd.notna(returns[i])
            }

    return result


# ============================================================
# FIXED BASE EVENTS
# ============================================================

def build_fixed_events(
    ref_data: Dict[str, pd.DataFrame],
    windows: List[int],
    threshold_pct: float,
    min_refs: int,
    start_ms: int,
    end_ms: int,
) -> Dict[Tuple[int, str], List[MultiRefEvent]]:

    common_times = build_common_reference_times(ref_data)

    common_times = common_times[
        (common_times >= start_ms)
        & (common_times <= end_ms)
    ]

    returns = build_return_arrays(
        ref_data,
        windows,
    )

    events: Dict[
        Tuple[int, str],
        List[MultiRefEvent]
    ] = {}

    for window in windows:
        for direction in ("UP", "DOWN"):
            events[(window, direction)] = []

    for t in common_times:
        t_int = int(t)

        for window in windows:
            moves = {}

            for ref in ref_data:
                value = returns[(window, ref)].get(t_int)

                if value is None:
                    moves = {}
                    break

                moves[ref] = value

            if len(moves) != len(ref_data):
                continue

            up_refs = [
                ref
                for ref, move in moves.items()
                if move >= threshold_pct
            ]

            down_refs = [
                ref
                for ref, move in moves.items()
                if move <= -threshold_pct
            ]

            if len(up_refs) >= min_refs:
                events[(window, "UP")].append(
                    MultiRefEvent(
                        end_ms=t_int,
                        direction="UP",
                        window_min=window,
                        threshold_pct=threshold_pct,
                        min_refs=min_refs,
                        ref_moves=moves,
                        matched_refs=up_refs,
                    )
                )

            if len(down_refs) >= min_refs:
                events[(window, "DOWN")].append(
                    MultiRefEvent(
                        end_ms=t_int,
                        direction="DOWN",
                        window_min=window,
                        threshold_pct=threshold_pct,
                        min_refs=min_refs,
                        ref_moves=moves,
                        matched_refs=down_refs,
                    )
                )

    return events


# ============================================================
# EQUALIZER
# ============================================================

def prepare_equalizer_matrix(
    ref_data: Dict[str, pd.DataFrame],
    window: int,
    start_ms: int,
    end_ms: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Einmalige, vektorisierte Return-Matrix für den Equalizer."""
    common = build_common_reference_times(ref_data)
    common = common[(common >= start_ms) & (common <= end_ms)]
    returns = build_return_arrays(ref_data, [window])

    times = []
    for t in common:
        if all(t in returns[(window, ref)] for ref in REF_SYMBOLS):
            times.append(int(t))

    if not times:
        return np.array([], dtype=np.int64), np.empty((0, len(REF_SYMBOLS))), REF_SYMBOLS

    times_arr = np.asarray(times, dtype=np.int64)
    matrix = np.column_stack([
        np.asarray([returns[(window, ref)][int(t)] for t in times_arr], dtype=np.float64)
        for ref in REF_SYMBOLS
    ])
    return times_arr, matrix, REF_SYMBOLS


def evaluate_thresholds_matrix(
    moves: np.ndarray,
    thresholds: np.ndarray,
    direction: str,
    quorum: int,
) -> dict:
    if moves.size == 0:
        return {
            "event_count": 0,
            "participation": {ref: 0.0 for ref in REF_SYMBOLS},
            "spread": float("inf"),
            "mean_participation": 0.0,
            "mask": np.zeros(0, dtype=bool),
        }

    if direction == "UP":
        votes = moves >= thresholds[None, :]
    else:
        votes = moves <= -thresholds[None, :]

    event_mask = votes.sum(axis=1) >= quorum
    event_count = int(event_mask.sum())

    if event_count == 0:
        participation = {ref: 0.0 for ref in REF_SYMBOLS}
        return {
            "event_count": 0,
            "participation": participation,
            "spread": float("inf"),
            "mean_participation": 0.0,
            "mask": event_mask,
        }

    selected = votes[event_mask]
    parts = selected.mean(axis=0)
    participation = {
        ref: float(parts[i]) for i, ref in enumerate(REF_SYMBOLS)
    }
    return {
        "event_count": event_count,
        "participation": participation,
        "spread": float(parts.max() - parts.min()),
        "mean_participation": float(parts.mean()),
        "mask": event_mask,
    }


def equalizer_objective(result: dict, baseline_count: int, target_participation: float) -> float:
    if baseline_count <= 0 or result["event_count"] <= 0:
        return float("inf")
    event_dev = abs(result["event_count"] - baseline_count) / baseline_count
    mean_dev = abs(result["mean_participation"] - target_participation)
    if event_dev <= MAX_EVENT_COUNT_DEVIATION:
        return result["spread"] + TARGET_PARTICIPATION_WEIGHT * mean_dev + 0.10 * event_dev
    return 2.0 + 20.0 * (event_dev - MAX_EVENT_COUNT_DEVIATION) + result["spread"] + TARGET_PARTICIPATION_WEIGHT * mean_dev


def clamp_equalizer_threshold(value: float) -> float:
    return max(EQUALIZER_MIN_PCT, min(EQUALIZER_MAX_PCT, value))


def random_neighbor(thresholds: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    candidate = dict(thresholds)
    symbols = rng.sample(REF_SYMBOLS, rng.randint(1, 2))
    for symbol in symbols:
        candidate[symbol] = clamp_equalizer_threshold(
            candidate[symbol] + rng.choice([-1, 1]) * rng.randint(EQUALIZER_RANDOM_MOVE_MIN, EQUALIZER_RANDOM_MOVE_MAX) * EQUALIZER_STEP_PCT
        )
    return candidate


def optimize_equalizer(
    moves: np.ndarray,
    window: int,
    direction: str,
    quorum: int,
    restarts: int,
    steps: int,
    seed_offset: int,
) -> EqualizedResult:
    baseline_thresholds = {ref: BASE_THRESHOLD_PCT for ref in REF_SYMBOLS}
    baseline_arr = np.full(len(REF_SYMBOLS), BASE_THRESHOLD_PCT, dtype=float)
    baseline = evaluate_thresholds_matrix(moves, baseline_arr, direction, quorum)
    baseline_count = baseline["event_count"]
    if baseline_count <= 0:
        raise RuntimeError(f"Keine BASE-Events für {window}m {direction} {quorum}/5.")

    target_participation = quorum / len(REF_SYMBOLS)
    if quorum == len(REF_SYMBOLS):
        return EqualizedResult(
            thresholds=baseline_thresholds,
            baseline_count=baseline_count,
            equalized_count=baseline_count,
            baseline_participation=baseline["participation"],
            equalized_participation=baseline["participation"],
            baseline_spread=baseline["spread"],
            equalized_spread=baseline["spread"],
            score=equalizer_objective(baseline, baseline_count, target_participation),
        )
    best_thresholds = dict(baseline_thresholds)
    best_result = baseline
    best_score = equalizer_objective(baseline, baseline_count, target_participation)
    eval_cache = {tuple(np.round(baseline_arr, 2)): baseline}
    rng = random.Random(EQUALIZER_RANDOM_SEED + window * 100 + quorum * 10 + (1 if direction == "UP" else 2) + seed_offset)

    for restart in range(restarts):
        current_thresholds = (
            dict(baseline_thresholds) if restart == 0 else {
                ref: clamp_equalizer_threshold(BASE_THRESHOLD_PCT + rng.randint(-8, 8) * EQUALIZER_STEP_PCT)
                for ref in REF_SYMBOLS
            }
        )
        current_arr = np.array([current_thresholds[r] for r in REF_SYMBOLS])
        current_key = tuple(np.round(current_arr, 2))
        current_result = eval_cache.get(current_key)
        if current_result is None:
            current_result = evaluate_thresholds_matrix(moves, current_arr, direction, quorum)
            eval_cache[current_key] = current_result
        current_score = equalizer_objective(current_result, baseline_count, target_participation)
        temperature = 0.003
        for step in range(steps):
            candidate_thresholds = random_neighbor(current_thresholds, rng)
            candidate_arr = np.array([candidate_thresholds[r] for r in REF_SYMBOLS])
            candidate_key = tuple(np.round(candidate_arr, 2))
            candidate_result = eval_cache.get(candidate_key)
            if candidate_result is None:
                candidate_result = evaluate_thresholds_matrix(moves, candidate_arr, direction, quorum)
                eval_cache[candidate_key] = candidate_result
            candidate_score = equalizer_objective(candidate_result, baseline_count, target_participation)
            delta = candidate_score - current_score
            temperature_now = max(0.00005, temperature * (1.0 - step / max(steps, 1)))
            if delta <= 0 or rng.random() < math.exp(-delta / temperature_now):
                current_thresholds, current_result, current_score = candidate_thresholds, candidate_result, candidate_score
            if current_score < best_score:
                best_score = current_score
                best_thresholds = dict(current_thresholds)
                best_result = current_result

    return EqualizedResult(
        thresholds=best_thresholds,
        baseline_count=baseline_count,
        equalized_count=best_result["event_count"],
        baseline_participation=baseline["participation"],
        equalized_participation=best_result["participation"],
        baseline_spread=baseline["spread"],
        equalized_spread=best_result["spread"],
        score=best_score,
    )


def build_equalized_events(
    ref_data: Dict[str, pd.DataFrame],
    window: int,
    direction: str,
    thresholds: Dict[str, float],
    quorum: int,
    start_ms: int,
    end_ms: int,
) -> List[MultiRefEvent]:
    common = build_common_reference_times(ref_data)
    common = common[(common >= start_ms) & (common <= end_ms)]
    returns = build_return_arrays(ref_data, [window])
    events = []
    threshold_arr = np.array([thresholds[r] for r in REF_SYMBOLS], dtype=float)
    for t in common:
        t_int = int(t)
        moves = {}
        for ref in REF_SYMBOLS:
            value = returns[(window, ref)].get(t_int)
            if value is None:
                moves = {}
                break
            moves[ref] = value
        if len(moves) != len(REF_SYMBOLS):
            continue
        vals = np.array([moves[r] for r in REF_SYMBOLS], dtype=float)
        votes = vals >= threshold_arr if direction == "UP" else vals <= -threshold_arr
        matched_refs = [ref for ref, vote in zip(REF_SYMBOLS, votes) if vote]
        if len(matched_refs) >= quorum:
            events.append(MultiRefEvent(
                end_ms=t_int, direction=direction, window_min=window,
                threshold_pct=BASE_THRESHOLD_PCT, min_refs=quorum,
                ref_moves=moves, matched_refs=matched_refs,
            ))
    return events


# ============================================================
# TARGET / TRADE
# ============================================================

def apply_costs(
    entry: float,
    exit_: float,
) -> Tuple[float, float]:

    cost = (
        FEE_BPS
        + SLIPPAGE_BPS
    ) / 10_000.0

    return (
        entry * (1.0 + cost),
        exit_ * (1.0 - cost),
    )


def exact_position(
    df: pd.DataFrame,
    t_ms: int,
) -> Optional[int]:

    times = df["open_time"].to_numpy(
        dtype=np.int64
    )

    i = int(
        np.searchsorted(
            times,
            int(t_ms),
        )
    )

    if (
        i < len(times)
        and int(times[i]) == int(t_ms)
    ):
        return i

    return None


def roi_for_hold(
    df: pd.DataFrame,
    entry_pos: int,
    hold_bars: int,
) -> Optional[float]:

    entry_time = int(
        df["open_time"].iat[entry_pos]
    )

    exit_time = (
        entry_time
        + int(hold_bars)
        * int(BAR_MIN)
        * 60_000
    )

    exit_pos = exact_position(
        df,
        exit_time,
    )

    if exit_pos is None:
        return None

    entry = float(
        df.iloc[entry_pos]["close"]
    )
    exit_ = float(
        df.iloc[exit_pos]["close"]
    )

    if entry <= 0 or exit_ <= 0:
        return None

    buy, sell = apply_costs(
        entry,
        exit_,
    )

    return (
        sell / buy - 1.0
    ) * 100.0


def simulate_events(
    target_df: pd.DataFrame,
    events: List[MultiRefEvent],
    target_pair: str,
) -> List[dict]:

    trades: List[dict] = []

    for event in events:

        entry_pos = exact_position(
            target_df,
            event.end_ms,
        )

        if entry_pos is None:
            continue

        entry_time = int(
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
            "direction": event.direction,
            "window_min": event.window_min,
            "threshold_pct": event.threshold_pct,
            "min_refs": event.min_refs,
            "score": event.score,
            "matched_refs": ",".join(
                event.matched_refs
            ),
            "event_time_ms": event.end_ms,
            "buy_time_ms": entry_time,
            "buy_time": ms_to_berlin_str(
                entry_time
                + int(BAR_MIN)
                * 60_000
            ),
            "ref_moves": event.ref_moves,
            "buy_price": entry_price * (
                1.0
                + (
                    FEE_BPS
                    + SLIPPAGE_BPS
                )
                / 10_000.0
            ),
        }

        any_ok = False

        for hname, bars in HOLD_BARS.items():
            roi = roi_for_hold(
                target_df,
                entry_pos,
                bars,
            )

            row[
                f"roi_{hname}"
            ] = roi

            if roi is not None:
                any_ok = True

        if any_ok:
            trades.append(row)

    return trades


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
        key=lambda x: int(
            x["buy_time_ms"]
        ),
    )

    kept: List[dict] = []
    last_buy_ms: Optional[int] = None

    for trade in ordered:

        roi = trade.get(col)

        if roi is None:
            continue

        buy_ms = int(
            trade["buy_time_ms"]
        )

        if (
            last_buy_ms is not None
            and buy_ms
            < last_buy_ms + hold_ms
        ):
            continue

        kept.append(trade)
        last_buy_ms = buy_ms

    return kept


def column_stats(
    trades: List[dict],
    col: str,
) -> dict:

    vals = [
        float(t[col])
        for t in trades
        if t.get(col) is not None
        and not (
            isinstance(
                t.get(col),
                float,
            )
            and np.isnan(
                t.get(col)
            )
        )
    ]

    if not vals:
        return {
            "n": 0,
            "n_pos": 0,
            "n_neg": 0,
            "pct_pos": None,
            "avg": None,
            "compound": None,
        }

    n_pos = sum(
        v > 0
        for v in vals
    )
    n_neg = sum(
        v < 0
        for v in vals
    )

    equity = 1.0

    for v in vals:
        equity *= (
            1.0 + v / 100.0
        )

    return {
        "n": len(vals),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pct_pos": (
            100.0
            * n_pos
            / (n_pos + n_neg)
            if n_pos + n_neg
            else None
        ),
        "avg": float(
            np.mean(vals)
        ),
        "compound": (
            equity - 1.0
        ) * 100.0,
    }


def summarize(
    trades: List[dict],
) -> Dict[str, dict]:

    result = {}

    for h in HORIZON_NAMES:
        filtered = filter_non_overlapping(
            trades,
            h,
        )

        result[h] = column_stats(
            filtered,
            f"roi_{h}",
        )

    return result


def buy_and_hold(
    df: pd.DataFrame,
    start_ms: int,
    end_ms: int,
) -> Optional[dict]:

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

    return {
        "roi": (
            sell / buy - 1.0
        ) * 100.0,
        "buy_time": ms_to_berlin_str(
            int(
                sub.iloc[0][
                    "open_time"
                ]
            )
            + int(BAR_MIN)
            * 60_000
        ),
        "sell_time": ms_to_berlin_str(
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
# RANDOM BASELINE
# ============================================================

def random_positions(
    df: pd.DataFrame,
    start_ms: int,
    end_ms: int,
    hold_bars: int,
    n_target: int,
    rng: np.random.Generator,
) -> List[int]:

    hold_ms = (
        hold_bars
        * BAR_MIN
        * 60_000
    )

    times = df[
        "open_time"
    ].to_numpy(
        dtype=np.int64
    )

    closes = df[
        "close"
    ].to_numpy(
        dtype=np.float64
    )

    exit_exists = np.isin(
        times + hold_ms,
        times,
    )

    candidates = [
        int(i)
        for i in np.flatnonzero(
            (times >= start_ms)
            & (times <= end_ms)
            & exit_exists
            & (closes > 0)
        )
    ]

    if (
        not candidates
        or n_target <= 0
    ):
        return []

    bar_ms = (
        BAR_MIN
        * 60_000
    )

    cand_arr = np.array(
        candidates,
        dtype=np.int64,
    )

    t0 = int(
        times[cand_arr[0]]
    )

    slot_of = {
        int(
            (
                int(times[i])
                - t0
            )
            // bar_ms
        ): int(i)
        for i in cand_arr
    }

    n_slots = max(
        slot_of
    ) + 1

    free = (
        n_slots
        - (n_target - 1)
        * hold_bars
    )

    if free >= n_target:
        offsets = (
            np.arange(
                n_target,
                dtype=np.int64,
            )
            * hold_bars
        )

        for _ in range(500):
            picks = (
                np.sort(
                    rng.choice(
                        free,
                        size=n_target,
                        replace=False,
                    )
                )
                + offsets
            )

            if all(
                int(k) in slot_of
                for k in picks
            ):
                return [
                    slot_of[int(k)]
                    for k in picks
                ]

    order = candidates.copy()
    rng.shuffle(order)

    accepted: List[int] = []

    for pos in order:
        t = int(times[pos])

        if all(
            abs(
                t
                - int(times[p])
            )
            >= hold_ms
            for p in accepted
        ):
            accepted.append(pos)

        if len(accepted) >= n_target:
            break

    return sorted(
        accepted,
        key=lambda p: int(
            times[p]
        ),
    )


def random_matched_stats(
    df: pd.DataFrame,
    target_n: Dict[str, int],
    start_ms: int,
    end_ms: int,
    rng: np.random.Generator,
) -> Dict[str, dict]:

    result = {}

    for h in HORIZON_NAMES:

        positions = random_positions(
            df,
            start_ms,
            end_ms,
            HOLD_BARS[h],
            target_n.get(h, 0),
            rng,
        )

        vals = []

        for pos in positions:
            roi = roi_for_hold(
                df,
                pos,
                HOLD_BARS[h],
            )

            if roi is not None:
                vals.append(roi)

        if not vals:
            result[h] = {
                "n": 0,
                "pct_pos": None,
                "avg": None,
                "compound": None,
            }
            continue

        equity = 1.0

        for v in vals:
            equity *= (
                1.0 + v / 100.0
            )

        result[h] = {
            "n": len(vals),
            "pct_pos": (
                100.0
                * sum(
                    v > 0
                    for v in vals
                )
                / len(vals)
            ),
            "avg": float(
                np.mean(vals)
            ),
            "compound": (
                equity - 1.0
            ) * 100.0,
        }

    return result


# ============================================================
# OUTPUT
# ============================================================

def fmt_pct(
    v: Optional[float],
) -> str:

    if v is None:
        return "n/a"

    if (
        isinstance(
            v,
            (float, np.floating),
        )
        and np.isnan(v)
    ):
        return "n/a"

    return f"{v:+.2f}%"


def fmt_pos(
    v: Optional[float],
) -> str:

    if v is None:
        return "n/a"

    return f"{v:.0f}%"


def min_ref_label(
    min_refs: int,
    total: int,
) -> str:

    if min_refs == 1:
        return f"ANY (1/{total})"

    if min_refs == total:
        return f"ALL ({total}/{total})"

    return f"{min_refs}/{total}"


def print_equalizer_report(
    eq: EqualizedResult,
    window: int,
    direction: str,
    quorum: int,
) -> None:
    """Druckt JEDE Equalized-Auswertung direkt gegen BASE."""

    print()
    print("=" * 132)
    print(
        f"EQUALIZED vs BASE | {direction} | Event {window}m | "
        f"{quorum}/5"
    )
    print("=" * 132)

    # --------------------------------------------------------
    # 1) THRESHOLD-TABELLE
    # --------------------------------------------------------
    print("\n[1] THRESHOLDS – BASE vs EQUALIZED")
    print("-" * 100)
    print(
        f"{'Coin':<10}"
        f"{'BASE':>14}"
        f"{'EQUALIZED':>16}"
        f"{'Delta':>16}"
        f"{'BASE Signal':>20}"
        f"{'EQ Signal':>20}"
    )
    print("-" * 100)

    for ref in REF_SYMBOLS:
        base = BASE_THRESHOLD_PCT
        new = eq.thresholds[ref]
        delta = new - base
        sign = "+" if direction == "UP" else "-"

        print(
            f"{ref:<10}"
            f"{base:>13.2f}%"
            f"{new:>15.2f}%"
            f"{delta:>+15.2f} pp"
            f"{sign}{base:.2f}%: 1"
            f"{sign}{new:.2f}%: 1"
        )

    print("-" * 100)
    print(
        f"{'Quorum':<10}"
        f"{quorum}/5"
        f"{'BASE':>16}"
        f"{'EQ':>16}"
    )

    # --------------------------------------------------------
    # 2) PARTICIPATION-TABELLE
    # --------------------------------------------------------
    print("\n[2] VOTING-PARTICIPATION – BASE vs EQUALIZED")
    print("-" * 100)
    print(
        f"{'Coin':<10}"
        f"{'BASE':>16}"
        f"{'EQUALIZED':>18}"
        f"{'Delta':>16}"
        f"{'Richtung':>20}"
    )
    print("-" * 100)

    for ref in REF_SYMBOLS:
        base = eq.baseline_participation[ref] * 100.0
        new = eq.equalized_participation[ref] * 100.0

        print(
            f"{ref:<10}"
            f"{base:>15.2f}%"
            f"{new:>17.2f}%"
            f"{new - base:>+15.2f} pp"
            f"{'→ Equalisierung':>20}"
        )

    print("-" * 100)
    print(
        f"{'Spread':<10}"
        f"{eq.baseline_spread * 100:>15.2f} pp"
        f"{eq.equalized_spread * 100:>17.2f} pp"
        f"{(eq.equalized_spread - eq.baseline_spread) * 100:>+15.2f} pp"
        f"{'kleiner = besser':>20}"
    )

    base_mean = (
        sum(eq.baseline_participation.values())
        / len(eq.baseline_participation)
        * 100.0
    )
    eq_mean = (
        sum(eq.equalized_participation.values())
        / len(eq.equalized_participation)
        * 100.0
    )

    print(
        f"{'Ø Participation':<10}"
        f"{base_mean:>15.2f}%"
        f"{eq_mean:>17.2f}%"
        f"{eq_mean - base_mean:>+15.2f} pp"
        f"{'Ziel 60%':>20}"
    )

    # --------------------------------------------------------
    # 3) EVENT-COUNT-TABELLE
    # --------------------------------------------------------
    print("\n[3] EVENTS – BASE vs EQUALIZED")
    print("-" * 100)

    event_delta = eq.equalized_count - eq.baseline_count
    event_delta_pct = (
        eq.equalized_count / eq.baseline_count - 1.0
        if eq.baseline_count
        else float("nan")
    )

    print(
        f"{'Kennzahl':<24}"
        f"{'BASE':>18}"
        f"{'EQUALIZED':>20}"
        f"{'Delta':>18}"
    )
    print("-" * 100)
    print(
        f"{'Raw Events':<24}"
        f"{eq.baseline_count:>18,}"
        f"{eq.equalized_count:>20,}"
        f"{event_delta:>+18,}"
    )
    print(
        f"{'Event Count Δ':<24}"
        f"{'0.00%':>18}"
        f"{event_delta_pct:>20.2%}"
        f"{event_delta_pct:>+18.2%}"
    )
    print(
        f"{'Event-Count-Toleranz':<24}"
        f"{MAX_EVENT_COUNT_DEVIATION:>17.2%}"
        f"{'präferiert':>20}"
    )

    # --------------------------------------------------------
    # 4) KOMPAKTE COIN-ZUSAMMENFASSUNG
    # --------------------------------------------------------
    print("\n[4] GESAMTÜBERSICHT JE REFCOIN")
    print("-" * 100)
    print(
        f"{'Coin':<10}"
        f"{'Base Thr.':>13}"
        f"{'Eq Thr.':>13}"
        f"{'Thr Δ':>13}"
        f"{'Base Part.':>15}"
        f"{'Eq Part.':>15}"
        f"{'Part. Δ':>15}"
    )
    print("-" * 100)

    for ref in REF_SYMBOLS:
        base_thr = BASE_THRESHOLD_PCT
        eq_thr = eq.thresholds[ref]
        base_part = eq.baseline_participation[ref] * 100.0
        eq_part = eq.equalized_participation[ref] * 100.0

        print(
            f"{ref:<10}"
            f"{base_thr:>12.2f}%"
            f"{eq_thr:>12.2f}%"
            f"{eq_thr - base_thr:>+12.2f} pp"
            f"{base_part:>14.2f}%"
            f"{eq_part:>14.2f}%"
            f"{eq_part - base_part:>+14.2f} pp"
        )

    # --------------------------------------------------------
    # 5) BACKTEST-TABELLE
    # --------------------------------------------------------
    # Die vollständige Hold-by-Hold-Tabelle wird unmittelbar danach
    # von print_comparison() ausgegeben, sobald BASE/EQUALIZED-Stats
    # vorliegen.


def print_comparison(
    target_pair: str,
    window: int,
    direction: str,
    quorum: int,
    base_stats: Dict[str, dict],
    eq_stats: Dict[str, dict],
    bh: Optional[dict],
    base_event_count: int,
    eq_event_count: int,
) -> None:

    print()
    print("=" * 132)
    print("[5] VOLLSTÄNDIGE BACKTEST-TABELLE – BASE vs EQUALIZED")
    print("=" * 132)
    print(
        f"EQUALIZED vs BASE | "
        f"{direction} | Event {window}m | "
        f"{quorum}/5 | Target {target_pair}"
    )
    print("=" * 132)

    print(
        f"{'Hold':>7}"
        f"{'Base n':>9}"
        f"{'Base %pos':>12}"
        f"{'Base Avg':>12}"
        f"{'Base Cmp':>13}"
        f"{'Eq n':>9}"
        f"{'Eq %pos':>12}"
        f"{'Eq Avg':>12}"
        f"{'Eq Cmp':>13}"
        f"{'Eq-Base':>13}"
        f"{'Eq-vs B&H':>14}"
    )

    print("-" * 132)

    bh_roi = (
        bh["roi"]
        if bh
        else None
    )

    for h in HORIZON_NAMES:

        b = base_stats[h]
        e = eq_stats[h]

        delta = (
            e["compound"]
            - b["compound"]
            if (
                e["compound"] is not None
                and b["compound"] is not None
            )
            else None
        )

        eq_vs_bh = (
            e["compound"]
            - bh_roi
            if (
                e["compound"] is not None
                and bh_roi is not None
            )
            else None
        )

        print(
            f"{h:>7}"
            f"{b['n']:>9}"
            f"{fmt_pos(b['pct_pos']):>12}"
            f"{fmt_pct(b['avg']):>12}"
            f"{fmt_pct(b['compound']):>13}"
            f"{e['n']:>9}"
            f"{fmt_pos(e['pct_pos']):>12}"
            f"{fmt_pct(e['avg']):>12}"
            f"{fmt_pct(e['compound']):>13}"
            f"{fmt_pct(delta):>13}"
            f"{fmt_pct(eq_vs_bh):>14}"
        )

    print("-" * 132)

    print(
        f"Raw Events: "
        f"BASE={base_event_count:,} | "
        f"EQUALIZED={eq_event_count:,} | "
        f"Diff="
        f"{(
            eq_event_count
            / base_event_count
            - 1.0
        ):+.2%}"
    )

    if bh:
        print(
            f"B&H: {fmt_pct(bh['roi'])} | "
            f"{bh['buy_time']} -> "
            f"{bh['sell_time']}"
        )


def print_normal_section(
    target_pair: str,
    direction: str,
    window: int,
    threshold: float,
    min_refs: int,
    stats: Dict[str, dict],
    bh: Optional[dict],
) -> None:

    label = min_ref_label(
        min_refs,
        len(REF_SYMBOLS),
    )

    print()
    print("=" * 118)
    print(
        f"{direction:4} | "
        f"Event {window}m | "
        f"{'+' if direction == 'UP' else '-'}"
        f"{threshold:g}% | "
        f"Refs {label} | "
        f"Target {target_pair}"
    )
    print("=" * 118)

    print(
        f"{'Hold':>7}"
        f"{'n':>7}"
        f"{'%pos':>8}"
        f"{'Ø ROI':>11}"
        f"{'Compound':>13}"
        f"{'vs B&H':>11}"
    )

    print("-" * 72)

    bh_roi = (
        bh["roi"]
        if bh
        else None
    )

    for h in HORIZON_NAMES:
        s = stats[h]

        vs_bh = (
            s["compound"] - bh_roi
            if (
                s["compound"] is not None
                and bh_roi is not None
            )
            else None
        )

        print(
            f"{h:>7}"
            f"{s['n']:>7}"
            f"{fmt_pos(s['pct_pos']):>8}"
            f"{fmt_pct(s['avg']):>11}"
            f"{fmt_pct(s['compound']):>13}"
            f"{fmt_pct(vs_bh):>11}"
        )


def print_trade_examples(
    trades: List[dict],
    max_rows: int = 5,
) -> None:

    if not SHOW_TRADES or not trades:
        return

    print("\nBeispiel-Events:")

    for t in trades[:max_rows]:

        moves = " ".join(
            f"{k}:{v:+.2f}%"
            for k, v in t[
                "ref_moves"
            ].items()
        )

        print(
            f"  {t['buy_time']} | "
            f"{t['direction']} | "
            f"{moves} | "
            f"matched={t['matched_refs']}"
        )


# ============================================================
# RANGE
# ============================================================

def clamp_end_to_closed_candle(
    end: datetime,
) -> datetime:

    bar_ms = (
        BAR_MIN
        * 60_000
    )

    now_ms = int(
        time.time()
        * 1000
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


def effective_range(
    start_ms: int,
    end_ms: int,
    *dfs: pd.DataFrame,
) -> Tuple[int, int]:

    lo = int(start_ms)
    hi = int(end_ms)

    for df in dfs:
        if df.empty:
            continue

        arr = df[
            "open_time"
        ].to_numpy(
            dtype=np.int64
        )

        lo = max(
            lo,
            int(arr.min()),
        )

        inside = arr[
            arr <= int(end_ms)
        ]

        if len(inside):
            hi = min(
                hi,
                int(inside.max()),
            )

    return lo, hi


# ============================================================
# MAIN
# ============================================================

def main(
    argv: Optional[List[str]] = None,
) -> None:

    args = parse_cli_args(argv)

    if args.equalizer_restarts < 1:
        raise SystemExit(
            "--equalizer-restarts muss >= 1 sein."
        )

    if args.equalizer_steps < 1:
        raise SystemExit(
            "--equalizer-steps muss >= 1 sein."
        )

    refs_raw = (
        args.ref
        or REF_SYMBOLS
    )

    target_raw = (
        args.target
        or TARGET_SYMBOL
    )

    run_random = (
        RUN_RANDOM
        if args.run_random is None
        else args.run_random
    )

    refs_raw = list(
        dict.fromkeys(
            str(x).strip().upper()
            for x in refs_raw
        )
    )

    if len(refs_raw) != 5:
        raise SystemExit(
            "Der Equalizer ist für genau 5 Refcoins "
            "konfiguriert. Bitte BTC SOL ETH XRP BNB verwenden."
        )

    if set(refs_raw) != set(REF_SYMBOLS):
        raise SystemExit(
            "Der Equalizer erwartet genau "
            "BTC SOL ETH XRP BNB."
        )

    usdt_set = binance_usdt_symbols()

    ref_bases: List[str] = []
    ref_pairs: Dict[str, str] = {}

    for raw in refs_raw:
        base, pair = resolve_pair(
            raw,
            usdt_set,
        )

        if base in ref_bases:
            continue

        ref_bases.append(base)
        ref_pairs[base] = pair

    target_base, target_pair = resolve_pair(
        target_raw,
        usdt_set,
    )

    if target_base in ref_bases:
        raise SystemExit(
            f"Target {target_base} darf nicht "
            "gleichzeitig Referenz sein."
        )

    start, end, window_label = resolve_event_window(
        from_s=args.from_date or FROM_DATE,
        to_s=args.to_date or TO_DATE,
        lookback_days=args.lookback,
        default_lookback=LOOKBACK_DAYS,
    )

    end = clamp_end_to_closed_candle(end)

    start_ms = utc_ms(start)
    end_ms = utc_ms(end)

    fetch_end_ms = utc_ms(
        end
        + pd.Timedelta(
            minutes=MAX_HOLD_MIN
        )
    )

    print("\n" + "=" * 118)
    print(
        "MULTI-REFERENCE BUY BACKTEST "
        "+ REFCOIN EQUALIZER"
    )
    print("=" * 118)

    print(
        f"Refs:       {', '.join(ref_bases)}"
    )
    print(
        f"Target:     {target_base} ({target_pair})"
    )
    print(
        f"Interval:   {KLINE_INTERVAL}"
    )
    print(
        f"Zeitraum:   "
        f"{start.strftime('%Y-%m-%d %H:%M %Z')} "
        f"-> "
        f"{end.strftime('%Y-%m-%d %H:%M %Z')}"
    )
    print(
        f"BASE-Scan:  Fenster={EVENT_WINDOWS_MIN} | "
        f"Thresholds={THRESHOLDS_PCT} | Quoren={MIN_REFS}"
    )
    print(
        "EQUALIZED:  dieselben Fenster + alle Quoren; "
        "individuelle Thresholds je Refcoin"
    )
    print(
        f"Equalizer:  "
        f"{args.equalizer_restarts} Restarts x "
        f"{args.equalizer_steps} Schritte"
    )
    print(
        f"Fee:        {FEE_BPS} bps/Seite | "
        f"Slippage: {SLIPPAGE_BPS} bps"
    )

    print("=" * 118)

    # --------------------------------------------------------
    # LOAD REFERENCES
    # --------------------------------------------------------

    ref_data: Dict[
        str,
        pd.DataFrame
    ] = {}

    print("\n1) Lade Referenzcoins:")

    for base in ref_bases:

        pair = ref_pairs[base]

        print(
            f"   {base:>8} ({pair}) ...",
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
                f"\nKeine Kerzen für Referenz {pair}."
            )

        ref_data[base] = df

        print(
            f" {len(df)} Kerzen"
        )

    # --------------------------------------------------------
    # LOAD TARGET
    # --------------------------------------------------------

    print(
        f"\n2) Lade Target {target_pair} ..."
    )

    target_df = fetch_klines(
        target_pair,
        KLINE_INTERVAL,
        start_ms,
        fetch_end_ms,
    )

    if target_df.empty:
        raise SystemExit(
            f"Keine Kerzen für Target {target_pair}."
        )

    print(
        f"   {len(target_df)} Kerzen"
    )

    bh = buy_and_hold(
        target_df,
        start_ms,
        end_ms,
    )

    # --------------------------------------------------------
    # EQUALIZER
    # --------------------------------------------------------

    print(
        "\n3) Berechne Refcoin-Equalizer ..."
    )

    equalized: Dict[Tuple[int, int, str], EqualizedResult] = {}
    eq_matrices: Dict[int, np.ndarray] = {}

    for window in EQUALIZER_WINDOWS_MIN:
        print(f"\n   Bereite {window}m Return-Matrix vor ...")
        _, moves, _ = prepare_equalizer_matrix(ref_data, window, start_ms, end_ms)
        eq_matrices[window] = moves
        for quorum in EQUALIZER_QUORUMS:
            for direction in ("UP", "DOWN"):
                print(f"   {window}m {direction} {quorum}/5 ...")
                eq = optimize_equalizer(
                    moves, window, direction, quorum,
                    args.equalizer_restarts, args.equalizer_steps, len(moves),
                )
                equalized[(window, quorum, direction)] = eq
                print_equalizer_report(eq, window, direction, quorum)

    # --------------------------------------------------------
    # BASE / EQUALIZED BACKTEST
    # --------------------------------------------------------

    print(
        "\n4) BASE vs EQUALIZED Backtests ..."
    )

    comparison_cache = {}

    for window in EQUALIZER_WINDOWS_MIN:
        base_dict = build_fixed_events(ref_data, [window], BASE_THRESHOLD_PCT, 1, start_ms, end_ms)
        for quorum in EQUALIZER_QUORUMS:
            for direction in ("UP", "DOWN"):
                base_events = [
                    e for e in base_dict[(window, direction)]
                    if len(e.matched_refs) >= quorum
                ]
                base_trades = simulate_events(target_df, base_events, target_pair)
                base_stats = summarize(base_trades)

                eq = equalized[(window, quorum, direction)]
                eq_events = build_equalized_events(
                    ref_data, window, direction, eq.thresholds, quorum, start_ms, end_ms
                )
                eq_trades = simulate_events(target_df, eq_events, target_pair)
                eq_stats = summarize(eq_trades)

                comparison_cache[(window, quorum, direction)] = (
                    base_events, base_stats, eq_events, eq_stats
                )

                print_comparison(
                    target_pair, window, direction, quorum,
                    base_stats, eq_stats, bh, len(base_events), len(eq_events)
                )

    # --------------------------------------------------------
    # OPTIONAL ORIGINAL FULL SCAN
    # --------------------------------------------------------

    if args.full_scan:
        print(
            "\n5) Vollständiger Original-Scan über ALLE konfigurierten "
            "Fenster / Thresholds / Quoren / Richtungen ..."
        )

        result_cache = {}

        for window in EVENT_WINDOWS_MIN:

            for threshold in THRESHOLDS_PCT:

                for min_refs in MIN_REFS:

                    for direction in (
                        "UP",
                        "DOWN",
                    ):

                        events_dict = build_fixed_events(
                            ref_data,
                            [window],
                            threshold,
                            min_refs,
                            start_ms,
                            end_ms,
                        )

                        events = events_dict[
                            (window, direction)
                        ]

                        trades = simulate_events(
                            target_df,
                            events,
                            target_pair,
                        )

                        stats = summarize(
                            trades
                        )

                        result_cache[
                            (
                                window,
                                threshold,
                                min_refs,
                                direction,
                            )
                        ] = (
                            trades,
                            stats,
                        )


    # --------------------------------------------------------
    # RANDOM BASELINE
    # --------------------------------------------------------

    random_cache = {}

    if run_random and args.full_scan:

        print(
            "\n6) RANDOM-Baselines ..."
        )

        for key, (
            trades,
            stats,
        ) in result_cache.items():

            target_n = {
                h: stats[h]["n"]
                for h in HORIZON_NAMES
            }

            seed = (
                RANDOM_SEED
                + key[0] * 1000
                + int(key[1] * 100)
                + key[2] * 10
                + (
                    1
                    if key[3] == "DOWN"
                    else 0
                )
            )

            rng = np.random.default_rng(
                seed
            )

            random_cache[key] = (
                random_matched_stats(
                    target_df,
                    target_n,
                    start_ms,
                    end_ms,
                    rng,
                )
            )

    # --------------------------------------------------------
    # COMPACT EQUALIZER OVERVIEW
    # --------------------------------------------------------

    print("\n" + "=" * 118)
    print("EQUALIZER-KURZÜBERSICHT")
    print("=" * 118)
    print(
        f"{'Event':>8} {'Vote':>6} {'Dir':>6} "
        f"{'Base N':>10} {'Eq N':>10} {'Event Δ':>10} "
        f"{'Base Spread':>14} {'Eq Spread':>14}"
    )
    print("-" * 118)
    for window in EQUALIZER_WINDOWS_MIN:
        for quorum in EQUALIZER_QUORUMS:
            for direction in ("UP", "DOWN"):
                eq = equalized[(window, quorum, direction)]
                print(
                    f"{window:>7}m {quorum:>2}/5 {direction:>6} "
                    f"{eq.baseline_count:>10,} {eq.equalized_count:>10,} "
                    f"{(eq.equalized_count / eq.baseline_count - 1.0):>+9.2%} "
                    f"{eq.baseline_spread * 100:>13.2f} pp "
                    f"{eq.equalized_spread * 100:>13.2f} pp"
                )

    # --------------------------------------------------------
    # ORIGINAL FULL SCAN OUTPUT (optional)
    # --------------------------------------------------------

    if args.full_scan:
        print("\n" + "=" * 118)
        print("ORIGINALER MULTI-REFERENCE VOLLSCAN")
        print("=" * 118)
        for window in EVENT_WINDOWS_MIN:
            for threshold in THRESHOLDS_PCT:
                for min_refs in MIN_REFS:
                    for direction in ("UP", "DOWN"):
                        trades, stats = result_cache[(window, threshold, min_refs, direction)]
                        print_normal_section(target_pair, direction, window, threshold, min_refs, stats, bh)
                        print_trade_examples(trades)

    # --------------------------------------------------------
    # FINAL A/B SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 118)
    print("FINALER A/B-VERGLEICH: BASE vs EQUALIZED")
    print("=" * 118)
    print(
        f"{'Event':>8} {'Vote':>6} {'Dir':>6} {'Hold':>7} "
        f"{'Base Cmp':>14} {'Eq Cmp':>14} {'Eq-Base':>14} "
        f"{'Base n':>9} {'Eq n':>9}"
    )
    print("-" * 118)
    for window in EQUALIZER_WINDOWS_MIN:
        for quorum in EQUALIZER_QUORUMS:
            for direction in ("UP", "DOWN"):
                base_events, base_stats, eq_events, eq_stats = comparison_cache[(window, quorum, direction)]
                for h in HORIZON_NAMES:
                    b = base_stats[h]
                    e = eq_stats[h]
                    delta = (e["compound"] - b["compound"] if e["compound"] is not None and b["compound"] is not None else None)
                    print(
                        f"{window:>7}m {quorum:>2}/5 {direction:>6} {h:>7} "
                        f"{fmt_pct(b['compound']):>14} {fmt_pct(e['compound']):>14} "
                        f"{fmt_pct(delta):>14} {b['n']:>9} {e['n']:>9}"
                    )

    print("=" * 118)
    print("FERTIG")
    print("=" * 118)

if __name__ == "__main__":
    main()
