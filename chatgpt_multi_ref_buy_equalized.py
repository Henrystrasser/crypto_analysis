#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Reference BUY Backtest + Reference-Coin Equalizer (Walk-forward)
======================================================================

1. BASE
   Event, wenn mindestens das Quorum (1/5 ... 5/5) der Refcoins
   BTC / SOL / ETH / XRP / BNB im Event-Fenster (Default 60m) die
   identische 2.00%-Schwelle reißt.

2. EQUALIZED
   Dieselben Quoren, aber mit individuellen Schwellen je Refcoin.
   Ziel der Kalibrierung:
     - Event-Anzahl möglichst nahe an BASE (+/- 5%)
     - BTC/SOL/ETH/XRP/BNB stimmen innerhalb der Events möglichst
       gleich häufig ab (Spread der Participation klein)
     - Ø Participation nahe Quorum/5
     - keine Gewichtung: jede Stimme ist binär und zählt exakt 1

Walk-forward (keine Look-ahead-Kalibrierung):
- Die Schwellen und die Ziel-Event-Anzahl (BASE +/- 5%) werden NUR aus dem
  zurückliegenden Trainingsfenster (WF_TRAIN_DAYS, Default 180 Tage)
  bestimmt; gehandelt wird der darauffolgende Kalendermonat
  (Europe/Berlin), danach rollt das Fenster einen Monat weiter.
- Kalibriert/gehandelt wird nur, wo Refs UND Target Daten haben.
  Monate ohne volle Trainings-Historie werden NICHT gehandelt (Ausgabe
  markiert sie).
- BASE läuft über exakt denselben Handelszeitraum.
- Die Suche ist deterministisch (Koordinatenabstieg auf dem
  0.50-6.00%-Grid): gleiche Daten => gleiches Ergebnis.

Der optionale Original-Vollscan (feste Schwellen, keine Kalibrierung)
läuft über den gesamten Zeitraum.

EQUALIZED-Beispiel:
    BTC >= 2.3%, SOL >= 1.8%, ETH >= 2.1%, XRP >= 1.7%, BNB >= 2.4%
    mindestens 3/5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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
TARGET_SYMBOL = "ENA"

# Ursprüngliche Testfenster.
EVENT_WINDOWS_MIN = [60]

# Ursprüngliche feste Schwellen für den normalen Scan.
THRESHOLDS_PCT = [1.0, 1.3,1.5,1.8, 2.0,2.3,2.6, 3.0]

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

# Deterministische Suche: Koordinatenabstieg auf dem Threshold-Grid,
# max. Anzahl kompletter Durchläufe über alle Refcoins.
EQUALIZER_MAX_PASSES = 20

# Walk-forward: Trainingsfenster (Tage) vor jedem Handelsmonat.
WF_TRAIN_DAYS = 180

# Event-Anzahl darf idealerweise max. +/- 5% von BASE (Training) abweichen.
MAX_EVENT_COUNT_DEVIATION = 0.05

# Ziel-Participation = Quorum/5 (z.B. 3/5 = 60%); Gewicht im Score.
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
        help="RANDOM-Baseline überspringen (überschreibt RUN_RANDOM)",
    )

    p.add_argument(
        "--random",
        dest="run_random",
        action="store_true",
        default=None,
        help=f"RANDOM-Baseline berechnen (Default: RUN_RANDOM={RUN_RANDOM})",
    )

    p.add_argument(
        "--train-days",
        type=int,
        default=None,
        help=f"Walk-forward Trainingsfenster in Tagen (Default: {WF_TRAIN_DAYS})",
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
    # EQUALIZED: individuelle Schwellen je Refcoin (BASE: None).
    thresholds: Optional[Dict[str, float]] = None

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


def threshold_grid() -> np.ndarray:
    """Threshold-Grid EQUALIZER_MIN_PCT..EQUALIZER_MAX_PCT, auf 2 Stellen gerundet."""
    n = int(round((EQUALIZER_MAX_PCT - EQUALIZER_MIN_PCT) / EQUALIZER_STEP_PCT)) + 1
    return np.round(EQUALIZER_MIN_PCT + np.arange(n) * EQUALIZER_STEP_PCT, 2)


def optimize_equalizer(
    moves: np.ndarray,
    direction: str,
    quorum: int,
) -> Optional[EqualizedResult]:
    """
    Deterministischer Koordinatenabstieg auf den Grid-Indizes.

    Start: alle Refcoins auf BASE_THRESHOLD_PCT. Pro Durchlauf wird für
    jeden Refcoin (feste Reihenfolge REF_SYMBOLS) jede Grid-Stufe bei
    festgehaltenen übrigen Schwellen bewertet; übernommen wird nur eine
    echte Verbesserung (Tie-Break: näher an BASE, dann kleinere Stufe).
    Ende, wenn ein Durchlauf nichts verbessert oder nach
    EQUALIZER_MAX_PASSES. Gleiche Daten => gleiches Ergebnis.

    None, wenn im Trainingsfenster keine BASE-Events existieren.
    """
    n_refs = len(REF_SYMBOLS)
    baseline_arr = np.full(n_refs, BASE_THRESHOLD_PCT, dtype=float)
    baseline = evaluate_thresholds_matrix(moves, baseline_arr, direction, quorum)
    baseline_count = baseline["event_count"]
    if baseline_count <= 0:
        return None

    target_participation = quorum / n_refs
    grid = threshold_grid()
    base_idx = int(np.argmin(np.abs(grid - BASE_THRESHOLD_PCT)))
    cache: Dict[Tuple[int, ...], Tuple[float, dict]] = {}

    def score_of(ix: Tuple[int, ...]) -> Tuple[float, dict]:
        if ix not in cache:
            res = evaluate_thresholds_matrix(moves, grid[list(ix)], direction, quorum)
            cache[ix] = (equalizer_objective(res, baseline_count, target_participation), res)
        return cache[ix]

    idx = tuple([base_idx] * n_refs)
    best_score, best_res = score_of(idx)

    # 5/5: jede Stimme ist immer dabei -> Participation immer 100%,
    # Equalisierung ohne Wirkung; BASE-Schwellen bleiben.
    if quorum < n_refs:
        for _ in range(EQUALIZER_MAX_PASSES):
            improved = False
            for k in range(n_refs):
                cur_key = (best_score, abs(idx[k] - base_idx), idx[k])
                choice = idx
                for j in range(len(grid)):
                    if j == idx[k]:
                        continue
                    cand = idx[:k] + (j,) + idx[k + 1:]
                    sc, _ = score_of(cand)
                    key = (sc, abs(j - base_idx), j)
                    if sc < best_score - 1e-12 and key < cur_key:
                        cur_key = key
                        choice = cand
                if choice != idx:
                    idx = choice
                    best_score, best_res = score_of(idx)
                    improved = True
            if not improved:
                break

    return EqualizedResult(
        thresholds={ref: float(grid[idx[i]]) for i, ref in enumerate(REF_SYMBOLS)},
        baseline_count=baseline_count,
        equalized_count=best_res["event_count"],
        baseline_participation=baseline["participation"],
        equalized_participation=best_res["participation"],
        baseline_spread=baseline["spread"],
        equalized_spread=best_res["spread"],
        score=best_score,
    )


def events_from_matrix(
    times: np.ndarray,
    moves: np.ndarray,
    window: int,
    direction: str,
    thresholds: Dict[str, float],
    quorum: int,
    lo_ms: int,
    hi_ms: int,
    equalized: bool,
) -> List[MultiRefEvent]:
    """Events im Zeitraum [lo_ms, hi_ms] aus der vorberechneten Return-Matrix."""
    if moves.size == 0:
        return []
    sel = (times >= lo_ms) & (times <= hi_ms)
    t_sel = times[sel]
    m_sel = moves[sel]
    thr = np.array([thresholds[r] for r in REF_SYMBOLS], dtype=float)
    votes = m_sel >= thr[None, :] if direction == "UP" else m_sel <= -thr[None, :]
    rows = np.flatnonzero(votes.sum(axis=1) >= quorum)
    events: List[MultiRefEvent] = []
    for i in rows:
        events.append(MultiRefEvent(
            end_ms=int(t_sel[i]),
            direction=direction,
            window_min=window,
            threshold_pct=float("nan") if equalized else BASE_THRESHOLD_PCT,
            min_refs=quorum,
            ref_moves={r: float(m_sel[i, k]) for k, r in enumerate(REF_SYMBOLS)},
            matched_refs=[r for k, r in enumerate(REF_SYMBOLS) if votes[i, k]],
            thresholds=dict(thresholds) if equalized else None,
        ))
    return events


def event_participation(events: List[MultiRefEvent]) -> Tuple[Dict[str, float], Optional[float]]:
    """Anteil der Events, in denen jeder Refcoin mitstimmt; Spread = max - min."""
    if not events:
        return {r: 0.0 for r in REF_SYMBOLS}, None
    part = {
        r: sum(r in e.matched_refs for e in events) / len(events)
        for r in REF_SYMBOLS
    }
    return part, max(part.values()) - min(part.values())


# ============================================================
# WALK-FORWARD
# ============================================================

BERLIN_TZ = ZoneInfo("Europe/Berlin")


@dataclass
class WFMonth:
    label: str
    trade_lo: int
    trade_hi: int
    train_lo: int
    train_hi: int
    tradeable: bool


def build_wf_plan(eff_lo: int, eff_hi: int, train_days: int) -> List[WFMonth]:
    """Kalendermonate (Europe/Berlin) im effektiven Zeitraum; Training = train_days davor."""
    plan: List[WFMonth] = []
    if eff_hi < eff_lo:
        return plan
    d = datetime.fromtimestamp(eff_lo / 1000.0, tz=BERLIN_TZ)
    m0 = datetime(d.year, d.month, 1, tzinfo=BERLIN_TZ)
    while True:
        m1 = datetime(m0.year + (m0.month == 12), m0.month % 12 + 1, 1, tzinfo=BERLIN_TZ)
        m0_ms, m1_ms = utc_ms(m0), utc_ms(m1)
        if m0_ms > eff_hi:
            break
        # Kalendertage in Europe/Berlin (Wandzeit, DST-korrekt).
        train_lo = utc_ms(m0 - timedelta(days=int(train_days)))
        plan.append(WFMonth(
            label=m0.strftime("%Y-%m"),
            trade_lo=max(m0_ms, eff_lo),
            trade_hi=min(m1_ms - 1, eff_hi),
            train_lo=train_lo,
            train_hi=m0_ms - 1,
            tradeable=train_lo >= eff_lo,
        ))
        m0 = m1
    return plan


def pp_fmt(v: float, width: int = 0) -> str:
    """Prozentpunkte gerundet, ohne '-0.00'."""
    r = round(float(v), 2) + 0.0
    return f"{r:>+{width}.2f} pp" if width else f"{r:+.2f} pp"


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

    if len(accepted) < n_target:
        print(
            f"   Hinweis RANDOM: nur {len(accepted)}/{n_target} "
            f"nicht-überlappende Trades möglich."
        )

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
    rec: dict,
    window: int,
    direction: str,
    quorum: int,
    period: str,
    train_days: int,
) -> None:
    """Walk-forward-Kalibrierung + Out-of-sample-Participation je Kombination."""

    print()
    print("=" * 132)
    print(
        f"EQUALIZED vs BASE | {direction} | Event {window}m | "
        f"{quorum}/5 | Handel {period}"
    )
    print("=" * 132)

    # 1) Schwellen je Handelsmonat (aus Training)
    print(
        f"\n[1] THRESHOLDS JE HANDELSMONAT – kalibriert auf den "
        f"{train_days} Tagen davor (BASE = {BASE_THRESHOLD_PCT:.2f}% für alle)"
    )
    print("-" * 132)
    print(
        f"{'Monat':<9}"
        + "".join(f"{r:>8}" for r in REF_SYMBOLS)
        + f"{'Train BASE N':>14}{'Train EQ N':>12}"
        + f"{'Train Spread B→E':>20}   Status"
    )
    print("-" * 132)
    for m in rec["months"]:
        eq = m["eq"]
        thr = m["thresholds"]
        cols = "".join(f"{thr[r]:>7.2f}%" for r in REF_SYMBOLS)
        if eq is None:
            print(
                f"{m['label']:<9}{cols}{0:>14}{'n/a':>12}{'n/a':>20}   "
                f"Fallback BASE-Schwellen (keine BASE-Events im Training)"
            )
        else:
            print(
                f"{m['label']:<9}{cols}{eq.baseline_count:>14,}"
                f"{eq.equalized_count:>12,}"
                f"{eq.baseline_spread * 100:>9.1f} → {eq.equalized_spread * 100:>5.1f} pp"
                f"   kalibriert"
            )

    # 2) Participation im Handelszeitraum (out-of-sample)
    base_part, base_spread = event_participation(rec["base_events"])
    eq_part, eq_spread = event_participation(rec["eq_events"])

    print("\n[2] VOTING-PARTICIPATION im Handelszeitraum (out-of-sample) – BASE vs EQUALIZED")
    print("-" * 100)
    print(f"{'Coin':<18}{'BASE':>14}{'EQUALIZED':>16}{'Delta':>16}")
    print("-" * 100)
    has_b = bool(rec["base_events"])
    has_e = bool(rec["eq_events"])

    def _pc(v: float, ok: bool, w: int) -> str:
        return f"{v:>{w - 1}.2f}%" if ok else f"{'n/a':>{w}}"

    def _dpp(b_: float, e_: float, w: int) -> str:
        return pp_fmt(e_ - b_, w - 3) if has_b and has_e else f"{'n/a':>{w}}"

    for ref in REF_SYMBOLS:
        b_ = base_part[ref] * 100.0
        e_ = eq_part[ref] * 100.0
        print(f"{ref:<18}{_pc(b_, has_b, 14)}{_pc(e_, has_e, 16)}{_dpp(b_, e_, 16):>16}")
    print("-" * 100)

    def _sp(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v * 100:.2f} pp"

    sp_delta = (
        pp_fmt((eq_spread - base_spread) * 100, 13)
        if base_spread is not None and eq_spread is not None
        else "n/a"
    )
    print(f"{'Spread':<18}{_sp(base_spread):>14}{_sp(eq_spread):>16}{sp_delta:>16}   kleiner = besser")

    b_mean = sum(base_part.values()) / len(base_part) * 100.0
    e_mean = sum(eq_part.values()) / len(eq_part) * 100.0
    print(
        f"{'Ø Participation':<18}{_pc(b_mean, has_b, 14)}{_pc(e_mean, has_e, 16)}"
        f"{_dpp(b_mean, e_mean, 16):>16}   Ziel {quorum / len(REF_SYMBOLS):.0%}"
    )

    # 3) Events im Handelszeitraum
    nb, ne = len(rec["base_events"]), len(rec["eq_events"])
    print("\n[3] EVENTS im Handelszeitraum – BASE vs EQUALIZED")
    print("-" * 100)
    print(f"{'Kennzahl':<24}{'BASE':>18}{'EQUALIZED':>20}{'Delta':>18}")
    print("-" * 100)
    print(f"{'Raw Events':<24}{nb:>18,}{ne:>20,}{ne - nb:>+18,}")
    print(f"{'Event Count Δ':<24}{'':>18}{'':>20}{fmt_ratio(ne, nb):>18}")
    print(
        f"{'Toleranz (Training)':<24}{MAX_EVENT_COUNT_DEVIATION:>17.2%} "
        f"{'präferiert':>20}"
    )

    # 4) Gesamtübersicht je Refcoin
    calibrated = [m for m in rec["months"] if m["eq"] is not None]
    print("\n[4] GESAMTÜBERSICHT JE REFCOIN (Schwellen über die kalibrierten Monate)")
    print("-" * 100)
    print(
        f"{'Coin':<10}{'Base Thr.':>11}{'Eq Thr. Ø':>12}{'Eq min':>10}{'Eq max':>10}"
        f"{'Base Part.':>13}{'Eq Part.':>12}{'Part. Δ':>14}"
    )
    print("-" * 100)
    for ref in REF_SYMBOLS:
        vals = [m["thresholds"][ref] for m in calibrated]
        b_ = base_part[ref] * 100.0
        e_ = eq_part[ref] * 100.0
        if vals:
            thr_s = f"{np.mean(vals):>11.2f}%{min(vals):>9.2f}%{max(vals):>9.2f}%"
        else:
            thr_s = f"{'n/a':>12}{'n/a':>10}{'n/a':>10}"
        print(
            f"{ref:<10}{BASE_THRESHOLD_PCT:>10.2f}%{thr_s}"
            f"{_pc(b_, has_b, 13)}{_pc(e_, has_e, 12)}{_dpp(b_, e_, 14):>14}"
        )
    print(
        f"Monate: {len(calibrated)} kalibriert, "
        f"{len(rec['months']) - len(calibrated)} mit Fallback BASE-Schwellen"
    )


def fmt_ratio(num: int, den: int) -> str:
    """num/den - 1 als Prozent; 'n/a' bei den == 0."""
    if not den:
        return "n/a"
    return f"{num / den - 1.0:+.2%}"


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
    period: str,
    rnd_base: Optional[Dict[str, dict]] = None,
    rnd_eq: Optional[Dict[str, dict]] = None,
) -> None:

    print()
    print("=" * 132)
    print("[5] VOLLSTÄNDIGE BACKTEST-TABELLE – BASE vs EQUALIZED")
    print("=" * 132)
    print(
        f"EQUALIZED vs BASE | "
        f"{direction} | Event {window}m | "
        f"{quorum}/5 | Target {target_pair} | {period}"
    )
    print("=" * 132)

    with_rnd = rnd_base is not None and rnd_eq is not None

    print(
        f"{'Hold':>7}"
        f"{'Base n':>8}"
        f"{'Base %pos':>10}"
        f"{'Base Avg':>10}"
        f"{'Base Cmp':>11}"
        f"{'Eq n':>7}"
        f"{'Eq %pos':>9}"
        f"{'Eq Avg':>10}"
        f"{'Eq Cmp':>11}"
        f"{'Eq-Base':>11}"
        f"{'Eq-vs B&H':>11}"
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
        eq_vs_bh = (
            e["compound"] - bh_roi
            if e["compound"] is not None and bh_roi is not None
            else None
        )

        rnd_cols = ""
        if with_rnd:
            rnd_cols = (
                f"{fmt_pct(rnd_base[h]['compound']):>12}"
                f"{fmt_pct(rnd_eq[h]['compound']):>12}"
            )

        print(
            f"{h:>7}"
            f"{b['n']:>8}"
            f"{fmt_pos(b['pct_pos']):>10}"
            f"{fmt_pct(b['avg']):>10}"
            f"{fmt_pct(b['compound']):>11}"
            f"{e['n']:>7}"
            f"{fmt_pos(e['pct_pos']):>9}"
            f"{fmt_pct(e['avg']):>10}"
            f"{fmt_pct(e['compound']):>11}"
            f"{fmt_pct(delta):>11}"
            f"{fmt_pct(eq_vs_bh):>11}"
            + rnd_cols
        )

    print("-" * 132)

    print(
        f"Raw Events: "
        f"BASE={base_event_count:,} | "
        f"EQUALIZED={eq_event_count:,} | "
        f"Diff={fmt_ratio(eq_event_count, base_event_count)}"
    )
    if with_rnd:
        print(
            "Rnd(B)/Rnd(E) = RANDOM-Einstiege im selben Handelszeitraum mit exakt "
            "derselben Trade-Anzahl wie BASE bzw. EQUALIZED (je Hold)."
        )

    if bh:
        print(
            f"B&H (Handelszeitraum): {fmt_pct(bh['roi'])} | "
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
    period: str = "",
    random_stats: Optional[Dict[str, dict]] = None,
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
        + (f" | {period}" if period else "")
    )
    print("=" * 118)

    print(
        f"{'Hold':>7}"
        f"{'n':>7}"
        f"{'%pos':>8}"
        f"{'Ø ROI':>11}"
        f"{'Compound':>13}"
        f"{'vs B&H':>11}"
        + (f"{'Rnd n':>8}{'Rnd Cmp':>13}" if random_stats else "")
    )

    print("-" * 72)

    bh_roi = bh["roi"] if bh else None

    for h in HORIZON_NAMES:
        s = stats[h]

        vs_bh = (
            s["compound"] - bh_roi
            if s["compound"] is not None and bh_roi is not None
            else None
        )

        rnd = ""
        if random_stats:
            r = random_stats[h]
            rnd = f"{r['n']:>8}{fmt_pct(r['compound']):>13}"

        print(
            f"{h:>7}"
            f"{s['n']:>7}"
            f"{fmt_pos(s['pct_pos']):>8}"
            f"{fmt_pct(s['avg']):>11}"
            f"{fmt_pct(s['compound']):>13}"
            f"{fmt_pct(vs_bh):>11}"
            + rnd
        )


def block_range(from_ms: int, to_ms: int) -> str:
    """'from YYYY-MM-DD to YYYY-MM-DD' (Europe/Berlin) für Block-Header."""
    a = datetime.fromtimestamp(from_ms / 1000.0, tz=BERLIN_TZ).strftime("%Y-%m-%d")
    b = datetime.fromtimestamp(to_ms / 1000.0, tz=BERLIN_TZ).strftime("%Y-%m-%d")
    return f"from {a} to {b}"


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

    train_days = (
        WF_TRAIN_DAYS
        if args.train_days is None
        else args.train_days
    )

    if train_days < 1:
        raise SystemExit(
            "--train-days muss >= 1 sein."
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

    # Kein Nachladen über TO_DATE hinaus: kein Trade darf nach dem
    # Enddatum schließen. Strategie, RANDOM und HODL enden am selben Punkt
    # (Close der letzten Kerze mit open_time <= end_ms).
    fetch_end_ms = end_ms

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
        f"Equalizer:  deterministischer Koordinatenabstieg | Grid "
        f"{EQUALIZER_MIN_PCT:.2f}-{EQUALIZER_MAX_PCT:.2f}% "
        f"(Schritt {EQUALIZER_STEP_PCT:.2f}) | max. {EQUALIZER_MAX_PASSES} Durchläufe"
    )
    print(
        f"Walk-fwd:   Training {train_days} Tage -> Handel Folgemonat "
        f"(Europe/Berlin), dann rollen"
    )
    print(
        f"RANDOM:     {'an' if run_random else 'aus'} "
        f"(--random / --no-random, Config RUN_RANDOM)"
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

        df = df[df["open_time"] <= end_ms].reset_index(drop=True)
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

    # Target-Daten auf <= Enddatum begrenzen (Exits nie nach TO_DATE).
    target_df = target_df[target_df["open_time"] <= end_ms].reset_index(drop=True)

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

    eff_lo, eff_hi = effective_range(
        start_ms, end_ms, target_df, *ref_data.values()
    )
    full_period = block_range(eff_lo, eff_hi)

    # --------------------------------------------------------
    # WALK-FORWARD PLAN
    # --------------------------------------------------------

    plan = build_wf_plan(eff_lo, eff_hi, train_days)
    trade_months = [m for m in plan if m.tradeable]

    print(
        f"\n3) Walk-forward-Plan (Training {train_days} Tage, "
        f"Handel = Folgemonat, Europe/Berlin)"
    )
    print(
        f"   Daten Refs + Target: {ms_to_berlin_str(eff_lo)} -> "
        f"{ms_to_berlin_str(eff_hi)}"
    )
    for m in plan:
        if m.tradeable:
            print(
                f"   {m.label}: Handel {ms_to_berlin_str(m.trade_lo)} -> "
                f"{ms_to_berlin_str(m.trade_hi)} | Training "
                f"{ms_to_berlin_str(m.train_lo)} -> {ms_to_berlin_str(m.train_hi)}"
            )
        else:
            print(
                f"   {m.label}: NICHT GEHANDELT – keine {train_days} Tage "
                f"Historie (Daten erst ab {ms_to_berlin_str(eff_lo)})"
            )

    # --------------------------------------------------------
    # EQUALIZER (walk-forward) + BASE vs EQUALIZED
    # --------------------------------------------------------

    equalized_recs: Dict[Tuple[int, int, str], dict] = {}
    skipped: Dict[Tuple[int, int, str], str] = {}
    comparison_cache: Dict[Tuple[int, int, str], tuple] = {}
    trade_period = ""

    if not trade_months:
        print(
            f"\n   !! Kein handelbarer Monat: Für keinen Monat liegen "
            f"{train_days} Tage Historie (Refs + Target) vor. "
            f"BASE/EQUALIZED-Vergleich entfällt."
        )
    else:
        trade_lo = trade_months[0].trade_lo
        trade_hi = trade_months[-1].trade_hi
        trade_period = block_range(trade_lo, trade_hi)
        bh_trade = buy_and_hold(target_df, trade_lo, trade_hi)
        base_thr = {r: BASE_THRESHOLD_PCT for r in REF_SYMBOLS}

        print(
            f"\n4) Walk-forward Equalizer + BASE vs EQUALIZED "
            f"(Handel {trade_period}, {len(trade_months)} Monate) ..."
        )

        for window in EQUALIZER_WINDOWS_MIN:
            print(f"\n   Bereite {window}m Return-Matrix vor ...")
            times_arr, moves, _ = prepare_equalizer_matrix(
                ref_data, window, start_ms, end_ms
            )

            for quorum in EQUALIZER_QUORUMS:
                for direction in ("UP", "DOWN"):
                    key = (window, quorum, direction)
                    print(f"   {window}m {direction} {quorum}/5 ...", flush=True)

                    months = []
                    eq_events: List[MultiRefEvent] = []
                    for m in trade_months:
                        train = (times_arr >= m.train_lo) & (times_arr <= m.train_hi)
                        eq = optimize_equalizer(moves[train], direction, quorum)
                        thr = eq.thresholds if eq is not None else dict(base_thr)
                        months.append({"label": m.label, "eq": eq, "thresholds": thr})
                        eq_events.extend(events_from_matrix(
                            times_arr, moves, window, direction, thr, quorum,
                            m.trade_lo, m.trade_hi, equalized=True,
                        ))

                    if all(x["eq"] is None for x in months):
                        skipped[key] = "keine BASE-Events in allen Trainingsfenstern"
                        print(f"   -> {window}m {direction} {quorum}/5 übersprungen: {skipped[key]}")
                        continue

                    base_events = events_from_matrix(
                        times_arr, moves, window, direction, base_thr, quorum,
                        trade_lo, trade_hi, equalized=False,
                    )

                    rec = {
                        "months": months,
                        "base_events": base_events,
                        "eq_events": eq_events,
                    }
                    equalized_recs[key] = rec
                    print_equalizer_report(
                        rec, window, direction, quorum, trade_period, train_days
                    )

                    base_stats = summarize(simulate_events(target_df, base_events, target_pair))
                    eq_stats = summarize(simulate_events(target_df, eq_events, target_pair))

                    rnd_b = rnd_e = None
                    if run_random:
                        seed = (
                            RANDOM_SEED + window * 1000 + quorum * 10
                            + (1 if direction == "DOWN" else 0)
                        )
                        rnd_b = random_matched_stats(
                            target_df, {h: base_stats[h]["n"] for h in HORIZON_NAMES},
                            trade_lo, trade_hi, np.random.default_rng(seed + 500_000),
                        )
                        rnd_e = random_matched_stats(
                            target_df, {h: eq_stats[h]["n"] for h in HORIZON_NAMES},
                            trade_lo, trade_hi, np.random.default_rng(seed + 600_000),
                        )

                    comparison_cache[key] = (
                        base_events, base_stats, eq_events, eq_stats, rnd_b, rnd_e
                    )

                    print_comparison(
                        target_pair, window, direction, quorum,
                        base_stats, eq_stats, bh_trade,
                        len(base_events), len(eq_events),
                        trade_period, rnd_b, rnd_e,
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
    print(
        "EQUALIZER-KURZÜBERSICHT (Handelszeitraum, out-of-sample"
        + (f", {trade_period})" if trade_period else ")")
    )
    print("=" * 118)
    print(
        f"{'Event':>8} {'Vote':>6} {'Dir':>6} "
        f"{'Base N':>10} {'Eq N':>10} {'Event Δ':>10} "
        f"{'Base Spread':>14} {'Eq Spread':>14} {'Monate kal./Fallback':>22}"
    )
    print("-" * 118)
    for window in EQUALIZER_WINDOWS_MIN:
        for quorum in EQUALIZER_QUORUMS:
            for direction in ("UP", "DOWN"):
                key = (window, quorum, direction)
                if key not in equalized_recs:
                    reason = skipped.get(key, "kein handelbarer Monat")
                    print(f"{window:>7}m {quorum:>2}/5 {direction:>6}   übersprungen: {reason}")
                    continue
                rec = equalized_recs[key]
                nb, ne = len(rec["base_events"]), len(rec["eq_events"])
                _, b_sp = event_participation(rec["base_events"])
                _, e_sp = event_participation(rec["eq_events"])
                n_cal = sum(m["eq"] is not None for m in rec["months"])
                n_fb = len(rec["months"]) - n_cal
                print(
                    f"{window:>7}m {quorum:>2}/5 {direction:>6} "
                    f"{nb:>10,} {ne:>10,} {fmt_ratio(ne, nb):>10} "
                    f"{('n/a' if b_sp is None else f'{b_sp * 100:.2f} pp'):>14} "
                    f"{('n/a' if e_sp is None else f'{e_sp * 100:.2f} pp'):>14} "
                    f"{f'{n_cal}/{n_fb}':>22}"
                )

    # --------------------------------------------------------
    # ORIGINAL FULL SCAN OUTPUT (optional)
    # --------------------------------------------------------

    if args.full_scan:
        print("\n" + "=" * 118)
        print(f"ORIGINALER MULTI-REFERENCE VOLLSCAN (feste Schwellen, {full_period})")
        print("=" * 118)
        for window in EVENT_WINDOWS_MIN:
            for threshold in THRESHOLDS_PCT:
                for min_refs in MIN_REFS:
                    for direction in ("UP", "DOWN"):
                        key = (window, threshold, min_refs, direction)
                        trades, stats = result_cache[key]
                        print_normal_section(
                            target_pair, direction, window, threshold, min_refs,
                            stats, bh, period=full_period,
                            random_stats=random_cache.get(key) if run_random else None,
                        )
                        print_trade_examples(trades)
        if bh:
            print(
                f"\nB&H (gesamter Zeitraum): {fmt_pct(bh['roi'])} | "
                f"{bh['buy_time']} -> {bh['sell_time']}"
            )

    # --------------------------------------------------------
    # FINAL A/B SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 118)
    print(
        "FINALER A/B-VERGLEICH: BASE vs EQUALIZED (walk-forward"
        + (f", {trade_period})" if trade_period else ")")
    )
    print("=" * 118)
    print(
        f"{'Event':>8} {'Vote':>6} {'Dir':>6} {'Hold':>7} "
        f"{'Base Cmp':>12} {'Eq Cmp':>12} {'Eq-Base':>12} "
        f"{'Base n':>7} {'Eq n':>7}"
        + (f" {'Rnd(B) Cmp':>12} {'Rnd(E) Cmp':>12}" if run_random else "")
    )
    print("-" * 118)
    for window in EQUALIZER_WINDOWS_MIN:
        for quorum in EQUALIZER_QUORUMS:
            for direction in ("UP", "DOWN"):
                key = (window, quorum, direction)
                if key not in comparison_cache:
                    reason = skipped.get(key, "kein handelbarer Monat")
                    print(f"{window:>7}m {quorum:>2}/5 {direction:>6}   übersprungen: {reason}")
                    continue
                _, base_stats, _, eq_stats, rnd_b, rnd_e = comparison_cache[key]
                for h in HORIZON_NAMES:
                    b = base_stats[h]
                    e = eq_stats[h]
                    delta = (
                        e["compound"] - b["compound"]
                        if e["compound"] is not None and b["compound"] is not None
                        else None
                    )
                    rnd = ""
                    if run_random and rnd_b is not None and rnd_e is not None:
                        rnd = (
                            f" {fmt_pct(rnd_b[h]['compound']):>12}"
                            f" {fmt_pct(rnd_e[h]['compound']):>12}"
                        )
                    print(
                        f"{window:>7}m {quorum:>2}/5 {direction:>6} {h:>7} "
                        f"{fmt_pct(b['compound']):>12} {fmt_pct(e['compound']):>12} "
                        f"{fmt_pct(delta):>12} {b['n']:>7} {e['n']:>7}"
                        + rnd
                    )

    print("=" * 118)
    print("FERTIG")
    print("=" * 118)

if __name__ == "__main__":
    from datetime import datetime as _start_dt
    from zoneinfo import ZoneInfo as _StartZoneInfo
    print(f"Start: {_start_dt.now(_StartZoneInfo('Europe/Berlin')):%Y-%m-%d %H:%M:%S %Z}", flush=True)
    main()
