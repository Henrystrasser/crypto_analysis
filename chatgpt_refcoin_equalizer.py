#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chatgpt_refcoin_equalizer.py

Ziel:
    3/5 Reference-Coin-Events sollen weiterhin ungefähr gleich häufig
    auftreten wie bei der normalen 2%-Variante.

    Gleichzeitig sollen BTC / SOL / ETH / XRP / BNB innerhalb dieser
    3/5-Events möglichst gleich häufig eine Stimme beitragen.

    Bei 5 Coins und einem 3/5-Event liegt das natürliche Ziel bei ~60 %
    Beteiligung je Coin.

Wichtig:
    - KEIN Coin ist auf 2 % fixiert.
    - 2 % ist nur die Ausgangsbasis.
    - Jeder Coin bekommt seine eigene Schwelle.
    - Jede Stimme zählt weiterhin exakt 1.
    - Es gibt KEINE Gewichtung von BTC, SOL usw.
    - UP wird optimiert. DOWN wird anschließend separat ausgewertet.
"""

import argparse
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

REF_SYMBOLS = [
    "BTCUSDT",
    "SOLUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "BNBUSDT",
]

TARGET_SYMBOL = "ENAUSDT"

INTERVAL = "1h"

START_DATE = "2024-01-01"
END_DATE = "2026-10-05"

EVENT_WINDOWS_MIN = [60, 180]

BASE_THRESHOLD = 0.02

QUORUM = 3

FEE_PER_SIDE = 0.00075

HOLDS_HOURS = [3, 6, 12, 24, 48, 60, 72, 78, 84, 90, 96, 120, 168, 192]

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

MAX_LIMIT = 1000

TZ = ZoneInfo("Europe/Berlin")


# ============================================================
# EQUALIZER CONFIG
# ============================================================

# Schrittweite, mit der Schwellen verändert werden.
#
# Beispiel:
#   2.00 % -> 2.10 %
#
# Kleinere Schritte = genauer, aber mehr Rechenzeit.
THRESHOLD_STEP = 0.0010       # 0.10 %

MIN_THRESHOLD = 0.0050        # 0.50 %
MAX_THRESHOLD = 0.0600        # 6.00 %

MAX_ITERATIONS = 100

# Wie stark wir die Event-Anzahl gegenüber der Baseline
# berücksichtigen.
EVENT_COUNT_WEIGHT = 2.0

# Wie stark wir die Ungleichheit der Coin-Beteiligung
# berücksichtigen.
BALANCE_WEIGHT = 1.0

# Ab wann wir eine Lösung als ausreichend gut betrachten.
MAX_EVENT_COUNT_DEVIATION = 0.05     # +/- 5 %
MAX_PARTICIPATION_SPREAD = 0.05      # max 5 Prozentpunkte


# ============================================================
# CLI
# ============================================================

def parse_cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"Startdatum, default: {START_DATE}",
    )

    parser.add_argument(
        "--end",
        default=END_DATE,
        help=f"Enddatum, default: {END_DATE}",
    )

    return parser.parse_args()


# ============================================================
# BINANCE DOWNLOAD
# ============================================================

def date_to_ms(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(tzinfo=TZ)
    return int(dt.timestamp() * 1000)


def ms_to_berlin_str(ms):
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.astimezone(TZ).strftime("%Y-%m-%d %H:%M")


def load_klines(symbol, interval, start_ms, end_ms):
    all_rows = []
    current = start_ms

    session = requests.Session()

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }

        success = False

        for attempt in range(8):
            try:
                r = session.get(
                    BINANCE_URL,
                    params=params,
                    timeout=30,
                )

                if r.status_code == 429:
                    wait = 2 ** attempt
                    print(
                        f"  {symbol}: Rate limit -> "
                        f"{wait}s warten..."
                    )
                    time.sleep(wait)
                    continue

                r.raise_for_status()

                data = r.json()

                success = True
                break

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                wait = min(30, 2 ** attempt)

                print(
                    f"  {symbol}: Netzwerkfehler "
                    f"(Versuch {attempt + 1}/8) -> "
                    f"{wait}s..."
                )

                time.sleep(wait)

            except Exception as e:
                print(
                    f"  {symbol}: Fehler: {e}"
                )

                time.sleep(2)

        if not success:
            raise RuntimeError(
                f"Download für {symbol} nach mehreren "
                f"Versuchen fehlgeschlagen."
            )

        if not data:
            break

        all_rows.extend(data)

        last_open = data[-1][0]

        if last_open >= end_ms:
            break

        next_current = last_open + 1

        if next_current <= current:
            break

        current = next_current

        time.sleep(0.15)

    if not all_rows:
        raise RuntimeError(
            f"Keine Daten für {symbol} erhalten."
        )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]

    df = pd.DataFrame(
        all_rows,
        columns=columns,
    )

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = df[
        ["open_time", "close"]
    ].dropna()

    df = df.drop_duplicates(
        subset="open_time"
    )

    df = df.sort_values(
        "open_time"
    )

    return df


# ============================================================
# DATA ALIGNMENT
# ============================================================

def load_all_data(start_ms, end_ms):
    data = {}

    symbols = REF_SYMBOLS + [TARGET_SYMBOL]

    for i, symbol in enumerate(symbols, 1):
        print(
            f"[{i}/{len(symbols)}] Lade {symbol}..."
        )

        df = load_klines(
            symbol,
            INTERVAL,
            start_ms,
            end_ms,
        )

        data[symbol] = df

        print(
            f"  {len(df):,} Candles"
        )

    return data


def build_common_dataframe(data):
    merged = None

    for symbol, df in data.items():

        tmp = df.copy()

        tmp = tmp.rename(
            columns={
                "close": symbol
            }
        )

        if merged is None:
            merged = tmp
        else:
            merged = merged.merge(
                tmp,
                on="open_time",
                how="inner",
            )

    merged = merged.sort_values(
        "open_time"
    ).reset_index(drop=True)

    return merged


# ============================================================
# RETURNS
# ============================================================

def add_returns(df, event_window_min):
    bars = event_window_min // 60

    if bars < 1:
        raise ValueError(
            "EVENT WINDOW muss mindestens 60m sein."
        )

    out = df.copy()

    for symbol in REF_SYMBOLS:

        out[f"{symbol}_ret"] = (
            out[symbol]
            / out[symbol].shift(bars)
            - 1.0
        )

    return out


# ============================================================
# NORMAL BASELINE
# ============================================================

def get_votes(
    df,
    thresholds,
    direction="UP",
):
    """
    Gibt pro Coin True/False zurück.

    thresholds:
        dict(symbol -> threshold)

    direction:
        UP:
            return >= +threshold

        DOWN:
            return <= -threshold
    """

    votes = {}

    for symbol in REF_SYMBOLS:

        ret = df[f"{symbol}_ret"]

        threshold = thresholds[symbol]

        if direction == "UP":
            votes[symbol] = ret >= threshold

        elif direction == "DOWN":
            votes[symbol] = ret <= -threshold

        else:
            raise ValueError(
                "direction muss UP oder DOWN sein."
            )

    return votes


def build_quorum_mask(votes):
    vote_df = pd.DataFrame(votes)

    count = vote_df.sum(axis=1)

    return count >= QUORUM


# ============================================================
# EVALUATION OF THRESHOLDS
# ============================================================

def evaluate_thresholds(
    df,
    thresholds,
    direction="UP",
):
    votes = get_votes(
        df,
        thresholds,
        direction,
    )

    vote_df = pd.DataFrame(votes)

    vote_count = vote_df.sum(axis=1)

    event_mask = vote_count >= QUORUM

    event_count = int(
        event_mask.sum()
    )

    participation = {}

    if event_count > 0:

        for symbol in REF_SYMBOLS:

            participation[symbol] = (
                vote_df.loc[event_mask, symbol]
                .mean()
            )

    else:

        for symbol in REF_SYMBOLS:
            participation[symbol] = 0.0

    return {
        "event_mask": event_mask,
        "vote_df": vote_df,
        "event_count": event_count,
        "participation": participation,
    }


# ============================================================
# EQUALIZER SCORE
# ============================================================

def equalizer_score(
    event_count,
    baseline_event_count,
    participation,
):
    """
    Niedriger = besser.

    Ziel:

    1. Event-Anzahl nahe Baseline
    2. Beteiligung der 5 Coins möglichst gleich

    Wichtig:
        Wir erzwingen NICHT exakt 60 %.
        Wenn die beste Lösung z.B.
        57 / 61 / 59 / 63 / 60 ist,
        ist das vollkommen okay.
    """

    if baseline_event_count <= 0:
        return float("inf")

    event_deviation = abs(
        event_count - baseline_event_count
    ) / baseline_event_count

    values = np.array(
        list(participation.values()),
        dtype=float,
    )

    if len(values) == 0:
        balance_error = float("inf")
    else:
        balance_error = (
            values.max() - values.min()
        )

    return (
        EVENT_COUNT_WEIGHT * event_deviation
        + BALANCE_WEIGHT * balance_error
    )


# ============================================================
# ITERATIVE EQUALIZER
# ============================================================

def optimize_thresholds(
    df,
    baseline_thresholds,
    direction="UP",
):
    baseline = evaluate_thresholds(
        df,
        baseline_thresholds,
        direction,
    )

    baseline_count = baseline[
        "event_count"
    ]

    print()
    print("=" * 78)
    print(
        f"EQUALIZER OPTIMIERUNG — {direction}"
    )
    print("=" * 78)

    print(
        f"Baseline 3/5 @ 2%: "
        f"{baseline_count:,} Events"
    )

    print()
    print("Baseline Participation:")

    for symbol in REF_SYMBOLS:

        p = baseline["participation"][symbol]

        print(
            f"  {symbol:<10} "
            f"{p * 100:6.2f}%"
        )

    thresholds = dict(
        baseline_thresholds
    )

    current = evaluate_thresholds(
        df,
        thresholds,
        direction,
    )

    current_score = equalizer_score(
        current["event_count"],
        baseline_count,
        current["participation"],
    )

    best_thresholds = dict(
        thresholds
    )

    best_result = current
    best_score = current_score

    print()
    print(
        f"Start Score: {current_score:.5f}"
    )

    for iteration in range(
        1,
        MAX_ITERATIONS + 1,
    ):

        improved = False

        # ----------------------------------------------------
        # Wir testen für jeden Coin:
        #
        #   Schwelle - Schritt
        #   Schwelle + Schritt
        #
        # und behalten die Variante mit dem besten Score.
        # ----------------------------------------------------

        for symbol in REF_SYMBOLS:

            candidates = []

            current_threshold = thresholds[
                symbol
            ]

            for delta in (
                -THRESHOLD_STEP,
                THRESHOLD_STEP,
            ):

                new_threshold = (
                    current_threshold + delta
                )

                if (
                    new_threshold
                    < MIN_THRESHOLD
                    or
                    new_threshold
                    > MAX_THRESHOLD
                ):
                    continue

                test_thresholds = dict(
                    thresholds
                )

                test_thresholds[
                    symbol
                ] = new_threshold

                result = evaluate_thresholds(
                    df,
                    test_thresholds,
                    direction,
                )

                score = equalizer_score(
                    result["event_count"],
                    baseline_count,
                    result["participation"],
                )

                candidates.append(
                    (
                        score,
                        new_threshold,
                        result,
                    )
                )

            if not candidates:
                continue

            candidates.sort(
                key=lambda x: x[0]
            )

            candidate_score = candidates[0][0]

            if candidate_score < best_score:

                (
                    best_score,
                    new_threshold,
                    result,
                ) = candidates[0]

                thresholds[
                    symbol
                ] = new_threshold

                best_thresholds = dict(
                    thresholds
                )

                best_result = result

                improved = True

        if not improved:
            break

        if iteration % 5 == 0:
            print(
                f"Iteration {iteration:3d}: "
                f"Events "
                f"{best_result['event_count']:,} | "
                f"Score "
                f"{best_score:.5f}"
            )

    return (
        baseline,
        best_result,
        best_thresholds,
    )


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    df,
    event_mask,
    hold_hours,
):
    """
    One-slot / non-overlap:

    Wenn ein Trade läuft, werden neue Events
    während dieser Zeit ignoriert.
    """

    hold_bars = hold_hours

    target = TARGET_SYMBOL

    prices = df[target].values
    timestamps = df["open_time"].values

    event_indices = np.flatnonzero(
        event_mask.values
    )

    trades = []

    next_free = -1

    for idx in event_indices:

        if idx < next_free:
            continue

        exit_idx = idx + hold_bars

        if exit_idx >= len(prices):
            continue

        entry_price = prices[idx]
        exit_price = prices[exit_idx]

        if entry_price <= 0:
            continue

        raw_return = (
            exit_price / entry_price
            - 1.0
        )

        net_return = (
            raw_return
            - 2 * FEE_PER_SIDE
        )

        trades.append(
            {
                "idx": idx,
                "entry": entry_price,
                "exit": exit_price,
                "raw_return": raw_return,
                "net_return": net_return,
                "entry_time": timestamps[idx],
                "exit_time": timestamps[exit_idx],
            }
        )

        next_free = exit_idx + 1

    if not trades:
        return {
            "n": 0,
            "win_rate": 0.0,
            "avg": 0.0,
            "compound": 0.0,
        }

    returns = np.array(
        [t["net_return"] for t in trades]
    )

    win_rate = (
        (returns > 0).mean()
    )

    avg_return = returns.mean()

    compound = (
        np.prod(1.0 + returns) - 1.0
    )

    return {
        "n": len(trades),
        "win_rate": win_rate,
        "avg": avg_return,
        "compound": compound,
    }


# ============================================================
# BUY & HOLD
# ============================================================

def buy_and_hold(df):
    prices = df[TARGET_SYMBOL].values

    if len(prices) < 2:
        return 0.0

    return (
        prices[-1] / prices[0]
        - 1.0
    )


# ============================================================
# PRINT THRESHOLDS
# ============================================================

def print_thresholds(
    thresholds,
    baseline_thresholds,
):
    print()
    print(
        "INDIVIDUELLE THRESHOLDS"
    )
    print("-" * 60)

    for symbol in REF_SYMBOLS:

        base = (
            baseline_thresholds[symbol]
            * 100
        )

        new = (
            thresholds[symbol]
            * 100
        )

        delta = new - base

        print(
            f"{symbol:<10} "
            f"{base:6.2f}% -> "
            f"{new:6.2f}% "
            f"({delta:+6.2f} pp)"
        )


# ============================================================
# PRINT PARTICIPATION
# ============================================================

def print_participation(
    baseline,
    equalized,
):
    print()
    print(
        "VOTING-BETEILIGUNG INNERHALB DER 3/5 EVENTS"
    )

    print("-" * 78)

    print(
        f"{'Coin':<10}"
        f"{'Normal':>12}"
        f"{'Equalized':>14}"
        f"{'Delta':>12}"
    )

    print("-" * 78)

    for symbol in REF_SYMBOLS:

        a = (
            baseline["participation"][symbol]
            * 100
        )

        b = (
            equalized["participation"][symbol]
            * 100
        )

        print(
            f"{symbol:<10}"
            f"{a:>11.2f}%"
            f"{b:>13.2f}%"
            f"{b - a:>+11.2f} pp"
        )

    print("-" * 78)

    baseline_values = [
        baseline["participation"][s]
        for s in REF_SYMBOLS
    ]

    equalized_values = [
        equalized["participation"][s]
        for s in REF_SYMBOLS
    ]

    print(
        f"{'Spread':<10}"
        f"{(max(baseline_values) - min(baseline_values))*100:>11.2f} pp"
        f"{(max(equalized_values) - min(equalized_values))*100:>13.2f} pp"
    )


# ============================================================
# RUN ONE DIRECTION
# ============================================================

def run_direction(
    df,
    event_window,
    direction,
):
    work = add_returns(
        df,
        event_window,
    )

    baseline_thresholds = {
        symbol: BASE_THRESHOLD
        for symbol in REF_SYMBOLS
    }

    (
        baseline,
        equalized,
        equalized_thresholds,
    ) = optimize_thresholds(
        work,
        baseline_thresholds,
        direction,
    )

    print_thresholds(
        equalized_thresholds,
        baseline_thresholds,
    )

    print_participation(
        baseline,
        equalized,
    )

    print()
    print(
        f"EVENT COUNT"
    )
    print("-" * 60)

    print(
        f"Normal 3/5 @ 2.00% : "
        f"{baseline['event_count']:,}"
    )

    print(
        f"Equalized 3/5      : "
        f"{equalized['event_count']:,}"
    )

    if baseline["event_count"]:

        deviation = (
            equalized["event_count"]
            / baseline["event_count"]
            - 1.0
        )

        print(
            f"Difference         : "
            f"{deviation:+.2%}"
        )

    # --------------------------------------------------------
    # BACKTEST TABLE
    # --------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    print(
        f"{direction} | Event {event_window}m | "
        f"Target {TARGET_SYMBOL}"
    )

    print(
        "=" * 78
    )

    print(
        f"{'Hold':>7}"
        f"{'Normal N':>12}"
        f"{'Normal W':>11}"
        f"{'Normal Avg':>13}"
        f"{'Normal Cmp':>14}"
        f"{'Eq N':>10}"
        f"{'Eq W':>10}"
        f"{'Eq Avg':>12}"
        f"{'Eq Cmp':>13}"
    )

    print("-" * 112)

    for hold in HOLDS_HOURS:

        normal_bt = backtest(
            work,
            baseline["event_mask"],
            hold,
        )

        equalized_bt = backtest(
            work,
            equalized["event_mask"],
            hold,
        )

        print(
            f"{hold:>5}h"
            f"{normal_bt['n']:>12}"
            f"{normal_bt['win_rate']*100:>10.1f}%"
            f"{normal_bt['avg']*100:>12.2f}%"
            f"{normal_bt['compound']*100:>13.2f}%"
            f"{equalized_bt['n']:>10}"
            f"{equalized_bt['win_rate']*100:>9.1f}%"
            f"{equalized_bt['avg']*100:>11.2f}%"
            f"{equalized_bt['compound']*100:>12.2f}%"
        )

    return {
        "baseline": baseline,
        "equalized": equalized,
        "thresholds": equalized_thresholds,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_cli_args()

    start_ms = date_to_ms(
        args.start
    )

    end_ms = date_to_ms(
        args.end
    )

    print()
    print("=" * 78)
    print("REFERENCE COIN EQUALIZER")
    print("=" * 78)

    print(
        f"Refs       : "
        f"{', '.join(REF_SYMBOLS)}"
    )

    print(
        f"Target     : "
        f"{TARGET_SYMBOL}"
    )

    print(
        f"Interval   : "
        f"{INTERVAL}"
    )

    print(
        f"Period     : "
        f"{args.start} -> {args.end}"
    )

    print(
        f"Quorum     : "
        f"{QUORUM}/{len(REF_SYMBOLS)}"
    )

    print(
        f"Baseline   : "
        f"{BASE_THRESHOLD * 100:.2f}%"
    )

    print(
        f"Fee        : "
        f"{FEE_PER_SIDE * 100:.3f}% / side"
    )

    print()

    data = load_all_data(
        start_ms,
        end_ms,
    )

    df = build_common_dataframe(
        data
    )

    print()
    print(
        f"Common timestamps: "
        f"{len(df):,}"
    )

    if len(df) == 0:
        raise RuntimeError(
            "Keine gemeinsamen Timestamps."
        )

    # --------------------------------------------------------
    # BUY & HOLD
    # --------------------------------------------------------

    bh = buy_and_hold(df)

    print()
    print(
        f"Buy & Hold {TARGET_SYMBOL}: "
        f"{bh * 100:+.2f}%"
    )

    # --------------------------------------------------------
    # EVENT WINDOWS
    # --------------------------------------------------------

    for event_window in EVENT_WINDOWS_MIN:

        print()
        print()
        print("#" * 90)
        print(
            f"# EVENT WINDOW {event_window}m"
        )
        print("#" * 90)

        # UP
        up_result = run_direction(
            df,
            event_window,
            "UP",
        )

        # DOWN
        down_result = run_direction(
            df,
            event_window,
            "DOWN",
        )

    print()
    print("=" * 78)
    print("FERTIG")
    print("=" * 78)


if __name__ == "__main__":
    main()