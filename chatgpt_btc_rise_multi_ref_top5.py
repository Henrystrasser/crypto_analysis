#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chatgpt_multiref_strength_momentum.py

MULTI-REF STRENGTH + TOP-5 MOMENTUM

Strategie:

    1. BTC + ETH + SOL beobachten.
    2. Ein Signal entsteht, wenn mindestens 2/3 oder 3/3
       der Referenzen einen Threshold innerhalb des Event-Fensters
       brechen.
    3. Genau zu diesem Zeitpunkt werden die 20 liquidesten
       Binance-USDT-Altcoins betrachtet.
    4. Die 5 stärksten Altcoins innerhalb des gleichen Event-Fensters
       werden gekauft.
    5. Kapital:
           5 x 20 %
    6. Globaler Non-Overlap pro Hold-Horizont.
    7. Compound Event für Event.

Beispiel:

    Event 30m, Threshold +1.0%

    BTC = +1.3%
    ETH = +1.1%
    SOL = +0.4%

    => 2/3 brechen +1.0%
    => Signal

    Danach Target-Momentum über dieselben 30m:

    XRP = +2.1%
    SUI = +1.9%
    DOGE = +1.8%
    AVAX = +1.6%
    LINK = +1.5%

    => Top 5 kaufen
    => 20% je Coin

WICHTIG:
    Die Top-5-Auswahl benutzt ausschließlich Daten
    bis zum Signalzeitpunkt. Kein Blick in die Zukunft.

Daten:
    Binance Spot
    15m Klines

Fees:
    7.5 bps Kauf
    7.5 bps Verkauf
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo


# ============================================================
# CONFIG
# ============================================================

BINANCE_BASE = "https://data-api.binance.vision"

INTERVAL = "15m"
INTERVAL_MINUTES = 15
INTERVAL_MS = INTERVAL_MINUTES * 60 * 1000

BERLIN = ZoneInfo("Europe/Berlin")


# ============================================================
# REFERENCES
# ============================================================

REF_SYMBOLS = [
    "BTC",
    "ETH",
    "SOL",
]


# ============================================================
# TARGET UNIVERSE
# ============================================================

# Genau 20 liquide Altcoins.
#
# Auswahl:
# Binance Spot USDT
# sortiert nach aktuellem 24h Quote Volume
TARGET_TOP_N = 20

# Pro Signal die 5 stärksten Targets.
TOP_K = 5

# 5 x 20%
ALLOCATION_PER_COIN = 1.0 / TOP_K


# ============================================================
# ZEITRAUM
# ============================================================

FROM_DATE = "2025-01-01"

# None = letzte abgeschlossene 15m-Kerze
TO_DATE = None


# ============================================================
# EVENT WINDOWS
# ============================================================

EVENT_WINDOWS_MIN = [
    30,
    60,
    120,
    240,
    360,
]


# ============================================================
# THRESHOLDS
# ============================================================

EVENT_THRESHOLDS_PCT = [
    1.0,
    2.0,
]


# ============================================================
# QUORUM
# ============================================================

# Wie viele von 3 Refs müssen den Threshold brechen?
#
# 2 = mindestens 2/3
# 3 = alle 3
#
# Beides wird im selben Lauf separat getestet.
QUORUMS = [
    2,
    3,
]


# ============================================================
# TARGET MOMENTUM
# ============================================================

# Für ein UP-Signal sollen die Targets ebenfalls positiv sein.
#
# True:
#   Nur Coins mit positiver Rendite werden als Top-5 betrachtet.
#
# False:
#   Immer die fünf höchsten Renditen nehmen,
#   auch wenn alle negativ sind.
REQUIRE_POSITIVE_TARGET_MOVE = True


# ============================================================
# HOLD HORIZONS
# ============================================================

HOLD_HORIZONS = [
    ("3h", 3),
    ("6h", 6),
    ("12h", 12),
    ("24h", 24),
    ("48h", 48),
    ("60h", 60),
    ("72h", 72),
    ("78h", 78),
    ("84h", 84),
    ("90h", 90),
    ("96h", 96),
    ("120h", 120),
    ("192h", 192),
]


# ============================================================
# FEES
# ============================================================

FEE_BPS_PER_SIDE = 7.5

# Optionaler zusätzlicher Slippage-Puffer.
SLIPPAGE_BPS_PER_SIDE = 0.0


# ============================================================
# TARGET FILTER
# ============================================================

STABLECOINS = {
    "USDT",
    "USDC",
    "FDUSD",
    "TUSD",
    "USDP",
    "DAI",
    "BUSD",
    "USD1",
    "USDE",
    "USDS",
    "PYUSD",
    "EUR",
    "EURI",
    "AEUR",
    "UST",
    "RLUSD",
}

EXCLUDED_TARGET_BASES = {
    "BTC",
    "ETH",
    "SOL",
    "XAUT",
    "ZEC",
    "XRP"
}


# ============================================================
# DOWNLOAD SETTINGS
# ============================================================

KLINE_LIMIT = 1000

MAX_WORKERS = 4

REQUEST_TIMEOUT = 30

MAX_RETRIES = 6

RETRY_SLEEP_SECONDS = 2.0


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PriceSeries:
    symbol: str
    times: np.ndarray
    closes: np.ndarray

    def exact_price(
        self,
        timestamp_ms: int,
    ) -> Optional[float]:

        idx = np.searchsorted(
            self.times,
            timestamp_ms,
        )

        if idx >= len(self.times):
            return None

        if int(self.times[idx]) != int(timestamp_ms):
            return None

        price = float(self.closes[idx])

        if (
            not math.isfinite(price)
            or price <= 0
        ):
            return None

        return price

    def first_price_at_or_after(
        self,
        timestamp_ms: int,
    ) -> Optional[float]:

        idx = np.searchsorted(
            self.times,
            timestamp_ms,
        )

        if idx >= len(self.times):
            return None

        price = float(
            self.closes[idx]
        )

        if (
            not math.isfinite(price)
            or price <= 0
        ):
            return None

        return price

    def last_price_at_or_before(
        self,
        timestamp_ms: int,
    ) -> Optional[float]:

        idx = np.searchsorted(
            self.times,
            timestamp_ms,
            side="right",
        ) - 1

        if idx < 0:
            return None

        price = float(
            self.closes[idx]
        )

        if (
            not math.isfinite(price)
            or price <= 0
        ):
            return None

        return price


@dataclass
class Event:
    start_ms: int
    end_ms: int

    ref_returns: Dict[str, float]

    ref_count_above_threshold: int


@dataclass
class TargetMomentum:
    symbol: str
    return_pct: float


@dataclass
class AcceptedTrade:
    event: Event
    selected: List[TargetMomentum]
    portfolio_roi: float


# ============================================================
# HTTP
# ============================================================

def create_session() -> requests.Session:

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                "chatgpt-multiref-strength-momentum/1.0",
            "Accept":
                "application/json",
        }
    )

    return session


def get_json(
    session: requests.Session,
    path: str,
    params: Optional[dict] = None,
) -> dict | list:

    url = BINANCE_BASE + path

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code in (
                418,
                429,
            ):

                wait = (
                    RETRY_SLEEP_SECONDS
                    * attempt
                )

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:

                    try:
                        wait = max(
                            wait,
                            float(retry_after),
                        )
                    except ValueError:
                        pass

                print(
                    f"  Binance Rate Limit "
                    f"({response.status_code}) "
                    f"-> warte {wait:.1f}s"
                )

                time.sleep(wait)

                continue

            response.raise_for_status()

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_SLEEP_SECONDS
                    * attempt
                )

            else:
                break

    raise RuntimeError(
        "Binance Request fehlgeschlagen: "
        f"{path} "
        f"params={params} "
        f"error={last_error}"
    )


# ============================================================
# DATE / TIME
# ============================================================

def berlin_date_to_utc_ms(
    date_string: str,
    end_of_day: bool = False,
) -> int:

    if len(date_string) == 10:

        dt = datetime.strptime(
            date_string,
            "%Y-%m-%d",
        ).replace(
            tzinfo=BERLIN
        )

        if end_of_day:

            dt += (
                timedelta(days=1)
                - timedelta(milliseconds=1)
            )

    else:

        dt = datetime.fromisoformat(
            date_string
        )

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=BERLIN
            )

    return int(
        dt.timestamp()
        * 1000
    )


def floor_to_interval_ms(
    timestamp_ms: int,
) -> int:

    return (
        timestamp_ms
        - timestamp_ms % INTERVAL_MS
    )


def latest_closed_candle_ms() -> int:

    now_ms = int(
        datetime.now(
            timezone.utc
        ).timestamp()
        * 1000
    )

    current_open = floor_to_interval_ms(
        now_ms
    )

    return (
        current_open
        - INTERVAL_MS
    )


def resolve_period() -> Tuple[int, int]:

    start_ms = berlin_date_to_utc_ms(
        FROM_DATE
    )

    if TO_DATE is None:

        end_ms = latest_closed_candle_ms()

    else:

        end_ms = berlin_date_to_utc_ms(
            TO_DATE,
            end_of_day=True,
        )

        end_ms = min(
            end_ms,
            latest_closed_candle_ms(),
        )

    if end_ms <= start_ms:

        raise ValueError(
            "Zeitraum ungültig."
        )

    return start_ms, end_ms


def format_berlin(
    timestamp_ms: int,
) -> str:

    dt = datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    ).astimezone(
        BERLIN
    )

    return dt.strftime(
        "%Y-%m-%d %H:%M"
    )


# ============================================================
# BINANCE UNIVERSE
# ============================================================

def build_target_universe(
    session: requests.Session,
    refs: Sequence[str],
    top_n: int,
) -> List[str]:

    print()
    print(
        "Prüfe Binance-Pairs..."
    )

    exchange_info = get_json(
        session,
        "/api/v3/exchangeInfo",
    )

    print(
        "Lade Binance 24h-Volumes..."
    )

    tickers = get_json(
        session,
        "/api/v3/ticker/24hr",
    )

    ref_set = {
        ref.upper()
        for ref in refs
    }

    volumes: Dict[
        str,
        float,
    ] = {}

    for ticker in tickers:

        symbol = ticker.get(
            "symbol"
        )

        if not symbol:
            continue

        try:

            quote_volume = float(
                ticker.get(
                    "quoteVolume",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            quote_volume = 0.0

        if quote_volume > 0:

            volumes[
                symbol
            ] = quote_volume

    candidates = []

    for item in exchange_info.get(
        "symbols",
        [],
    ):

        if item.get(
            "status"
        ) != "TRADING":

            continue

        if item.get(
            "quoteAsset"
        ) != "USDT":

            continue

        if (
            item.get(
                "isSpotTradingAllowed"
            )
            is False
        ):
            continue

        symbol = item.get(
            "symbol"
        )

        base = item.get(
            "baseAsset"
        )

        if not symbol or not base:
            continue

        base = base.upper()

        if base in STABLECOINS:
            continue

        if base in EXCLUDED_TARGET_BASES:
            continue

        if base in ref_set:
            continue

        volume = volumes.get(
            symbol,
            0.0,
        )

        if volume <= 0:
            continue

        candidates.append(
            (
                volume,
                symbol,
            )
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        symbol
        for _, symbol
        in candidates[:top_n]
    ]


# ============================================================
# KLINES
# ============================================================

def fetch_klines(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> PriceSeries:

    rows = []

    current_start = start_ms

    while current_start <= end_ms:

        data = get_json(
            session,
            "/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": KLINE_LIMIT,
            },
        )

        if not data:
            break

        for row in data:

            open_time = int(
                row[0]
            )

            if open_time < start_ms:
                continue

            if open_time > end_ms:
                continue

            close_price = float(
                row[4]
            )

            if close_price <= 0:
                continue

            rows.append(
                (
                    open_time,
                    close_price,
                )
            )

        last_open_time = int(
            data[-1][0]
        )

        next_start = (
            last_open_time
            + INTERVAL_MS
        )

        if next_start <= current_start:
            break

        current_start = next_start

        time.sleep(0.02)

        if len(data) < KLINE_LIMIT:
            break

    if not rows:

        raise RuntimeError(
            f"Keine Klines für {symbol}"
        )

    unique = {}

    for timestamp, close in rows:

        unique[timestamp] = close

    sorted_rows = sorted(
        unique.items(),
        key=lambda x: x[0],
    )

    times = np.array(
        [
            x[0]
            for x in sorted_rows
        ],
        dtype=np.int64,
    )

    closes = np.array(
        [
            x[1]
            for x in sorted_rows
        ],
        dtype=np.float64,
    )

    return PriceSeries(
        symbol=symbol,
        times=times,
        closes=closes,
    )


# ============================================================
# TARGET LOADING
# ============================================================

def load_one_target(
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> Tuple[
    str,
    Optional[PriceSeries],
    Optional[str],
]:

    session = create_session()

    try:

        series = fetch_klines(
            session,
            symbol,
            start_ms,
            end_ms,
        )

        return (
            symbol,
            series,
            None,
        )

    except Exception as exc:

        return (
            symbol,
            None,
            str(exc),
        )

    finally:

        session.close()


def load_target_data(
    targets: Sequence[str],
    start_ms: int,
    end_ms: int,
) -> Dict[str, PriceSeries]:

    print()
    print(
        f"Lade {len(targets)} Target-Coins "
        f"mit {MAX_WORKERS} parallelen Workern..."
    )

    loaded = {}

    completed = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                load_one_target,
                symbol,
                start_ms,
                end_ms,
            ): symbol
            for symbol in targets
        }

        for future in as_completed(
            futures
        ):

            symbol, series, error = (
                future.result()
            )

            completed += 1

            if series is None:

                print(
                    f"  [{completed:2d}/"
                    f"{len(targets):2d}] "
                    f"{symbol:<14} "
                    f"FEHLER: {error}"
                )

                continue

            loaded[symbol] = series

            print(
                f"  [{completed:2d}/"
                f"{len(targets):2d}] "
                f"{symbol:<14} "
                f"{len(series.times):6d} Kerzen"
            )

    return loaded


# ============================================================
# REFERENCE DATA
# ============================================================

def load_reference_data(
    session: requests.Session,
    refs: Sequence[str],
    start_ms: int,
    end_ms: int,
) -> Dict[str, PriceSeries]:

    loaded = {}

    for base in refs:

        symbol = (
            base.upper()
            + "USDT"
        )

        print()
        print(
            f"Lade Ref {base}..."
        )

        series = fetch_klines(
            session,
            symbol,
            start_ms,
            end_ms,
        )

        loaded[
            base.upper()
        ] = series

        print(
            f"  {len(series.times)} Kerzen"
        )

    return loaded


# ============================================================
# REFERENCE FRAME
# ============================================================

def build_reference_frame(
    refs_data: Dict[str, PriceSeries],
) -> pd.DataFrame:

    common_times = None

    for series in refs_data.values():

        idx = pd.Index(
            series.times,
            dtype="int64",
        )

        if common_times is None:

            common_times = idx

        else:

            common_times = (
                common_times.intersection(
                    idx
                )
            )

    if (
        common_times is None
        or len(common_times) == 0
    ):

        raise RuntimeError(
            "Keine gemeinsamen "
            "Zeitstempel der Refs."
        )

    common_times = (
        common_times.sort_values()
    )

    data = {
        "time":
            common_times.to_numpy(
                dtype=np.int64
            )
    }

    for ref, series in refs_data.items():

        lookup = pd.Series(
            series.closes,
            index=series.times,
        )

        data[ref] = (
            lookup
            .reindex(
                common_times
            )
            .to_numpy()
        )

    frame = pd.DataFrame(
        data
    )

    return frame.dropna()


# ============================================================
# EVENTS
# ============================================================

def create_events(
    refs_data: Dict[str, PriceSeries],
    window_minutes: int,
    threshold_pct: float,
    quorum: int,
) -> List[Event]:

    if window_minutes % INTERVAL_MINUTES != 0:

        raise ValueError(
            f"{window_minutes}m ist nicht "
            f"durch {INTERVAL_MINUTES}m teilbar."
        )

    if quorum < 1 or quorum > len(refs_data):

        raise ValueError(
            f"Quorum {quorum} ist ungültig."
        )

    frame = build_reference_frame(
        refs_data
    )

    bars = (
        window_minutes
        // INTERVAL_MINUTES
    )

    threshold = (
        threshold_pct
        / 100.0
    )

    returns = {}

    for ref in refs_data.keys():

        prices = frame[
            ref
        ].to_numpy(
            dtype=np.float64
        )

        ref_return = (
            prices
            / np.roll(
                prices,
                bars,
            )
        ) - 1.0

        ref_return[
            :bars
        ] = np.nan

        returns[ref] = ref_return

    returns_df = pd.DataFrame(
        returns
    )

    # Wie viele Refs haben den Threshold gebrochen?
    threshold_count = (
        returns_df >= threshold
    ).sum(
        axis=1
    )

    condition = (
        threshold_count
        >= quorum
    )

    # Nur beim ersten Eintritt in das Quorum
    # ein neues Event erzeugen.
    previous_condition = (
        condition
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    crossing = (
        condition
        & ~previous_condition
    )

    times = frame[
        "time"
    ].to_numpy(
        dtype=np.int64
    )

    events = []

    for idx in np.flatnonzero(
        crossing.to_numpy()
    ):

        if idx < bars:
            continue

        start_ms = int(
            times[
                idx - bars
            ]
        )

        end_ms = int(
            times[idx]
        )

        ref_returns = {}

        valid = True

        for ref in refs_data.keys():

            value = float(
                returns_df.iloc[
                    idx
                ][ref]
            )

            if not math.isfinite(
                value
            ):

                valid = False
                break

            ref_returns[
                ref
            ] = value

        if not valid:
            continue

        events.append(
            Event(
                start_ms=start_ms,
                end_ms=end_ms,
                ref_returns=ref_returns,
                ref_count_above_threshold=int(
                    threshold_count.iloc[idx]
                ),
            )
        )

    return events


# ============================================================
# TARGET MOMENTUM
# ============================================================

def find_top_momentum(
    event: Event,
    target_data: Dict[str, PriceSeries],
) -> List[TargetMomentum]:

    candidates = []

    for symbol, series in target_data.items():

        start_price = series.exact_price(
            event.start_ms
        )

        end_price = series.exact_price(
            event.end_ms
        )

        if (
            start_price is None
            or end_price is None
        ):
            continue

        target_return = (
            end_price
            / start_price
        ) - 1.0

        if (
            REQUIRE_POSITIVE_TARGET_MOVE
            and target_return <= 0
        ):
            continue

        candidates.append(
            TargetMomentum(
                symbol=symbol,
                return_pct=(
                    target_return
                ),
            )
        )

    candidates.sort(
        key=lambda x: x.return_pct,
        reverse=True,
    )

    return candidates[:TOP_K]


# ============================================================
# ROI
# ============================================================

def calculate_net_roi(
    entry_price: float,
    exit_price: float,
) -> float:

    if (
        entry_price <= 0
        or exit_price <= 0
    ):
        return float("nan")

    buy_fee = (
        FEE_BPS_PER_SIDE
        / 10000.0
    )

    sell_fee = (
        FEE_BPS_PER_SIDE
        / 10000.0
    )

    buy_slippage = (
        SLIPPAGE_BPS_PER_SIDE
        / 10000.0
    )

    sell_slippage = (
        SLIPPAGE_BPS_PER_SIDE
        / 10000.0
    )

    effective_entry = (
        entry_price
        * (1.0 + buy_slippage)
    )

    effective_exit = (
        exit_price
        * (1.0 - sell_slippage)
    )

    quantity = (
        (1.0 - buy_fee)
        / effective_entry
    )

    final_value = (
        quantity
        * effective_exit
        * (1.0 - sell_fee)
    )

    return final_value - 1.0


# ============================================================
# SIMULATION
# ============================================================

def simulate_hold(
    events: Sequence[Event],
    target_data: Dict[str, PriceSeries],
    hold_hours: int,
) -> Tuple[
    List[AcceptedTrade],
    Dict[str, float],
]:

    hold_ms = (
        hold_hours
        * 60
        * 60
        * 1000
    )

    accepted = []

    last_entry_ms: Optional[int] = None

    for event in events:

        # ----------------------------------------------------
        # GLOBAL NON-OVERLAP
        # ----------------------------------------------------

        if last_entry_ms is not None:

            if (
                event.end_ms
                < (
                    last_entry_ms
                    + hold_ms
                )
            ):
                continue

        # ----------------------------------------------------
        # TOP 5 MOMENTUM
        # ----------------------------------------------------

        selected = find_top_momentum(
            event,
            target_data,
        )

        if len(selected) < TOP_K:
            continue

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        coin_rois = []

        valid = True

        for target in selected:

            series = target_data.get(
                target.symbol
            )

            if series is None:

                valid = False
                break

            entry_price = series.exact_price(
                event.end_ms
            )

            exit_price = series.exact_price(
                event.end_ms
                + hold_ms
            )

            if (
                entry_price is None
                or exit_price is None
            ):

                valid = False
                break

            roi = calculate_net_roi(
                entry_price,
                exit_price,
            )

            if not math.isfinite(roi):

                valid = False
                break

            coin_rois.append(
                roi
            )

        if not valid:
            continue

        # 5 x 20 %
        portfolio_roi = float(
            np.mean(
                coin_rois
            )
        )

        accepted.append(
            AcceptedTrade(
                event=event,
                selected=selected,
                portfolio_roi=portfolio_roi,
            )
        )

        last_entry_ms = (
            event.end_ms
        )

    stats = calculate_stats(
        accepted
    )

    return (
        accepted,
        stats,
    )


# ============================================================
# STATS
# ============================================================

def calculate_stats(
    trades: Sequence[AcceptedTrade],
) -> Dict[str, float]:

    if not trades:

        return {
            "n": 0,
            "positive_pct": float("nan"),
            "avg_roi": float("nan"),
            "median_roi": float("nan"),
            "compound": float("nan"),
            "best": float("nan"),
            "worst": float("nan"),
        }

    rois = np.array(
        [
            trade.portfolio_roi
            for trade in trades
        ],
        dtype=np.float64,
    )

    positive_pct = (
        np.mean(
            rois > 0
        )
        * 100.0
    )

    avg_roi = (
        np.mean(rois)
        * 100.0
    )

    median_roi = (
        np.median(rois)
        * 100.0
    )

    compound = (
        np.prod(
            1.0 + rois
        )
        - 1.0
    ) * 100.0

    best = (
        np.max(rois)
        * 100.0
    )

    worst = (
        np.min(rois)
        * 100.0
    )

    return {
        "n": int(
            len(rois)
        ),
        "positive_pct": float(
            positive_pct
        ),
        "avg_roi": float(
            avg_roi
        ),
        "median_roi": float(
            median_roi
        ),
        "compound": float(
            compound
        ),
        "best": float(
            best
        ),
        "worst": float(
            worst
        ),
    }


# ============================================================
# BUY & HOLD
# ============================================================

def calculate_buy_hold(
    target_data: Dict[str, PriceSeries],
    start_ms: int,
    end_ms: int,
) -> Tuple[
    float,
    int,
]:

    rois = []

    buy_fee = (
        FEE_BPS_PER_SIDE
        / 10000.0
    )

    sell_fee = (
        FEE_BPS_PER_SIDE
        / 10000.0
    )

    for series in target_data.values():

        entry_price = (
            series.first_price_at_or_after(
                start_ms
            )
        )

        exit_price = (
            series.last_price_at_or_before(
                end_ms
            )
        )

        if (
            entry_price is None
            or exit_price is None
        ):
            continue

        quantity = (
            (1.0 - buy_fee)
            / entry_price
        )

        final_value = (
            quantity
            * exit_price
            * (1.0 - sell_fee)
        )

        roi = (
            final_value
            - 1.0
        )

        if math.isfinite(roi):

            rois.append(
                roi
            )

    if not rois:

        return (
            float("nan"),
            0,
        )

    benchmark = (
        np.mean(rois)
        * 100.0
    )

    return (
        float(benchmark),
        len(rois),
    )


# ============================================================
# OUTPUT
# ============================================================

def fmt_pct(
    value: float,
) -> str:

    if not math.isfinite(value):
        return "n/a"

    return f"{value:9.2f}%"


def print_header(
    start_ms: int,
    end_ms: int,
    targets: Sequence[str],
) -> None:

    print()
    print("=" * 118)

    print(
        "MULTI-REF STRENGTH + TOP-5 MOMENTUM"
    )

    print("=" * 118)

    print(
        f"Refs: {', '.join(REF_SYMBOLS)}"
    )

    print(
        f"Interval: {INTERVAL}"
    )

    print(
        f"Zeitraum: "
        f"{format_berlin(start_ms)} → "
        f"{format_berlin(end_ms)}"
    )

    print(
        f"Target Universe: "
        f"{len(targets)} liquide Binance-USDT-Alts"
    )

    print(
        f"Top Momentum: "
        f"{TOP_K} × "
        f"{ALLOCATION_PER_COIN * 100:.1f}%"
    )

    print(
        "Quorum: 2/3 und 3/3 separat"
    )

    print(
        "Non-overlap: GLOBAL pro Hold"
    )

    print(
        f"Fee: "
        f"{FEE_BPS_PER_SIDE:.1f} bps je Seite"
    )

    print()

    print(
        "Targets:"
    )

    print(
        "  "
        + ", ".join(targets)
    )

    print("=" * 118)


def print_results_table(
    results: Sequence[
        Tuple[
            str,
            Dict[str, float],
        ]
    ],
    buy_hold: float,
) -> None:

    print()

    print(
        f"{'Hold':>6} "
        f"{'Trades':>8} "
        f"{'Win%':>9} "
        f"{'Ø Event':>12} "
        f"{'Median':>12} "
        f"{'Compound':>13} "
        f"{'vs B&H':>11} "
        f"{'Best':>10} "
        f"{'Worst':>10}"
    )

    print("-" * 118)

    for hold_name, stats in results:

        compound = stats[
            "compound"
        ]

        if (
            math.isfinite(
                compound
            )
            and math.isfinite(
                buy_hold
            )
        ):

            vs_bh = (
                compound
                - buy_hold
            )

        else:

            vs_bh = float("nan")

        print(
            f"{hold_name:>6} "
            f"{stats['n']:>8d} "
            f"{fmt_pct(stats['positive_pct']):>9} "
            f"{fmt_pct(stats['avg_roi']):>12} "
            f"{fmt_pct(stats['median_roi']):>12} "
            f"{fmt_pct(compound):>13} "
            f"{fmt_pct(vs_bh):>11} "
            f"{fmt_pct(stats['best']):>10} "
            f"{fmt_pct(stats['worst']):>10}"
        )


# ============================================================
# SAMPLE EVENTS
# ============================================================

def print_sample_events(
    events: Sequence[Event],
    target_data: Dict[str, PriceSeries],
) -> None:

    shown = 0

    print()
    print(
        "Beispiele der ersten Momentum-Events:"
    )

    for event in events:

        selected = find_top_momentum(
            event,
            target_data,
        )

        if len(selected) < TOP_K:
            continue

        shown += 1

        ref_text = " | ".join(
            f"{ref} "
            f"{ret * 100:+.2f}%"
            for ref, ret
            in event.ref_returns.items()
        )

        target_text = " | ".join(
            f"{target.symbol} "
            f"{target.return_pct * 100:+.2f}%"
            for target
            in selected
        )

        print()

        print(
            f"  {format_berlin(event.end_ms)}"
        )

        print(
            f"  Refs: {ref_text}"
        )

        print(
            f"  Quorum: "
            f"{event.ref_count_above_threshold}/"
            f"{len(REF_SYMBOLS)}"
        )

        print(
            f"  Top 5: "
            f"{target_text}"
        )

        if shown >= 3:
            break


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Multi-Ref Strength + "
            "Top-5 Momentum Backtest"
        )
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=TARGET_TOP_N,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
    )

    parser.add_argument(
        "--from-date",
        default=FROM_DATE,
    )

    parser.add_argument(
        "--to-date",
        default=TO_DATE,
    )

    parser.add_argument(
        "--refs",
        default=",".join(
            REF_SYMBOLS
        ),
    )

    parser.add_argument(
        "--debug-events",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    global TARGET_TOP_N
    global TOP_K
    global ALLOCATION_PER_COIN
    global MAX_WORKERS
    global FROM_DATE
    global TO_DATE
    global REF_SYMBOLS

    args = parse_args()

    TARGET_TOP_N = max(
        5,
        int(args.top_n),
    )

    TOP_K = max(
        1,
        int(args.top_k),
    )

    if TOP_K > TARGET_TOP_N:

        raise ValueError(
            "TOP_K darf nicht größer "
            "als TARGET_TOP_N sein."
        )

    ALLOCATION_PER_COIN = (
        1.0 / TOP_K
    )

    MAX_WORKERS = max(
        1,
        int(args.workers),
    )

    FROM_DATE = args.from_date

    TO_DATE = args.to_date

    REF_SYMBOLS = [
        x.strip().upper()
        for x in args.refs.split(",")
        if x.strip()
    ]

    if len(REF_SYMBOLS) != 3:

        raise ValueError(
            "Diese Version erwartet "
            "genau 3 Referenz-Coins."
        )

    start_ms, end_ms = (
        resolve_period()
    )

    max_hold_hours = max(
        hours
        for _, hours
        in HOLD_HORIZONS
    )

    # Targets müssen auch bis nach dem
    # längsten Exit verfügbar sein.
    target_data_end_ms = (
        end_ms
        + max_hold_hours
        * 60
        * 60
        * 1000
    )

    # --------------------------------------------------------
    # BINANCE
    # --------------------------------------------------------

    session = create_session()

    try:

        targets = build_target_universe(
            session,
            REF_SYMBOLS,
            TARGET_TOP_N,
        )

        if not targets:

            raise RuntimeError(
                "Target Universe ist leer."
            )

        print()
        print(
            f"Target Universe: "
            f"{len(targets)} Coins"
        )

        for idx, symbol in enumerate(
            targets,
            start=1,
        ):

            print(
                f"  {idx:2d}. {symbol}"
            )

        print()
        print(
            "Lade Reference-Coins..."
        )

        refs_data = load_reference_data(
            session,
            REF_SYMBOLS,
            start_ms,
            end_ms,
        )

    finally:

        session.close()

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    target_data = load_target_data(
        targets,
        start_ms,
        target_data_end_ms,
    )

    print()
    print(
        f"Geladen: "
        f"{len(target_data)} / "
        f"{len(targets)} Targets"
    )

    if len(target_data) < TOP_K:

        raise RuntimeError(
            f"Nur {len(target_data)} Targets geladen, "
            f"TOP_K={TOP_K}."
        )

    targets = [
        symbol
        for symbol in targets
        if symbol in target_data
    ]

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print_header(
        start_ms,
        end_ms,
        targets,
    )

    # --------------------------------------------------------
    # BUY & HOLD
    # --------------------------------------------------------

    buy_hold, bh_n = (
        calculate_buy_hold(
            target_data,
            start_ms,
            end_ms,
        )
    )

    print()
    print(
        "Buy&Hold Referenz:"
    )

    if math.isfinite(
        buy_hold
    ):

        print(
            f"  Gleichgewichteter "
            f"Buy&Hold der "
            f"{bh_n} Targets: "
            f"{buy_hold:+.2f}%"
        )

    else:

        print(
            "  n/a"
        )

    # --------------------------------------------------------
    # SCENARIOS
    # --------------------------------------------------------

    total_scenarios = (
        len(EVENT_WINDOWS_MIN)
        * len(EVENT_THRESHOLDS_PCT)
        * len(QUORUMS)
    )

    scenario_counter = 0

    for window_minutes in (
        EVENT_WINDOWS_MIN
    ):

        for threshold_pct in (
            EVENT_THRESHOLDS_PCT
        ):

            for quorum in QUORUMS:

                scenario_counter += 1

                print()
                print("=" * 118)

                print(
                    f"UP | Event "
                    f"{window_minutes}m | "
                    f"+{threshold_pct:.1f}% | "
                    f"{quorum}/3 REFS"
                )

                print("=" * 118)

                print(
                    f"Berechne "
                    f"UP | {window_minutes}m | "
                    f"{threshold_pct:.1f}% | "
                    f"{quorum}/3..."
                )

                events = create_events(
                    refs_data,
                    window_minutes,
                    threshold_pct,
                    quorum,
                )

                print(
                    f"  Raw Events: "
                    f"{len(events)}"
                )

                # Wie viele Events haben
                # überhaupt 5 positive Targets?
                usable_events = 0

                for event in events:

                    selected = (
                        find_top_momentum(
                            event,
                            target_data,
                        )
                    )

                    if len(selected) >= TOP_K:
                        usable_events += 1

                print(
                    f"  Events mit "
                    f"Top-{TOP_K} Momentum: "
                    f"{usable_events}"
                )

                if args.debug_events:

                    print_sample_events(
                        events,
                        target_data,
                    )

                scenario_results = []

                for (
                    hold_name,
                    hold_hours,
                ) in HOLD_HORIZONS:

                    _, stats = (
                        simulate_hold(
                            events,
                            target_data,
                            hold_hours,
                        )
                    )

                    scenario_results.append(
                        (
                            hold_name,
                            stats,
                        )
                    )

                print_results_table(
                    scenario_results,
                    buy_hold,
                )

                print()

                print(
                    f"Fortschritt: "
                    f"{scenario_counter}/"
                    f"{total_scenarios}"
                )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 118)
    print("FERTIG")
    print("=" * 118)

    print(
        f"Refs: "
        f"{', '.join(REF_SYMBOLS)}"
    )

    print(
        f"Targets: "
        f"{len(targets)}"
    )

    print(
        f"Top {TOP_K} Momentum = "
        f"{ALLOCATION_PER_COIN * 100:.1f}% je Coin"
    )

    print(
        "Quorums getestet: "
        "2/3 und 3/3"
    )

    print(
        "Non-overlap wurde GLOBAL "
        "pro Hold-Horizont angewendet."
    )

    print(
        "Die Top-5-Auswahl basiert nur "
        "auf der Kursentwicklung bis zum "
        "jeweiligen Signalzeitpunkt."
    )

    print(
        "Compound: gesamtes Kapital wird "
        "nach jedem abgeschlossenen "
        "Portfolio-Trade erneut eingesetzt."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Abbruch durch Benutzer."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 118)

        print(
            "FEHLER:"
        )

        print(
            str(exc)
        )

        print("=" * 118)

        sys.exit(1)