#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chatgpt_alt_laggard_confirmation.py

MULTI-REF LAGGARD + CONFIRMATION TOP-5

Idee:
    1. BTC + ETH + SOL steigen gemeinsam.
    2. Unter den 20 liquidesten Altcoins werden echte Laggards gesucht.
    3. Noch NICHT kaufen.
    4. In den naechsten CONFIRM_WINDOW_MIN Minuten muss der Laggard
       anfangen aufzuholen.
    5. Nur bestaetigte Laggards kommen in die Top-5-Auswahl.
    6. Einstieg erfolgt am Ende der Bestaetigungsphase.

Beispiel:

    BTC  +1.8 %
    ETH  +1.5 %
    SOL  +1.7 %

    Ref Average = +1.67 %

    ALT A = +0.1 %
    Gap    = +1.57pp

    Danach 30m Confirmation:

    ALT A = +0.8 %
    Refs   = +0.2 %

    ALT A outperformt die Refs in der Confirmation
    und ist damit ein bestaetigter Laggard.

Portfolio:
    Top 5 = 5 x 20 %
    Event-ROI = Durchschnitt der 5 Einzel-ROIs
    Compound ueber alle akzeptierten Events.

Non-overlap:
    GLOBAL pro Hold-Horizont.

Fees:
    Binance Spot
    7.5 bps Kauf
    7.5 bps Verkauf

Daten:
    Binance Spot
    15m Klines
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
# REFERENCE COINS
# ============================================================

REF_SYMBOLS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB"
]


# ============================================================
# TARGET UNIVERSE
# ============================================================

# Genau 20 liquide Binance-USDT-Altcoins.
#
# Auswahl erfolgt direkt ueber Binance 24h Quote Volume.
TARGET_TOP_N = 20

# Pro Event werden 5 Coins gekauft.
TOP_K = 5

# 5 x 20 %
ALLOCATION_PER_COIN = 1.0 / TOP_K


# ============================================================
# ZEITRAUM
# ============================================================

FROM_DATE = "2024-01-01"

# None = automatisch bis zur letzten vollstaendig
# abgeschlossenen 15m-Kerze.
TO_DATE = None


# ============================================================
# EVENT SETTINGS
# ============================================================

# Die Ref-Bewegung wird fuer mehrere Fenster untersucht.
EVENT_WINDOWS_MIN = [
    30,
    60,
    120,
    240,
    360,
]

# Separat:
# +1 %
# +2 %
EVENT_THRESHOLDS_PCT = [
    1.0,
    2.0,
]


# ============================================================
# CONFIRMATION SETTINGS
# ============================================================

# Nach dem Laggard-Event wird zusaetzlich 30 Minuten gewartet.
CONFIRM_WINDOW_MIN = 30

# Der Laggard muss in dieser Phase mindestens +0.50 %
# steigen.
CONFIRM_MOVE_PCT = 0.50

# Der Coin muss beim urspruenglichen Event mindestens
# 0.50 Prozentpunkte hinter dem Ref-Durchschnitt liegen.
MIN_INITIAL_GAP_PCT = 0.50

# Nach der Confirmation soll noch etwas Gap uebrig sein.
# Wenn der Coin schon komplett aufgeholt hat, steigen wir
# nicht mehr ein.
MIN_REMAINING_GAP_PCT = 0.25


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

# Zusätzliche Slippage.
# 0.0 = keine zusätzliche Slippage.
SLIPPAGE_BPS_PER_SIDE = 0.0


# ============================================================
# TARGET FILTER
# ============================================================

# Stablecoins und ähnliche USD-Coins ausschliessen.
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

# Referenz-Coins und XAUT nicht als Targets verwenden.
EXCLUDED_TARGET_BASES = {
    "BTC",
    "ETH",
    "SOL",
    "XAUT",
}


# ============================================================
# DOWNLOAD SETTINGS
# ============================================================

KLINE_LIMIT = 1000

# Moderate Parallelisierung.
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

        if int(self.times[idx]) != int(
            timestamp_ms
        ):
            return None

        price = float(
            self.closes[idx]
        )

        if not math.isfinite(price) or price <= 0:
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

        if not math.isfinite(price) or price <= 0:
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

        if not math.isfinite(price) or price <= 0:
            return None

        return price


@dataclass
class Event:
    start_ms: int
    end_ms: int
    confirm_end_ms: int

    ref_returns: Dict[str, float]
    ref_strength: float

    ref_confirm_returns: Dict[str, float]
    ref_confirm_strength: float


@dataclass
class SelectedTarget:
    symbol: str

    initial_gap: float
    initial_target_return: float

    confirm_return: float
    confirm_excess: float

    remaining_gap: float


@dataclass
class AcceptedTrade:
    event: Event
    selected: List[SelectedTarget]
    portfolio_roi: float


# ============================================================
# HTTP
# ============================================================

def create_session() -> requests.Session:

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "chatgpt-alt-laggard-confirmation/1.0"
            ),
            "Accept": "application/json",
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

    current_open = (
        floor_to_interval_ms(
            now_ms
        )
    )

    return (
        current_open
        - INTERVAL_MS
    )


def resolve_period() -> Tuple[int, int]:

    start_ms = berlin_date_to_utc_ms(
        FROM_DATE,
        end_of_day=False,
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
            "Zeitraum ungueltig: "
            "TO_DATE liegt vor FROM_DATE."
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
# TARGET UNIVERSE
# ============================================================

def load_exchange_info(
    session: requests.Session,
) -> dict:

    return get_json(
        session,
        "/api/v3/exchangeInfo",
    )


def load_24h_tickers(
    session: requests.Session,
) -> list:

    return get_json(
        session,
        "/api/v3/ticker/24hr",
    )


def build_target_universe(
    session: requests.Session,
    refs: Sequence[str],
    top_n: int,
) -> List[str]:

    exchange_info = load_exchange_info(
        session
    )

    tickers = load_24h_tickers(
        session
    )

    ref_set = {
        x.upper()
        for x in refs
    }

    ticker_volume: Dict[
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

            volume = float(
                ticker.get(
                    "quoteVolume",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            volume = 0.0

        if volume > 0:

            ticker_volume[
                symbol
            ] = volume

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

        base = item.get(
            "baseAsset"
        )

        symbol = item.get(
            "symbol"
        )

        if not base or not symbol:
            continue

        base_upper = base.upper()

        if base_upper in STABLECOINS:
            continue

        if base_upper in EXCLUDED_TARGET_BASES:
            continue

        if base_upper in ref_set:
            continue

        volume = ticker_volume.get(
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
        for _, symbol in candidates[:top_n]
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
            f"Keine Klines fuer {symbol}"
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
# PARALLEL TARGET LOADING
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

    loaded: Dict[
        str,
        PriceSeries,
    ] = {}

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
                f"{len(series.times):6d} "
                f"Kerzen"
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

    refs_data = {}

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

        refs_data[
            base.upper()
        ] = series

        print(
            f"  {len(series.times)} "
            f"Kerzen"
        )

    return refs_data


# ============================================================
# REFERENCE FRAME
# ============================================================

def build_reference_frame(
    refs_data: Dict[str, PriceSeries],
) -> pd.DataFrame:

    common_times = None

    for series in refs_data.values():

        times = pd.Index(
            series.times,
            dtype="int64",
        )

        if common_times is None:

            common_times = times

        else:

            common_times = (
                common_times.intersection(
                    times
                )
            )

    if (
        common_times is None
        or len(common_times) == 0
    ):

        raise RuntimeError(
            "Keine gemeinsamen Zeitstempel "
            "der Referenz-Coins."
        )

    common_times = (
        common_times.sort_values()
    )

    data = {
        "time": common_times.to_numpy(
            dtype=np.int64
        )
    }

    for base, series in refs_data.items():

        lookup = pd.Series(
            series.closes,
            index=series.times,
        )

        data[base] = (
            lookup
            .reindex(common_times)
            .to_numpy()
        )

    return pd.DataFrame(
        data
    ).dropna()


# ============================================================
# EVENT CREATION
# ============================================================

def create_events(
    refs_data: Dict[str, PriceSeries],
    window_minutes: int,
    threshold_pct: float,
    confirm_window_min: int,
) -> List[Event]:

    if (
        window_minutes
        % INTERVAL_MINUTES
        != 0
    ):

        raise ValueError(
            f"{window_minutes}m ist nicht durch "
            f"{INTERVAL_MINUTES}m teilbar."
        )

    if (
        confirm_window_min
        % INTERVAL_MINUTES
        != 0
    ):

        raise ValueError(
            f"{confirm_window_min}m ist nicht durch "
            f"{INTERVAL_MINUTES}m teilbar."
        )

    frame = build_reference_frame(
        refs_data
    )

    event_bars = (
        window_minutes
        // INTERVAL_MINUTES
    )

    confirm_bars = (
        confirm_window_min
        // INTERVAL_MINUTES
    )

    threshold = (
        threshold_pct / 100.0
    )

    return_matrix = {}

    for ref in refs_data.keys():

        prices = frame[ref].to_numpy(
            dtype=np.float64
        )

        returns = (
            prices
            / np.roll(
                prices,
                event_bars,
            )
        ) - 1.0

        returns[
            :event_bars
        ] = np.nan

        return_matrix[ref] = returns

    returns_df = pd.DataFrame(
        return_matrix
    )

    # ALL:
    # Jede Referenz muss die Schwelle erreicht haben.
    condition = (
        returns_df >= threshold
    ).all(
        axis=1
    )

    # Nur beim ersten Eintritt in die Bedingung
    # entsteht ein Event.
    previous_condition = (
        condition.shift(1)
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

        if idx < event_bars:
            continue

        confirm_end_idx = (
            idx
            + confirm_bars
        )

        if (
            confirm_end_idx
            >= len(frame)
        ):
            continue

        start_ms = int(
            times[
                idx - event_bars
            ]
        )

        end_ms = int(
            times[idx]
        )

        confirm_end_ms = int(
            times[
                confirm_end_idx
            ]
        )

        ref_returns = {}
        ref_confirm_returns = {}

        valid = True

        for ref in refs_data.keys():

            initial_return = float(
                returns_df.iloc[
                    idx
                ][ref]
            )

            if not math.isfinite(
                initial_return
            ):

                valid = False
                break

            ref_price_end = float(
                frame.iloc[
                    idx
                ][ref]
            )

            ref_price_confirm = float(
                frame.iloc[
                    confirm_end_idx
                ][ref]
            )

            if (
                ref_price_end <= 0
                or ref_price_confirm <= 0
            ):

                valid = False
                break

            confirm_return = (
                ref_price_confirm
                / ref_price_end
            ) - 1.0

            if not math.isfinite(
                confirm_return
            ):

                valid = False
                break

            ref_returns[
                ref
            ] = initial_return

            ref_confirm_returns[
                ref
            ] = confirm_return

        if not valid:
            continue

        ref_strength = float(
            np.mean(
                list(
                    ref_returns.values()
                )
            )
        )

        ref_confirm_strength = float(
            np.mean(
                list(
                    ref_confirm_returns.values()
                )
            )
        )

        events.append(
            Event(
                start_ms=start_ms,
                end_ms=end_ms,
                confirm_end_ms=confirm_end_ms,
                ref_returns=ref_returns,
                ref_strength=ref_strength,
                ref_confirm_returns=ref_confirm_returns,
                ref_confirm_strength=ref_confirm_strength,
            )
        )

    return events


# ============================================================
# CONFIRMED LAGGARD SELECTION
# ============================================================

def find_confirmed_top_laggards(
    event: Event,
    target_data: Dict[str, PriceSeries],
    hold_ms: int,
) -> List[SelectedTarget]:

    candidates = []

    for symbol, series in target_data.items():

        initial_start = series.exact_price(
            event.start_ms
        )

        initial_end = series.exact_price(
            event.end_ms
        )

        confirm_end = series.exact_price(
            event.confirm_end_ms
        )

        exit_price = series.exact_price(
            event.confirm_end_ms
            + hold_ms
        )

        if (
            initial_start is None
            or initial_end is None
            or confirm_end is None
            or exit_price is None
        ):
            continue

        # ----------------------------------------------------
        # 1. INITIAL LAG
        # ----------------------------------------------------

        initial_target_return = (
            initial_end
            / initial_start
        ) - 1.0

        initial_gap = (
            event.ref_strength
            - initial_target_return
        )

        # Muss wirklich ein Laggard sein.
        if (
            initial_gap
            < MIN_INITIAL_GAP_PCT / 100.0
        ):
            continue

        # ----------------------------------------------------
        # 2. CONFIRMATION
        # ----------------------------------------------------

        confirm_return = (
            confirm_end
            / initial_end
        ) - 1.0

        # Coin muss selbst deutlich steigen.
        if (
            confirm_return
            < CONFIRM_MOVE_PCT / 100.0
        ):
            continue

        # Coin muss in dieser Phase stärker sein
        # als der Ref-Durchschnitt.
        confirm_excess = (
            confirm_return
            - event.ref_confirm_strength
        )

        if confirm_excess <= 0:
            continue

        # ----------------------------------------------------
        # 3. REST-GAP
        # ----------------------------------------------------

        # Gap zu Beginn:
        #
        # Ref initial - Target initial
        #
        # Danach:
        #
        # alter Gap + Ref Confirmation - Target Confirmation
        #
        remaining_gap = (
            initial_gap
            + event.ref_confirm_strength
            - confirm_return
        )

        # Wenn alles schon aufgeholt wurde,
        # ist der Einstieg fuer uns zu spaet.
        if (
            remaining_gap
            < MIN_REMAINING_GAP_PCT / 100.0
        ):
            continue

        candidates.append(
            SelectedTarget(
                symbol=symbol,
                initial_gap=initial_gap,
                initial_target_return=(
                    initial_target_return
                ),
                confirm_return=confirm_return,
                confirm_excess=confirm_excess,
                remaining_gap=remaining_gap,
            )
        )

    # Der groesste noch vorhandene Gap zuerst.
    candidates.sort(
        key=lambda x: x.remaining_gap,
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

    # Start mit 1.0 Kapital.
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
                event.confirm_end_ms
                <
                last_entry_ms
                + hold_ms
            ):
                continue

        # ----------------------------------------------------
        # TOP-5 CONFIRMED LAGGARDS
        # ----------------------------------------------------

        selected = (
            find_confirmed_top_laggards(
                event,
                target_data,
                hold_ms,
            )
        )

        # Genau 5 x 20 %
        if len(selected) < TOP_K:
            continue

        # ----------------------------------------------------
        # 5 EINZEL-ROIs
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
                event.confirm_end_ms
            )

            exit_price = series.exact_price(
                event.confirm_end_ms
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

            coin_rois.append(roi)

        if not valid:
            continue

        # 5 x 20 %
        portfolio_roi = float(
            np.mean(coin_rois)
        )

        accepted.append(
            AcceptedTrade(
                event=event,
                selected=selected,
                portfolio_roi=portfolio_roi,
            )
        )

        # Entry blockiert neue Events.
        last_entry_ms = (
            event.confirm_end_ms
        )

    return (
        accepted,
        calculate_stats(
            accepted
        ),
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

    # Compound Event für Event.
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

        roi = final_value - 1.0

        if math.isfinite(roi):
            rois.append(roi)

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
        "MULTI-REF LAGGARD + CONFIRMATION TOP-5"
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
        f"{format_berlin(start_ms)} CET/CEST → "
        f"{format_berlin(end_ms)} CET/CEST"
    )

    print(
        f"Target Universe: "
        f"{len(targets)} liquide Binance-USDT-Alts"
    )

    print(
        f"Top Laggards: "
        f"{TOP_K} × "
        f"{ALLOCATION_PER_COIN * 100:.1f}%"
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
        "Confirmation:"
    )

    print(
        f"  Window: "
        f"{CONFIRM_WINDOW_MIN}m"
    )

    print(
        f"  Mindest-Move: "
        f"+{CONFIRM_MOVE_PCT:.2f}%"
    )

    print(
        f"  Initial Gap: "
        f"mindestens "
        f"{MIN_INITIAL_GAP_PCT:.2f}pp"
    )

    print(
        f"  Rest-Gap: "
        f"mindestens "
        f"{MIN_REMAINING_GAP_PCT:.2f}pp"
    )

    print()
    print(
        f"Targets: "
        f"{', '.join(targets)}"
    )

    print("=" * 118)


def print_results_table(
    results: Sequence[
        Tuple[str, Dict[str, float]]
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
            math.isfinite(compound)
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
            f"{fmt_pct(stats['compound']):>13} "
            f"{fmt_pct(vs_bh):>11} "
            f"{fmt_pct(stats['best']):>10} "
            f"{fmt_pct(stats['worst']):>10}"
        )


def print_sample_events(
    events: Sequence[Event],
    target_data: Dict[str, PriceSeries],
) -> None:

    if not events:
        return

    print()
    print(
        "Beispiele für bestätigte Laggards:"
    )

    shown = 0

    for event in events:

        selected = (
            find_confirmed_top_laggards(
                event,
                target_data,
                hold_ms=(
                    3
                    * 60
                    * 60
                    * 1000
                ),
            )
        )

        if len(selected) < TOP_K:
            continue

        shown += 1

        print()
        print(
            f"  Event "
            f"{format_berlin(event.end_ms)} "
            f"-> Entry "
            f"{format_berlin(event.confirm_end_ms)}"
        )

        print(
            "  Refs Event: "
            + " | ".join(
                f"{ref} {ret * 100:+.2f}%"
                for ref, ret
                in event.ref_returns.items()
            )
        )

        print(
            f"  Ref Avg Event: "
            f"{event.ref_strength * 100:+.2f}%"
        )

        print(
            "  Refs Confirm: "
            + " | ".join(
                f"{ref} {ret * 100:+.2f}%"
                for ref, ret
                in event.ref_confirm_returns.items()
            )
        )

        print(
            f"  Ref Avg Confirm: "
            f"{event.ref_confirm_strength * 100:+.2f}%"
        )

        for target in selected:

            print(
                f"    {target.symbol:<12} "
                f"initial gap="
                f"{target.initial_gap * 100:+.2f}pp | "
                f"confirm="
                f"{target.confirm_return * 100:+.2f}% | "
                f"excess="
                f"{target.confirm_excess * 100:+.2f}pp | "
                f"rest-gap="
                f"{target.remaining_gap * 100:+.2f}pp"
            )

        if shown >= 3:
            break


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Multi-Ref Laggard + "
            "Confirmation Top-5 Backtest"
        )
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=TARGET_TOP_N,
        help=(
            "Target Universe "
            f"(default: {TARGET_TOP_N})"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=(
            "Top Laggards pro Event "
            f"(default: {TOP_K})"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=(
            "Parallele Downloads "
            f"(default: {MAX_WORKERS})"
        ),
    )

    parser.add_argument(
        "--from-date",
        default=FROM_DATE,
        help=(
            "Startdatum "
            f"(default: {FROM_DATE})"
        ),
    )

    parser.add_argument(
        "--to-date",
        default=TO_DATE,
        help=(
            "Enddatum YYYY-MM-DD; "
            "leer = letzte abgeschlossene Kerze"
        ),
    )

    parser.add_argument(
        "--refs",
        default=",".join(
            REF_SYMBOLS
        ),
        help=(
            "Refs, z.B. BTC,ETH,SOL"
        ),
    )

    parser.add_argument(
        "--debug-events",
        action="store_true",
        help=(
            "Zeigt Beispiel-Events "
            "mit Confirmation"
        ),
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

    FROM_DATE = (
        args.from_date
    )

    TO_DATE = (
        args.to_date
    )

    REF_SYMBOLS = [
        x.strip().upper()
        for x in args.refs.split(",")
        if x.strip()
    ]

    if len(REF_SYMBOLS) < 2:

        raise ValueError(
            "Mindestens zwei Referenz-Coins erforderlich."
        )

    # --------------------------------------------------------
    # Zeitraum
    # --------------------------------------------------------

    start_ms, end_ms = (
        resolve_period()
    )

    confirm_ms = (
        CONFIRM_WINDOW_MIN
        * 60
        * 1000
    )

    max_hold_hours = max(
        hours
        for _, hours
        in HOLD_HORIZONS
    )

    # Refs brauchen bis zum Ende
    # der Confirmation.
    refs_data_end_ms = (
        end_ms
        + confirm_ms
    )

    # Targets brauchen bis zum
    # Ende der Confirmation +
    # maximalem Hold.
    targets_data_end_ms = (
        end_ms
        + confirm_ms
        + max_hold_hours
        * 60
        * 60
        * 1000
    )

    # --------------------------------------------------------
    # Binance
    # --------------------------------------------------------

    session = create_session()

    try:

        # ----------------------------------------------------
        # Target Universe
        # ----------------------------------------------------

        targets = (
            build_target_universe(
                session,
                REF_SYMBOLS,
                TARGET_TOP_N,
            )
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

        # ----------------------------------------------------
        # References
        # ----------------------------------------------------

        print()
        print(
            "Lade Reference-Coins..."
        )

        refs_data = (
            load_reference_data(
                session,
                REF_SYMBOLS,
                start_ms,
                refs_data_end_ms,
            )
        )

    finally:

        session.close()

    # --------------------------------------------------------
    # Targets laden
    # --------------------------------------------------------

    target_data = (
        load_target_data(
            targets,
            start_ms,
            targets_data_end_ms,
        )
    )

    print()
    print(
        f"Geladen: "
        f"{len(target_data)} / "
        f"{len(targets)} Targets"
    )

    if len(target_data) < TOP_K:

        raise RuntimeError(
            f"Nur {len(target_data)} Targets "
            f"geladen, aber TOP_K={TOP_K}."
        )

    targets = [
        symbol
        for symbol in targets
        if symbol in target_data
    ]

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print_header(
        start_ms,
        end_ms,
        targets,
    )

    # --------------------------------------------------------
    # Buy & Hold
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
            f"Buy&Hold der {bh_n} Targets: "
            f"{buy_hold:+.2f}%"
        )

    else:

        print(
            "  n/a"
        )

    # --------------------------------------------------------
    # Scenarios
    # --------------------------------------------------------

    total_scenarios = (
        len(EVENT_WINDOWS_MIN)
        * len(EVENT_THRESHOLDS_PCT)
    )

    scenario_counter = 0

    for window_minutes in (
        EVENT_WINDOWS_MIN
    ):

        for threshold_pct in (
            EVENT_THRESHOLDS_PCT
        ):

            scenario_counter += 1

            print()
            print(
                "=" * 118
            )

            print(
                f"UP | Event "
                f"{window_minutes}m | "
                f"+{threshold_pct:.1f}% | "
                f"LAGGARD + CONFIRMATION"
            )

            print(
                "=" * 118
            )

            print(
                f"Berechne UP | "
                f"{window_minutes}m | "
                f"{threshold_pct:.1f}%..."
            )

            events = create_events(
                refs_data,
                window_minutes,
                threshold_pct,
                CONFIRM_WINDOW_MIN,
            )

            print(
                f"  Raw Events: "
                f"{len(events)}"
            )

            # Nur einmal fuer die Übersicht:
            # Wie viele Events liefern überhaupt
            # 5 bestätigte Laggards?
            confirmed_count = 0

            for event in events:

                confirmed = (
                    find_confirmed_top_laggards(
                        event,
                        target_data,
                        hold_ms=(
                            3
                            * 60
                            * 60
                            * 1000
                        ),
                    )
                )

                if len(confirmed) >= TOP_K:
                    confirmed_count += 1

            print(
                f"  Events mit "
                f"Top-{TOP_K} Confirmation: "
                f"{confirmed_count}"
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
    # Final
    # --------------------------------------------------------

    print()
    print(
        "=" * 118
    )

    print(
        "FERTIG"
    )

    print(
        "=" * 118
    )

    print(
        f"Refs: "
        f"{', '.join(REF_SYMBOLS)}"
    )

    print(
        f"Target Universe: "
        f"{len(targets)} liquideste "
        f"Binance-USDT-Alts"
    )

    print(
        f"Top {TOP_K} = "
        f"{ALLOCATION_PER_COIN * 100:.1f}% je Coin"
    )

    print(
        f"Entry erfolgt erst nach "
        f"{CONFIRM_WINDOW_MIN}m Confirmation."
    )

    print(
        f"Confirmation verlangt mindestens "
        f"+{CONFIRM_MOVE_PCT:.2f}% Coin-Move "
        f"und Outperformance der Refs."
    )

    print(
        f"Initial Gap mindestens "
        f"{MIN_INITIAL_GAP_PCT:.2f}pp."
    )

    print(
        f"Nach Confirmation mindestens "
        f"{MIN_REMAINING_GAP_PCT:.2f}pp Rest-Gap."
    )

    print(
        "Non-overlap wurde GLOBAL pro "
        "Hold-Horizont angewendet."
    )

    print(
        "Compound setzt voraus, dass das gesamte "
        "Kapital nach jedem abgeschlossenen "
        "Portfolio-Trade wieder eingesetzt wird."
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
        print(
            "=" * 118
        )

        print(
            "FEHLER:"
        )

        print(
            str(exc)
        )

        print(
            "=" * 118
        )

        sys.exit(1)