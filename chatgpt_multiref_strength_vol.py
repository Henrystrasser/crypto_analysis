#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chatgpt_multiref_single.py

MULTI-REF STRENGTH
SINGLE TARGET COIN
VOLATILITÄTS-ADAPTIVER THRESHOLD

============================================================
STRATEGIE
============================================================

Wir beobachten mehrere Reference-Coins.

Beispiel mit 5 Refs:

    BTC
    ETH
    SOL
    BNB
    DOGE

Für ein Event-Fenster, z.B. 30 Minuten, berechnen wir:

    BTC: +1.3 %
    ETH: +1.1 %
    SOL: +0.4 %
    BNB: +1.2 %
    DOGE: +0.9 %

Wenn der dynamische Threshold z.B. +1.0 % beträgt:

    BTC  -> über Threshold
    ETH  -> über Threshold
    SOL  -> nein
    BNB  -> über Threshold
    DOGE -> nein

=> 3/5 Refs

Mit --quorums 3,4,5 kann man solche Quoren
separat testen.

============================================================
VOLATILITÄTS-ADAPTIVER THRESHOLD
============================================================

    effective_threshold =
        base_threshold
        * volatility_factor
        * THRESHOLD_INTENSITY

Beispiel:

    Base Threshold = 1.0 %
    Volatility Factor = 1.8
    Intensity = 1.0

    => effektiver Threshold = 1.8 %

THRESHOLD_INTENSITY:

    0.75 = lockerer
    1.00 = neutral
    1.25 = strenger

============================================================
TARGET
============================================================

Es gibt nur EINEN Targetcoin.

Beispiel:

    TARGET_SYMBOL = "XRPUSDT"

Wenn das Multi-Ref-Signal entsteht:

    -> Target wird gekauft
    -> nach dem gewählten Hold verkauft

Es gibt keine Top-5-Selektion mehr.

============================================================
NON-OVERLAP
============================================================

GLOBAL pro Hold-Horizont.

Ein neuer Trade darf erst starten,
wenn der vorherige Trade desselben
Horizonts abgeschlossen ist.

============================================================
COMPOUND
============================================================

Nach jedem abgeschlossenen Trade wird
das gesamte Kapital erneut eingesetzt.

============================================================
FEES
============================================================

Binance Spot:

    Kauf: 7.5 bps
    Verkauf: 7.5 bps

============================================================
DATEN
============================================================

Binance Spot
15m Klines

Kein Lookahead:

    - Threshold basiert nur auf Daten
      bis zum Signalzeitpunkt.
    - Volatilitätsbaseline verwendet SHIFT(1).
"""


from __future__ import annotations

import argparse
import math
import sys
import time
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

BINANCE_BASE = (
    "https://data-api.binance.vision"
)

INTERVAL = "15m"

INTERVAL_MINUTES = 15

INTERVAL_MS = (
    INTERVAL_MINUTES
    * 60
    * 1000
)

BERLIN = ZoneInfo(
    "Europe/Berlin"
)


# ============================================================
# TARGET
# ============================================================

# Hier deinen Targetcoin eintragen.
#
# Alternativ kannst du ihn beim Start setzen:
#
# python3 chatgpt_multiref_single.py --target XRPUSDT
#
TARGET_SYMBOL = "AAVEUSDT"


# ============================================================
# REFERENCES
# ============================================================

# Beliebig viele Refs möglich:
#
# 3 Refs:
# BTC,ETH,SOL
#
# 4 Refs:
# BTC,ETH,SOL,BNB
#
# 5 Refs:
# BTC,ETH,SOL,BNB,DOGE
#
REF_SYMBOLS = [
    "BTC",
    "ETH",
    "SOL",
    "XRP"
]


# ============================================================
# QUORUM
# ============================================================

# None bedeutet:
# automatisch die letzten beiden Quoren testen.
#
# Bei 3 Refs:
#   2/3 und 3/3
#
# Bei 4 Refs:
#   3/4 und 4/4
#
# Bei 5 Refs:
#   4/5 und 5/5
#
# Über CLI kannst du alles frei setzen:
#
# --quorums 2,3
# --quorums 3,4,5
# --quorums 2,3,4,5
#
QUORUMS = None


# ============================================================
# ZEITRAUM
# ============================================================

FROM_DATE = "2024-01-01"

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
# BASE THRESHOLDS
# ============================================================

EVENT_THRESHOLDS_PCT = [
    1.0,
    2.0,
]


# ============================================================
# THRESHOLD INTENSITY
# ============================================================

# Haupt-Stellhebel.
#
# 1.00 = neutral
# 0.80 = 20% niedriger
# 1.20 = 20% höher
#
THRESHOLD_INTENSITY = 0.8 


# ============================================================
# VOLATILITY SETTINGS
# ============================================================

# Aktuelle Marktvolatilität:
# 96 x 15m = 24 Stunden
VOL_CURRENT_LOOKBACK_BARS = 96

# Historische Normalvolatilität:
# 30 Tage
VOL_BASELINE_LOOKBACK_DAYS = 30

VOL_BASELINE_LOOKBACK_BARS = (
    VOL_CURRENT_LOOKBACK_BARS
    * VOL_BASELINE_LOOKBACK_DAYS
)

# 1.0 = linear
#
# Beispiel:
#
# Volatilität 2x normal
#
# Power 1.0:
# Threshold-Faktor = 2.0
#
# Power 0.5:
# Threshold-Faktor = sqrt(2) = 1.41
#
VOLATILITY_POWER = 1.0

# Schutz gegen extreme Ausreißer.
VOL_FACTOR_MIN = 0.50
VOL_FACTOR_MAX = 3.00


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
    ("150h", 150),
    ("192h", 192),
    ("250h", 250)
]


# ============================================================
# FEES
# ============================================================

FEE_BPS_PER_SIDE = 7.5

SLIPPAGE_BPS_PER_SIDE = 0.0


# ============================================================
# DOWNLOAD
# ============================================================

KLINE_LIMIT = 1000

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

        if int(
            self.times[idx]
        ) != int(timestamp_ms):

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

    base_threshold_pct: float

    effective_threshold_pct: float

    volatility_factor: float


@dataclass
class AcceptedTrade:
    event: Event
    portfolio_roi: float


# ============================================================
# HTTP
# ============================================================

def create_session() -> requests.Session:

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                "chatgpt-multiref-single/1.0",
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

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                if retry_after:

                    try:

                        wait = max(
                            wait,
                            float(
                                retry_after
                            ),
                        )

                    except ValueError:
                        pass

                print(
                    f"  Binance Rate Limit "
                    f"({response.status_code}) "
                    f"-> warte "
                    f"{wait:.1f}s"
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
# SYMBOL HELPERS
# ============================================================

def normalize_symbol(
    value: str,
) -> str:

    value = (
        value.strip()
        .upper()
    )

    if not value:
        raise ValueError(
            "Leeres Symbol."
        )

    if value.endswith(
        "USDT"
    ):

        return value

    return (
        value
        + "USDT"
    )


def normalize_refs(
    refs: Sequence[str],
) -> List[str]:

    result = []

    for ref in refs:

        base = (
            ref.strip()
            .upper()
        )

        if not base:
            continue

        result.append(
            base
        )

    # Duplikate entfernen,
    # Reihenfolge behalten.
    result = list(
        dict.fromkeys(
            result
        )
    )

    return result


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
                - timedelta(
                    milliseconds=1
                )
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
        - timestamp_ms
        % INTERVAL_MS
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

    start_ms = (
        berlin_date_to_utc_ms(
            FROM_DATE
        )
    )

    if TO_DATE is None:

        end_ms = (
            latest_closed_candle_ms()
        )

    else:

        end_ms = (
            berlin_date_to_utc_ms(
                TO_DATE,
                end_of_day=True,
            )
        )

        end_ms = min(
            end_ms,
            latest_closed_candle_ms(),
        )

    if end_ms <= start_ms:

        raise ValueError(
            "Zeitraum ungültig."
        )

    return (
        start_ms,
        end_ms,
    )


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

        unique[
            timestamp
        ] = close

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
# FRAME
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
# VOLATILITY
# ============================================================

def add_market_volatility(
    frame: pd.DataFrame,
    refs: Sequence[str],
) -> pd.DataFrame:

    # --------------------------------------------------------
    # 15m Return pro Ref
    # --------------------------------------------------------

    for ref in refs:

        frame[
            f"{ref}_ret15"
        ] = (
            frame[ref]
            / frame[ref].shift(1)
        ) - 1.0

    # --------------------------------------------------------
    # 24h Rolling Volatilität
    # --------------------------------------------------------

    vol_columns = []

    for ref in refs:

        col = (
            f"{ref}_vol"
        )

        frame[col] = (
            frame[
                f"{ref}_ret15"
            ]
            .rolling(
                VOL_CURRENT_LOOKBACK_BARS
            )
            .std()
        )

        vol_columns.append(
            col
        )

    # Median über alle Refs.
    #
    # Dadurch kann ein einzelner extremer Ref
    # den Faktor nicht komplett dominieren.
    frame[
        "market_vol"
    ] = frame[
        vol_columns
    ].median(
        axis=1
    )

    # --------------------------------------------------------
    # Historische Normalvolatilität
    #
    # SHIFT(1) verhindert Lookahead.
    # --------------------------------------------------------

    frame[
        "baseline_vol"
    ] = (
        frame[
            "market_vol"
        ]
        .rolling(
            VOL_BASELINE_LOOKBACK_BARS
        )
        .median()
        .shift(1)
    )

    # --------------------------------------------------------
    # Volatilitätsverhältnis
    # --------------------------------------------------------

    frame[
        "vol_ratio"
    ] = (
        frame[
            "market_vol"
        ]
        / frame[
            "baseline_vol"
        ]
    )

    frame[
        "vol_ratio"
    ] = frame[
        "vol_ratio"
    ].clip(
        lower=VOL_FACTOR_MIN,
        upper=VOL_FACTOR_MAX,
    )

    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    frame[
        "vol_factor"
    ] = (
        frame[
            "vol_ratio"
        ]
        ** VOLATILITY_POWER
    )

    return frame


# ============================================================
# EVENTS
# ============================================================

def create_events(
    refs_data: Dict[str, PriceSeries],
    window_minutes: int,
    base_threshold_pct: float,
    quorum: int,
) -> List[Event]:

    if (
        window_minutes
        % INTERVAL_MINUTES
        != 0
    ):

        raise ValueError(
            f"{window_minutes}m ist nicht "
            f"durch {INTERVAL_MINUTES}m teilbar."
        )

    if (
        quorum < 1
        or quorum > len(refs_data)
    ):

        raise ValueError(
            f"Quorum {quorum} ungültig."
        )

    frame = build_reference_frame(
        refs_data
    )

    frame = add_market_volatility(
        frame,
        list(
            refs_data.keys()
        ),
    )

    event_bars = (
        window_minutes
        // INTERVAL_MINUTES
    )

    # --------------------------------------------------------
    # Dynamischer Threshold
    # --------------------------------------------------------

    frame[
        "effective_threshold_pct"
    ] = (
        base_threshold_pct
        * frame["vol_factor"]
        * THRESHOLD_INTENSITY
    )

    # --------------------------------------------------------
    # Returns über Event Window
    # --------------------------------------------------------

    return_columns = {}

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
                event_bars,
            )
        ) - 1.0

        ref_return[
            :event_bars
        ] = np.nan

        return_columns[
            ref
        ] = ref_return

    returns_df = pd.DataFrame(
        return_columns
    )

    # --------------------------------------------------------
    # Wie viele Refs brechen den Threshold?
    #
    # WICHTIG:
    # Kein Addieren der Renditen.
    # Wir zählen nur die Anzahl der Refs,
    # deren individueller Return über dem
    # jeweiligen dynamischen Threshold liegt.
    # --------------------------------------------------------

    threshold_matrix = pd.DataFrame(
        {
            ref:
                (
                    returns_df[ref]
                    >=
                    (
                        frame[
                            "effective_threshold_pct"
                        ]
                        / 100.0
                    )
                )
            for ref in refs_data.keys()
        }
    )

    threshold_count = (
        threshold_matrix.sum(
            axis=1
        )
    )

    condition = (
        threshold_count
        >= quorum
    )

    # --------------------------------------------------------
    # Nur erster Eintritt in das Quorum.
    # --------------------------------------------------------

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

        if idx < event_bars:
            continue

        effective_threshold = float(
            frame.iloc[
                idx
            ][
                "effective_threshold_pct"
            ]
        )

        volatility_factor = float(
            frame.iloc[
                idx
            ][
                "vol_factor"
            ]
        )

        if not math.isfinite(
            effective_threshold
        ):

            continue

        if not math.isfinite(
            volatility_factor
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
                base_threshold_pct=(
                    base_threshold_pct
                ),
                effective_threshold_pct=(
                    effective_threshold
                ),
                volatility_factor=(
                    volatility_factor
                ),
            )
        )

    return events


# ============================================================
# TARGET ROI
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
        * (
            1.0
            + buy_slippage
        )
    )

    effective_exit = (
        exit_price
        * (
            1.0
            - sell_slippage
        )
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

    return (
        final_value
        - 1.0
    )


# ============================================================
# SIMULATION
# ============================================================

def simulate_hold(
    events: Sequence[Event],
    target: PriceSeries,
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
                <
                (
                    last_entry_ms
                    + hold_ms
                )
            ):

                continue

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        entry_price = target.exact_price(
            event.end_ms
        )

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        exit_price = target.exact_price(
            event.end_ms
            + hold_ms
        )

        if (
            entry_price is None
            or exit_price is None
        ):

            continue

        roi = calculate_net_roi(
            entry_price,
            exit_price,
        )

        if not math.isfinite(
            roi
        ):

            continue

        accepted.append(
            AcceptedTrade(
                event=event,
                portfolio_roi=roi,
            )
        )

        last_entry_ms = (
            event.end_ms
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
    target: PriceSeries,
    start_ms: int,
    end_ms: int,
) -> float:

    entry_price = (
        target.first_price_at_or_after(
            start_ms
        )
    )

    exit_price = (
        target.last_price_at_or_before(
            end_ms
        )
    )

    if (
        entry_price is None
        or exit_price is None
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

    quantity = (
        (1.0 - buy_fee)
        / entry_price
    )

    final_value = (
        quantity
        * exit_price
        * (1.0 - sell_fee)
    )

    return (
        final_value
        - 1.0
    ) * 100.0


# ============================================================
# FORMAT
# ============================================================

def fmt_pct(
    value: float,
) -> str:

    if not math.isfinite(
        value
    ):

        return "n/a"

    return f"{value:9.2f}%"


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
# DEBUG EVENTS
# ============================================================

def print_sample_events(
    events: Sequence[Event],
) -> None:

    print()
    print(
        "Beispiele der ersten Signal-Events:"
    )

    shown = 0

    for event in events:

        shown += 1

        ref_text = " | ".join(
            f"{ref} "
            f"{ret * 100:+.2f}%"
            for ref, ret
            in event.ref_returns.items()
        )

        print()

        print(
            f"  {format_berlin(event.end_ms)}"
        )

        print(
            f"  Refs: "
            f"{ref_text}"
        )

        print(
            f"  Quorum: "
            f"{event.ref_count_above_threshold}/"
            f"{len(event.ref_returns)}"
        )

        print(
            f"  Base Threshold: "
            f"{event.base_threshold_pct:.2f}%"
        )

        print(
            f"  Effective Threshold: "
            f"{event.effective_threshold_pct:.2f}%"
        )

        print(
            f"  Vol Factor: "
            f"{event.volatility_factor:.2f}x"
        )

        if shown >= 5:
            break


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Multi-Ref Strength "
            "Single Target"
        )
    )

    parser.add_argument(
        "--target",
        default=TARGET_SYMBOL,
        help=(
            "Targetcoin, z.B. XRPUSDT"
        ),
    )

    parser.add_argument(
        "--refs",
        default=",".join(
            REF_SYMBOLS
        ),
        help=(
            "Refs, z.B. "
            "BTC,ETH,SOL,BNB,DOGE"
        ),
    )

    parser.add_argument(
        "--quorums",
        default=None,
        help=(
            "Quoren, z.B. "
            "2,3 oder 3,4,5"
        ),
    )

    parser.add_argument(
        "--intensity",
        type=float,
        default=THRESHOLD_INTENSITY,
        help=(
            "Threshold Intensity"
        ),
    )

    parser.add_argument(
        "--vol-power",
        type=float,
        default=VOLATILITY_POWER,
        help=(
            "Volatility Power"
        ),
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
        "--debug-events",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    global TARGET_SYMBOL
    global REF_SYMBOLS
    global QUORUMS
    global THRESHOLD_INTENSITY
    global VOLATILITY_POWER
    global FROM_DATE
    global TO_DATE

    args = parse_args()

    TARGET_SYMBOL = (
        normalize_symbol(
            args.target
        )
    )

    REF_SYMBOLS = normalize_refs(
        [
            x
            for x
            in args.refs.split(",")
        ]
    )

    if len(REF_SYMBOLS) < 2:

        raise ValueError(
            "Mindestens 2 Refs erforderlich."
        )

    target_base = TARGET_SYMBOL[
        :-4
    ]

    # Target darf nicht gleichzeitig
    # als Reference verwendet werden.
    if target_base in REF_SYMBOLS:

        raise ValueError(
            f"Target {TARGET_SYMBOL} "
            f"darf nicht gleichzeitig "
            f"Reference sein."
        )

    if args.quorums:

        QUORUMS = [
            int(x.strip())
            for x
            in args.quorums.split(",")
            if x.strip()
        ]

    else:

        # Default:
        # letzte zwei Quoren.
        n_refs = len(
            REF_SYMBOLS
        )

        if n_refs >= 3:

            QUORUMS = [
                n_refs - 2,
                n_refs - 1,
                n_refs,
            ]

        else:

            QUORUMS = [
                n_refs - 1,
                n_refs,
            ]

    QUORUMS = sorted(
        set(
            int(q)
            for q
            in QUORUMS
        )
    )

    for quorum in QUORUMS:

        if (
            quorum < 1
            or quorum > len(
                REF_SYMBOLS
            )
        ):

            raise ValueError(
                f"Quorum {quorum} ist "
                f"bei {len(REF_SYMBOLS)} "
                f"Refs ungültig."
            )

    THRESHOLD_INTENSITY = (
        float(
            args.intensity
        )
    )

    VOLATILITY_POWER = (
        float(
            args.vol_power
        )
    )

    if THRESHOLD_INTENSITY <= 0:

        raise ValueError(
            "Threshold Intensity "
            "muss > 0 sein."
        )

    if VOLATILITY_POWER < 0:

        raise ValueError(
            "Volatility Power "
            "muss >= 0 sein."
        )

    FROM_DATE = (
        args.from_date
    )

    TO_DATE = (
        args.to_date
    )

    # --------------------------------------------------------
    # Zeitraum
    # --------------------------------------------------------

    start_ms, end_ms = (
        resolve_period()
    )

    max_hold_hours = max(
        hours
        for _, hours
        in HOLD_HORIZONS
    )

    target_data_end_ms = (
        end_ms
        + max_hold_hours
        * 60
        * 60
        * 1000
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 120)

    print(
        "MULTI-REF STRENGTH | SINGLE TARGET"
    )

    print(
        "VOLATILITÄTS-ADAPTIVER THRESHOLD"
    )

    print("=" * 120)

    print(
        f"Target: "
        f"{TARGET_SYMBOL}"
    )

    print(
        f"Refs ({len(REF_SYMBOLS)}): "
        f"{', '.join(REF_SYMBOLS)}"
    )

    print(
        f"Quorums: "
        + ", ".join(
            f"{q}/{len(REF_SYMBOLS)}"
            for q in QUORUMS
        )
    )

    print(
        f"Interval: "
        f"{INTERVAL}"
    )

    print(
        f"Zeitraum: "
        f"{format_berlin(start_ms)} → "
        f"{format_berlin(end_ms)}"
    )

    print(
        f"Threshold Intensity: "
        f"{THRESHOLD_INTENSITY:.2f}"
    )

    print(
        f"Volatility Power: "
        f"{VOLATILITY_POWER:.2f}"
    )

    print(
        f"Vol Factor Cap: "
        f"{VOL_FACTOR_MIN:.2f} - "
        f"{VOL_FACTOR_MAX:.2f}"
    )

    print(
        f"Fee: "
        f"{FEE_BPS_PER_SIDE:.1f} bps je Seite"
    )

    print(
        "Non-overlap: GLOBAL pro Hold"
    )

    print("=" * 120)

    # --------------------------------------------------------
    # Load refs
    # --------------------------------------------------------

    session = create_session()

    try:

        print()
        print(
            "Lade Reference-Coins..."
        )

        refs_data = {}

        for ref in REF_SYMBOLS:

            symbol = normalize_symbol(
                ref
            )

            print(
                f"Lade Ref {ref}..."
            )

            series = fetch_klines(
                session,
                symbol,
                start_ms,
                end_ms,
            )

            refs_data[
                ref
            ] = series

            print(
                f"  {len(series.times)} "
                f"Kerzen"
            )

    finally:

        session.close()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    print()
    print(
        f"Lade Target "
        f"{TARGET_SYMBOL}..."
    )

    target_session = create_session()

    try:

        target = fetch_klines(
            target_session,
            TARGET_SYMBOL,
            start_ms,
            target_data_end_ms,
        )

    finally:

        target_session.close()

    print(
        f"  {len(target.times)} Kerzen"
    )

    # --------------------------------------------------------
    # Buy & Hold
    # --------------------------------------------------------

    buy_hold = calculate_buy_hold(
        target,
        start_ms,
        end_ms,
    )

    print()
    print(
        "Buy&Hold Referenz:"
    )

    if math.isfinite(
        buy_hold
    ):

        print(
            f"  {TARGET_SYMBOL}: "
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
        * len(QUORUMS)
    )

    scenario_counter = 0

    for window_minutes in (
        EVENT_WINDOWS_MIN
    ):

        for base_threshold_pct in (
            EVENT_THRESHOLDS_PCT
        ):

            for quorum in QUORUMS:

                scenario_counter += 1

                print()
                print("=" * 120)

                print(
                    f"{TARGET_SYMBOL} | "
                    f"Event "
                    f"{window_minutes}m | "
                    f"Base "
                    f"{base_threshold_pct:.1f}% | "
                    f"Quorum "
                    f"{quorum}/{len(REF_SYMBOLS)}"
                )

                print("=" * 120)

                print(
                    f"Berechne "
                    f"{window_minutes}m | "
                    f"{base_threshold_pct:.1f}% | "
                    f"{quorum}/"
                    f"{len(REF_SYMBOLS)}..."
                )

                events = create_events(
                    refs_data,
                    window_minutes,
                    base_threshold_pct,
                    quorum,
                )

                print(
                    f"  Raw Events: "
                    f"{len(events)}"
                )

                if events:

                    effective_thresholds = np.array(
                        [
                            e.effective_threshold_pct
                            for e
                            in events
                        ],
                        dtype=np.float64,
                    )

                    volatility_factors = np.array(
                        [
                            e.volatility_factor
                            for e
                            in events
                        ],
                        dtype=np.float64,
                    )

                    print(
                        f"  Effective Threshold: "
                        f"Ø "
                        f"{np.mean(effective_thresholds):.2f}% | "
                        f"Min "
                        f"{np.min(effective_thresholds):.2f}% | "
                        f"Max "
                        f"{np.max(effective_thresholds):.2f}%"
                    )

                    print(
                        f"  Vol Factor: "
                        f"Ø "
                        f"{np.mean(volatility_factors):.2f}x | "
                        f"Min "
                        f"{np.min(volatility_factors):.2f}x | "
                        f"Max "
                        f"{np.max(volatility_factors):.2f}x"
                    )

                if args.debug_events:

                    print_sample_events(
                        events
                    )

                # ------------------------------------------------
                # Holds
                # ------------------------------------------------

                scenario_results = []

                for (
                    hold_name,
                    hold_hours,
                ) in HOLD_HORIZONS:

                    _, stats = (
                        simulate_hold(
                            events,
                            target,
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
    print("=" * 120)

    print(
        "FERTIG"
    )

    print("=" * 120)

    print(
        f"Target: "
        f"{TARGET_SYMBOL}"
    )

    print(
        f"Refs ({len(REF_SYMBOLS)}): "
        f"{', '.join(REF_SYMBOLS)}"
    )

    print(
        "Quorums: "
        + ", ".join(
            f"{q}/{len(REF_SYMBOLS)}"
            for q in QUORUMS
        )
    )

    print(
        f"Threshold Intensity: "
        f"{THRESHOLD_INTENSITY:.2f}"
    )

    print(
        f"Volatility Power: "
        f"{VOLATILITY_POWER:.2f}"
    )

    print(
        "Der Threshold passt sich "
        "an die aktuelle Volatilität "
        "relativ zur historischen "
        "Normalvolatilität an."
    )

    print(
        "Multi-Ref Strength zählt "
        "die Anzahl der Refs über "
        "dem Threshold."
    )

    print(
        "Es werden keine Ref-Renditen "
        "addiert."
    )

    print(
        "Non-overlap wurde GLOBAL "
        "pro Hold-Horizont angewendet."
    )

    print(
        "Compound setzt das gesamte "
        "Kapital nach jedem Trade "
        "erneut ein."
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
        print("=" * 120)

        print(
            "FEHLER:"
        )

        print(
            str(exc)
        )

        print("=" * 120)

        sys.exit(1)