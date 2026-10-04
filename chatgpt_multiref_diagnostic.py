#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chatgpt_multiref_diagnostic.py

MULTI-REF STRENGTH
SINGLE TARGET
VOLATILITY-ADAPTIVE THRESHOLD
+ OVERLAP / SIGNAL CLUSTER DIAGNOSTIC


============================================================
IDEE
============================================================

Wir haben einen einzelnen Targetcoin, z.B. XRPUSDT.

Mehrere Referenz-Coins, z.B.:

    BTC
    ETH
    SOL
    BNB
    DOGE

lösen ein Signal aus, wenn mindestens X von Y Refs
einen dynamischen Threshold brechen.

Beispiel:

    4 Refs

    BTC  +1.4%
    ETH  +1.2%
    SOL  +0.5%
    BNB  +1.3%

    Threshold = 1.0%

    => 3/4 Refs über Threshold


============================================================
VOLATILITÄTS-ADAPTIVER THRESHOLD
============================================================

    effective_threshold =
        base_threshold
        * volatility_factor
        * THRESHOLD_INTENSITY


============================================================
WICHTIG:
SIGNAL- UND TRADING-ANALYSE GETRENNT
============================================================

Wir untersuchen zuerst ALLE Signale.

Danach simulieren wir, welche davon mit nur einem
Portfolio-Slot tatsächlich gehandelt werden können.


============================================================
SIGNAL CLUSTER
============================================================

Für einen bestimmten Hold:

    Event A
    Event B
    Event C
    Event D

Wenn B/C/D noch innerhalb der Haltezeit von A liegen,
gehören sie zu demselben Cluster.

Beispiel bei Hold = 24h:

    10:00 Signal A   -> handelbar
    14:00 Signal B   -> Overlap
    18:00 Signal C   -> Overlap
    22:00 Signal D   -> Overlap

    10:00 = Cluster-Signal #1
    14:00 = Cluster-Signal #2
    18:00 = Cluster-Signal #3
    22:00 = Cluster-Signal #4

Das erlaubt uns zu prüfen:

    Wie gut ist Signal #1?
    Wie gut ist Signal #2?
    Wie gut ist Signal #3?
    Wie gut ist Signal #4?


============================================================
AUSFÜHRBARE STRATEGIE
============================================================

Wir handeln immer nur:

    erstes Signal eines neuen Clusters

Das entspricht unserem bisherigen:

    GLOBALER NON-OVERLAP


============================================================
AUSWERTUNG
============================================================

Für jeden Hold werden ausgegeben:

    Raw Signals
    Raw Win%
    Raw Ø ROI

    Cluster:
        #1
        #2
        #3
        #4
        #5+

    Accepted Trades
    Skipped Signals
    Skip-Rate

    Ø ROI Accepted
    Ø ROI Skipped

    Compound
    Best
    Worst
    vs Buy&Hold

    Jahresperformance


============================================================
KEIN LOOKAHEAD FÜR DAS SIGNAL
============================================================

Der dynamische Threshold benutzt nur Daten bis
zum Signalzeitpunkt.

Die Signalbedingung selbst verwendet keine Zukunftsdaten.

Die spätere Trade-ROI ist natürlich für die Backtest-Analyse
aus zukünftigen Kerzen berechnet.


============================================================
TARGET
============================================================

Ein einzelner Targetcoin.

Standard:

    XRPUSDT


============================================================
REFS
============================================================

Beliebig viele Refs.

Beispiel:

    --refs BTC,ETH,SOL,BNB

oder:

    --refs BTC,ETH,SOL,BNB,DOGE


============================================================
QUORUMS
============================================================

Beispiel bei 5 Refs:

    --quorums 3,4,5

testet:

    3/5
    4/5
    5/5


============================================================
Signal-Schalter (Config oben): VOL_ADAPTIVE (False = fester
Threshold), EVENT_MODE ("cross" = nur Neueintritt, "level" = jede
Kerze mit erfüllter Bedingung), DIRECTION ("UP" | "DOWN").
Bei VOL_ADAPTIVE wird die Vol-Historie vor FROM_DATE nachgeladen.

Zeitangaben: Europe/Berlin; Entry/Exit = Kerzen-CLOSE.
Ein neuer Trade darf exakt am Hold-Ende des vorherigen starten.
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

def _interval_to_minutes(interval: str) -> int:
    """'15m' -> 15, '1h' -> 60, '4h' -> 240, '1d' -> 1440 (aus INTERVAL abgeleitet)."""
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    return int(interval[:-1]) * units[interval[-1]]


INTERVAL = "15m"

# Aus INTERVAL abgeleitet - nicht von Hand koppeln.
INTERVAL_MINUTES = _interval_to_minutes(INTERVAL)

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

TARGET_SYMBOL = "PEPEUSDT"


# ============================================================
# REFERENCES
# ============================================================

REF_SYMBOLS = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
]


# ============================================================
# QUORUMS
# ============================================================

# None:
# Bei 3 Refs automatisch:
#     2/3 und 3/3
#
# Bei 4 Refs:
#     3/4 und 4/4
#
# Bei 5 Refs:
#     4/5 und 5/5
#
# Über CLI frei definierbar.
QUORUMS = None


# ============================================================
# ZEITRAUM
# ============================================================

FROM_DATE = "2024-01-01"

# None = letzte vollständig abgeschlossene 15m-Kerze
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

# 1.00 = neutral
# 0.80 = 20% niedriger
# 1.20 = 20% höher

THRESHOLD_INTENSITY = 1.00


# ============================================================
# VOLATILITY
# ============================================================

# Aktuelle Marktvolatilität: 24 Stunden (Kerzenzahl wird berechnet)
VOL_CURRENT_LOOKBACK_MINUTES = 24 * 60
VOL_CURRENT_LOOKBACK_BARS = (
    VOL_CURRENT_LOOKBACK_MINUTES
    // INTERVAL_MINUTES
)

# 30 Tage Normalvolatilität
VOL_BASELINE_LOOKBACK_DAYS = 30

VOL_BASELINE_LOOKBACK_BARS = (
    VOL_BASELINE_LOOKBACK_DAYS
    * 24 * 60
    // INTERVAL_MINUTES
)

# ------------------------------------------------------------
# SIGNAL-SCHALTER
# ------------------------------------------------------------

# VOL_ADAPTIVE:
#   True  = volatilitäts-adaptiver Threshold. Die Vol-Historie
#           (Baseline + aktuelles Fenster) wird automatisch VOR
#           FROM_DATE nachgeladen (Warmup); Events erst ab FROM_DATE.
#   False = fester Threshold (Faktor 1.0), kein Warmup nötig.
VOL_ADAPTIVE = True

# EVENT_MODE:
#   "cross" = nur wenn die Bedingung NEU erfüllt ist
#             (vorherige Kerze hat sie nicht erfüllt)
#   "level" = jede Kerze, auf der die Bedingung gilt
#             (nach Hold-Ende wird erneut gekauft, falls weiter erfüllt)
EVENT_MODE = "cross"

# DIRECTION:
#   "UP"   = Refs steigen um >= Threshold
#   "DOWN" = Refs fallen um >= Threshold (Return <= -Threshold)
DIRECTION = "UP"


def vol_warmup_ms() -> int:
    """Vorlauf für die Ref-Daten (nur bei VOL_ADAPTIVE)."""
    if not VOL_ADAPTIVE:
        return 0
    return (
        VOL_BASELINE_LOOKBACK_BARS
        + VOL_CURRENT_LOOKBACK_BARS
        + 1
    ) * INTERVAL_MS

# Volatilitätsreaktion:
#
# 1.0 = linear
# 0.5 = abgeschwächt
# 1.5 = verstärkt
VOLATILITY_POWER = 1.0

VOL_FACTOR_MIN = 0.50
VOL_FACTOR_MAX = 3.00


# ============================================================
# HOLDS
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
class SignalObservation:

    event: Event

    roi: float

    cluster_id: int

    cluster_rank: int

    accepted: bool


# ============================================================
# HTTP
# ============================================================

def create_session() -> requests.Session:

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                "chatgpt-multiref-diagnostic/1.0",
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
# SYMBOL
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

    return list(
        dict.fromkeys(
            result
        )
    )


# ============================================================
# DATETIME
# ============================================================

def berlin_date_to_utc_ms(
    date_string: str,
    end_of_day: bool = False,
) -> int:

    if len(
        date_string
    ) == 10:

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

    return (
        pd.DataFrame(
            data
        )
        .dropna()
    )


# ============================================================
# VOLATILITY
# ============================================================

def add_market_volatility(
    frame: pd.DataFrame,
    refs: Sequence[str],
) -> pd.DataFrame:

    if not VOL_ADAPTIVE:
        # Fester Threshold: Faktor 1.0, keine Vol-Historie nötig.
        frame["vol_factor"] = 1.0
        return frame

    for ref in refs:

        frame[
            f"{ref}_ret15"
        ] = (
            frame[ref]
            / frame[ref].shift(1)
        ) - 1.0

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

    frame[
        "market_vol"
    ] = frame[
        vol_columns
    ].median(
        axis=1
    )

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
# EVENT GENERATION
# ============================================================

def _threshold_hit(
    returns: pd.Series,
    threshold: pd.Series,
    direction: str,
) -> pd.Series:
    """UP: Return >= +Threshold | DOWN: Return <= -Threshold."""
    if direction == "UP":
        return returns >= threshold
    if direction == "DOWN":
        return returns <= -threshold
    raise ValueError(f"DIRECTION {direction!r} ungültig (UP|DOWN).")


def create_events(
    refs_data: Dict[str, PriceSeries],
    window_minutes: int,
    base_threshold_pct: float,
    quorum: int,
    direction: Optional[str] = None,
) -> List[Event]:

    if direction is None:
        direction = DIRECTION

    if EVENT_MODE not in ("cross", "level"):
        raise ValueError(
            f"EVENT_MODE {EVENT_MODE!r} ungültig (cross|level)."
        )

    if (
        window_minutes
        % INTERVAL_MINUTES
        != 0
    ):

        raise ValueError(
            f"{window_minutes}m nicht "
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

    # Dynamischer Threshold
    frame[
        "effective_threshold_pct"
    ] = (
        base_threshold_pct
        * frame[
            "vol_factor"
        ]
        * THRESHOLD_INTENSITY
    )

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

    threshold_matrix = pd.DataFrame(
        {
            ref:
                (
                    _threshold_hit(
                        returns_df[ref],
                        frame["effective_threshold_pct"] / 100.0,
                        direction,
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

    # Nur erster Eintritt
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

    if EVENT_MODE == "level":
        # LEVEL: jede Kerze, auf der die Bedingung gilt.
        crossing = condition.copy()

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
                start_ms=int(
                    times[
                        idx - event_bars
                    ]
                ),
                end_ms=int(
                    times[idx]
                ),
                ref_returns=ref_returns,
                ref_count_above_threshold=int(
                    threshold_count.iloc[
                        idx
                    ]
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


def calculate_event_roi(
    event: Event,
    target: PriceSeries,
    hold_ms: int,
) -> Optional[float]:

    entry_price = target.exact_price(
        event.end_ms
    )

    exit_price = target.exact_price(
        event.end_ms
        + hold_ms
    )

    if (
        entry_price is None
        or exit_price is None
    ):

        return None

    roi = calculate_net_roi(
        entry_price,
        exit_price,
    )

    if not math.isfinite(roi):
        return None

    return roi


# ============================================================
# STATS
# ============================================================

def stats_from_rois(
    rois: Sequence[float],
) -> Dict[str, float]:

    if not rois:

        return {
            "n": 0,
            "win_pct": float("nan"),
            "avg": float("nan"),
            "median": float("nan"),
            "compound": float("nan"),
            "best": float("nan"),
            "worst": float("nan"),
        }

    arr = np.array(
        rois,
        dtype=np.float64,
    )

    return {
        "n": int(
            len(arr)
        ),
        "win_pct": float(
            np.mean(
                arr > 0
            )
            * 100.0
        ),
        "avg": float(
            np.mean(arr)
            * 100.0
        ),
        "median": float(
            np.median(arr)
            * 100.0
        ),
        "compound": float(
            (
                np.prod(
                    1.0 + arr
                )
                - 1.0
            )
            * 100.0
        ),
        "best": float(
            np.max(arr)
            * 100.0
        ),
        "worst": float(
            np.min(arr)
            * 100.0
        ),
    }


# ============================================================
# HOLD ANALYSIS
# ============================================================

def analyze_hold(
    events: Sequence[Event],
    target: PriceSeries,
    hold_hours: int,
) -> Dict:

    hold_ms = (
        hold_hours
        * 60
        * 60
        * 1000
    )

    observations = []

    cluster_id = 0

    cluster_end_ms = -1

    cluster_rank = 0

    # --------------------------------------------------------
    # Alle gültigen Raw Signals durchlaufen
    # --------------------------------------------------------

    for event in events:

        roi = calculate_event_roi(
            event,
            target,
            hold_ms,
        )

        if roi is None:
            continue

        # Neues Cluster?
        #
        # Wenn das neue Signal bereits nach dem Ende
        # des vorherigen Holds kommt, ist es ein neues Cluster.
        if (
            event.end_ms
            >= cluster_end_ms
        ):

            cluster_id += 1

            cluster_rank = 1

            cluster_end_ms = (
                event.end_ms
                + hold_ms
            )

        else:

            cluster_rank += 1

        observations.append(
            SignalObservation(
                event=event,
                roi=roi,
                cluster_id=cluster_id,
                cluster_rank=cluster_rank,
                accepted=(
                    cluster_rank == 1
                ),
            )
        )

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    raw_rois = [
        obs.roi
        for obs in observations
    ]

    raw_stats = stats_from_rois(
        raw_rois
    )

    # --------------------------------------------------------
    # ACCEPTED
    # --------------------------------------------------------

    accepted = [
        obs
        for obs in observations
        if obs.accepted
    ]

    accepted_rois = [
        obs.roi
        for obs in accepted
    ]

    accepted_stats = (
        stats_from_rois(
            accepted_rois
        )
    )

    # --------------------------------------------------------
    # SKIPPED
    # --------------------------------------------------------

    skipped = [
        obs
        for obs in observations
        if not obs.accepted
    ]

    skipped_rois = [
        obs.roi
        for obs in skipped
    ]

    skipped_stats = (
        stats_from_rois(
            skipped_rois
        )
    )

    # --------------------------------------------------------
    # CLUSTER RANKS
    # --------------------------------------------------------

    rank_stats = {}

    for rank_label, rank_min, rank_max in [
        ("1", 1, 1),
        ("2", 2, 2),
        ("3", 3, 3),
        ("4", 4, 4),
        ("5+", 5, 999999),
    ]:

        rank_rois = [
            obs.roi
            for obs in observations
            if (
                rank_min
                <= obs.cluster_rank
                <= rank_max
            )
        ]

        rank_stats[
            rank_label
        ] = stats_from_rois(
            rank_rois
        )

    # --------------------------------------------------------
    # CLUSTER SIZE
    # --------------------------------------------------------

    cluster_sizes = {}

    for obs in observations:

        cluster_sizes.setdefault(
            obs.cluster_id,
            0,
        )

        cluster_sizes[
            obs.cluster_id
        ] += 1

    cluster_size_values = list(
        cluster_sizes.values()
    )

    if cluster_size_values:

        avg_cluster_size = float(
            np.mean(
                cluster_size_values
            )
        )

        median_cluster_size = float(
            np.median(
                cluster_size_values
            )
        )

        max_cluster_size = int(
            np.max(
                cluster_size_values
            )
        )

    else:

        avg_cluster_size = float("nan")
        median_cluster_size = float("nan")
        max_cluster_size = 0

    # --------------------------------------------------------
    # YEARLY ACCEPTED PERFORMANCE
    # --------------------------------------------------------

    yearly = {}

    for obs in accepted:

        dt = datetime.fromtimestamp(
            obs.event.end_ms / 1000,
            tz=timezone.utc,
        ).astimezone(
            BERLIN
        )

        year = dt.year

        yearly.setdefault(
            year,
            [],
        )

        yearly[
            year
        ].append(
            obs.roi
        )

    yearly_stats = {}

    for year, rois in sorted(
        yearly.items()
    ):

        yearly_stats[
            year
        ] = stats_from_rois(
            rois
        )

    # --------------------------------------------------------
    # YEAR ROBUSTNESS
    # --------------------------------------------------------

    yearly_compounds = [
        data["compound"]
        for data
        in yearly_stats.values()
        if math.isfinite(
            data["compound"]
        )
    ]

    if yearly_compounds:

        worst_year = float(
            np.min(
                yearly_compounds
            )
        )

        best_year = float(
            np.max(
                yearly_compounds
            )
        )

        positive_years = int(
            np.sum(
                np.array(
                    yearly_compounds
                )
                > 0
            )
        )

    else:

        worst_year = float("nan")
        best_year = float("nan")
        positive_years = 0

    # --------------------------------------------------------
    # RAW SIGNAL RETENTION
    # --------------------------------------------------------

    raw_count = len(
        observations
    )

    accepted_count = len(
        accepted
    )

    skipped_count = len(
        skipped
    )

    if raw_count:

        accepted_pct = (
            accepted_count
            / raw_count
            * 100.0
        )

        skipped_pct = (
            skipped_count
            / raw_count
            * 100.0
        )

    else:

        accepted_pct = float("nan")
        skipped_pct = float("nan")

    # --------------------------------------------------------
    # DECAY
    # --------------------------------------------------------

    rank1_avg = rank_stats[
        "1"
    ]["avg"]

    rank2_avg = rank_stats[
        "2"
    ]["avg"]

    rank3_avg = rank_stats[
        "3"
    ]["avg"]

    if (
        math.isfinite(rank1_avg)
        and math.isfinite(rank2_avg)
    ):

        rank2_minus_rank1 = (
            rank2_avg
            - rank1_avg
        )

    else:

        rank2_minus_rank1 = float(
            "nan"
        )

    if (
        math.isfinite(rank1_avg)
        and math.isfinite(rank3_avg)
    ):

        rank3_minus_rank1 = (
            rank3_avg
            - rank1_avg
        )

    else:

        rank3_minus_rank1 = float(
            "nan"
        )

    return {
        "hold_hours": hold_hours,
        "observations": observations,
        "raw_stats": raw_stats,
        "accepted_stats": accepted_stats,
        "skipped_stats": skipped_stats,
        "rank_stats": rank_stats,
        "cluster_count": len(
            cluster_sizes
        ),
        "avg_cluster_size":
            avg_cluster_size,
        "median_cluster_size":
            median_cluster_size,
        "max_cluster_size":
            max_cluster_size,
        "accepted_pct":
            accepted_pct,
        "skipped_pct":
            skipped_pct,
        "yearly_stats":
            yearly_stats,
        "worst_year":
            worst_year,
        "best_year":
            best_year,
        "positive_years":
            positive_years,
        "rank2_minus_rank1":
            rank2_minus_rank1,
        "rank3_minus_rank1":
            rank3_minus_rank1,
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
# FORMATTING
# ============================================================

def fmt_pct(
    value: float,
) -> str:

    if not math.isfinite(
        value
    ):

        return "n/a"

    return (
        f"{value:9.2f}%"
    )


def fmt_num(
    value: float,
) -> str:

    if not math.isfinite(
        value
    ):

        return "n/a"

    return (
        f"{value:9.2f}"
    )


# ============================================================
# PRINT CONFIG
# ============================================================

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
    a = datetime.fromtimestamp(from_ms / 1000.0, tz=BERLIN).strftime("%Y-%m-%d")
    b = datetime.fromtimestamp(to_ms / 1000.0, tz=BERLIN).strftime("%Y-%m-%d")
    return f"from {a} to {b}"


def print_header(
    start_ms: int,
    end_ms: int,
    buy_hold: float,
) -> None:

    print()
    print("=" * 120)

    print(
        "MULTI-REF STRENGTH "
        "SINGLE TARGET DIAGNOSTIC"
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
        "Quorums: "
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
        f"Vol adaptiv: {VOL_ADAPTIVE} | "
        f"Event-Modus: {EVENT_MODE} | "
        f"Richtung: {DIRECTION} | "
        f"Warmup: {vol_warmup_ms() / 86_400_000:.1f} Tage vor Start"
    )

    print(
        f"Vol Factor: "
        f"{VOL_FACTOR_MIN:.2f} - "
        f"{VOL_FACTOR_MAX:.2f}"
    )

    print(
        f"Fee: "
        f"{FEE_BPS_PER_SIDE:.1f} bps je Seite"
    )

    print(
        "Non-overlap: "
        "GLOBAL pro Hold"
    )

    print(
        "Buy&Hold: "
        f"{fmt_pct(buy_hold)}"
    )

    print("=" * 120)


# ============================================================
# PRINT RAW / CLUSTER DIAGNOSTIC
# ============================================================

def print_diagnostic(
    result: Dict,
) -> None:

    raw = result[
        "raw_stats"
    ]

    accepted = result[
        "accepted_stats"
    ]

    skipped = result[
        "skipped_stats"
    ]

    print()
    print(
        "SCHRITT 1 – RAW SIGNALS"
    )

    print(
        f"  Raw Signals: "
        f"{raw['n']}"
    )

    print(
        f"  Raw Win%: "
        f"{fmt_pct(raw['win_pct'])}"
    )

    print(
        f"  Raw Ø ROI: "
        f"{fmt_pct(raw['avg'])}"
    )

    print(
        f"  Raw Median: "
        f"{fmt_pct(raw['median'])}"
    )

    print()
    print(
        "SCHRITT 2 – SIGNAL CLUSTER"
    )

    print(
        f"  Cluster: "
        f"{result['cluster_count']}"
    )

    print(
        f"  Ø Clustergröße: "
        f"{fmt_num(result['avg_cluster_size'])}"
    )

    print(
        f"  Median Clustergröße: "
        f"{fmt_num(result['median_cluster_size'])}"
    )

    print(
        f"  Größter Cluster: "
        f"{result['max_cluster_size']}"
    )

    print()
    print(
        "SCHRITT 3 – SIGNAL-POSITION IM CLUSTER"
    )

    print()

    print(
        f"{'#':>4} "
        f"{'N':>8} "
        f"{'Win%':>10} "
        f"{'Ø ROI':>12} "
        f"{'Median':>12}"
    )

    print("-" * 55)

    for rank in [
        "1",
        "2",
        "3",
        "4",
        "5+",
    ]:

        stats = result[
            "rank_stats"
        ][rank]

        print(
            f"{rank:>4} "
            f"{stats['n']:>8d} "
            f"{fmt_pct(stats['win_pct']):>10} "
            f"{fmt_pct(stats['avg']):>12} "
            f"{fmt_pct(stats['median']):>12}"
        )

    print()
    print(
        "SCHRITT 4 – TATSÄCHLICH HANDELBARE TRADES"
    )

    print(
        f"  Accepted Trades: "
        f"{accepted['n']}"
    )

    print(
        f"  Ø Accepted ROI: "
        f"{fmt_pct(accepted['avg'])}"
    )

    print(
        f"  Win%: "
        f"{fmt_pct(accepted['win_pct'])}"
    )

    print(
        f"  Median: "
        f"{fmt_pct(accepted['median'])}"
    )

    print(
        f"  Compound: "
        f"{fmt_pct(accepted['compound'])}"
    )

    print(
        f"  Best: "
        f"{fmt_pct(accepted['best'])}"
    )

    print(
        f"  Worst: "
        f"{fmt_pct(accepted['worst'])}"
    )

    print()
    print(
        "SCHRITT 5 – WAS WURDE DURCH OVERLAP VERWORFEN?"
    )

    print(
        f"  Skipped Signals: "
        f"{skipped['n']}"
    )

    print(
        f"  Skip-Rate: "
        f"{fmt_pct(result['skipped_pct'])}"
    )

    print(
        f"  Ø ROI skipped: "
        f"{fmt_pct(skipped['avg'])}"
    )

    print(
        f"  Win% skipped: "
        f"{fmt_pct(skipped['win_pct'])}"
    )

    print(
        f"  Accepted Anteil: "
        f"{fmt_pct(result['accepted_pct'])}"
    )

    print()
    print(
        "SCHRITT 6 – JAHR / MARKTPHASE"
    )

    if not result[
        "yearly_stats"
    ]:

        print(
            "  Keine Trades."
        )

    else:

        print()

        print(
            f"{'Jahr':>8} "
            f"{'Trades':>8} "
            f"{'Win%':>10} "
            f"{'Ø ROI':>12} "
            f"{'Compound':>13}"
        )

        print("-" * 58)

        for year, stats in (
            result[
                "yearly_stats"
            ].items()
        ):

            print(
                f"{year:>8} "
                f"{stats['n']:>8d} "
                f"{fmt_pct(stats['win_pct']):>10} "
                f"{fmt_pct(stats['avg']):>12} "
                f"{fmt_pct(stats['compound']):>13}"
            )

        print()
        print(
            f"  Positive Jahre: "
            f"{result['positive_years']}/"
            f"{len(result['yearly_stats'])}"
        )

        print(
            f"  Bestes Jahr: "
            f"{fmt_pct(result['best_year'])}"
        )

        print(
            f"  Schlechtestes Jahr: "
            f"{fmt_pct(result['worst_year'])}"
        )


# ============================================================
# PRINT HOLD TABLE
# ============================================================

def print_hold_table(
    hold_results: Dict[str, Dict],
    buy_hold: float,
) -> None:

    print()

    print(
        f"{'Hold':>6} "
        f"{'Raw':>7} "
        f"{'Trade':>8} "
        f"{'Win%':>9} "
        f"{'Ø Trade':>11} "
        f"{'Compound':>13} "
        f"{'vs B&H':>11} "
        f"{'Skip%':>9} "
        f"{'Skip Ø':>11} "
        f"{'Worst Yr':>11}"
    )

    print("-" * 120)

    for hold_name, result in (
        hold_results.items()
    ):

        raw = result[
            "raw_stats"
        ]

        accepted = result[
            "accepted_stats"
        ]

        skipped = result[
            "skipped_stats"
        ]

        compound = accepted[
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
            f"{raw['n']:>7d} "
            f"{accepted['n']:>8d} "
            f"{fmt_pct(accepted['win_pct']):>9} "
            f"{fmt_pct(accepted['avg']):>11} "
            f"{fmt_pct(compound):>13} "
            f"{fmt_pct(vs_bh):>11} "
            f"{fmt_pct(result['skipped_pct']):>9} "
            f"{fmt_pct(skipped['avg']):>11} "
            f"{fmt_pct(result['worst_year']):>11}"
        )


# ============================================================
# VERDICT
# ============================================================

def create_verdict(
    result: Dict,
    buy_hold: float,
) -> List[str]:

    messages = []

    raw = result[
        "raw_stats"
    ]

    accepted = result[
        "accepted_stats"
    ]

    skipped = result[
        "skipped_stats"
    ]

    # --------------------------------------------------------
    # 1. Edge allgemein
    # --------------------------------------------------------

    if (
        math.isfinite(
            accepted["avg"]
        )
        and accepted["avg"] > 0
    ):

        messages.append(
            "Der handelbare Trade-Ø ist positiv."
        )

    else:

        messages.append(
            "Der handelbare Trade-Ø ist nicht positiv."
        )

    # --------------------------------------------------------
    # 2. Overlap
    # --------------------------------------------------------

    if (
        math.isfinite(
            skipped["avg"]
        )
        and math.isfinite(
            accepted["avg"]
        )
    ):

        if (
            skipped["avg"]
            > accepted["avg"]
            + 0.10
        ):

            messages.append(
                "Die verworfenen Overlap-Signale "
                "sind im Durchschnitt stärker als "
                "die tatsächlich gehandelten Trades."
            )

        elif (
            skipped["avg"]
            < accepted["avg"]
            - 0.10
        ):

            messages.append(
                "Der Non-Overlap scheint sinnvoll: "
                "spätere Signale sind schwächer."
            )

        else:

            messages.append(
                "Die verworfenen Signale haben "
                "eine ähnliche Qualität wie die "
                "handelbaren Trades."
            )

    # --------------------------------------------------------
    # 3. Signal Decay
    # --------------------------------------------------------

    rank1 = result[
        "rank_stats"
    ]["1"]["avg"]

    rank2 = result[
        "rank_stats"
    ]["2"]["avg"]

    rank3 = result[
        "rank_stats"
    ]["3"]["avg"]

    if (
        math.isfinite(rank1)
        and math.isfinite(rank2)
        and math.isfinite(rank3)
    ):

        if (
            rank2 < rank1
            and rank3 < rank2
        ):

            messages.append(
                "Der Edge fällt innerhalb "
                "eines Signal-Clusters kontinuierlich ab."
            )

        elif (
            rank2 > rank1
            or rank3 > rank2
        ):

            messages.append(
                "Der Edge bleibt bei späteren "
                "Signalen erhalten oder wird teilweise stärker."
            )

        else:

            messages.append(
                "Kein klarer Signal-Decay innerhalb "
                "der Cluster erkennbar."
            )

    # --------------------------------------------------------
    # 4. Year Robustness
    # --------------------------------------------------------

    if result[
        "yearly_stats"
    ]:

        years = len(
            result[
                "yearly_stats"
            ]
        )

        positive = result[
            "positive_years"
        ]

        if positive == years:

            messages.append(
                "In allen verfügbaren Jahren "
                "war der Compound positiv."
            )

        elif positive == 0:

            messages.append(
                "In keinem verfügbaren Jahr "
                "war der Compound positiv."
            )

        else:

            messages.append(
                f"Die Strategie ist jahrabhängig: "
                f"{positive}/{years} Jahre positiv."
            )

    # --------------------------------------------------------
    # 5. Buy & Hold
    # --------------------------------------------------------

    if (
        math.isfinite(
            accepted["compound"]
        )
        and math.isfinite(
            buy_hold
        )
    ):

        if (
            accepted["compound"]
            > buy_hold
        ):

            messages.append(
                "Compound liegt über Buy & Hold."
            )

        else:

            messages.append(
                "Compound liegt unter Buy & Hold."
            )

    return messages


# ============================================================
# GLOBAL SUMMARY
# ============================================================

def build_global_summary_rows(
    all_results: List[Dict],
    buy_hold: float,
) -> pd.DataFrame:

    rows = []

    for item in all_results:

        result = item[
            "result"
        ]

        accepted = result[
            "accepted_stats"
        ]

        raw = result[
            "raw_stats"
        ]

        skipped = result[
            "skipped_stats"
        ]

        compound = accepted[
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

        rows.append(
            {
                "window_min":
                    item["window_minutes"],

                "threshold_pct":
                    item["threshold_pct"],

                "quorum":
                    item["quorum"],

                "hold_h":
                    item["hold_hours"],

                "raw_signals":
                    raw["n"],

                "trades":
                    accepted["n"],

                "win_pct":
                    accepted["win_pct"],

                "avg_roi_pct":
                    accepted["avg"],

                "median_roi_pct":
                    accepted["median"],

                "compound_pct":
                    compound,

                "vs_bh_pct":
                    vs_bh,

                "best_pct":
                    accepted["best"],

                "worst_pct":
                    accepted["worst"],

                "skip_pct":
                    result[
                        "skipped_pct"
                    ],

                "skipped_avg_pct":
                    skipped["avg"],

                "worst_year_pct":
                    result[
                        "worst_year"
                    ],

                "positive_years":
                    result[
                        "positive_years"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT GLOBAL TOP RESULTS
# ============================================================

def print_global_top(
    summary_df: pd.DataFrame,
) -> None:

    if summary_df.empty:
        return

    valid = summary_df[
        summary_df[
            "compound_pct"
        ].notna()
    ].copy()

    if valid.empty:
        return

    valid = valid.sort_values(
        "compound_pct",
        ascending=False,
    ).head(15)

    print()
    print("=" * 120)

    print(
        "SCHRITT 7 – TOP-ERGEBNISSE ÜBER ALLE VARIANTEN"
    )

    print("=" * 120)

    print()

    print(
        f"{'Event':>7} "
        f"{'Base':>7} "
        f"{'Q':>5} "
        f"{'Hold':>6} "
        f"{'Trades':>8} "
        f"{'Win%':>9} "
        f"{'Ø ROI':>11} "
        f"{'Compound':>13} "
        f"{'vs B&H':>11} "
        f"{'Worst Yr':>11}"
    )

    print("-" * 120)

    for _, row in valid.iterrows():

        print(
            f"{int(row['window_min']):>4}m "
            f"{row['threshold_pct']:>6.1f}% "
            f"{int(row['quorum']):>2}/"
            f"{int(row['quorum']) if False else ''}"
        )

        # Neu formatieren in zweiter Zeile nicht nötig.
        # Ein kompakter eigener Print folgt weiter unten.
        break

    for _, row in valid.iterrows():

        q = int(
            row["quorum"]
        )

        # Gesamtzahl Refs kann aus dem aktuellen globalen
        # Setup entnommen werden.
        q_text = (
            f"{q}/{len(REF_SYMBOLS)}"
        )

        print(
            f"{str(int(row['window_min'])) + 'm':>7} "
            f"{row['threshold_pct']:>6.1f}% "
            f"{q_text:>5} "
            f"{str(int(row['hold_h'])) + 'h':>6} "
            f"{int(row['trades']):>8d} "
            f"{fmt_pct(row['win_pct']):>9} "
            f"{fmt_pct(row['avg_roi_pct']):>11} "
            f"{fmt_pct(row['compound_pct']):>13} "
            f"{fmt_pct(row['vs_bh_pct']):>11} "
            f"{fmt_pct(row['worst_year_pct']):>11}"
        )


# ============================================================
# BEST RESULT DETAIL
# ============================================================

def print_best_result_detail(
    all_results: List[Dict],
    buy_hold: float,
) -> None:

    valid = [
        item
        for item in all_results
        if math.isfinite(
            item["result"][
                "accepted_stats"
            ]["compound"]
        )
    ]

    if not valid:
        return

    best = max(
        valid,
        key=lambda item:
            item["result"][
                "accepted_stats"
            ]["compound"]
    )

    result = best[
        "result"
    ]

    print()
    print("=" * 120)

    print(
        "DETAIL-AUSWERTUNG DES BESTEN COMPOUND-RESULTATS"
    )

    print("=" * 120)

    print(
        f"Event Window: "
        f"{best['window_minutes']}m"
    )

    print(
        f"Base Threshold: "
        f"{best['threshold_pct']:.1f}%"
    )

    print(
        f"Quorum: "
        f"{best['quorum']}/{len(REF_SYMBOLS)}"
    )

    print(
        f"Hold: "
        f"{best['hold_hours']}h"
    )

    print_diagnostic(
        result
    )

    print()
    print(
        "AUTOMATISCHE BEWERTUNG"
    )

    print()

    for idx, message in enumerate(
        create_verdict(
            result,
            buy_hold,
        ),
        start=1,
    ):

        print(
            f"{idx}. {message}"
        )


# ============================================================
# DEBUG SAMPLE EVENTS
# ============================================================

def print_sample_events(
    events: Sequence[Event],
) -> None:

    if not events:
        return

    print()
    print(
        "BEISPIEL-SIGNALE"
    )

    print()

    for idx, event in enumerate(
        events[:5],
        start=1,
    ):

        refs = " | ".join(
            f"{ref} "
            f"{ret * 100:+.2f}%"
            for ref, ret
            in event.ref_returns.items()
        )

        print(
            f"{idx}. "
            f"{format_berlin(event.end_ms)} | "
            f"Refs: {refs}"
        )

        print(
            f"   Quorum: "
            f"{event.ref_count_above_threshold}/"
            f"{len(REF_SYMBOLS)} | "
            f"Base: "
            f"{event.base_threshold_pct:.2f}% | "
            f"Effektiv: "
            f"{event.effective_threshold_pct:.2f}% | "
            f"Vol: "
            f"{event.volatility_factor:.2f}x"
        )


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

    TARGET_SYMBOL = normalize_symbol(
        args.target
    )

    REF_SYMBOLS = normalize_refs(
        args.refs.split(",")
    )

    if len(REF_SYMBOLS) < 2:

        raise ValueError(
            "Mindestens zwei Refs erforderlich."
        )

    # Target darf keine Ref sein.
    target_base = TARGET_SYMBOL[
        :-4
    ]

    if target_base in REF_SYMBOLS:

        raise ValueError(
            f"Target {TARGET_SYMBOL} "
            f"darf nicht gleichzeitig Ref sein."
        )

    # --------------------------------------------------------
    # Quorums
    # --------------------------------------------------------

    if args.quorums:

        QUORUMS = sorted(
            set(
                int(q.strip())
                for q
                in args.quorums.split(",")
                if q.strip()
            )
        )

    else:

        n_refs = len(
            REF_SYMBOLS
        )

        if n_refs >= 3:

            QUORUMS = [
                n_refs - 1,
                n_refs,
            ]

        else:

            QUORUMS = [
                n_refs
            ]

    for quorum in QUORUMS:

        if (
            quorum < 1
            or quorum > len(
                REF_SYMBOLS
            )
        ):

            raise ValueError(
                f"Quorum {quorum} ist bei "
                f"{len(REF_SYMBOLS)} Refs ungültig."
            )

    # --------------------------------------------------------
    # Parameter
    # --------------------------------------------------------

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

    FROM_DATE = (
        args.from_date
    )

    TO_DATE = (
        args.to_date
    )

    if THRESHOLD_INTENSITY <= 0:

        raise ValueError(
            "Intensity muss > 0 sein."
        )

    if VOLATILITY_POWER < 0:

        raise ValueError(
            "Volatility Power muss >= 0 sein."
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

    target_end_ms = (
        end_ms
        + max_hold_hours
        * 60
        * 60
        * 1000
    )

    # --------------------------------------------------------
    # References laden
    # --------------------------------------------------------

    session = create_session()

    try:

        print()
        print(
            "Lade Reference-Coins..."
        )

        ref_fetch_start_ms = start_ms - vol_warmup_ms()

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
                ref_fetch_start_ms,
                end_ms,
            )

            refs_data[
                ref
            ] = series

            print(
                f"  {len(series.times)} Kerzen"
            )

    finally:

        session.close()

    # --------------------------------------------------------
    # Target laden
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
            target_end_ms,
        )

    finally:

        target_session.close()

    print(
        f"  {len(target.times)} Kerzen"
    )

    # --------------------------------------------------------
    # B&H
    # --------------------------------------------------------

    buy_hold = calculate_buy_hold(
        target,
        start_ms,
        end_ms,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print_header(
        start_ms,
        end_ms,
        buy_hold,
    )

    # --------------------------------------------------------
    # Alle Ergebnisse
    # --------------------------------------------------------

    # Effektiver Zeitraum fuer die Block-Header (Refs + Target vorhanden).
    block_period = _block_range(*_effective_range_ms(
        start_ms,
        end_ms,
        target.times,
        *[s.times for s in refs_data.values()],
    ))

    all_results = []

    # --------------------------------------------------------
    # Szenarien
    # --------------------------------------------------------

    total = (
        len(EVENT_WINDOWS_MIN)
        * len(EVENT_THRESHOLDS_PCT)
        * len(QUORUMS)
    )

    counter = 0

    for window_minutes in (
        EVENT_WINDOWS_MIN
    ):

        for threshold_pct in (
            EVENT_THRESHOLDS_PCT
        ):

            for quorum in QUORUMS:

                counter += 1

                print()
                print("=" * 120)

                print(
                    f"{TARGET_SYMBOL} | "
                    f"{window_minutes}m | "
                    f"Base "
                    f"{threshold_pct:.1f}% | "
                    f"Quorum "
                    f"{quorum}/"
                    f"{len(REF_SYMBOLS)} | "
                    f"{DIRECTION} | {EVENT_MODE} | "
                    f"{block_period}"
                )

                print("=" * 120)

                events = create_events(
                    refs_data,
                    window_minutes,
                    threshold_pct,
                    quorum,
                )

                # Warmup-Kerzen liefern nur Vol-Historie: Events erst ab FROM_DATE.
                events = [
                    e for e in events
                    if e.end_ms >= start_ms
                ]

                print(
                    f"Raw Event-Trigger: "
                    f"{len(events)}"
                )

                if args.debug_events:

                    print_sample_events(
                        events
                    )

                hold_results = {}

                for (
                    hold_name,
                    hold_hours,
                ) in HOLD_HORIZONS:

                    result = analyze_hold(
                        events,
                        target,
                        hold_hours,
                    )

                    hold_results[
                        hold_name
                    ] = result

                    all_results.append(
                        {
                            "window_minutes":
                                window_minutes,
                            "threshold_pct":
                                threshold_pct,
                            "quorum":
                                quorum,
                            "hold_hours":
                                hold_hours,
                            "result":
                                result,
                        }
                    )

                print_hold_table(
                    hold_results,
                    buy_hold,
                )

                print()
                print(
                    f"Fortschritt: "
                    f"{counter}/{total}"
                )

    # --------------------------------------------------------
    # GLOBAL SUMMARY
    # --------------------------------------------------------

    summary_df = (
        build_global_summary_rows(
            all_results,
            buy_hold,
        )
    )

    print_global_top(
        summary_df
    )

    # --------------------------------------------------------
    # BEST DETAIL
    # --------------------------------------------------------

    print_best_result_detail(
        all_results,
        buy_hold,
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if args.csv:

        summary_df.to_csv(
            args.csv,
            index=False,
        )

        print()
        print(
            f"CSV gespeichert: "
            f"{args.csv}"
        )

    # --------------------------------------------------------
    # FINAL
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
        f"Refs: "
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

    print()
    print(
        "Die Analyse trennt bewusst:"
    )

    print(
        "  1. Qualität aller Raw Signals"
    )

    print(
        "  2. Signal-Cluster und Overlap"
    )

    print(
        "  3. Qualität des 1./2./3./4./5+ Signals"
    )

    print(
        "  4. tatsächlich handelbare Trades"
    )

    print(
        "  5. durch Non-Overlap verworfene Signals"
    )

    print(
        "  6. Performance nach Jahren"
    )

    print(
        "  7. Gesamtvergleich aller Varianten"
    )

    print(
        "Wichtig: Die Raw-Signal-Statistik ist "
        "keine handelbare Performance."
    )


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Multi-Ref Strength Single Target "
            "Diagnostic"
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
        help=(
            "Startdatum YYYY-MM-DD"
        ),
    )

    parser.add_argument(
        "--to-date",
        default=TO_DATE,
        help=(
            "Enddatum YYYY-MM-DD"
        ),
    )

    parser.add_argument(
        "--debug-events",
        action="store_true",
        help=(
            "Zeigt Beispiel-Events"
        ),
    )

    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Speichert globale Summary "
            "als CSV"
        ),
    )

    return parser.parse_args()


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