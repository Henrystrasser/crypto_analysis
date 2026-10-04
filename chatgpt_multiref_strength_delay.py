#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chatgpt_multiref_strength_delay.py

MULTI-REF STRENGTH
SINGLE TARGET
VOLATILITY-ADAPTIVER THRESHOLD
+ ENTRY DELAY TEST

============================================================
IDEE
============================================================

Die eigentliche Signaldefinition bleibt unverändert.

Mehrere Referenz-Coins werden beobachtet.

Beispiel mit 4 Refs:

    BTC
    ETH
    SOL
    BNB

Wenn mindestens X/4 Refs innerhalb des Event-Fensters
den dynamischen Threshold brechen:

    -> Signal

Danach testen wir unterschiedliche Entry Delays
(Default ENTRY_DELAYS_MIN):

    0m
    60m
    120m
    300m
    600m

============================================================
BEISPIEL
============================================================

Signal-Kerze schließt um 10:00
(alle Zeiten = Kerzen-CLOSE, Europe/Berlin)

Delay 0m:

    Entry 10:00 (Close der Signal-Kerze)
    Exit  + Hold

Delay 60m:

    Entry 11:00
    Exit  + Hold

Delay 120m:

    Entry 12:00
    Exit  + Hold

Der Hold beginnt immer beim tatsächlichen Entry.

============================================================
VOLATILITY-ADAPTIVER THRESHOLD
============================================================

    effective_threshold =
        base_threshold
        * volatility_factor
        * THRESHOLD_INTENSITY

============================================================
MULTI-REF STRENGTH
============================================================

Es werden weiterhin KEINE Renditen addiert.

Wir zählen:

    Wie viele Refs überschreiten
    den jeweiligen Threshold?

Beispiel:

    BTC +1.4%
    ETH +1.2%
    SOL +0.6%
    BNB +1.3%

Threshold = 1.0%

=> 3/4

============================================================
ENTRY DELAY
============================================================

Getestet werden standardmäßig (ENTRY_DELAYS_MIN):

    0m
    60m
    120m
    300m
    600m

Der Delay verändert NICHT:

    - Event Window
    - Threshold
    - Quorum

Er verändert ausschließlich den tatsächlichen Entry-Zeitpunkt.

============================================================
NON-OVERLAP
============================================================

GLOBAL pro Hold.

Wir blockieren ab dem tatsächlichen Entry:

    actual_entry + hold

Ein neues Event wird nur berücksichtigt,
wenn sein Signalzeitpunkt nach diesem Block
liegt.

WICHTIG:

Ein Event mit Signal vor dem tatsächlichen Entry
eines bereits reservierten Trades wird ebenfalls
nicht zusätzlich gehandelt.

Damit bleibt das Verhalten eines einzelnen
Tradingbots realistisch.

============================================================
COMPOUND
============================================================

Nach jedem abgeschlossenen Trade wird das komplette
Kapital wieder eingesetzt.

============================================================
BUY & HOLD
============================================================

Buy & Hold des einzelnen Targetcoins
über den gesamten Backtest-Zeitraum.

============================================================
KEIN LOOKAHEAD
============================================================

Die Volatilitätsbaseline wird mit SHIFT(1)
berechnet.

Die Signalentscheidung verwendet nur Informationen
bis zum Signalzeitpunkt.

Der Delay darf natürlich zukünftige Kursdaten für
den tatsächlichen Entry verwenden – genau das ist
der Sinn des Tests.

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

TARGET_SYMBOL = "BCHUSDT"


# ============================================================
# REFERENCES
# ============================================================

# Frei konfigurierbar.
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
    "BNB",
    "XRP"
]


# ============================================================
# QUORUM
# ============================================================

# Standard:
# 3/4
#
# Du kannst per CLI z.B. auch:
#
# --quorum 4
#
# setzen.

BASE_QUORUM = 3


# ============================================================
# ZEITRAUM
# ============================================================

FROM_DATE = "2024-01-01"

TO_DATE = None


# ============================================================
# EVENT WINDOWS
# ============================================================

EVENT_WINDOWS_MIN = [
    30,
    60,
    120,
]


# ============================================================
# BASE THRESHOLDS
# ============================================================

EVENT_THRESHOLDS_PCT = [
    1.0,
    2.0,
]


# ============================================================
# ENTRY DELAYS
# ============================================================

# DAS ist der Test.

ENTRY_DELAYS_MIN = [
    0,
    60,
    120,
    300,
    600,
]


# ============================================================
# THRESHOLD INTENSITY
# ============================================================

# 1.00 = neutral
# 0.80 = lockerer
# 1.20 = strenger

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

# 1.0 = linear
VOLATILITY_POWER = 1.0

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

    signal_ms: int

    ref_returns: Dict[str, float]

    ref_count_above_threshold: int

    base_threshold_pct: float

    effective_threshold_pct: float

    volatility_factor: float


@dataclass
class Trade:

    event: Event

    entry_ms: int

    roi: float


# ============================================================
# HTTP
# ============================================================

def create_session() -> requests.Session:

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                "chatgpt-multiref-strength-delay/1.0",
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

        value = (
            ref.strip()
            .upper()
        )

        if value:
            result.append(value)

    return list(
        dict.fromkeys(
            result
        )
    )


# ============================================================
# DATE / TIME
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


def resolve_period() -> Tuple[
    int,
    int,
]:

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
    refs_data: Dict[
        str,
        PriceSeries,
    ],
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

    # Median statt Mittelwert.
    # Ein einzelner Ausreißer-Ref dominiert
    # den Markt-Volatilitätsfaktor dadurch weniger.

    frame[
        "market_vol"
    ] = frame[
        vol_columns
    ].median(
        axis=1
    )

    # Historische Normalvolatilität.
    #
    # SHIFT(1):
    # der aktuelle Wert darf seine eigene
    # Baseline nicht beeinflussen.

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
        /
        frame[
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
# EVENTS
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
    refs_data: Dict[
        str,
        PriceSeries,
    ],
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
    # DYNAMISCHER THRESHOLD
    # --------------------------------------------------------

    frame[
        "effective_threshold_pct"
    ] = (
        base_threshold_pct
        * frame[
            "vol_factor"
        ]
        * THRESHOLD_INTENSITY
    )

    # --------------------------------------------------------
    # REF RETURNS
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
    # COUNT DER REFS ÜBER THRESHOLD
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ONLY FIRST ENTRY INTO CONDITION
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
                signal_ms=int(
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


# ============================================================
# SIMULATION
# ============================================================

def simulate(
    events: Sequence[Event],
    target: PriceSeries,
    hold_hours: int,
    entry_delay_minutes: int,
) -> List[Trade]:

    hold_ms = (
        hold_hours
        * 60
        * 60
        * 1000
    )

    delay_ms = (
        entry_delay_minutes
        * 60
        * 1000
    )

    trades = []

    # Ende des aktuell reservierten Trades.
    blocked_until_ms: Optional[
        int
    ] = None

    for event in events:

        actual_entry_ms = (
            event.signal_ms
            + delay_ms
        )

        # ----------------------------------------------------
        # GLOBALER NON-OVERLAP
        # ----------------------------------------------------
        #
        # Der Bot hat sich ab dem Signal bereits
        # für den Trade entschieden.
        #
        # Deshalb darf ein neues Signal, das in die
        # laufende Reservation / Position fällt,
        # keinen zweiten Trade auslösen.
        # ----------------------------------------------------

        if (
            blocked_until_ms is not None
            and event.signal_ms
            < blocked_until_ms
        ):

            continue

        entry_price = (
            target.exact_price(
                actual_entry_ms
            )
        )

        if entry_price is None:

            continue

        exit_ms = (
            actual_entry_ms
            + hold_ms
        )

        exit_price = (
            target.exact_price(
                exit_ms
            )
        )

        if exit_price is None:

            continue

        roi = calculate_net_roi(
            entry_price,
            exit_price,
        )

        if not math.isfinite(
            roi
        ):

            continue

        trades.append(
            Trade(
                event=event,
                entry_ms=actual_entry_ms,
                roi=roi,
            )
        )

        blocked_until_ms = (
            actual_entry_ms
            + hold_ms
        )

    return trades


# ============================================================
# STATS
# ============================================================

def calculate_stats(
    trades: Sequence[Trade],
) -> Dict[str, float]:

    if not trades:

        return {
            "n": 0,
            "win_pct": float("nan"),
            "avg": float("nan"),
            "median": float("nan"),
            "compound": float("nan"),
            "best": float("nan"),
            "worst": float("nan"),
        }

    rois = np.array(
        [
            trade.roi
            for trade in trades
        ],
        dtype=np.float64,
    )

    return {
        "n": int(
            len(rois)
        ),

        "win_pct": float(
            np.mean(
                rois > 0
            )
            * 100.0
        ),

        "avg": float(
            np.mean(rois)
            * 100.0
        ),

        "median": float(
            np.median(rois)
            * 100.0
        ),

        "compound": float(
            (
                np.prod(
                    1.0 + rois
                )
                - 1.0
            )
            * 100.0
        ),

        "best": float(
            np.max(rois)
            * 100.0
        ),

        "worst": float(
            np.min(rois)
            * 100.0
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
# OUTPUT
# ============================================================

def fmt_pct(
    value: float,
) -> str:

    if not math.isfinite(
        value
    ):

        return "n/a"

    return f"{value:9.2f}%"


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
        "MULTI-REF STRENGTH | "
        "SINGLE TARGET | ENTRY DELAY TEST"
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
        f"Quorum: "
        f"{BASE_QUORUM}/{len(REF_SYMBOLS)}"
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
        f"Entry Delays: "
        f"{', '.join(str(x) + 'm' for x in ENTRY_DELAYS_MIN)}"
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
        f"Buy&Hold: "
        f"{fmt_pct(buy_hold)}"
    )

    print("=" * 120)


def print_scenario_table(
    rows: Sequence[
        Tuple[
            int,
            Dict[str, float],
        ]
    ],
    buy_hold: float,
) -> None:

    print()

    print(
        f"{'Delay':>7} "
        f"{'Hold':>6} "
        f"{'Trades':>8} "
        f"{'Win%':>9} "
        f"{'Ø ROI':>11} "
        f"{'Median':>11} "
        f"{'Compound':>13} "
        f"{'vs B&H':>11} "
        f"{'Best':>10} "
        f"{'Worst':>10}"
    )

    print("-" * 120)

    for delay, hold_name, stats in rows:

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
            f"{str(delay) + 'm':>7} "
            f"{hold_name:>6} "
            f"{stats['n']:>8d} "
            f"{fmt_pct(stats['win_pct']):>9} "
            f"{fmt_pct(stats['avg']):>11} "
            f"{fmt_pct(stats['median']):>11} "
            f"{fmt_pct(stats['compound']):>13} "
            f"{fmt_pct(vs_bh):>11} "
            f"{fmt_pct(stats['best']):>10} "
            f"{fmt_pct(stats['worst']):>10}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global TARGET_SYMBOL
    global REF_SYMBOLS
    global BASE_QUORUM
    global THRESHOLD_INTENSITY
    global VOLATILITY_POWER
    global FROM_DATE
    global TO_DATE
    global ENTRY_DELAYS_MIN

    args = parse_args()

    TARGET_SYMBOL = normalize_symbol(
        args.target
    )

    REF_SYMBOLS = normalize_refs(
        args.refs.split(",")
    )

    if len(REF_SYMBOLS) < 2:

        raise ValueError(
            "Mindestens 2 Refs erforderlich."
        )

    BASE_QUORUM = (
        int(args.quorum)
    )

    if not (
        1
        <= BASE_QUORUM
        <= len(REF_SYMBOLS)
    ):

        raise ValueError(
            f"Quorum {BASE_QUORUM} ungültig."
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
            "Intensity muss > 0 sein."
        )

    if VOLATILITY_POWER < 0:

        raise ValueError(
            "Volatility Power muss >= 0 sein."
        )

    FROM_DATE = (
        args.from_date
    )

    TO_DATE = (
        args.to_date
    )

    if args.delays:

        ENTRY_DELAYS_MIN = sorted(
            set(
                int(x.strip())
                for x
                in args.delays.split(",")
                if x.strip()
            )
        )

    for delay in ENTRY_DELAYS_MIN:

        if delay < 0:

            raise ValueError(
                "Entry Delay darf nicht negativ sein."
            )

        if (
            delay
            % INTERVAL_MINUTES
            != 0
        ):

            raise ValueError(
                f"Entry Delay {delay}m muss "
                f"ein Vielfaches von "
                f"{INTERVAL_MINUTES}m sein."
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

    max_delay_min = max(
        ENTRY_DELAYS_MIN
    )

    target_end_ms = (
        end_ms
        + max_delay_min
        * 60
        * 1000
        + max_hold_hours
        * 60
        * 60
        * 1000
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    # Temporär noch nicht bekannt.
    # Buy&Hold wird erst nach dem Target-Download
    # berechnet.

    # --------------------------------------------------------
    # References
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
            target_end_ms,
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

    print_header(
        start_ms,
        end_ms,
        buy_hold,
    )

    # --------------------------------------------------------
    # GLOBAL RESULTS
    # --------------------------------------------------------

    # Effektiver Zeitraum fuer die Block-Header (Refs + Target vorhanden).
    block_period = _block_range(*_effective_range_ms(
        start_ms,
        end_ms,
        target.times,
        *[s.times for s in refs_data.values()],
    ))

    global_rows = []

    total_scenarios = (
        len(EVENT_WINDOWS_MIN)
        * len(EVENT_THRESHOLDS_PCT)
    )

    scenario_counter = 0

    # --------------------------------------------------------
    # EVENT SCENARIOS
    # --------------------------------------------------------

    for window_minutes in (
        EVENT_WINDOWS_MIN
    ):

        for base_threshold_pct in (
            EVENT_THRESHOLDS_PCT
        ):

            scenario_counter += 1

            print()
            print("=" * 120)

            print(
                f"Event "
                f"{window_minutes}m | "
                f"Base "
                f"{base_threshold_pct:.1f}% | "
                f"Quorum "
                f"{BASE_QUORUM}/"
                f"{len(REF_SYMBOLS)} | "
                f"{DIRECTION} | {EVENT_MODE} | "
                f"{block_period}"
            )

            print("=" * 120)

            events = create_events(
                refs_data,
                window_minutes,
                base_threshold_pct,
                BASE_QUORUM,
            )

            # Warmup-Kerzen liefern nur Vol-Historie: Events erst ab FROM_DATE.
            events = [
                e for e in events
                if e.signal_ms >= start_ms
            ]

            print(
                f"Raw Events: "
                f"{len(events)}"
            )

            if events:

                threshold_values = np.array(
                    [
                        event.effective_threshold_pct
                        for event
                        in events
                    ],
                    dtype=np.float64,
                )

                vol_values = np.array(
                    [
                        event.volatility_factor
                        for event
                        in events
                    ],
                    dtype=np.float64,
                )

                print(
                    f"Effective Threshold: "
                    f"Ø "
                    f"{np.mean(threshold_values):.2f}% | "
                    f"Min "
                    f"{np.min(threshold_values):.2f}% | "
                    f"Max "
                    f"{np.max(threshold_values):.2f}%"
                )

                print(
                    f"Vol Factor: "
                    f"Ø "
                    f"{np.mean(vol_values):.2f}x | "
                    f"Min "
                    f"{np.min(vol_values):.2f}x | "
                    f"Max "
                    f"{np.max(vol_values):.2f}x"
                )

            # ------------------------------------------------
            # Holds + Delays
            # ------------------------------------------------

            rows = []

            for (
                hold_name,
                hold_hours,
            ) in HOLD_HORIZONS:

                for delay in (
                    ENTRY_DELAYS_MIN
                ):

                    trades = simulate(
                        events,
                        target,
                        hold_hours,
                        delay,
                    )

                    stats = calculate_stats(
                        trades
                    )

                    rows.append(
                        (
                            delay,
                            hold_name,
                            stats,
                        )
                    )

                    global_rows.append(
                        {
                            "window_min":
                                window_minutes,

                            "base_threshold_pct":
                                base_threshold_pct,

                            "quorum":
                                BASE_QUORUM,

                            "delay_min":
                                delay,

                            "hold_h":
                                hold_hours,

                            "trades":
                                stats["n"],

                            "win_pct":
                                stats["win_pct"],

                            "avg_roi_pct":
                                stats["avg"],

                            "median_roi_pct":
                                stats["median"],

                            "compound_pct":
                                stats["compound"],

                            "best_pct":
                                stats["best"],

                            "worst_pct":
                                stats["worst"],
                        }
                    )

            print_scenario_table(
                rows,
                buy_hold,
            )

            print()
            print(
                f"Fortschritt: "
                f"{scenario_counter}/"
                f"{total_scenarios}"
            )

    # --------------------------------------------------------
    # GLOBAL TOP RESULTS
    # --------------------------------------------------------

    df = pd.DataFrame(
        global_rows
    )

    valid = df[
        df[
            "compound_pct"
        ].notna()
    ].copy()

    if not valid.empty:

        valid = valid.sort_values(
            "compound_pct",
            ascending=False,
        )

        print()
        print("=" * 120)

        print(
            "TOP ERGEBNISSE – ENTRY DELAY"
        )

        print("=" * 120)

        print()

        print(
            f"{'Event':>7} "
            f"{'Base':>7} "
            f"{'Q':>5} "
            f"{'Delay':>7} "
            f"{'Hold':>6} "
            f"{'Trades':>8} "
            f"{'Win%':>9} "
            f"{'Ø ROI':>11} "
            f"{'Compound':>13} "
            f"{'Worst':>10}"
        )

        print("-" * 120)

        for _, row in valid.head(
            15
        ).iterrows():

            q_text = (
                f"{int(row['quorum'])}/"
                f"{len(REF_SYMBOLS)}"
            )

            print(
                f"{str(int(row['window_min'])) + 'm':>7} "
                f"{row['base_threshold_pct']:>6.1f}% "
                f"{q_text:>5} "
                f"{str(int(row['delay_min'])) + 'm':>7} "
                f"{str(int(row['hold_h'])) + 'h':>6} "
                f"{int(row['trades']):>8d} "
                f"{fmt_pct(row['win_pct']):>9} "
                f"{fmt_pct(row['avg_roi_pct']):>11} "
                f"{fmt_pct(row['compound_pct']):>13} "
                f"{fmt_pct(row['worst_pct']):>10}"
            )

    # --------------------------------------------------------
    # DELAY-VERGLEICH
    # --------------------------------------------------------

    print()
    print("=" * 120)

    print(
        "DELAY-VERGLEICH"
    )

    print("=" * 120)

    print()

    print(
        "Hier sehen wir, wie stark der Edge "
        "vom sofortigen Einstieg abhängt."
    )

    print()

    # Für jedes Szenario/Hold die Änderung
    # vom sofortigen Entry auf die übrigen ENTRY_DELAYS_MIN ausgeben.
    comparison_rows = []

    group_cols = [
        "window_min",
        "base_threshold_pct",
        "quorum",
        "hold_h",
    ]

    for group_values, group in (
        df.groupby(group_cols)
    ):

        by_delay = {
            int(row["delay_min"]):
                row
            for _, row
            in group.iterrows()
        }

        if 0 not in by_delay:
            continue

        base = by_delay[0]

        for delay in (
            ENTRY_DELAYS_MIN
        ):

            if delay == 0:
                continue

            if delay not in by_delay:
                continue

            delayed = by_delay[
                delay
            ]

            base_cmp = float(
                base[
                    "compound_pct"
                ]
            )

            delay_cmp = float(
                delayed[
                    "compound_pct"
                ]
            )

            delta = (
                delay_cmp
                - base_cmp
            )

            comparison_rows.append(
                {
                    "window":
                        group_values[0],

                    "threshold":
                        group_values[1],

                    "quorum":
                        group_values[2],

                    "hold":
                        group_values[3],

                    "delay":
                        delay,

                    "base_compound":
                        base_cmp,

                    "delay_compound":
                        delay_cmp,

                    "delta":
                        delta,
                }
            )

    if comparison_rows:

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        print()

        print(
            f"{'Event':>7} "
            f"{'Base':>7} "
            f"{'Q':>5} "
            f"{'Hold':>6} "
            f"{'Delay':>7} "
            f"{'0m Cmp':>12} "
            f"{'Delay Cmp':>12} "
            f"{'Delta':>11}"
        )

        print(
            "-" * 100
        )

        for _, row in (
            comparison_df
            .sort_values(
                "delta",
                ascending=False,
            )
            .head(20)
            .iterrows()
        ):

            q_text = (
                f"{int(row['quorum'])}/"
                f"{len(REF_SYMBOLS)}"
            )

            print(
                f"{str(int(row['window'])) + 'm':>7} "
                f"{row['threshold']:>6.1f}% "
                f"{q_text:>5} "
                f"{str(int(row['hold'])) + 'h':>6} "
                f"{str(int(row['delay'])) + 'm':>7} "
                f"{fmt_pct(row['base_compound']):>12} "
                f"{fmt_pct(row['delay_compound']):>12} "
                f"{fmt_pct(row['delta']):>11}"
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
        f"Quorum: "
        f"{BASE_QUORUM}/"
        f"{len(REF_SYMBOLS)}"
    )

    print(
        f"Entry Delays: "
        + ", ".join(
            f"{x}m"
            for x
            in ENTRY_DELAYS_MIN
        )
    )

    print()
    print(
        "Wichtig:"
    )

    print(
        "Der Hold beginnt immer beim "
        "tatsächlichen Entry nach dem Delay."
    )

    print(
        "Die Signaldefinition bleibt für "
        + ", ".join(f"{x}m" for x in ENTRY_DELAYS_MIN)
        + " exakt gleich."
    )

    print(
        "Es wird nur der Einstieg verschoben."
    )

    print(
        "Non-overlap wird anhand des tatsächlichen "
        "Entry-Zeitpunkts und der Hold-Dauer umgesetzt."
    )


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Multi-Ref Strength "
            "Entry Delay Test"
        )
    )

    parser.add_argument(
        "--target",
        default=TARGET_SYMBOL,
        help="Targetcoin, z.B. XRPUSDT",
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
        "--quorum",
        type=int,
        default=BASE_QUORUM,
        help=(
            "Anzahl Refs über Threshold"
        ),
    )

    parser.add_argument(
        "--delays",
        default=None,
        help=(
            "Entry Delays, z.B. "
            "0,60,120,300,600"
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
        help="Startdatum",
    )

    parser.add_argument(
        "--to-date",
        default=TO_DATE,
        help="Enddatum",
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