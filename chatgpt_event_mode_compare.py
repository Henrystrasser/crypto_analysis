import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

API_URL = "https://data-api.binance.vision/api/v3/klines"

TARGET = "ENAUSDT"

REFS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
]

INTERVAL = "1h"
INTERVAL_MINUTES = 60

START_DATE = "2024-01-01"
END_DATE = None


# ============================================================
# FEES
# ============================================================

FEE_PER_SIDE = 0.00075      # 7.5 bps = 0.075%


# ============================================================
# EVENT TESTS
# ============================================================

EVENT_WINDOWS_MIN = [
    60,
    120,
    240,
    360,
]

BASE_THRESHOLDS = [
    0.01,       # 1%
    0.02,       # 2%
]


# ============================================================
# HOLD TESTS
# ============================================================

HOLDS_HOURS = [
    3,
    6,
    12,
    24,
    48,
    60,
    72,
    78,
    84,
    90,
    96,
    120,
    192,
]


# ============================================================
# VOLATILITY ADAPTIVE THRESHOLD
# ============================================================

USE_VOL_ADAPTIVE_THRESHOLD = True

VOL_LOOKBACK_BARS = 96       # 24h
VOL_BASELINE_BARS = 2880     # 30 Tage

VOL_FACTOR_MIN = 0.5
VOL_FACTOR_MAX = 3.0

THRESHOLD_INTENSITY = 1.0


# ============================================================
# TIME CONSTANTS
# ============================================================

MS_PER_MINUTE = 60_000


# ============================================================
# DATE HELPERS
# ============================================================

def date_to_ms(date_str):
    dt = datetime.strptime(
        date_str,
        "%Y-%m-%d",
    ).replace(
        tzinfo=timezone.utc
    )

    return int(dt.timestamp() * 1000)


def get_last_closed_candle_ms():
    now_ms = int(time.time() * 1000)

    interval_ms = (
        INTERVAL_MINUTES * MS_PER_MINUTE
    )

    current_open = (
        now_ms // interval_ms
    ) * interval_ms

    return current_open - interval_ms


# ============================================================
# BINANCE DOWNLOAD
# ============================================================

def fetch_klines(
    symbol,
    start_ms,
    end_ms,
):
    """
    Robuster Binance Downloader mit Retries.
    """

    all_rows = []
    current = start_ms

    session = requests.Session()

    max_retries = 8

    while current < end_ms:

        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }

        rows = None

        for attempt in range(
            1,
            max_retries + 1,
        ):

            try:

                response = session.get(
                    API_URL,
                    params=params,
                    timeout=(10, 60),
                )

                response.raise_for_status()

                rows = response.json()

                break

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:

                if attempt >= max_retries:
                    raise RuntimeError(
                        f"{symbol}: Download nach "
                        f"{max_retries} Versuchen fehlgeschlagen.\n"
                        f"Letzter Fehler: {exc}"
                    ) from exc

                wait_seconds = min(
                    2 ** (attempt - 1),
                    30,
                )

                print(
                    f"      Netzwerkproblem "
                    f"(Versuch {attempt}/{max_retries}) "
                    f"-> Retry in {wait_seconds}s ..."
                )

                time.sleep(wait_seconds)

            except requests.exceptions.HTTPError as exc:

                status = response.status_code

                if status in (
                    418,
                    429,
                    500,
                    502,
                    503,
                    504,
                ):

                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"{symbol}: HTTP {status} "
                            f"nach {max_retries} Versuchen."
                        ) from exc

                    wait_seconds = min(
                        2 ** attempt,
                        60,
                    )

                    print(
                        f"      Binance HTTP {status} "
                        f"(Versuch {attempt}/{max_retries}) "
                        f"-> Retry in {wait_seconds}s ..."
                    )

                    time.sleep(wait_seconds)

                else:
                    raise

        if rows is None:
            raise RuntimeError(
                f"{symbol}: Kein gültiger Download."
            )

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = rows[-1][0]

        next_start = (
            last_open_time
            + INTERVAL_MINUTES * MS_PER_MINUTE
        )

        if next_start <= current:
            break

        current = next_start

        if len(rows) < 1000:
            break

        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

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
        "taker_base_volume",
        "taker_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(
        all_rows,
        columns=columns,
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = df[
        [
            "open_time",
            "close",
        ]
    ].copy()

    df = df.drop_duplicates(
        subset="open_time"
    )

    df = df.sort_values(
        "open_time"
    )

    df = df.set_index(
        "open_time"
    )

    return df


# ============================================================
# DATA DOWNLOAD
# ============================================================

def load_all_data():

    start_ms = date_to_ms(
        START_DATE
    )

    if END_DATE is None:
        end_ms = get_last_closed_candle_ms()
    else:
        end_ms = date_to_ms(
            END_DATE
        )

    symbols = []

    for symbol in [TARGET] + REFS:
        if symbol not in symbols:
            symbols.append(symbol)

    print()
    print("=" * 80)
    print("DOWNLOAD")
    print("=" * 80)

    print(
        f"Target : {TARGET}"
    )

    print(
        f"Refs   : {', '.join(REFS)}"
    )

    print(
        f"From   : {START_DATE}"
    )

    print(
        f"To     : "
        f"{END_DATE if END_DATE else 'now'}"
    )

    print()

    data = {}

    for i, symbol in enumerate(
        symbols,
        start=1,
    ):

        print(
            f"[{i:>2}/{len(symbols):>2}] "
            f"Downloading {symbol} ..."
        )

        df = fetch_klines(
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        if df.empty:
            raise RuntimeError(
                f"Keine Daten für {symbol}"
            )

        data[symbol] = df

        print(
            f"      {len(df):,} candles"
        )

    return data


# ============================================================
# BUILD CORE DATAFRAME
# ============================================================

def build_dataframe(data):

    frames = []

    for symbol in [
        TARGET
    ] + REFS:

        frame = data[symbol][
            ["close"]
        ].rename(
            columns={
                "close": symbol
            }
        )

        frames.append(frame)

    df = pd.concat(
        frames,
        axis=1,
        join="inner",
    )

    df = df.dropna()
    df = df.sort_index()

    return df


# ============================================================
# VOLATILITY
# ============================================================

def add_volatility_data(df):

    if not USE_VOL_ADAPTIVE_THRESHOLD:

        df["market_vol"] = np.nan
        df["vol_baseline"] = np.nan
        df["vol_factor"] = 1.0

        return df

    ref_vols = []

    for ref in REFS:

        ret = df[ref].pct_change()

        rolling_std = (
            ret
            .rolling(
                VOL_LOOKBACK_BARS
            )
            .std()
        )

        ref_vols.append(
            rolling_std.rename(ref)
        )

    vol_df = pd.concat(
        ref_vols,
        axis=1,
    )

    df["market_vol"] = (
        vol_df.median(axis=1)
    )

    df["vol_baseline"] = (
        df["market_vol"]
        .rolling(
            VOL_BASELINE_BARS
        )
        .median()
        .shift(1)
    )

    df["vol_factor"] = (
        df["market_vol"]
        / df["vol_baseline"]
    )

    df["vol_factor"] = (
        df["vol_factor"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .clip(
            lower=VOL_FACTOR_MIN,
            upper=VOL_FACTOR_MAX,
        )
        .fillna(1.0)
    )

    return df


# ============================================================
# BUILD MULTI-REF SIGNAL
# ============================================================

def build_signals(
    df,
    event_window_min,
    base_threshold,
):
    """
    Returns:

    level_signal:
        Bedingung aktuell erfüllt

    cross_signal:
        Übergang FALSE -> TRUE

    ref_count:
        Anzahl der Refs oberhalb der Schwelle

    effective_threshold:
        volatilitätsangepasste Schwelle
    """

    window_bars = (
        event_window_min
        // INTERVAL_MINUTES
    )

    if window_bars < 1:
        raise ValueError(
            "Event Window muss mindestens 15 Minuten sein."
        )

    effective_threshold = (
        base_threshold
        * THRESHOLD_INTENSITY
        * df["vol_factor"]
    )

    ref_returns = pd.DataFrame(
        index=df.index
    )

    for ref in REFS:

        ref_returns[ref] = (
            df[ref]
            / df[ref].shift(window_bars)
            - 1.0
        )

    ref_count = (
        ref_returns.gt(
            effective_threshold,
            axis=0,
        )
        .sum(axis=1)
    )

    level_signal = (
        ref_count
        >= len(REFS) * 0 + 3
    )

    # --------------------------------------------------------
    # Cross = FALSE -> TRUE
    # --------------------------------------------------------

    previous_level = (
        level_signal
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    cross_signal = (
        level_signal
        & ~previous_level
    )

    result = pd.DataFrame(
        index=df.index
    )

    result["ref_count"] = ref_count

    result["effective_threshold"] = (
        effective_threshold
    )

    result["level_signal"] = (
        level_signal
    )

    result["cross_signal"] = (
        cross_signal
    )

    return result


# ============================================================
# STATISTICS
# ============================================================

def compound_return(trades):

    equity = 1.0

    for trade in trades:

        equity *= (
            1.0
            + trade["net_return"]
        )

    return equity - 1.0


def summarize_trades(trades):

    if not trades:

        return {
            "n": 0,
            "win": np.nan,
            "avg": np.nan,
            "median": np.nan,
            "compound": np.nan,
        }

    returns = np.array(
        [
            trade["net_return"]
            for trade in trades
        ],
        dtype=float,
    )

    return {
        "n":
            len(trades),

        "win":
            np.mean(
                returns > 0
            ),

        "avg":
            np.mean(returns),

        "median":
            np.median(returns),

        "compound":
            compound_return(trades),
    }


def format_pct(
    value,
    decimals=2,
):

    if pd.isna(value):
        return "n/a"

    return (
        f"{value * 100:.{decimals}f}%"
    )


# ============================================================
# COMMON TRADE CREATION
# ============================================================

def create_trade(
    prices,
    timestamps,
    entry_pos,
    exit_pos,
):
    entry_price = prices[entry_pos]
    exit_price = prices[exit_pos]

    if (
        not np.isfinite(entry_price)
        or not np.isfinite(exit_price)
    ):
        return None

    gross_return = (
        exit_price
        / entry_price
        - 1.0
    )

    net_return = (
        (1.0 + gross_return)
        * (1.0 - FEE_PER_SIDE)
        * (1.0 - FEE_PER_SIDE)
        - 1.0
    )

    return {
        "entry_time":
            timestamps[entry_pos],

        "exit_time":
            timestamps[exit_pos],

        "gross_return":
            gross_return,

        "net_return":
            net_return,

        "entry_pos":
            entry_pos,

        "exit_pos":
            exit_pos,
    }


# ============================================================
# LEVEL TRADES
# ============================================================

def get_level_trades(
    df,
    level_signal,
    hold_hours,
):
    """
    LEVEL:

    Sobald flat und level_signal == True:
    Einstieg.

    Während eines Trades werden weitere
    Signale ignoriert.
    """

    hold_bars = int(
        round(
            hold_hours
            * 60
            / INTERVAL_MINUTES
        )
    )

    signals = (
        level_signal
        .to_numpy(dtype=bool)
    )

    prices = (
        df[TARGET]
        .to_numpy()
    )

    timestamps = (
        df.index
        .to_numpy()
    )

    trades = []

    blocked_until = -1

    for pos in np.flatnonzero(signals):

        if pos < blocked_until:
            continue

        exit_pos = (
            pos + hold_bars
        )

        if exit_pos >= len(df):
            continue

        trade = create_trade(
            prices,
            timestamps,
            pos,
            exit_pos,
        )

        if trade is None:
            continue

        trades.append(trade)

        blocked_until = exit_pos

    return trades


# ============================================================
# CROSS TRADES
# ============================================================

def get_cross_trades(
    df,
    cross_signal,
    hold_hours,
):
    """
    CROSS:

    Einstieg nur beim FALSE -> TRUE Übergang.
    """

    hold_bars = int(
        round(
            hold_hours
            * 60
            / INTERVAL_MINUTES
        )
    )

    signals = (
        cross_signal
        .to_numpy(dtype=bool)
    )

    prices = (
        df[TARGET]
        .to_numpy()
    )

    timestamps = (
        df.index
        .to_numpy()
    )

    trades = []

    blocked_until = -1

    for pos in np.flatnonzero(signals):

        if pos < blocked_until:
            continue

        exit_pos = (
            pos + hold_bars
        )

        if exit_pos >= len(df):
            continue

        trade = create_trade(
            prices,
            timestamps,
            pos,
            exit_pos,
        )

        if trade is None:
            continue

        trades.append(trade)

        blocked_until = exit_pos

    return trades


# ============================================================
# EXIT CHECK / FRESH CROSS
# ============================================================

def get_exit_check_trades(
    df,
    cross_signal,
    event_window_min,
    hold_hours,
):
    """
    EXIT CHECK:

    1. Erster Einstieg nur über CROSS.

    2. Trade läuft für die komplette Hold-Zeit.

    3. Beim EXIT wird geprüft:

       Hat innerhalb des letzten EVENT WINDOWS
       ein neuer CROSS stattgefunden?

       JA:
           sofort beim Exit erneut kaufen.

       NEIN:
           flat bleiben.

    4. Sobald wieder ein neuer CROSS auftaucht,
       wird erneut eingestiegen.

    Damit wird ein dauerhaftes LEVEL != automatisch
    als neues Event interpretiert.
    """

    hold_bars = int(
        round(
            hold_hours
            * 60
            / INTERVAL_MINUTES
        )
    )

    event_window_bars = int(
        round(
            event_window_min
            / INTERVAL_MINUTES
        )
    )

    if event_window_bars < 1:
        raise ValueError(
            "Event Window muss mindestens 15 Minuten sein."
        )

    cross = (
        cross_signal
        .to_numpy(dtype=bool)
    )

    prices = (
        df[TARGET]
        .to_numpy()
    )

    timestamps = (
        df.index
        .to_numpy()
    )

    trades = []

    pos = 0

    while pos < len(df):

        # ----------------------------------------------------
        # Suche nächsten Cross
        # ----------------------------------------------------

        future_cross_positions = np.flatnonzero(
            cross[pos:]
        )

        if len(future_cross_positions) == 0:
            break

        entry_pos = (
            pos
            + future_cross_positions[0]
        )

        exit_pos = (
            entry_pos
            + hold_bars
        )

        if exit_pos >= len(df):
            break

        trade = create_trade(
            prices,
            timestamps,
            entry_pos,
            exit_pos,
        )

        if trade is None:
            pos = entry_pos + 1
            continue

        trades.append(trade)

        # ----------------------------------------------------
        # Exit Check
        #
        # Prüfe Crosses innerhalb des letzten Event Windows
        # einschließlich des Exit-Bars.
        # ----------------------------------------------------

        window_start = max(
            0,
            exit_pos
            - event_window_bars
            + 1,
        )

        fresh_cross = np.any(
            cross[
                window_start:
                exit_pos + 1
            ]
        )

        if fresh_cross:

            # Direkt am Exit wieder rein
            pos = exit_pos

        else:

            # Erst wieder auf einen neuen Cross warten
            pos = exit_pos + 1

    return trades


# ============================================================
# BUY & HOLD
# ============================================================

def calculate_buy_hold(df):

    first_price = (
        df[TARGET].iloc[0]
    )

    last_price = (
        df[TARGET].iloc[-1]
    )

    gross_return = (
        last_price
        / first_price
        - 1.0
    )

    net_return = (
        (1.0 + gross_return)
        * (1.0 - FEE_PER_SIDE)
        * (1.0 - FEE_PER_SIDE)
        - 1.0
    )

    return net_return


# ============================================================
# ONE TEST
# ============================================================

def run_test(
    df,
    event_window_min,
    base_threshold,
):

    signals = build_signals(
        df=df,
        event_window_min=
            event_window_min,
        base_threshold=
            base_threshold,
    )

    level_signal = (
        signals["level_signal"]
    )

    cross_signal = (
        signals["cross_signal"]
    )

    raw_level_count = int(
        level_signal.sum()
    )

    raw_cross_count = int(
        cross_signal.sum()
    )

    print()
    print()
    print("=" * 115)
    print(
        f"EVENT WINDOW {event_window_min}m"
        f" | BASE THRESHOLD "
        f"{base_threshold * 100:.1f}%"
        f" | REFS "
        f"3/{len(REFS)}"
    )
    print("=" * 115)

    print(
        f"Raw Level Events : "
        f"{raw_level_count:,}"
    )

    print(
        f"Raw Cross Events : "
        f"{raw_cross_count:,}"
    )

    print()

    results = []

    for hold_hours in HOLDS_HOURS:

        # ----------------------------------------------------
        # LEVEL
        # ----------------------------------------------------

        level_trades = (
            get_level_trades(
                df=df,
                level_signal=
                    level_signal,
                hold_hours=
                    hold_hours,
            )
        )

        level_stats = (
            summarize_trades(
                level_trades
            )
        )

        # ----------------------------------------------------
        # CROSS
        # ----------------------------------------------------

        cross_trades = (
            get_cross_trades(
                df=df,
                cross_signal=
                    cross_signal,
                hold_hours=
                    hold_hours,
            )
        )

        cross_stats = (
            summarize_trades(
                cross_trades
            )
        )

        # ----------------------------------------------------
        # EXIT CHECK
        # ----------------------------------------------------

        exit_check_trades = (
            get_exit_check_trades(
                df=df,
                cross_signal=
                    cross_signal,
                event_window_min=
                    event_window_min,
                hold_hours=
                    hold_hours,
            )
        )

        exit_check_stats = (
            summarize_trades(
                exit_check_trades
            )
        )

        # ----------------------------------------------------
        # Delta
        # ----------------------------------------------------

        delta_cross_vs_level = np.nan
        delta_exit_vs_level = np.nan
        delta_exit_vs_cross = np.nan

        if (
            not pd.isna(
                level_stats["compound"]
            )
            and
            not pd.isna(
                cross_stats["compound"]
            )
        ):
            delta_cross_vs_level = (
                cross_stats["compound"]
                - level_stats["compound"]
            )

        if (
            not pd.isna(
                level_stats["compound"]
            )
            and
            not pd.isna(
                exit_check_stats["compound"]
            )
        ):
            delta_exit_vs_level = (
                exit_check_stats["compound"]
                - level_stats["compound"]
            )

        if (
            not pd.isna(
                cross_stats["compound"]
            )
            and
            not pd.isna(
                exit_check_stats["compound"]
            )
        ):
            delta_exit_vs_cross = (
                exit_check_stats["compound"]
                - cross_stats["compound"]
            )

        print(
            f"{hold_hours:>4}h | "
            f"LEVEL "
            f"N={level_stats['n']:>3} "
            f"W={format_pct(level_stats['win']):>7} "
            f"Ø={format_pct(level_stats['avg']):>8} "
            f"C={format_pct(level_stats['compound']):>10}"
            f" || "
            f"CROSS "
            f"N={cross_stats['n']:>3} "
            f"W={format_pct(cross_stats['win']):>7} "
            f"Ø={format_pct(cross_stats['avg']):>8} "
            f"C={format_pct(cross_stats['compound']):>10}"
            f" || "
            f"EXIT "
            f"N={exit_check_stats['n']:>3} "
            f"W={format_pct(exit_check_stats['win']):>7} "
            f"Ø={format_pct(exit_check_stats['avg']):>8} "
            f"C={format_pct(exit_check_stats['compound']):>10}"
        )

        results.append(
            {
                "event_window":
                    event_window_min,

                "threshold":
                    base_threshold,

                "hold":
                    hold_hours,

                "level":
                    level_stats,

                "cross":
                    cross_stats,

                "exit":
                    exit_check_stats,

                "delta_cross_level":
                    delta_cross_vs_level,

                "delta_exit_level":
                    delta_exit_vs_level,

                "delta_exit_cross":
                    delta_exit_vs_cross,
            }
        )

    return results


# ============================================================
# TOP RESULTS
# ============================================================

def print_top_results(
    all_results
):

    valid = [
        row
        for row in all_results
        if not pd.isna(
            row["exit"]["compound"]
        )
    ]

    if not valid:
        return

    valid = sorted(
        valid,
        key=lambda row:
            row["exit"]["compound"],
        reverse=True,
    )

    print()
    print()
    print("=" * 120)
    print("TOP EXIT-CHECK RESULTS")
    print("=" * 120)

    print(
        f"{'Event':>7} "
        f"{'Base':>7} "
        f"{'Hold':>7} "
        f"{'N':>5} "
        f"{'Win':>8} "
        f"{'Avg':>9} "
        f"{'Compound':>11} "
        f"{'Δ vs Level':>12} "
        f"{'Δ vs Cross':>12}"
    )

    print("-" * 120)

    for row in valid[:25]:

        print(
            f"{row['event_window']:>6}m "
            f"{row['threshold'] * 100:>6.1f}% "
            f"{row['hold']:>6}h "
            f"{row['exit']['n']:>5} "
            f"{format_pct(row['exit']['win']):>8} "
            f"{format_pct(row['exit']['avg']):>9} "
            f"{format_pct(row['exit']['compound']):>11} "
            f"{format_pct(row['delta_exit_level']):>12} "
            f"{format_pct(row['delta_exit_cross']):>12}"
        )


# ============================================================
# BEST RESULT FOR EACH METHOD
# ============================================================

def print_method_winners(
    all_results
):

    print()
    print()
    print("=" * 120)
    print("BEST RESULT JE METHODE")
    print("=" * 120)

    methods = [
        ("LEVEL", "level"),
        ("CROSS", "cross"),
        ("EXIT CHECK", "exit"),
    ]

    for name, key in methods:

        valid = [
            row
            for row in all_results
            if not pd.isna(
                row[key]["compound"]
            )
        ]

        if not valid:
            continue

        best = max(
            valid,
            key=lambda row:
                row[key]["compound"]
        )

        print(
            f"{name:<11} | "
            f"Event {best['event_window']}m"
            f" | Base "
            f"{best['threshold'] * 100:.1f}%"
            f" | Hold "
            f"{best['hold']}h"
            f" | N "
            f"{best[key]['n']}"
            f" | Win "
            f"{format_pct(best[key]['win'])}"
            f" | Avg "
            f"{format_pct(best[key]['avg'])}"
            f" | Compound "
            f"{format_pct(best[key]['compound'])}"
        )


# ============================================================
# DIRECT HOLD-BY-HOLD COMPARISON
# ============================================================

def print_comparison_table(
    all_results
):

    print()
    print()
    print("=" * 120)
    print("METHODENVERGLEICH")
    print("=" * 120)

    print(
        f"{'Event':>7} "
        f"{'Base':>7} "
        f"{'Hold':>7} "
        f"{'Level Cmp':>11} "
        f"{'Cross Cmp':>11} "
        f"{'Exit Cmp':>11} "
        f"{'Exit-Level':>12}"
    )

    print("-" * 120)

    valid = [
        row
        for row in all_results
        if not pd.isna(
            row["level"]["compound"]
        )
    ]

    # Pro Event/Threshold nur die besten Holds
    # für Exit-Check anzeigen.
    valid = sorted(
        valid,
        key=lambda row: (
            row["event_window"],
            row["threshold"],
            row["hold"],
        )
    )

    for row in valid:

        print(
            f"{row['event_window']:>6}m "
            f"{row['threshold'] * 100:>6.1f}% "
            f"{row['hold']:>6}h "
            f"{format_pct(row['level']['compound']):>11} "
            f"{format_pct(row['cross']['compound']):>11} "
            f"{format_pct(row['exit']['compound']):>11} "
            f"{format_pct(row['delta_exit_level']):>12}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("MULTI-REF: LEVEL vs CROSS vs EXIT CHECK")
    print("=" * 80)

    print(
        "EXIT CHECK:"
    )

    print(
        "  Initial entry = CROSS"
    )

    print(
        "  Beim Exit wird geprüft, ob "
        "im letzten Event Window ein neuer CROSS "
        "stattgefunden hat."
    )

    print(
        "  Falls ja -> direkt neuer Trade."
    )

    print(
        "  Falls nein -> warten auf neuen CROSS."
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    data = load_all_data()

    # --------------------------------------------------------
    # Dataframe
    # --------------------------------------------------------

    df = build_dataframe(
        data
    )

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        f"Rows  : {len(df):,}"
    )

    print(
        f"Start : {df.index.min()}"
    )

    print(
        f"End   : {df.index.max()}"
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df = add_volatility_data(
        df
    )

    if USE_VOL_ADAPTIVE_THRESHOLD:

        valid_vol = (
            df["vol_factor"]
            .dropna()
        )

        print()
        print("=" * 80)
        print("VOLATILITY")
        print("=" * 80)

        print(
            f"Median Vol Factor : "
            f"{valid_vol.median():.2f}"
        )

        print(
            f"Min Vol Factor    : "
            f"{valid_vol.min():.2f}"
        )

        print(
            f"Max Vol Factor    : "
            f"{valid_vol.max():.2f}"
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    buy_hold = (
        calculate_buy_hold(df)
    )

    print()
    print("=" * 80)
    print("BENCHMARK")
    print("=" * 80)

    print(
        f"{TARGET} Buy & Hold: "
        f"{format_pct(buy_hold)}"
    )

    # --------------------------------------------------------
    # Tests
    # --------------------------------------------------------

    all_results = []

    for event_window in (
        EVENT_WINDOWS_MIN
    ):

        for threshold in (
            BASE_THRESHOLDS
        ):

            results = run_test(
                df=df,
                event_window_min=
                    event_window,
                base_threshold=
                    threshold,
            )

            all_results.extend(
                results
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_comparison_table(
        all_results
    )

    print_top_results(
        all_results
    )

    print_method_winners(
        all_results
    )

    print()
    print("=" * 80)
    print("ENDE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()