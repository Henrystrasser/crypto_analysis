import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

API_URL = "https://data-api.binance.vision/api/v3/klines"

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

TARGET = "ENAUSDT"

# ------------------------------------------------------------
# MULTI-REF STRENGTH
# ------------------------------------------------------------

REFS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
]

BASE_QUORUM = 3

# ------------------------------------------------------------
# MARKET BREADTH
#
# Fixed basket of 20 established / liquid alts.
#
# TARGET is deliberately NOT included in breadth.
# ------------------------------------------------------------

BREADTH_UNIVERSE = [
    "BNBUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ETCUSDT",
    "XLMUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "FILUSDT",
    "AAVEUSDT",
    "ALGOUSDT",
    "VETUSDT",
    "EOSUSDT",
    "TRXUSDT",
]

# Breadth definition:
# percentage of breadth coins whose return is > this threshold
#
# First test = simply positive:
# 0.0% = coin must be green over the event window
BREADTH_RETURN_THRESHOLD = 0.0

# We test several minimum breadth levels
BREADTH_MIN_LEVELS = [
    0.50,   # >= 50% positive
    0.60,   # >= 60% positive
    0.70,   # >= 70% positive
    0.80,   # >= 80% positive
]

# Require at least this many valid breadth coins at a timestamp.
# Normally all 20 should be available.
MIN_BREADTH_SYMBOLS = 18

# ------------------------------------------------------------
# TIMEFRAME
# ------------------------------------------------------------

INTERVAL = "15m"

START_DATE = "2024-01-01"
END_DATE = None

INTERVAL_MINUTES = 15

MS_PER_MINUTE = 60_000
MS_PER_DAY = 24 * MS_PER_MINUTE

# ------------------------------------------------------------
# FEES
# ------------------------------------------------------------

FEE_PER_SIDE = 0.00075      # 7.5 bps = 0.075%

# ------------------------------------------------------------
# TEST GRID
# ------------------------------------------------------------

EVENT_WINDOWS_MIN = [
    30,
    60,
    120,
    240,
    360,
]

BASE_THRESHOLDS = [
    0.01,     # 1%
    0.02,     # 2%
]

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

# ------------------------------------------------------------
# VOLATILITY ADAPTIVE THRESHOLD
# ------------------------------------------------------------

USE_VOL_ADAPTIVE_THRESHOLD = True

VOL_LOOKBACK_BARS = 96         # 24h
VOL_BASELINE_BARS = 2880       # 30 days

VOL_FACTOR_MIN = 0.5
VOL_FACTOR_MAX = 3.0

THRESHOLD_INTENSITY = 1.0


# ============================================================
# HELPERS
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

    # One candle back = last fully closed candle
    return current_open - interval_ms


def fetch_klines(symbol, start_ms, end_ms):
    """
    Robust Binance kline pagination.
    """

    all_rows = []
    current = start_ms

    while current < end_ms:

        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        rows = response.json()

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

        # Small delay to be friendly to API rate limits
        time.sleep(0.05)

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
# DOWNLOAD
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

    print()
    print("=" * 80)
    print("DOWNLOAD")
    print("=" * 80)

    print(f"Target : {TARGET}")
    print(
        f"Refs   : {', '.join(REFS)}"
    )

    print(
        f"Breadth: {len(BREADTH_UNIVERSE)} coins"
    )

    print(
        f"From   : {START_DATE}"
    )

    print(
        f"To     : "
        f"{END_DATE if END_DATE else 'now'}"
    )

    print()

    all_symbols = []

    # Target
    all_symbols.append(TARGET)

    # Refs
    for symbol in REFS:
        if symbol not in all_symbols:
            all_symbols.append(symbol)

    # Breadth universe
    for symbol in BREADTH_UNIVERSE:
        if symbol not in all_symbols:
            all_symbols.append(symbol)

    data = {}

    for i, symbol in enumerate(
        all_symbols,
        start=1,
    ):

        print(
            f"[{i:>2}/{len(all_symbols):>2}] "
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
# TARGET + REFS
# ============================================================

def build_core_dataframe(data):

    symbols = [
        TARGET
    ] + REFS

    frames = []

    for symbol in symbols:

        df = data[symbol][
            ["close"]
        ].rename(
            columns={
                "close": symbol
            }
        )

        frames.append(df)

    result = pd.concat(
        frames,
        axis=1,
        join="inner",
    )

    result = result.dropna()
    result = result.sort_index()

    return result


# ============================================================
# BREADTH DATA
# ============================================================

def build_breadth_dataframe(data):

    frames = []

    for symbol in BREADTH_UNIVERSE:

        if symbol == TARGET:
            continue

        if symbol not in data:
            continue

        df = data[symbol][
            ["close"]
        ].rename(
            columns={
                "close": symbol
            }
        )

        frames.append(df)

    breadth = pd.concat(
        frames,
        axis=1,
        join="outer",
    )

    breadth = breadth.sort_index()

    return breadth


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

        ret = (
            df[ref].pct_change()
        )

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

    # Median volatility across refs
    df["market_vol"] = (
        vol_df.median(axis=1)
    )

    # Historical baseline, shifted by one candle
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
    )

    df["vol_factor"] = (
        df["vol_factor"]
        .clip(
            lower=VOL_FACTOR_MIN,
            upper=VOL_FACTOR_MAX,
        )
        .fillna(1.0)
    )

    return df


# ============================================================
# SIGNALS
# ============================================================

def build_base_signal(
    df,
    event_window_min,
    base_threshold,
):

    window_bars = (
        event_window_min
        // INTERVAL_MINUTES
    )

    if window_bars < 1:
        raise ValueError(
            "Event Window muss mindestens 15m sein."
        )

    # Dynamic threshold
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

    ref_counts = (
        ref_returns.gt(
            effective_threshold,
            axis=0,
        )
        .sum(axis=1)
    )

    base_signal = (
        ref_counts
        >= BASE_QUORUM
    )

    result = pd.DataFrame(
        index=df.index
    )

    result["ref_count"] = ref_counts

    result["effective_threshold"] = (
        effective_threshold
    )

    result["base_signal"] = (
        base_signal
    )

    return result


# ============================================================
# MARKET BREADTH
# ============================================================

def calculate_breadth(
    breadth_df,
    event_window_min,
):

    window_bars = (
        event_window_min
        // INTERVAL_MINUTES
    )

    breadth_returns = (
        breadth_df
        / breadth_df.shift(window_bars)
        - 1.0
    )

    valid_count = (
        breadth_returns.notna()
        .sum(axis=1)
    )

    positive_count = (
        breadth_returns.gt(
            BREADTH_RETURN_THRESHOLD
        )
        .sum(axis=1)
    )

    breadth_pct = (
        positive_count
        / valid_count
    )

    result = pd.DataFrame(
        index=breadth_df.index
    )

    result["breadth_pct"] = (
        breadth_pct
    )

    result["breadth_valid"] = (
        valid_count
    )

    result["positive_count"] = (
        positive_count
    )

    return result


# ============================================================
# TRADES
# ============================================================

def get_trades(
    df,
    signals,
    hold_hours,
):

    hold_bars = int(
        round(
            hold_hours
            * 60
            / INTERVAL_MINUTES
        )
    )

    signal_positions = np.flatnonzero(
        signals.to_numpy(
            dtype=bool
        )
    )

    if len(signal_positions) == 0:
        return []

    target_prices = (
        df[TARGET].to_numpy()
    )

    timestamps = (
        df.index.to_numpy()
    )

    trades = []

    blocked_until_pos = -1

    for pos in signal_positions:

        exit_pos = (
            pos + hold_bars
        )

        if exit_pos >= len(df):
            continue

        # Global one-slot / no overlap
        if pos < blocked_until_pos:
            continue

        entry_price = (
            target_prices[pos]
        )

        exit_price = (
            target_prices[exit_pos]
        )

        if (
            not np.isfinite(entry_price)
            or not np.isfinite(exit_price)
        ):
            continue

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

        trades.append(
            {
                "entry_time":
                    timestamps[pos],

                "exit_time":
                    timestamps[exit_pos],

                "gross_return":
                    gross_return,

                "net_return":
                    net_return,

                "signal_position":
                    pos,
            }
        )

        blocked_until_pos = (
            exit_pos
        )

    return trades


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
                returns > 0.0
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
# BUY & HOLD
# ============================================================

def calculate_buy_hold(df):

    first_price = (
        df[TARGET].iloc[0]
    )

    last_price = (
        df[TARGET].iloc[-1]
    )

    gross = (
        last_price
        / first_price
        - 1.0
    )

    net = (
        (1.0 + gross)
        * (1.0 - FEE_PER_SIDE)
        * (1.0 - FEE_PER_SIDE)
        - 1.0
    )

    return net


# ============================================================
# ONE PARAMETER TEST
# ============================================================

def run_parameter_test(
    df,
    breadth_df,
    event_window_min,
    base_threshold,
):

    # --------------------------------------------------------
    # Base Multi-Ref signal
    # --------------------------------------------------------

    signals = build_base_signal(
        df=df,
        event_window_min=
            event_window_min,
        base_threshold=
            base_threshold,
    )

    base_signal = (
        signals["base_signal"]
    )

    raw_base_events = int(
        base_signal.sum()
    )

    # --------------------------------------------------------
    # Breadth
    # --------------------------------------------------------

    breadth = calculate_breadth(
        breadth_df=breadth_df,
        event_window_min=
            event_window_min,
    )

    # Align to core dataframe
    breadth = breadth.reindex(
        df.index
    )

    # --------------------------------------------------------
    # Print header
    # --------------------------------------------------------

    print()
    print("=" * 110)

    print(
        f"EVENT {event_window_min:>3}m"
        f" | BASE "
        f"{base_threshold * 100:.1f}%"
        f" | REFS "
        f"{BASE_QUORUM}/{len(REFS)}"
    )

    print("=" * 110)

    print(
        f"Raw Base Events : "
        f"{raw_base_events:,}"
    )

    print(
        f"Breadth rule    : "
        f"{BREADTH_RETURN_THRESHOLD * 100:.1f}%"
        f" per coin"
    )

    # --------------------------------------------------------
    # Breadth event counts
    # --------------------------------------------------------

    for level in BREADTH_MIN_LEVELS:

        breadth_pass = (
            breadth["breadth_valid"]
            >= MIN_BREADTH_SYMBOLS
        ) & (
            breadth["breadth_pct"]
            >= level
        )

        filtered_signal = (
            base_signal
            & breadth_pass
        )

        raw_filtered = int(
            filtered_signal.sum()
        )

        removed = (
            raw_base_events
            - raw_filtered
        )

        removed_pct = (
            removed
            / raw_base_events
            * 100
            if raw_base_events > 0
            else np.nan
        )

        print(
            f"Breadth >= "
            f"{level * 100:.0f}% : "
            f"{raw_filtered:,} events "
            f"| removed "
            f"{removed_pct:.1f}%"
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result_rows = []

    for breadth_level in (
        BREADTH_MIN_LEVELS
    ):

        breadth_pass = (
            breadth["breadth_valid"]
            >= MIN_BREADTH_SYMBOLS
        ) & (
            breadth["breadth_pct"]
            >= breadth_level
        )

        filtered_signal = (
            base_signal
            & breadth_pass
        )

        for hold_hours in HOLDS_HOURS:

            # ----------------------------------------------
            # Base
            # ----------------------------------------------

            base_trades = get_trades(
                df=df,
                signals=base_signal,
                hold_hours=hold_hours,
            )

            base_stats = summarize_trades(
                base_trades
            )

            # ----------------------------------------------
            # Breadth filtered
            # ----------------------------------------------

            breadth_trades = get_trades(
                df=df,
                signals=filtered_signal,
                hold_hours=hold_hours,
            )

            breadth_stats = summarize_trades(
                breadth_trades
            )

            # ----------------------------------------------
            # Delta
            # ----------------------------------------------

            delta = np.nan

            if (
                not pd.isna(
                    breadth_stats["compound"]
                )
                and not pd.isna(
                    base_stats["compound"]
                )
            ):

                delta = (
                    breadth_stats["compound"]
                    - base_stats["compound"]
                )

            result_rows.append(
                {
                    "event":
                        event_window_min,

                    "base_threshold":
                        base_threshold,

                    "breadth_level":
                        breadth_level,

                    "hold":
                        hold_hours,

                    "base":
                        base_stats,

                    "breadth":
                        breadth_stats,

                    "delta":
                        delta,
                }
            )

            print(
                f"{breadth_level * 100:>3.0f}% "
                f"| {hold_hours:>4}h "
                f"| "
                f"Base "
                f"N={base_stats['n']:>3} "
                f"Win={format_pct(base_stats['win']):>7} "
                f"Ø={format_pct(base_stats['avg']):>8} "
                f"Cmp={format_pct(base_stats['compound']):>10}"
                f" || "
                f"Breadth "
                f"N={breadth_stats['n']:>3} "
                f"Win={format_pct(breadth_stats['win']):>7} "
                f"Ø={format_pct(breadth_stats['avg']):>8} "
                f"Cmp={format_pct(breadth_stats['compound']):>10}"
                f" || "
                f"ΔCmp={format_pct(delta):>9}"
            )

    return result_rows


# ============================================================
# TOP RESULTS
# ============================================================

def print_top_results(
    all_results
):

    valid = []

    for row in all_results:

        compound = (
            row["breadth"]["compound"]
        )

        if pd.isna(compound):
            continue

        valid.append(row)

    if not valid:
        return

    valid = sorted(
        valid,
        key=lambda x:
            x["breadth"]["compound"],
        reverse=True,
    )

    print()
    print()
    print("=" * 110)
    print("TOP MARKET-BREADTH RESULTS")
    print("=" * 110)

    print(
        f"{'Event':>7} "
        f"{'Base':>7} "
        f"{'Breadth':>9} "
        f"{'Hold':>7} "
        f"{'N':>5} "
        f"{'Win':>8} "
        f"{'Avg':>9} "
        f"{'Compound':>11} "
        f"{'Δ vs Base':>11}"
    )

    print("-" * 110)

    for row in valid[:30]:

        print(
            f"{row['event']:>6}m "
            f"{row['base_threshold'] * 100:>6.1f}% "
            f"{row['breadth_level'] * 100:>8.0f}% "
            f"{row['hold']:>6}h "
            f"{row['breadth']['n']:>5} "
            f"{format_pct(row['breadth']['win']):>8} "
            f"{format_pct(row['breadth']['avg']):>9} "
            f"{format_pct(row['breadth']['compound']):>11} "
            f"{format_pct(row['delta']):>11}"
        )


# ============================================================
# BEST RESULT PER BREADTH LEVEL
# ============================================================

def print_best_by_breadth(
    all_results
):

    print()
    print()
    print("=" * 110)
    print("BEST RESULT FOR EACH BREADTH LEVEL")
    print("=" * 110)

    for level in BREADTH_MIN_LEVELS:

        candidates = [
            row
            for row in all_results
            if row["breadth_level"]
            == level
            and not pd.isna(
                row["breadth"]["compound"]
            )
        ]

        if not candidates:
            continue

        best = max(
            candidates,
            key=lambda x:
                x["breadth"]["compound"]
        )

        print(
            f"Breadth >= "
            f"{level * 100:.0f}% : "
            f"Event {best['event']}m"
            f" | Base "
            f"{best['base_threshold'] * 100:.1f}%"
            f" | Hold "
            f"{best['hold']}h"
            f" | N "
            f"{best['breadth']['n']}"
            f" | Win "
            f"{format_pct(best['breadth']['win'])}"
            f" | Avg "
            f"{format_pct(best['breadth']['avg'])}"
            f" | Compound "
            f"{format_pct(best['breadth']['compound'])}"
            f" | Δ "
            f"{format_pct(best['delta'])}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if TARGET in BREADTH_UNIVERSE:
        print(
            "Hinweis: Target ist in der Breadth-Liste."
        )
        print(
            "Das Script ignoriert das Target dort bewusst."
        )

    if len(
        set(BREADTH_UNIVERSE)
    ) != len(
        BREADTH_UNIVERSE
    ):

        raise ValueError(
            "BREADTH_UNIVERSE enthält Duplikate."
        )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    data = load_all_data()

    # --------------------------------------------------------
    # Core dataframe
    # --------------------------------------------------------

    df = build_core_dataframe(
        data
    )

    print()
    print("=" * 80)
    print("CORE DATASET")
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
    # Breadth dataframe
    # --------------------------------------------------------

    breadth_df = build_breadth_dataframe(
        data
    )

    print()
    print("=" * 80)
    print("BREADTH DATASET")
    print("=" * 80)

    print(
        f"Coins : {breadth_df.shape[1]}"
    )

    print(
        f"Rows  : {len(breadth_df):,}"
    )

    print(
        f"Start : {breadth_df.index.min()}"
    )

    print(
        f"End   : {breadth_df.index.max()}"
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
    # Buy & Hold
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
    # Run tests
    # --------------------------------------------------------

    all_results = []

    for event_window in (
        EVENT_WINDOWS_MIN
    ):

        for base_threshold in (
            BASE_THRESHOLDS
        ):

            results = run_parameter_test(
                df=df,
                breadth_df=breadth_df,
                event_window_min=
                    event_window,
                base_threshold=
                    base_threshold,
            )

            all_results.extend(
                results
            )

    # --------------------------------------------------------
    # Top results
    # --------------------------------------------------------

    print_top_results(
        all_results
    )

    # --------------------------------------------------------
    # Best per breadth level
    # --------------------------------------------------------

    print_best_by_breadth(
        all_results
    )

    # --------------------------------------------------------
    # End
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ENDE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
