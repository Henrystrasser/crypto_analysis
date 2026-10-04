import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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
    "BNBUSDT",
    "XRPUSDT"
]

# Falls du keine doppelten Refs willst:
REFS = list(dict.fromkeys(REFS))

INTERVAL = "15m"

# Zeitraum
START_DATE = "2024-01-01"
END_DATE = None              # None = bis jetzt

# Binance Spot Fee:
# 0.075% = 7.5 bps
FEE_PER_SIDE = 0.00075

# ------------------------------------------------------------
# Test A / OR
#
# BASE:
#   z.B. 3 von 5 Refs > 2%
#
# STRONG:
#   z.B. 2 von 5 Refs > 3%
#
# SIGNAL:
#   BASE ODER STRONG
# ------------------------------------------------------------

BASE_QUORUM = 3

STRONG_QUORUM = 2
STRONG_THRESHOLD_MULTIPLIER = 1.5

# ------------------------------------------------------------
# Volatilitätsanpassung
# ------------------------------------------------------------

USE_VOL_ADAPTIVE_THRESHOLD = True

VOL_LOOKBACK_HOURS = 24      # aktuelle Vol (Kerzenzahl wird aus INTERVAL berechnet)
VOL_BASELINE_DAYS = 30       # Baseline-Vol

VOL_FACTOR_MIN = 0.5
VOL_FACTOR_MAX = 3.0

THRESHOLD_INTENSITY = 1.0

# EVENT_MODE:
#   "level" = jede Kerze, auf der die Bedingung gilt
#             (nach Hold-Ende wird erneut gekauft, falls weiter erfüllt)
#   "cross" = nur wenn die Bedingung NEU erfüllt ist
#             (vorherige Kerze hat sie nicht erfüllt)
EVENT_MODE = "level"

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

EVENT_WINDOWS_MIN = [
    30,
    60,
    120,
    240,
    360,
]

BASE_THRESHOLDS = [
    0.01,    # 1%
    0.02,    # 2%
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


# ============================================================
# HELPERS
# ============================================================

INTERVAL_MINUTES = int(INTERVAL[:-1]) * {"m": 1, "h": 60, "d": 1440}[INTERVAL[-1]]  # aus INTERVAL abgeleitet
MS_PER_MINUTE = 60_000
MS_PER_DAY = 24 * 60 * MS_PER_MINUTE

# Aus INTERVAL / Zeitangaben abgeleitet (nicht von Hand koppeln).
INTERVAL_MS = INTERVAL_MINUTES * MS_PER_MINUTE
VOL_LOOKBACK_BARS = VOL_LOOKBACK_HOURS * 60 // INTERVAL_MINUTES
VOL_BASELINE_BARS = VOL_BASELINE_DAYS * 24 * 60 // INTERVAL_MINUTES

BERLIN = ZoneInfo("Europe/Berlin")

# Werden zur Laufzeit gesetzt (load_all_data / main).
ANALYSIS_START = None   # pd.Timestamp (Berlin) = START_DATE 00:00
BLOCK_PERIOD = ""       # "from YYYY-MM-DD to YYYY-MM-DD"



def date_to_ms(date_str, end_of_day=False):
    """
    YYYY-MM-DD als Europe/Berlin-Datum.
    end_of_day=True -> Open-Zeit der letzten Kerze dieses Tages.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=BERLIN)
    if end_of_day:
        dt = dt + timedelta(days=1)
        return int(dt.timestamp() * 1000) - INTERVAL_MS
    return int(dt.timestamp() * 1000)


def get_now_closed_candle_ms():
    """
    Liefert den Timestamp der letzten vollständig geschlossenen
    Kerze (INTERVAL).
    """
    now_ms = int(time.time() * 1000)
    interval_ms = INTERVAL_MINUTES * MS_PER_MINUTE

    current_candle_open = (now_ms // interval_ms) * interval_ms

    # Eine Kerze zurück = letzte sicher geschlossene Kerze
    return current_candle_open - interval_ms


def fetch_klines(symbol, start_ms, end_ms):
    """
    Robustes Pagination für Binance Spot Klines.
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

        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()

        rows = response.json()

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = rows[-1][0]

        next_start = last_open_time + MS_PER_MINUTE * INTERVAL_MINUTES

        if next_start <= current:
            break

        current = next_start

        if len(rows) < 1000:
            break

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

    df = pd.DataFrame(all_rows, columns=columns)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df[["open_time", "close"]].copy()
    df = df.drop_duplicates("open_time")
    df = df.sort_values("open_time")
    df = df.set_index("open_time")

    df.index = df.index.tz_convert(BERLIN)
    return df


def load_all_data():
    global ANALYSIS_START
    start_ms = date_to_ms(START_DATE)
    last_closed_ms = get_now_closed_candle_ms()
    if END_DATE is None:
        end_ms = last_closed_ms
    else:
        # END_DATE inklusive, aber nie nach der letzten geschlossenen Kerze
        end_ms = min(date_to_ms(END_DATE, end_of_day=True), last_closed_ms)
    ANALYSIS_START = pd.Timestamp(start_ms, unit="ms", tz="UTC").tz_convert(BERLIN)
    # Warmup: Vol-Historie vor START_DATE nachladen (nur bei Vol-Anpassung)
    warmup_bars = (
        VOL_BASELINE_BARS + VOL_LOOKBACK_BARS + 1
        if USE_VOL_ADAPTIVE_THRESHOLD else 0
    )
    fetch_start_ms = start_ms - warmup_bars * INTERVAL_MS
    print(f"Warmup : {warmup_bars * INTERVAL_MINUTES / 1440:.1f} Tage vor START_DATE (Europe/Berlin)")

    print()
    print("=" * 80)
    print("DOWNLOAD")
    print("=" * 80)
    print(f"Target : {TARGET}")
    print(f"Refs   : {', '.join(REFS)}")
    print(f"From   : {START_DATE}")
    print(f"To     : {END_DATE if END_DATE else 'now'}")
    print()

    symbols = [TARGET] + REFS

    data = {}

    for symbol in symbols:
        print(f"Downloading {symbol} ...")

        df = fetch_klines(
            symbol=symbol,
            start_ms=fetch_start_ms,
            end_ms=end_ms,
        )

        if df.empty:
            raise RuntimeError(f"Keine Daten für {symbol}")

        data[symbol] = df

        print(
            f"  -> {len(df):,} candles "
            f"({df.index.min()} -> {df.index.max()})"
        )

    return data


def build_aligned_dataframe(data):
    """
    Gemeinsame Zeitachse für Target + alle Refs.
    """

    close_frames = []

    target_df = data[TARGET][["close"]].rename(
        columns={"close": TARGET}
    )
    close_frames.append(target_df)

    for ref in REFS:
        ref_df = data[ref][["close"]].rename(
            columns={"close": ref}
        )
        close_frames.append(ref_df)

    df = pd.concat(close_frames, axis=1, join="inner")
    df = df.dropna().sort_index()

    return df


def add_volatility_data(df):
    """
    Marktvolatilität:
    Median der rolling std der 15m-Returns über alle Refs.

    Baseline:
    rolling median über 30 Tage,
    um 1 Kerze verschoben => kein Lookahead.
    """

    if not USE_VOL_ADAPTIVE_THRESHOLD:
        df["market_vol"] = np.nan
        df["vol_baseline"] = np.nan
        df["vol_factor"] = 1.0
        return df

    ref_returns = []

    for ref in REFS:
        ret = df[ref].pct_change()
        rolling_std = ret.rolling(VOL_LOOKBACK_BARS).std()
        ref_returns.append(rolling_std.rename(ref))

    vol_df = pd.concat(ref_returns, axis=1)

    df["market_vol"] = vol_df.median(axis=1)

    df["vol_baseline"] = (
        df["market_vol"]
        .rolling(VOL_BASELINE_BARS)
        .median()
        .shift(1)
    )

    df["vol_factor"] = (
        df["market_vol"] / df["vol_baseline"]
    )

    df["vol_factor"] = df["vol_factor"].replace(
        [np.inf, -np.inf],
        np.nan,
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


def build_signal_masks(
    df,
    event_window_min,
    base_threshold,
):
    """
    Erzeugt:

      BASE:
        BASE_QUORUM refs > effective threshold

      STRONG:
        STRONG_QUORUM refs > effective threshold * multiplier

      OR:
        BASE OR STRONG

    Zusätzlich:
      only_base
      only_strong
      both
    """

    window_bars = event_window_min // INTERVAL_MINUTES

    if window_bars < 1:
        raise ValueError(f"Event Window muss mindestens {INTERVAL_MINUTES}m sein.")

    # --------------------------------------------------------
    # Referenzrenditen
    # --------------------------------------------------------

    ref_returns = pd.DataFrame(index=df.index)

    for ref in REFS:
        ref_returns[ref] = (
            df[ref] / df[ref].shift(window_bars) - 1.0
        )

    # --------------------------------------------------------
    # Dynamische Schwelle
    # --------------------------------------------------------

    effective_threshold = (
        base_threshold
        * THRESHOLD_INTENSITY
        * df["vol_factor"]
    )

    # --------------------------------------------------------
    # Base
    # --------------------------------------------------------

    base_counts = (
        ref_returns.gt(
            effective_threshold,
            axis=0,
        )
        .sum(axis=1)
    )

    base_signal = base_counts >= BASE_QUORUM

    # --------------------------------------------------------
    # Strong
    # --------------------------------------------------------

    strong_threshold = (
        effective_threshold
        * STRONG_THRESHOLD_MULTIPLIER
    )

    strong_counts = (
        ref_returns.gt(
            strong_threshold,
            axis=0,
        )
        .sum(axis=1)
    )

    strong_signal = strong_counts >= STRONG_QUORUM

    # --------------------------------------------------------
    # OR
    # --------------------------------------------------------

    or_signal = base_signal | strong_signal

    # --------------------------------------------------------
    # Signal-Kategorien
    # --------------------------------------------------------

    only_base = base_signal & ~strong_signal
    only_strong = strong_signal & ~base_signal
    both = base_signal & strong_signal

    result = pd.DataFrame(index=df.index)

    result["base_count"] = base_counts
    result["strong_count"] = strong_counts

    result["base_signal"] = base_signal
    result["strong_signal"] = strong_signal
    result["or_signal"] = or_signal

    result["only_base"] = only_base
    result["only_strong"] = only_strong
    result["both"] = both

    result["effective_threshold"] = effective_threshold

    return result


def event_signal(signal, df):
    """
    EVENT_MODE anwenden ("level" = jede Kerze, "cross" = nur Neueintritt)
    und nur Signale ab START_DATE zulassen (Warmup-Kerzen werden nie gehandelt).
    """
    s = signal.astype(bool)
    if EVENT_MODE == "cross":
        s = s & ~s.shift(1, fill_value=False)
    elif EVENT_MODE != "level":
        raise ValueError(f"EVENT_MODE {EVENT_MODE!r} ungültig (level|cross).")
    if ANALYSIS_START is not None:
        s = s & (df.index >= ANALYSIS_START)
    return s


def get_trades(
    df,
    signals,
    hold_hours,
):
    """
    Globaler One-Slot-Ansatz:
    - chronologische Verarbeitung
    - sobald ein Trade läuft, werden überschneidende
      neue Signale übersprungen (Re-Entry exakt am Exit erlaubt)
    - Exit exakt auf der Kerze Entry + Hold; fehlt sie
      (Datenende / Lücke) -> kein Trade
    """
    signal_positions = np.flatnonzero(
        signals.to_numpy(dtype=bool)
    )
    if len(signal_positions) == 0:
        return []
    target_prices = df[TARGET].to_numpy()
    timestamps = df.index.to_numpy()
    exit_positions = df.index.get_indexer(
        df.index[signal_positions]
        + pd.Timedelta(hours=hold_hours)
    )
    trades = []
    blocked_until_pos = -1
    for pos, exit_pos in zip(signal_positions, exit_positions):
        # Keine exakte Exit-Kerze im Datensatz
        if exit_pos < 0:
            continue
        # Overlap vermeiden
        if pos < blocked_until_pos:
            continue
        entry_price = target_prices[pos]
        exit_price = target_prices[exit_pos]
        if not np.isfinite(entry_price) or not np.isfinite(exit_price):
            continue
        gross_return = exit_price / entry_price - 1.0
        # Gebühr beim Entry + Exit
        net_return = (
            (1.0 + gross_return)
            * (1.0 - FEE_PER_SIDE)
            * (1.0 - FEE_PER_SIDE)
            - 1.0
        )
        trades.append(
            {
                "entry_time": timestamps[pos],
                "exit_time": timestamps[exit_pos],
                "gross_return": gross_return,
                "net_return": net_return,
                "signal_position": pos,
            }
        )
        blocked_until_pos = exit_pos
    return trades


def compound_return(trades):
    equity = 1.0

    for trade in trades:
        equity *= 1.0 + trade["net_return"]

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
        [t["net_return"] for t in trades],
        dtype=float,
    )

    win = np.mean(returns > 0.0)

    return {
        "n": len(trades),
        "win": win,
        "avg": np.mean(returns),
        "median": np.median(returns),
        "compound": compound_return(trades),
    }


def format_pct(x, decimals=2):
    if pd.isna(x):
        return "n/a"

    return f"{x * 100:.{decimals}f}%"


def calculate_buy_hold(df):
    """
    Buy & Hold Target über den gesamten Zeitraum,
    ebenfalls mit Entry- und Exit-Fee.
    """

    first_price = df[TARGET].iloc[0]
    last_price = df[TARGET].iloc[-1]

    gross = last_price / first_price - 1.0

    net = (
        (1.0 + gross)
        * (1.0 - FEE_PER_SIDE)
        * (1.0 - FEE_PER_SIDE)
        - 1.0
    )

    return net


def print_config_header(
    event_window_min,
    base_threshold,
):
    print()
    print("-" * 80)
    print(
        f"EVENT {event_window_min:>3}m"
        f" | BASE {base_threshold * 100:.1f}%"
        f" | Base {BASE_QUORUM}/{len(REFS)}"
        f" | Strong {STRONG_QUORUM}/{len(REFS)}"
        f" @ {base_threshold * STRONG_THRESHOLD_MULTIPLIER * 100:.1f}%"
        f" | OR"
        f" | {EVENT_MODE} | {BLOCK_PERIOD}"
    )
    print("-" * 80)


def run_test(
    df,
    event_window_min,
    base_threshold,
):
    signals = build_signal_masks(
        df=df,
        event_window_min=event_window_min,
        base_threshold=base_threshold,
    )
    # EVENT_MODE + nur Signale ab START_DATE (Warmup nie handeln)
    for col in ("base_signal", "strong_signal", "or_signal"):
        signals[col] = event_signal(signals[col], df)
    signals["only_base"] = signals["base_signal"] & ~signals["strong_signal"]
    signals["only_strong"] = signals["strong_signal"] & ~signals["base_signal"]
    signals["both"] = signals["base_signal"] & signals["strong_signal"]

    base_count = int(signals["base_signal"].sum())
    strong_count = int(signals["strong_signal"].sum())
    or_count = int(signals["or_signal"].sum())

    only_base_count = int(signals["only_base"].sum())
    only_strong_count = int(signals["only_strong"].sum())
    both_count = int(signals["both"].sum())

    print_config_header(
        event_window_min,
        base_threshold,
    )

    print(
        f"Raw Base Events   : {base_count:,}"
    )

    print(
        f"Raw Strong Events : {strong_count:,}"
    )

    print(
        f"Raw OR Events     : {or_count:,}"
    )

    print(
        f"  only Base       : {only_base_count:,}"
    )

    print(
        f"  only Strong     : {only_strong_count:,}"
    )

    print(
        f"  both            : {both_count:,}"
    )

    if base_count > 0:
        print(
            f"Strong-only extra : "
            f"{only_strong_count / base_count * 100:.2f}% "
            f"of Base Events"
        )

    print()

    results = []

    for hold_hours in HOLDS_HOURS:

        base_trades = get_trades(
            df=df,
            signals=signals["base_signal"],
            hold_hours=hold_hours,
        )

        strong_trades = get_trades(
            df=df,
            signals=signals["strong_signal"],
            hold_hours=hold_hours,
        )

        or_trades = get_trades(
            df=df,
            signals=signals["or_signal"],
            hold_hours=hold_hours,
        )

        base_stats = summarize_trades(base_trades)
        strong_stats = summarize_trades(strong_trades)
        or_stats = summarize_trades(or_trades)

        delta_vs_base = np.nan

        if (
            not pd.isna(or_stats["compound"])
            and not pd.isna(base_stats["compound"])
        ):
            delta_vs_base = (
                or_stats["compound"]
                - base_stats["compound"]
            )

        print(
            f"{hold_hours:>4}h | "
            f"Base N={base_stats['n']:>3} "
            f"Win={format_pct(base_stats['win']):>7} "
            f"Ø={format_pct(base_stats['avg']):>8} "
            f"Cmp={format_pct(base_stats['compound']):>9}"
            f" || "
            f"Strong N={strong_stats['n']:>3} "
            f"Win={format_pct(strong_stats['win']):>7} "
            f"Ø={format_pct(strong_stats['avg']):>8} "
            f"Cmp={format_pct(strong_stats['compound']):>9}"
            f" || "
            f"OR N={or_stats['n']:>3} "
            f"Win={format_pct(or_stats['win']):>7} "
            f"Ø={format_pct(or_stats['avg']):>8} "
            f"Cmp={format_pct(or_stats['compound']):>9}"
            f" || "
            f"ΔCmp={format_pct(delta_vs_base):>9}"
        )

        results.append(
            {
                "event_window": event_window_min,
                "base_threshold": base_threshold,
                "hold_hours": hold_hours,
                "base": base_stats,
                "strong": strong_stats,
                "or": or_stats,
                "delta": delta_vs_base,
            }
        )

    return results


def print_top_results(all_results):
    valid = []

    for r in all_results:
        if pd.isna(r["or"]["compound"]):
            continue

        valid.append(r)

    if not valid:
        return

    valid = sorted(
        valid,
        key=lambda x: x["or"]["compound"],
        reverse=True,
    )

    print()
    print("=" * 110)
    print("TOP OR RESULTS")
    print("=" * 110)

    print(
        f"{'Event':>7} "
        f"{'Base':>7} "
        f"{'Hold':>7} "
        f"{'N':>5} "
        f"{'Win':>8} "
        f"{'Avg':>9} "
        f"{'Compound':>11} "
        f"{'Δ vs Base':>11}"
    )

    print("-" * 110)

    for r in valid[:20]:

        print(
            f"{r['event_window']:>6}m "
            f"{r['base_threshold'] * 100:>6.1f}% "
            f"{r['hold_hours']:>6}h "
            f"{r['or']['n']:>5} "
            f"{format_pct(r['or']['win']):>8} "
            f"{format_pct(r['or']['avg']):>9} "
            f"{format_pct(r['or']['compound']):>11} "
            f"{format_pct(r['delta']):>11}"
        )


def main():

    # --------------------------------------------------------
    # Daten laden
    # --------------------------------------------------------

    data = load_all_data()

    # --------------------------------------------------------
    # Gemeinsame Zeitachse
    # --------------------------------------------------------

    df = build_aligned_dataframe(data)

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(f"Rows   : {len(df):,}")
    print(f"Start  : {df.index.min()}")
    print(f"End    : {df.index.max()}")
    print()

    # --------------------------------------------------------
    # Volatilität
    # --------------------------------------------------------

    df = add_volatility_data(df)

    if USE_VOL_ADAPTIVE_THRESHOLD:
        valid_vol = df["vol_factor"].dropna()

        print(
            f"Vol Factor median : "
            f"{valid_vol.median():.2f}"
        )

        print(
            f"Vol Factor min    : "
            f"{valid_vol.min():.2f}"
        )

        print(
            f"Vol Factor max    : "
            f"{valid_vol.max():.2f}"
        )

    # --------------------------------------------------------
    # Analysezeitraum (ohne Warmup) fuer B&H und Block-Header
    # --------------------------------------------------------
    global BLOCK_PERIOD
    analysis_df = df[df.index >= ANALYSIS_START]
    BLOCK_PERIOD = (
        f"from {analysis_df.index.min():%Y-%m-%d} "
        f"to {analysis_df.index.max():%Y-%m-%d}"
    )
    print()
    print(f"Analyse ab : {analysis_df.index.min()} (Europe/Berlin, Warmup davor nur für Vol)")
    print(f"Analyse bis: {analysis_df.index.max()}")
    print(f"Event-Modus: {EVENT_MODE}")

    # --------------------------------------------------------
    # Buy & Hold
    # --------------------------------------------------------

    buy_hold = calculate_buy_hold(analysis_df)

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

    for event_window_min in EVENT_WINDOWS_MIN:

        for base_threshold in BASE_THRESHOLDS:

            results = run_test(
                df=df,
                event_window_min=event_window_min,
                base_threshold=base_threshold,
            )

            all_results.extend(results)

    # --------------------------------------------------------
    # Top Results
    # --------------------------------------------------------

    print_top_results(all_results)

    print()
    print("=" * 80)
    print("ENDE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()