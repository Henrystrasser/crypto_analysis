#!/usr/bin/env python3
"""
Feature-Pipeline: BTC-Anstiegs-/Abfall-Fenster → Coin-Move im selben Fenster

Definition (Option A):
  UP:   BTC steigt um ≥ threshold in window_min Minuten
  DOWN: BTC fällt  um ≤ −threshold in window_min Minuten

Defaults: siehe Config (Fenster, Intervall, LOOKBACK_DAYS, Top-N, Excludes).
Zeitraum: primär Config FROM_DATE/TO_DATE oder LOOKBACK_DAYS (in VS Code ändern); CLI --from/--to/--lookback als optionaler Override (deutsche Zeit, Europe/Berlin).

Pro Event × Coin u.a.:
  direction, btc_rise_window_min, btc_rise_threshold_pct,
  btc_rise_start, btc_rise_end, btc_move_pct,
  coin, coin_rise_pct_in_btc_window,
  plus Post-Lags ab Fensterende (je Lag = eigene Spalte/Metrik).

Nach jeder Coin-Tabelle (je Richtung):
  Fußzeile „% positiv“ und „Ø Return“ unter Coin% und jeder Post-Lag-Spalte.

Ausgabe: nur Konsole (keine CSV).
Universum: Top-N ohne Stables/BTC, ohne EXCLUDE_SYMBOLS, Binance Spot USDT.
Kein BTC als Referenz-Coin in der Ausgabe.

Alle Uhrzeiten/Datumsgrenzen: deutsche Zeit (Europe/Berlin, CET/CEST).

Keine Handelsempfehlung — Feature-/Analyse-Skript.
Abhängigkeiten: pip install requests pandas numpy
"""

from __future__ import annotations

import argparse
import sys

from laggard_common import (
    add_time_range_arguments,
    fetch_top_coins_robust,
    ms_to_berlin_str,
    resolve_event_window,
)
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

BTC_RISE_WINDOW_MIN = 60
BTC_RISE_THRESHOLD_PCT = 1.0

# --- Zeitraum (deutsche Zeit, Europe/Berlin inkl. Sommer-/Winterzeit) — hier in VS Code ändern ---
# Wenn FROM_DATE gesetzt: fester Zeitraum nutzen.
#   TO_DATE=None → Ende = jetzt.
#   Date-only: FROM = 00:00:00 Berlin, TO = inklusiv bis 23:59:59.999 Berlin.
# Wenn FROM_DATE=None: LOOKBACK_DAYS rückwärts ab jetzt.
# FROM_DATE und LOOKBACK_DAYS nicht gleichzeitig "aktiv" (FROM hat Vorrang).
FROM_DATE: Optional[str] = None  # z.B. "2026-09-01" oder "2026-09-01 12:00"
TO_DATE: Optional[str] = None    # z.B. "2026-09-02"
LOOKBACK_DAYS = 764
KLINE_INTERVAL = "1h"

# Werte = Anzahl Kerzen bei KLINE_INTERVAL (1h). Sub-Stunden-Lags entfallen.
POST_WINDOW_LAGS: Dict[str, int] = {
    "post_2h": 2,
    "post_12h": 12,
    "post_24h": 24,
    "post_48h": 48,
}

TOP_N = 100
COIN_PAIRS: List[str] = []

STABLE_SYMBOLS = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "USDD",
    "FRAX", "PYUSD", "EURC", "EURT", "GUSD", "LUSD", "SUSD", "USDP",
    "USD1", "USDY", "RLUSD", "CRVUSD", "GHO", "USDS",
}

EXCLUDE_SYMBOLS = {
    "HYPE",
    "ZEC",
}

USE_THRESHOLD_CROSSING = True

COINGECKO = "https://api.coingecko.com/api/v3"
BINANCE = "https://api.binance.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-rise-window-features/0.2"})


def interval_to_minutes(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    if unit == "m":
        return n
    if unit == "h":
        return n * 60
    if unit == "d":
        return n * 60 * 24
    raise ValueError(f"Unsupported interval: {interval}")


BAR_MIN = interval_to_minutes(KLINE_INTERVAL)
if BTC_RISE_WINDOW_MIN % BAR_MIN != 0:
    raise SystemExit(
        f"btc_rise_window_min ({BTC_RISE_WINDOW_MIN}) muss durch "
        f"KLINE_INTERVAL ({KLINE_INTERVAL} = {BAR_MIN} Min) teilbar sein."
    )
WINDOW_BARS = BTC_RISE_WINDOW_MIN // BAR_MIN


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# Anzeige aller Zeiten in deutscher Zeit (Europe/Berlin, CET/CEST) über
# laggard_common.ms_to_berlin_str. Die Event-Erkennung (±X% im rollierenden
# Fenster) arbeitet auf absoluten Zeitstempeln und ist zeitzonenunabhängig.


def sleep_polite(seconds: float = 0.3) -> None:
    time.sleep(seconds)


def fetch_top_coins_no_stables(n: int = TOP_N) -> List[dict]:
    """CoinGecko → Cache → Binance-Volumen (Fallback); siehe laggard_common.fetch_top_coins_robust."""
    return fetch_top_coins_robust(
        n,
        exclude_base="BTC",
        stable_symbols=STABLE_SYMBOLS,
        exclude_symbols=EXCLUDE_SYMBOLS,
        session=SESSION,
        coingecko=COINGECKO,
    )


def binance_usdt_symbols() -> set:
    r = SESSION.get(f"{BINANCE}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    return {
        s["symbol"]
        for s in r.json()["symbols"]
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
    }


def map_to_binance_pair(symbol: str, usdt_set: set) -> Optional[str]:
    aliases = {
        "RNDR": "RENDERUSDT",
        "RENDER": "RENDERUSDT",
        "MATIC": "MATICUSDT",
        "POL": "POLUSDT",
    }
    if symbol in aliases and aliases[symbol] in usdt_set:
        return aliases[symbol]
    pair = f"{symbol}USDT"
    return pair if pair in usdt_set else None


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
        sleep_polite(0.2)
        if len(data) < 1000:
            break

    if not rows:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype(np.int64)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


@dataclass
class BtcMoveEvent:
    start_ms: int
    end_ms: int
    btc_move_pct: float
    direction: str  # "UP" | "DOWN"


def detect_btc_move_events(btc: pd.DataFrame) -> Tuple[List[BtcMoveEvent], List[BtcMoveEvent]]:
    df = btc.copy()
    df["move_pct"] = df["close"].pct_change(WINDOW_BARS) * 100.0
    ups: List[BtcMoveEvent] = []
    downs: List[BtcMoveEvent] = []

    for i in range(len(df)):
        ret = df.at[i, "move_pct"]
        if pd.isna(ret):
            continue
        prev = df.at[i - 1, "move_pct"] if i > 0 else np.nan
        start_i = i - WINDOW_BARS
        if start_i < 0:
            continue

        if ret >= BTC_RISE_THRESHOLD_PCT:
            if USE_THRESHOLD_CROSSING and pd.notna(prev) and prev >= BTC_RISE_THRESHOLD_PCT:
                pass
            else:
                ups.append(
                    BtcMoveEvent(
                        start_ms=int(df.at[start_i, "open_time"]),
                        end_ms=int(df.at[i, "open_time"]),
                        btc_move_pct=float(ret),
                        direction="UP",
                    )
                )

        if ret <= -BTC_RISE_THRESHOLD_PCT:
            if USE_THRESHOLD_CROSSING and pd.notna(prev) and prev <= -BTC_RISE_THRESHOLD_PCT:
                pass
            else:
                downs.append(
                    BtcMoveEvent(
                        start_ms=int(df.at[start_i, "open_time"]),
                        end_ms=int(df.at[i, "open_time"]),
                        btc_move_pct=float(ret),
                        direction="DOWN",
                    )
                )

    return ups, downs


def index_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("open_time", drop=False)


def price_at(df_idx: pd.DataFrame, t_ms: int) -> Optional[float]:
    if t_ms in df_idx.index:
        return float(df_idx.loc[t_ms, "close"])
    later = df_idx.index[df_idx.index >= t_ms]
    if len(later) == 0:
        return None
    return float(df_idx.loc[later[0], "close"])


def loc_at(df_idx: pd.DataFrame, t_ms: int) -> Optional[int]:
    if t_ms in df_idx.index:
        pos = df_idx.index.get_loc(t_ms)
    else:
        later = df_idx.index[df_idx.index >= t_ms]
        if len(later) == 0:
            return None
        pos = df_idx.index.get_loc(later[0])
    if isinstance(pos, slice):
        return pos.start
    if isinstance(pos, np.ndarray):
        return int(pos[0])
    return int(pos)


def rise_in_window(df_idx: pd.DataFrame, start_ms: int, end_ms: int) -> Optional[float]:
    c0 = price_at(df_idx, start_ms)
    c1 = price_at(df_idx, end_ms)
    if c0 is None or c1 is None or c0 <= 0:
        return None
    return (c1 / c0 - 1.0) * 100.0


def post_window_return(df_idx: pd.DataFrame, end_ms: int, lag_bars: int) -> Optional[float]:
    pos = loc_at(df_idx, end_ms)
    if pos is None:
        return None
    j = pos + lag_bars
    if j >= len(df_idx):
        return None
    c0 = float(df_idx.iloc[pos]["close"])
    c1 = float(df_idx.iloc[j]["close"])
    if c0 <= 0:
        return None
    return (c1 / c0 - 1.0) * 100.0


def fmt_pct(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   n/a"
    return f"{v:+6.2f}%"


def column_positivity(rows: List[dict], col: str) -> dict:
    """
    Positiv = Wert > 0, Negativ = < 0.
    Null / n/a fließen nicht in die %-Quote ein.
    Ø Return = Mittelwert aller gültigen Werte (inkl. 0).
    """
    n_pos = n_neg = n_zero = n_na = 0
    vals: List[float] = []
    for row in rows:
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            n_na += 1
        else:
            fv = float(v)
            vals.append(fv)
            if fv > 0:
                n_pos += 1
            elif fv < 0:
                n_neg += 1
            else:
                n_zero += 1
    denom = n_pos + n_neg
    pct_pos = (100.0 * n_pos / denom) if denom > 0 else None
    avg_ret = float(np.mean(vals)) if vals else None
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "n_na": n_na,
        "pct_positive": pct_pos,
        "avg_return": avg_ret,
    }


def positivity_by_columns(rows: List[dict], cols: List[str]) -> Dict[str, dict]:
    return {c: column_positivity(rows, c) for c in cols}


def build_rows_for_coin(
    coin: str,
    df: pd.DataFrame,
    events: List[BtcMoveEvent],
) -> List[dict]:
    idx = index_by_time(df)
    rows: List[dict] = []
    for e in events:
        coin_rise = rise_in_window(idx, e.start_ms, e.end_ms)
        row: dict = {
            "direction": e.direction,
            "btc_rise_window_min": BTC_RISE_WINDOW_MIN,
            "btc_rise_threshold_pct": BTC_RISE_THRESHOLD_PCT,
            "btc_rise_start": ms_to_berlin_str(e.start_ms),
            "btc_rise_end": ms_to_berlin_str(e.end_ms),
            "btc_move_pct": e.btc_move_pct,
            "coin": coin,
            "coin_rise_pct_in_btc_window": coin_rise,
            "kline_interval": KLINE_INTERVAL,
        }
        for lag_name, lag_bars in POST_WINDOW_LAGS.items():
            row[lag_name] = post_window_return(idx, e.end_ms, lag_bars)
        rows.append(row)
    return rows


def fmt_pos_pct(stats: dict) -> str:
    p = stats.get("pct_positive")
    if p is None:
        return "   n/a"
    return f"{p:5.0f}%"


def fmt_avg_ret(stats: dict) -> str:
    a = stats.get("avg_return")
    if a is None:
        return "   n/a"
    return f"{a:+6.2f}%"


def print_coin_table(coin: str, rows: List[dict], direction: str) -> Dict[str, dict]:
    label = (
        f"BTC ≥ +{BTC_RISE_THRESHOLD_PCT}% / {BTC_RISE_WINDOW_MIN}m"
        if direction == "UP"
        else f"BTC ≤ −{BTC_RISE_THRESHOLD_PCT}% / {BTC_RISE_WINDOW_MIN}m"
    )
    post_cols = list(POST_WINDOW_LAGS.keys())
    # Zeit-/Return-Spalten für %-positiv (nicht BTC% — dort ist das Vorzeichen per Definition klar)
    value_cols = ["coin_rise_pct_in_btc_window"] + post_cols
    stats_by_col = positivity_by_columns(rows, value_cols)

    print()
    print("=" * 120)
    print(
        f"COIN {coin} | {direction} ({label}) | {len(rows)} Events | "
        f"Post-Lags ab Fensterende ({KLINE_INTERVAL})"
    )
    print("=" * 120)
    header = (
        f"{'Start (Berlin)':<24} {'Ende (Berlin)':<24} {'BTC%':>7} {'Coin%':>7}"
        + "".join(f"{c:>9}" for c in post_cols)
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row['btc_rise_start']:<24} {row['btc_rise_end']:<24} "
            f"{fmt_pct(row['btc_move_pct']):>7} "
            f"{fmt_pct(row.get('coin_rise_pct_in_btc_window')):>7}"
        )
        for c in post_cols:
            line += f"{fmt_pct(row.get(c)):>9}"
        #print(line)

    # Fußzeilen: % positiv und darunter Ø Return (Gain/Loss) je Spalte
    print("-" * len(header))
    foot = f"{'% positiv':<24} {'':<24} {'':>7} "
    foot += f"{fmt_pos_pct(stats_by_col['coin_rise_pct_in_btc_window']):>7}"
    for c in post_cols:
        foot += f"{fmt_pos_pct(stats_by_col[c]):>9}"
    print(foot)

    foot_avg = f"{'Ø Return':<24} {'':<24} {'':>7} "
    foot_avg += f"{fmt_avg_ret(stats_by_col['coin_rise_pct_in_btc_window']):>7}"
    for c in post_cols:
        foot_avg += f"{fmt_avg_ret(stats_by_col[c]):>9}"
    print(foot_avg)

    # Kurz darunter: pos/neg Counts nur für Coin%-Spalte (Fenster)
    cs = stats_by_col["coin_rise_pct_in_btc_window"]
    print(
        f"  (Coin%-Spalte: {cs['n_pos']} pos / {cs['n_neg']} neg"
        + (f" / {cs['n_zero']} null" if cs["n_zero"] else "")
        + (f" / {cs['n_na']} n/a" if cs["n_na"] else "")
        + "; % = pos/(pos+neg); Ø = Mittelwert je Spalte)"
    )
    print(flush=True)
    sys.stdout.flush()
    return stats_by_col




def resolve_universe() -> List[str]:
    usdt_set = binance_usdt_symbols()
    if COIN_PAIRS:
        pairs = []
        for p in COIN_PAIRS:
            p = p.upper()
            if not p.endswith("USDT"):
                p = f"{p}USDT"
            if p in usdt_set:
                pairs.append(p)
            else:
                print(f"   überspringe {p} (nicht auf Binance Spot USDT)", flush=True)
        return pairs

    coins = fetch_top_coins_no_stables(TOP_N)
    pairs = []
    skipped = []
    for c in coins:
        pair = map_to_binance_pair(c["symbol"], usdt_set)
        if pair:
            pairs.append(pair)
        else:
            skipped.append(c["symbol"])
    if skipped:
        print(f"   ohne Binance-Pair: {skipped}", flush=True)
    return pairs


def list_events(title: str, events: List[BtcMoveEvent]) -> None:
    print(f"\n   {title} ({len(events)}):", flush=True)
    show = events[:20]
    for e in show:
        print(
            f"   - {ms_to_berlin_str(e.start_ms)} → {ms_to_berlin_str(e.end_ms)} | "
            f"BTC {e.btc_move_pct:+.2f}%",
            flush=True,
        )
    if len(events) > 20:
        print(f"   … +{len(events) - 20} weitere", flush=True)


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BTC ±Fenster → Coin-Features (ohne BTC-Referenz)"
    )
    add_time_range_arguments(p)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_cli_args(argv)
    start, end, window_label = resolve_event_window(
        from_s=args.from_date or FROM_DATE,
        to_s=args.to_date or TO_DATE,
        lookback_days=args.lookback,  # only CLI; None → use LOOKBACK when no FROM
        default_lookback=LOOKBACK_DAYS,
    )
    print("=== BTC ±Fenster → Coin-Features (ohne BTC-Referenz) ===\n", flush=True)
    start_ms, end_ms = utc_ms(start), utc_ms(end)

    print(
        f"Zeitraum: {start.strftime('%Y-%m-%d %H:%M %Z')} → {end.strftime('%Y-%m-%d %H:%M %Z')} (Europe/Berlin)",
        flush=True,
    )
    print(
        f"UP:   BTC ≥ +{BTC_RISE_THRESHOLD_PCT}% in {BTC_RISE_WINDOW_MIN} Min "
        f"({WINDOW_BARS} × {KLINE_INTERVAL})",
        flush=True,
    )
    print(
        f"DOWN: BTC ≤ −{BTC_RISE_THRESHOLD_PCT}% in {BTC_RISE_WINDOW_MIN} Min",
        flush=True,
    )
    print(
        f"{window_label} | Intervall: {KLINE_INTERVAL} | "
        f"Fenster: {BTC_RISE_WINDOW_MIN} Min ({WINDOW_BARS} Bars)",
        flush=True,
    )
    print(
        f"Exclude: {sorted(EXCLUDE_SYMBOLS)} | Top-N: {TOP_N}",
        flush=True,
    )
    print(
        f"Post-Fenster-Lags (je Spalte eigene Metrik): {list(POST_WINDOW_LAGS)}\n",
        flush=True,
    )

    print("1) Coin-Universum…", flush=True)
    universe = resolve_universe()
    print(f"   {len(universe)} Paare\n", flush=True)

    print("2) BTC-Kerzen (nur für Event-Erkennung)…", flush=True)
    btc = fetch_klines("BTCUSDT", KLINE_INTERVAL, start_ms, end_ms)
    if btc.empty:
        raise SystemExit("Keine BTC-Daten.")
    print(f"   {len(btc)} Kerzen", flush=True)

    ups, downs = detect_btc_move_events(btc)
    print(f"\n3) Events: {len(ups)} UP  |  {len(downs)} DOWN", flush=True)
    if not ups and not downs:
        raise SystemExit("Keine Events — Schwelle/Fenster/LOOKBACK anpassen.")
    if ups:
        list_events("UP", ups)
    if downs:
        list_events("DOWN", downs)
    print(flush=True)

    summary_rows: List[dict] = []

    print(f"4) Coins einzeln ({len(universe)}) — kein BTC in der Ausgabe\n", flush=True)
    for i, pair in enumerate(universe, 1):
        print(f"--- [{i}/{len(universe)}] lade {pair} …", flush=True)
        try:
            df = fetch_klines(pair, KLINE_INTERVAL, start_ms, end_ms)
        except Exception as exc:  # noqa: BLE001
            print(f"    Fehler: {exc}\n", flush=True)
            continue
        if df.empty:
            print("    keine Kerzen\n", flush=True)
            continue
        print(f"    {len(df)} Kerzen — Features…", flush=True)

        for direction, events in (("UP", ups), ("DOWN", downs)):
            if not events:
                continue
            rows = build_rows_for_coin(pair, df, events)
            stats_by_col = print_coin_table(pair, rows, direction)
            summary: dict = {
                "coin": pair,
                "direction": direction,
                "n_events": len(rows),
            }
            for col, st in stats_by_col.items():
                key = "coin_window" if col == "coin_rise_pct_in_btc_window" else col
                summary[f"pct_pos_{key}"] = st["pct_positive"]
                summary[f"avg_{key}"] = st["avg_return"]
                summary[f"n_pos_{key}"] = st["n_pos"]
                summary[f"n_neg_{key}"] = st["n_neg"]
            summary_rows.append(summary)

        del df
        print(flush=True)

    if summary_rows:
        print("\nKurz-Übersicht % positiv | Ø Return je Spalte:", flush=True)
        lag_keys = ["coin_window"] + list(POST_WINDOW_LAGS.keys())
        hdr = f"{'Coin':<14} {'Dir':<5} {'n':>4}" + "".join(f"{k:>14}" for k in lag_keys)
        print(hdr, flush=True)
        for s in summary_rows:
            line = f"{s['coin']:<14} {s['direction']:<5} {s['n_events']:>4}"
            for k in lag_keys:
                pv = s.get(f"pct_pos_{k}")
                av = s.get(f"avg_{k}")
                p_s = f"{pv:.0f}%" if pv is not None else "n/a"
                a_s = f"{av:+.2f}%" if av is not None else "n/a"
                line += f"{(p_s + '|' + a_s):>14}"
            print(line, flush=True)

    print(
        "\nFertig. Nur Konsolen-Ausgabe — keine CSV.\n"
        "Positiv = Wert > 0; Negativ = < 0; % = pos/(pos+neg); "
        "Ø Return = Mittelwert je Spalte.\n"
        "Post-Lags = getrennte Metriken ab Fensterende.\n"
        "Keine Handelsempfehlung — Feature-/Analyse-Skript.\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
