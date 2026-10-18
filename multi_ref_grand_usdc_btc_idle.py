#!/usr/bin/env python3
"""
Multi-Reference Grand | Vergleich Idle-USDT vs Idle-BTC
=======================================================

Gleiche Logik wie chatgpt_multi_ref_single_target_grand:
  Multi-Ref Events → Target, beste Variante nach Compound.

Zusätzlich wird für die beste Variante jedes Targets der Compound
berechnet, wenn zwischen den Trades BTC gehalten wird (statt Cash/USDT).
Gehandelt werden USDT-Paare.

Sortierung nach USDT-Idle-Compound.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import chatgpt_multi_ref_single_target as base
from laggard_common import resolve_event_window


# ============================================================
# CONFIG
# ============================================================

FROM_DATE: Optional[str] = "2024-01-01"
TO_DATE: Optional[str] = "2026-12-28"
LOOKBACK_DAYS = 400
TOP_N = 30
MIN_TRADES = 20

REF_SYMBOLS: List[str] = ["BTC", "ETH", "BCH", "DOGE", "TRX", "SOL"]
EVENT_WINDOWS_MIN: List[int] = [60]
THRESHOLDS_PCT: List[float] = [1.0, 1.5, 2.0]
MIN_REFS: List[int] = [1, 2, 3, 4]

HOLD_HORIZONS: Dict[str, int] = {
    "3h": 180, "6h": 360, "12h": 720, "24h": 1440,
    "48h": 2880, "60h": 3600, "72h": 4320, "78h": 4680,
    "84h": 5040, "90h": 5400, "96h": 5760,
    "120h": 7200, "168h": 10080, "192h": 11520,
}

KLINE_INTERVAL = "1h"
FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0

COINGECKO = "https://api.coingecko.com/api/v3"
COINGECKO_FETCH = 250
EXTRA_STABLE_SYMBOLS = {
    "USDE", "USD1", "PYUSD", "USDS", "BFUSD", "FDUSD", "USDX",
}
WRAPPED_STAKED_SYMBOLS = {
    "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBBTC", "BTCB", "WBNB", "WTRX",
}
WRAPPED_NAME_WORDS = ("wrapped", "staked", "bridged", "restaked")


def apply_config_to_base() -> None:
    bar_min = base.interval_to_minutes(KLINE_INTERVAL)
    if any(w % bar_min != 0 for w in EVENT_WINDOWS_MIN):
        raise SystemExit("Windows müssen durch Intervall teilbar sein.")
    if any(h % bar_min != 0 for h in HOLD_HORIZONS.values()):
        raise SystemExit("Holds müssen durch Intervall teilbar sein.")
    base.KLINE_INTERVAL = KLINE_INTERVAL
    base.BAR_MIN = bar_min
    base.EVENT_WINDOWS_MIN = list(EVENT_WINDOWS_MIN)
    base.THRESHOLDS_PCT = list(THRESHOLDS_PCT)
    base.MIN_REFS = list(MIN_REFS)
    base.HOLD_HORIZONS = dict(HOLD_HORIZONS)
    base.WINDOW_BARS = {w: w // bar_min for w in EVENT_WINDOWS_MIN}
    base.HOLD_BARS = {h: mins // bar_min for h, mins in HOLD_HORIZONS.items()}
    base.MAX_HOLD_MIN = max(HOLD_HORIZONS.values())
    base.HORIZON_NAMES = list(HOLD_HORIZONS.keys())
    base.FEE_BPS = FEE_BPS
    base.SLIPPAGE_BPS = SLIPPAGE_BPS


BERLIN = ZoneInfo("Europe/Berlin")


def progress_bar(current: int, total: int, prefix: str = "", length: int = 30) -> None:
    if total <= 0:
        return
    filled = int(length * current / total)
    bar = "█" * filled + "░" * (length - filled)
    pct = 100.0 * current / total
    print(f"\r{prefix} |{bar}| {pct:5.1f}% ({current}/{total})", end="", flush=True)
    if current >= total:
        print()


def effective_range_ms(start_ms: int, end_ms: int, *time_arrays) -> Tuple[int, int]:
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


def binance_pair_for(base_sym: str, usdt_set: set) -> Optional[str]:
    pair = f"{base_sym}USDT"
    return pair if pair in usdt_set else None


def is_stable(sym: str, name: str = "") -> bool:
    s = sym.upper()
    if s in getattr(base, "STABLE_SYMBOLS", set()) or s in EXTRA_STABLE_SYMBOLS or "USD" in s:
        return True
    return "stable" in name.lower()


def is_wrapped(sym: str, name: str = "") -> bool:
    if sym.upper() in WRAPPED_STAKED_SYMBOLS:
        return True
    return any(w in name.lower() for w in WRAPPED_NAME_WORDS)


def fetch_coingecko_markets(n: int) -> List[dict]:
    r = requests.get(
        f"{COINGECKO}/coins/markets",
        params={"vs_currency": "usd", "order": "market_cap_desc",
                "per_page": min(n, 250), "page": 1, "sparkline": "false"},
        timeout=30,
    )
    r.raise_for_status()
    return [{"symbol": str(x.get("symbol", "")).upper(),
             "name": str(x.get("name", ""))} for x in r.json()]


def build_universe(top_n: int, ref_bases: List[str], usdt_set: set):
    try:
        raw = fetch_coingecko_markets(COINGECKO_FETCH)
        source = "CoinGecko"
    except Exception:
        raw = []
        source = "Fallback"
    targets, skipped, seen = [], [], set()
    ref_set = {r.upper() for r in ref_bases}
    for item in raw:
        if len(targets) >= top_n:
            break
        sym, name = item["symbol"], item["name"]
        if not sym or sym in seen or sym == "BTC" or sym in ref_set:
            continue
        seen.add(sym)
        if is_stable(sym, name) or is_wrapped(sym, name):
            skipped.append(sym)
            continue
        pair = binance_pair_for(sym, usdt_set)
        if pair:
            targets.append((sym, pair))
    return targets, skipped, source


def compound_btc_idle(btc_df: pd.DataFrame, kept_trades: List[dict], hold_name: str,
                      start_ms: int, end_ms: int) -> Optional[float]:
    """Compound der Strategie, wenn zwischen den Trades BTC gehalten wird.

    Verwendet exakt dieselben Trades wie der USDT-Compound (base.summarize:
    non-overlapping, Re-Entry genau beim Exit erlaubt). Zwischen den Trades
    und vor dem ersten / nach dem letzten Trade liegt das Kapital in BTC,
    über genau den Zeitraum start..end (BTC-Close, keine Kerze nach end).
    """
    if len(kept_trades) < MIN_TRADES:
        return None
    b = btc_df[(btc_df["open_time"] >= start_ms) & (btc_df["open_time"] <= end_ms)]
    if b.empty:
        return None
    b_times = b["open_time"].to_numpy(np.int64)
    b_closes = b["close"].to_numpy(np.float64)

    def btc_at(t_ms: int) -> float:
        # BTC-Close der Kerze mit open_time == t, sonst letzte bekannte (<= t).
        i = int(np.searchsorted(b_times, t_ms, side="right")) - 1
        return float(b_closes[max(i, 0)])

    hold_ms = HOLD_HORIZONS[hold_name] * 60_000
    col = f"roi_{hold_name}"
    equity = 1.0
    last_price = float(b_closes[0])          # Start: BTC ab start
    for tr in kept_trades:
        buy_ms = int(tr["buy_time_ms"])
        p = btc_at(buy_ms)
        if last_price > 0:
            equity *= p / last_price          # BTC-Idle bis zum Entry
        equity *= 1.0 + float(tr[col]) / 100.0  # Target-Trade (mit Fees, wie USDT)
        last_price = btc_at(buy_ms + hold_ms)  # nach Exit wieder BTC
    if last_price > 0:
        equity *= float(b_closes[-1]) / last_price  # BTC bis end
    return (equity - 1.0) * 100.0


def main():
    t0 = time.time()
    apply_config_to_base()

    usdt_set = base.binance_usdt_symbols()
    ref_bases, ref_pairs = [], {}
    for raw in REF_SYMBOLS:
        b, pair = base.resolve_pair(raw, usdt_set)
        if b not in ref_bases:
            ref_bases.append(b)
            ref_pairs[b] = pair

    start, end, _ = resolve_event_window(
        from_s=FROM_DATE, to_s=TO_DATE, lookback_days=None, default_lookback=LOOKBACK_DAYS
    )
    end = base.clamp_end_to_closed_candle(end)
    start_ms, end_ms = base.utc_ms(start), base.utc_ms(end)
    # Keine Daten / Exits nach TO_DATE (wie btc_switch / single_target).
    fetch_end_ms = end_ms

    print("MULTI-REF GRAND | USDT-Idle vs BTC-Idle")
    print(f"Refs: {', '.join(ref_bases)} | MinTrades {MIN_TRADES}")

    print("\n1) Universe...")
    targets, _, source = build_universe(TOP_N, ref_bases, usdt_set)
    print(f"   {source} | {len(targets)} Targets")

    print("\n2) Load refs + BTC...")
    ref_data = {}
    for b in ref_bases:
        print(f"   {b}", end="", flush=True)
        df = base.fetch_klines(ref_pairs[b], KLINE_INTERVAL, start_ms, end_ms)
        df = df[df["open_time"] <= end_ms].reset_index(drop=True)
        ref_data[b] = df
        print(f" {len(df)}")
    btc_df = ref_data.get("BTC")
    if btc_df is None:
        print("   BTC extra...", end="", flush=True)
        btc_df = base.fetch_klines("BTCUSDT", KLINE_INTERVAL, start_ms, fetch_end_ms)
        btc_df = btc_df[btc_df["open_time"] <= end_ms].reset_index(drop=True)
        print(f" {len(btc_df)}")

    print("\n3) Events...")
    all_events = base.detect_multi_ref_events(
        ref_data, ref_pairs, EVENT_WINDOWS_MIN, THRESHOLDS_PCT, MIN_REFS, start_ms, end_ms
    )

    print("\n4) Targets\n")
    rows = []
    total_targets = len(targets)
    for i, (tbase, tpair) in enumerate(targets, 1):
        progress_bar(i, total_targets, prefix="Targets")
        print(f"  {tpair:<12}", end="", flush=True)
        try:
            tdf = base.fetch_klines(tpair, KLINE_INTERVAL, start_ms, fetch_end_ms)
        except Exception:
            print(" load error")
            continue
        tdf = tdf[tdf["open_time"] <= end_ms].reset_index(drop=True)
        if tdf.empty:
            print(" empty")
            continue

        best = None
        for window in EVENT_WINDOWS_MIN:
            for threshold in THRESHOLDS_PCT:
                for min_refs in MIN_REFS:
                    for direction in ("UP", "DOWN"):
                        key = (window, threshold, min_refs, direction)
                        events = all_events.get(key, [])
                        if not events:
                            continue
                        trades = base.simulate_events(tdf, events, tpair)
                        stats = base.summarize(trades)
                        for h in base.HORIZON_NAMES:
                            s = stats[h]
                            if s["n"] < MIN_TRADES or s["compound"] is None:
                                continue
                            cand = {
                                "direction": direction, "window": window,
                                "threshold": threshold, "min_refs": min_refs,
                                "hold": h, "n": s["n"],
                                "compound_usdc": s["compound"],
                                "trades": trades,
                            }
                            if best is None or cand["compound_usdc"] > best["compound_usdc"]:
                                best = cand
        if best is None:
            print(" no variant")
            continue
        kept = base.filter_non_overlapping(best.pop("trades"), best["hold"])
        best["compound_btc"] = compound_btc_idle(btc_df, kept, best["hold"], start_ms, end_ms)
        rows.append({"pair": tpair, "best": best})
        btc_s = f"{best['compound_btc']:+.1f}%" if best["compound_btc"] is not None else "n/a"
        print(f" USDT {best['compound_usdc']:+.1f}% | BTC-Idle {btc_s}")

    rows.sort(key=lambda r: r["best"]["compound_usdc"], reverse=True)
    print("\n" + "=" * 110)
    print(f"BESTE VARIANTE | sortiert nach USDT-Idle | from {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    print("=" * 110)
    print(f"{'#':>3} {'Coin':<12} {'Dir':<5} {'Win':>5} {'Thr':>6} {'Refs':>6} {'Hold':>6} "
          f"{'n':>5} {'USDT':>9} {'BTC-Idle':>10} {'Δ':>8}")
    print("-" * 85)
    for i, r in enumerate(rows, 1):
        b = r["best"]
        delta = ""
        if b["compound_btc"] is not None:
            delta = f"{b['compound_btc'] - b['compound_usdc']:+.1f}%"
        btc_s = f"{b['compound_btc']:+.1f}%" if b["compound_btc"] is not None else "n/a"
        print(f"{i:>3} {r['pair']:<12} {b['direction']:<5} {b['window']:>4}m "
              f"{b['threshold']:>5.1f}% {b['min_refs']}/{len(ref_bases):<2} {b['hold']:>6} "
              f"{b['n']:>5} {b['compound_usdc']:>+8.1f}% {btc_s:>10} {delta:>8}")
    print("-" * 85)
    print("USDT = nur Trade-Returns (Kapital zwischen Trades in USDT) | BTC-Idle = gleiche Trades + BTC zwischen den Trades (ohne Fees für BTC-Wechsel)")
    print(f"Fertig in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    print(f"Start: {datetime.now(BERLIN):%Y-%m-%d %H:%M:%S %Z}")
    main()
