#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Ref Asymmetric | Combined UP + DOWN Signals
=================================================

Eine gemeinsame Simulation:
  - UP-Signale (BTC + Refs steigen über UP-Schwelle)
  - DOWN-Signale (BTC + Refs fallen über DOWN-Schwelle)
  Beide dürfen einen Kauf auslösen, solange keine Position offen ist.

Schwellen für UP und DOWN sind getrennt.
Non-overlapping Holds. Grand über Top-N Targets.
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

FROM_DATE = "2024-01-01"
TO_DATE = "2026-12-28"
LOOKBACK_DAYS = 400
TOP_N = 30
MIN_TRADES = 15

REF_SYMBOLS = ["BTC", "ETH", "BCH", "DOGE", "TRX", "SOL"]
EVENT_WINDOWS_MIN = [60]

THRESHOLDS_UP = [0.5,1.0, 1.5, 2.0,2.5]
THRESHOLDS_DOWN = [0.5,1.0, 1.5, 2.0, 2.5]
MIN_REFS = [2, 3, 4]

HOLD_HORIZONS = {
    "12h": 720, "24h": 1440, "48h": 2880,
    "72h": 4320, "96h": 5760, "168h": 10080,
}

KLINE_INTERVAL = "1h"
FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0

COINGECKO = "https://api.coingecko.com/api/v3"
EXTRA_STABLE = {"USDE", "USD1", "PYUSD", "USDS", "BFUSD", "FDUSD"}
WRAPPED = {"WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBBTC", "BTCB", "WBNB", "WTRX"}


def apply_config():
    bar = base.interval_to_minutes(KLINE_INTERVAL)
    base.KLINE_INTERVAL = KLINE_INTERVAL
    base.BAR_MIN = bar
    base.EVENT_WINDOWS_MIN = list(EVENT_WINDOWS_MIN)
    base.HOLD_HORIZONS = dict(HOLD_HORIZONS)
    base.WINDOW_BARS = {w: w // bar for w in EVENT_WINDOWS_MIN}
    base.HOLD_BARS = {h: m // bar for h, m in HOLD_HORIZONS.items()}
    base.MAX_HOLD_MIN = max(HOLD_HORIZONS.values())
    base.HORIZON_NAMES = list(HOLD_HORIZONS.keys())
    base.FEE_BPS = FEE_BPS
    base.SLIPPAGE_BPS = SLIPPAGE_BPS


BERLIN = ZoneInfo("Europe/Berlin")


def is_bad(sym: str, name: str = "") -> bool:
    s = sym.upper()
    if s in getattr(base, "STABLE_SYMBOLS", set()) or s in EXTRA_STABLE or "USD" in s:
        return True
    if s in WRAPPED or any(w in name.lower() for w in ("wrapped", "staked")):
        return True
    return False


def fetch_top(n: int) -> List[dict]:
    r = requests.get(
        f"{COINGECKO}/coins/markets",
        params={"vs_currency": "usd", "order": "market_cap_desc",
                "per_page": min(n + 40, 250), "page": 1, "sparkline": "false"},
        timeout=30,
    )
    r.raise_for_status()
    return [{"symbol": str(x.get("symbol", "")).upper(),
             "name": str(x.get("name", ""))} for x in r.json()]


def build_universe(top_n: int, refs: List[str], usdt: set) -> List[Tuple[str, str]]:
    try:
        raw = fetch_top(top_n + 30)
    except Exception:
        raw = []
    out, seen = [], set()
    ref_set = set(refs)
    for item in raw:
        if len(out) >= top_n:
            break
        sym, name = item["symbol"], item["name"]
        if not sym or sym in seen or sym == "BTC" or sym in ref_set or is_bad(sym, name):
            continue
        seen.add(sym)
        pair = f"{sym}USDT"
        if pair in usdt:
            out.append((sym, pair))
    return out


def event_time(ev) -> Optional[int]:
    """Versucht die Zeit eines Events zu extrahieren (robust)."""
    if isinstance(ev, (int, np.integer)):
        return int(ev)
    if isinstance(ev, (float, np.floating)) and ev > 1e12:
        return int(ev)
    # MultiRefEvent und ähnliche Objekte
    for attr in ("end_ms", "time", "open_time", "timestamp", "ts", "t", "event_time"):
        v = getattr(ev, attr, None)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)) and v > 1e12:
            return int(v)
    # Dict
    if isinstance(ev, dict):
        for k in ("end_ms", "time", "open_time", "timestamp", "ts", "t"):
            if k in ev:
                v = ev[k]
                if isinstance(v, (int, np.integer)):
                    return int(v)
    # Tuple / list
    if isinstance(ev, (tuple, list)) and len(ev) > 0:
        v = ev[0]
        if isinstance(v, (int, np.integer)):
            return int(v)
    return None


def simulate_combined(target_df: pd.DataFrame, up_events: list, down_events: list,
                      hold_min: int) -> List[float]:
    """Non-overlapping Simulation: UP oder DOWN darf einsteigen, solange frei."""
    t_times = target_df["open_time"].to_numpy(np.int64)
    t_closes = target_df["close"].to_numpy(np.float64)
    hold_ms = hold_min * 60_000
    cost = FEE_BPS / 10_000.0

    # Alle Events mit Zeit sammeln
    timed = []
    for ev in up_events:
        ts = event_time(ev)
        if ts is not None:
            timed.append((ts, "UP"))
    for ev in down_events:
        ts = event_time(ev)
        if ts is not None:
            timed.append((ts, "DOWN"))
    timed.sort()

    rois = []
    last_exit = -1
    for ts, _dir in timed:
        if ts <= last_exit:
            continue
        e = int(np.searchsorted(t_times, ts))
        if e >= len(t_times) or t_times[e] != ts:
            continue
        x = int(np.searchsorted(t_times, ts + hold_ms))
        if x >= len(t_times) or t_times[x] != ts + hold_ms:
            continue
        entry = float(t_closes[e])
        exit_ = float(t_closes[x])
        if entry <= 0:
            continue
        roi = (exit_ * (1 - cost) / (entry * (1 + cost)) - 1.0) * 100.0
        rois.append(roi)
        last_exit = ts + hold_ms
    return rois


def compound_of(vals: List[float]) -> Optional[float]:
    if len(vals) < MIN_TRADES:
        return None
    eq = 1.0
    for v in vals:
        eq *= 1.0 + v / 100.0
    return (eq - 1.0) * 100.0


def main():
    t0 = time.time()
    apply_config()

    usdt = base.binance_usdt_symbols()
    refs, ref_pairs = [], {}
    for raw in REF_SYMBOLS:
        b, pair = base.resolve_pair(raw, usdt)
        if b not in refs:
            refs.append(b)
            ref_pairs[b] = pair

    start, end, _ = resolve_event_window(
        from_s=FROM_DATE, to_s=TO_DATE, lookback_days=None, default_lookback=LOOKBACK_DAYS
    )
    end = base.clamp_end_to_closed_candle(end)
    start_ms, end_ms = base.utc_ms(start), base.utc_ms(end)
    fetch_end = base.utc_ms(end + pd.Timedelta(minutes=base.MAX_HOLD_MIN))

    print("MULTI-REF COMBINED UP+DOWN | asymmetrische Schwellen")
    print(f"Refs: {', '.join(refs)}")
    print(f"UP: {THRESHOLDS_UP} | DOWN: {THRESHOLDS_DOWN}")
    print(f"Min Refs: {MIN_REFS} | Min Trades: {MIN_TRADES}\n")

    targets = build_universe(TOP_N, refs, usdt)
    print(f"Targets: {', '.join(b for b, _ in targets)}\n")

    print("Lade Refs...")
    ref_data = {}
    for b in refs:
        print(f"  {b}", end="", flush=True)
        df = base.fetch_klines(ref_pairs[b], KLINE_INTERVAL, start_ms, end_ms)
        ref_data[b] = df
        print(f" {len(df)}")

    print("\nErzeuge Events...")
    ev_up = base.detect_multi_ref_events(
        ref_data, ref_pairs, EVENT_WINDOWS_MIN, THRESHOLDS_UP, MIN_REFS, start_ms, end_ms
    )
    ev_down = base.detect_multi_ref_events(
        ref_data, ref_pairs, EVENT_WINDOWS_MIN, THRESHOLDS_DOWN, MIN_REFS, start_ms, end_ms
    )

    # Diagnose
    n_up = sum(len(v) for v in ev_up.values())
    n_down = sum(len(v) for v in ev_down.values())
    print(f"  UP-Events: {n_up} | DOWN-Events: {n_down}")
    sample = next((v[0] for v in ev_up.values() if v), None)
    if sample is not None:
        print(f"  Beispiel-Event Typ: {type(sample).__name__}")
        print(f"  Beispiel-Event Inhalt: {sample}")
        print(f"  event_time() erkannt: {event_time(sample)}")
    else:
        print("  Keine UP-Events gefunden – prüfe Schwellen / Refs.")
    print()

    # Grid über Kombinationen von UP-Thr, DOWN-Thr, MinRefs, Hold
    print("\nSimulation...\n")
    rows = []
    for i, (tbase, tpair) in enumerate(targets, 1):
        print(f"[{i:>2}/{len(targets)}] {tpair:<12}", end="", flush=True)
        try:
            tdf = base.fetch_klines(tpair, KLINE_INTERVAL, start_ms, fetch_end)
        except Exception:
            print(" error")
            continue
        if tdf.empty:
            print(" empty")
            continue

        best = None
        for up_thr in THRESHOLDS_UP:
            for down_thr in THRESHOLDS_DOWN:
                for q in MIN_REFS:
                    # Events für diese Kombination
                    up_key = (EVENT_WINDOWS_MIN[0], up_thr, q, "UP")
                    down_key = (EVENT_WINDOWS_MIN[0], down_thr, q, "DOWN")
                    up_evs = ev_up.get(up_key, [])
                    down_evs = ev_down.get(down_key, [])
                    if not up_evs and not down_evs:
                        continue
                    for h, hold_min in HOLD_HORIZONS.items():
                        rois = simulate_combined(tdf, up_evs, down_evs, hold_min)
                        cmp_ = compound_of(rois)
                        if cmp_ is None:
                            continue
                        n_pos = sum(v > 0 for v in rois)
                        cand = {
                            "up_thr": up_thr,
                            "down_thr": down_thr,
                            "min_refs": q,
                            "hold": h,
                            "n": len(rois),
                            "pct_pos": 100.0 * n_pos / len(rois),
                            "avg": float(np.mean(rois)),
                            "compound": cmp_,
                        }
                        if best is None or cand["compound"] > best["compound"]:
                            best = cand

        if best is None:
            print(" keine Variante")
            continue
        rows.append({"pair": tpair, "best": best})
        print(f" UP≥{best['up_thr']:g}% DOWN≥{best['down_thr']:g}% | "
              f"{best['min_refs']}/{len(refs)} | {best['hold']} → {best['compound']:+.1f}% (n={best['n']})")

    rows.sort(key=lambda r: r["best"]["compound"], reverse=True)
    print("\n" + "=" * 110)
    print("BESTE KOMBINATION JE COIN | UP + DOWN gemeinsam (non-overlapping)")
    print("=" * 110)
    print(f"{'#':>3} {'Coin':<12} {'UP≥':>6} {'DOWN≥':>7} {'Refs':>6} {'Hold':>6} "
          f"{'n':>5} {'%pos':>6} {'Avg':>8} {'Compound':>10}")
    print("-" * 80)
    for i, r in enumerate(rows, 1):
        b = r["best"]
        print(f"{i:>3} {r['pair']:<12} {b['up_thr']:>5.1f}% {b['down_thr']:>6.1f}% "
              f"{b['min_refs']}/{len(refs):<2} {b['hold']:>6} {b['n']:>5} "
              f"{b['pct_pos']:>5.0f}% {b['avg']:>+7.2f}% {b['compound']:>+9.1f}%")
    print("-" * 80)
    print("UP und DOWN können beide einen Einstieg auslösen, solange keine Position offen ist.")
    print(f"Fertig in {time.time()-t0:.0f}s. Keine Handelsempfehlung.")


if __name__ == "__main__":
    print(f"Start: {datetime.now(BERLIN):%Y-%m-%d %H:%M:%S %Z}")
    main()
