#!/usr/bin/env python3
"""
Multi-Reference Buy Backtest -> GRAND über die Top-N Alt-Coins

Basis: chatgpt_btc_rise_multi_ref_buy.py (wird importiert, NICHT verändert).
Simulation, Gebühren, Signal- und Hold-Logik sind identisch, weil die
Funktionen direkt aus dem Basisscript verwendet werden:
  detect_multi_ref_events, simulate_events, summarize (Non-overlap je Hold),
  buy_and_hold, apply_costs, fetch_klines.

Ablauf:
  1. Target-Universum: Top-N Alt-Coins nach Market-Cap (CoinGecko),
     ohne BTC, Stablecoins, Wrapped/Staked-Token, Gold-Token,
     EXCLUDE_SYMBOLS (HYPE, ZEC wie im Basisscript) und ohne Coins
     ohne Binance-USDT-Pair. Refs werden als Target übersprungen und
     die Liste wird aufgefüllt, sodass TOP_N Coins getestet werden.
     Fallback ohne CoinGecko: Binance 24h-Quote-Volumen.
  2. Referenzdaten einmal laden, Events einmal erzeugen (gelten für alle Targets).
  3. Pro Target: alle Varianten
       Event-Fenster x Threshold x Min-Refs x UP/DOWN x Hold
     simulieren und die beste Variante nach Compound wählen
     (nur Varianten mit n >= MIN_TRADES).
  4. Am Ende eine Tabelle, eine Zeile pro Coin, sortiert nach Compound.

Nur Konsolen-Ausgabe, keine CSV, kein Cache.

CLI (optional, überschreibt die Config oben):
  python chatgpt_multi_ref_single_target_grand.py
  python chatgpt_multi_ref_single_target_grand.py --top-n 10 --min-trades 30
  python chatgpt_multi_ref_single_target_grand.py --from 2025-06-01 --to 2026-09-30
  python chatgpt_multi_ref_single_target_grand.py --coins AAVE UNI LINK
  python chatgpt_multi_ref_single_target_grand.py --refs BTC ETH SOL
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import chatgpt_btc_rise_multi_ref_buy as base
from laggard_common import add_time_range_arguments, resolve_event_window


# ============================================================
# CONFIG (hier in VS Code ändern)
# ============================================================

# Zeitraum (Europe/Berlin). TO_DATE=None -> bis jetzt.
# FROM_DATE=None -> LOOKBACK_DAYS rückwärts ab jetzt.
FROM_DATE: Optional[str] = "2024-01-01"
TO_DATE: Optional[str] = "2026-12-28"
LOOKBACK_DAYS = 400

# Anzahl getesteter Target-Coins (Top-N Alts nach Market-Cap).
TOP_N = 30

# Mindestanzahl Trades (nach Non-overlap), damit eine Variante zählt.
MIN_TRADES = 20

# --- Grid / Refs: Default = Werte aus chatgpt_btc_rise_multi_ref_buy.py ---
REF_SYMBOLS: List[str] = ["BTC", "SOL", "ETH", "XRP"]

EVENT_WINDOWS_MIN: List[int] = [60, 180]

THRESHOLDS_PCT: List[float] = [1.0, 2.0]

# Wie viele Refs müssen gleichzeitig in dieselbe Richtung laufen?
MIN_REFS: List[int] = [1, 2, 3]

HOLD_HORIZONS: Dict[str, int] = {
    "3h": 180,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
    "48h": 2880,
    "60h": 3600,
    "72h": 4320,
    "78h": 4680,
    "84h": 5040,
    "90h": 5400,
    "96h": 5760,
    "120h": 7200,
    "168h": 10080,
    "192h": 11520,
}

KLINE_INTERVAL = "1h"

FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0

# --- Universum ---
COINGECKO = "https://api.coingecko.com/api/v3"

# Zusätzlich zu base.STABLE_SYMBOLS / base.EXCLUDE_SYMBOLS (HYPE, ZEC).
EXTRA_STABLE_SYMBOLS = {
    "USDE", "USD1", "PYUSD", "USDS", "USDTB", "BFUSD", "USDF", "USDG",
    "RLUSD", "FDUSD", "USDX", "USDB", "USR", "USDO", "AUSD", "EURS",
    "EURC", "EURT", "XUSD", "USDA", "DOLA", "MIM", "SUSDS", "SUSDE",
    "U", "USDD", "USDGO",
}

# Wrapped / Staked / Bridged / Gold-Token -> keine eigenständigen Alts.
WRAPPED_STAKED_SYMBOLS = {
    "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "EETH", "RETH", "CBETH",
    "CBBTC", "LBTC", "SOLVBTC", "BTCB", "WBETH", "BETH", "METH", "EZETH",
    "RSETH", "JITOSOL", "MSOL", "BNSOL", "JUPSOL", "BBSOL", "STSOL",
    "TBTC", "FBTC", "CLBTC", "UNIBTC", "PUMPBTC", "WBNB", "WTRX", "WSOL",
    "WAVAX", "WMATIC", "WPOL", "STHYPE", "KHYPE", "OSETH", "SWETH",
    "BUIDL", "USYC", "OUSG", "XAUT", "PAXG", "KAG", "XAUM",
}
WRAPPED_NAME_WORDS = ("wrapped", "staked", "bridged", "restaked", "liquid staking", "tokenized")

# Wie viele CoinGecko-Coins anfragen (Puffer für Filter).
COINGECKO_FETCH = 250


# ============================================================
# BASE-KONFIG ÜBERNEHMEN
# ============================================================

def apply_config_to_base() -> None:
    """Grid/Fees in das Basismodul schreiben, damit dessen Funktionen
    exakt diese Werte verwenden (gleiche Mathematik wie im Basisscript)."""
    bar_min = base.interval_to_minutes(KLINE_INTERVAL)
    if any(w % bar_min != 0 for w in EVENT_WINDOWS_MIN):
        raise SystemExit("Alle EVENT_WINDOWS_MIN müssen durch KLINE_INTERVAL teilbar sein.")
    if any(h % bar_min != 0 for h in HOLD_HORIZONS.values()):
        raise SystemExit("Alle HOLD_HORIZONS müssen durch KLINE_INTERVAL teilbar sein.")
    if any(x <= 0 for x in THRESHOLDS_PCT):
        raise SystemExit("THRESHOLDS_PCT müssen > 0 sein.")

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


# ============================================================
# HELPERS
# ============================================================

BERLIN = ZoneInfo("Europe/Berlin")


def berlin_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=BERLIN).strftime("%Y-%m-%d")


def effective_range_ms(start_ms: int, end_ms: int, *time_arrays) -> Tuple[int, int]:
    """[start, end] begrenzt auf tatsächlich vorhandene Kerzen."""
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


def block_range(from_ms: int, to_ms: int) -> str:
    return f"from {berlin_date(from_ms)} to {berlin_date(to_ms)}"


def binance_pair_for(base_sym: str, usdt_set: set) -> Optional[str]:
    aliases = {
        "RNDR": "RENDERUSDT",
        "RENDER": "RENDERUSDT",
        "MATIC": "MATICUSDT",
        "POL": "POLUSDT",
    }
    if base_sym in aliases and aliases[base_sym] in usdt_set:
        return aliases[base_sym]
    pair = f"{base_sym}USDT"
    return pair if pair in usdt_set else None


def is_stable(sym: str, name: str = "") -> bool:
    s = sym.upper()
    if s in base.STABLE_SYMBOLS or s in EXTRA_STABLE_SYMBOLS:
        return True
    if "USD" in s or "EUR" in s:
        return True
    n = name.lower()
    words = n.replace("(", " ").replace(")", " ").split()
    return (
        "stable" in n          # z.B. "United Stables" (U)
        or "dollar" in n
        or "usd" in words
    )


def is_wrapped_or_staked(sym: str, name: str = "") -> bool:
    if sym.upper() in WRAPPED_STAKED_SYMBOLS:
        return True
    n = name.lower()
    return any(w in n for w in WRAPPED_NAME_WORDS)


# ============================================================
# UNIVERSUM
# ============================================================

def fetch_coingecko_markets(n: int) -> List[dict]:
    r = requests.get(
        f"{COINGECKO}/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(int(n), 250),
            "page": 1,
            "sparkline": "false",
        },
        headers={"User-Agent": "multi-ref-grand/1.0"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError("CoinGecko lieferte keine Liste")
    return [
        {"symbol": str(x.get("symbol", "")).upper(), "name": str(x.get("name", ""))}
        for x in data
    ]


def fetch_binance_volume_candidates() -> List[dict]:
    r = base.SESSION.get(f"{base.BINANCE}/api/v3/ticker/24hr", timeout=30)
    r.raise_for_status()
    rows = []
    for t in r.json():
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        try:
            vol = float(t.get("quoteVolume", 0) or 0)
        except (TypeError, ValueError):
            vol = 0.0
        rows.append((vol, sym[:-4]))
    rows.sort(reverse=True)
    return [{"symbol": b, "name": ""} for _, b in rows]


def build_universe(
    top_n: int,
    ref_bases: List[str],
    usdt_set: set,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """Liefert (targets[(base, pair)], skipped[(symbol, grund)], quelle)."""
    try:
        raw = fetch_coingecko_markets(COINGECKO_FETCH)
        source = "CoinGecko Market-Cap"
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"   CoinGecko fehlgeschlagen: {str(exc)[:160]} -> Binance-Volumen", flush=True)
        raw = fetch_binance_volume_candidates()
        source = "Binance 24h-Quote-Volumen (Fallback)"

    targets: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    seen = set()
    ref_set = {r.upper() for r in ref_bases}

    for item in raw:
        if len(targets) >= top_n:
            break
        sym, name = item["symbol"], item["name"]
        if not sym or sym in seen:
            continue
        seen.add(sym)
        if sym == "BTC":
            continue
        if is_stable(sym, name):
            skipped.append((sym, "Stablecoin"))
            continue
        if is_wrapped_or_staked(sym, name):
            skipped.append((sym, "Wrapped/Staked/Gold-Token"))
            continue
        if sym in base.EXCLUDE_SYMBOLS:
            skipped.append((sym, "EXCLUDE_SYMBOLS"))
            continue
        if sym in ref_set:
            skipped.append((sym, "ist Referenz"))
            continue
        pair = binance_pair_for(sym, usdt_set)
        if pair is None:
            skipped.append((sym, "kein Binance-USDT-Pair"))
            continue
        targets.append((sym, pair))

    return targets, skipped, source


# ============================================================
# BEST VARIANT
# ============================================================

def best_variant_for_target(
    target_df: pd.DataFrame,
    target_pair: str,
    all_events: Dict[Tuple[int, float, int, str], List],
    bh: Optional[dict],
    min_trades: int,
) -> Tuple[Optional[dict], int, int]:
    """Alle Varianten simulieren, beste nach Compound (n >= min_trades).
    Rückgabe: (best, anzahl_varianten, anzahl_qualifiziert)."""
    best: Optional[dict] = None
    n_variants = 0
    n_ok = 0
    bh_roi = bh["roi"] if bh else None

    for window in EVENT_WINDOWS_MIN:
        for threshold in THRESHOLDS_PCT:
            for min_refs in MIN_REFS:
                for direction in ("UP", "DOWN"):
                    key = (window, threshold, min_refs, direction)
                    events = all_events.get(key, [])
                    trades = base.simulate_events(target_df, events, target_pair) if events else []
                    stats = base.summarize(trades)

                    for h in base.HORIZON_NAMES:
                        n_variants += 1
                        s = stats[h]
                        if s["compound"] is None or s["n"] < min_trades:
                            continue
                        n_ok += 1
                        cand = {
                            "direction": direction,
                            "window": window,
                            "threshold": threshold,
                            "min_refs": min_refs,
                            "hold": h,
                            "n": s["n"],
                            "pct_pos": s["pct_pos"],
                            "avg": s["avg"],
                            "compound": s["compound"],
                            "vs_bh": (s["compound"] - bh_roi) if bh_roi is not None else None,
                        }
                        if (
                            best is None
                            or cand["compound"] > best["compound"]
                            or (cand["compound"] == best["compound"] and cand["n"] > best["n"])
                        ):
                            best = cand

    return best, n_variants, n_ok


# ============================================================
# OUTPUT
# ============================================================

def variant_label(b: dict, total_refs: int) -> str:
    sign = "+" if b["direction"] == "UP" else "-"
    return (
        f"{b['direction']} | Event {b['window']}m | {sign}{b['threshold']:g}% | "
        f"Refs {base.min_ref_label(b['min_refs'], total_refs)} | Hold {b['hold']}"
    )


def print_summary_table(
    rows: List[dict],
    total_refs: int,
    ref_bases: List[str],
    period: str,
    min_trades: int,
) -> None:
    rows = sorted(rows, key=lambda r: r["best"]["compound"], reverse=True)
    width = 132

    print()
    print("=" * width)
    print(
        f"BESTE VARIANTE JE COIN | Refs {', '.join(ref_bases)} | "
        f"min n={min_trades} | {period}"
    )
    print("=" * width)
    header = (
        f"{'#':>3} "
        f"{'Coin':<12} "
        f"{'Dir':<5} "
        f"{'Event':>6} "
        f"{'Thresh':>7} "
        f"{'Refs':>10} "
        f"{'Hold':>6} "
        f"{'n':>5} "
        f"{'%pos':>6} "
        f"{'Ø ROI':>9} "
        f"{'Compound':>11} "
        f"{'vs B&H':>11} "
        f"{'B&H':>10}  "
        f"{'Daten ab':<10}"
    )
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows, 1):
        b = r["best"]
        sign = "+" if b["direction"] == "UP" else "-"
        print(
            f"{i:>3} "
            f"{r['pair']:<12} "
            f"{b['direction']:<5} "
            f"{str(b['window']) + 'm':>6} "
            f"{sign + format(b['threshold'], 'g') + '%':>7} "
            f"{base.min_ref_label(b['min_refs'], total_refs):>10} "
            f"{b['hold']:>6} "
            f"{b['n']:>5} "
            f"{base.fmt_pos(b['pct_pos']):>6} "
            f"{base.fmt_pct(b['avg']):>9} "
            f"{base.fmt_pct(b['compound']):>11} "
            f"{base.fmt_pct(b['vs_bh']):>11} "
            f"{base.fmt_pct(r['bh_roi']):>10}  "
            f"{r['late_start'] or '':<10}"
        )

    print("-" * len(header))
    print(
        "Compound = Produkt der non-overlapping Trade-Returns der besten Variante "
        "(gleiche Logik wie chatgpt_btc_rise_multi_ref_buy.py)."
    )
    print("vs B&H = Compound minus Buy&Hold des Coins im selben Zeitraum.")
    print("Daten ab = nur gesetzt, wenn der Coin erst nach dem Startdatum Kerzen hat (späteres Listing).")
    print("UP und DOWN sind beide BUY-Signale. Jede Variante = eigene Strategie.")


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-Reference Buy Backtest über Top-N Alts, beste Variante je Coin"
    )
    p.add_argument("--top-n", type=int, default=None, help=f"Anzahl Coins (Default: {TOP_N})")
    p.add_argument("--min-trades", type=int, default=None, help=f"Min. n (Default: {MIN_TRADES})")
    p.add_argument("--refs", "--ref", nargs="+", default=None, metavar="COIN",
                   help=f"Referenzcoins (Default: {' '.join(REF_SYMBOLS)})")
    p.add_argument("--coins", nargs="+", default=None, metavar="COIN",
                   help="Explizite Target-Liste statt Top-N")
    add_time_range_arguments(p)
    return p.parse_args(argv)


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[List[str]] = None) -> None:
    t0 = time.time()
    args = parse_args(argv)
    apply_config_to_base()

    top_n = TOP_N if args.top_n is None else max(1, int(args.top_n))
    min_trades = MIN_TRADES if args.min_trades is None else max(1, int(args.min_trades))
    refs_raw = list(dict.fromkeys(
        str(x).strip().upper() for x in (args.refs or REF_SYMBOLS)
    ))
    if not refs_raw:
        raise SystemExit("Mindestens ein Referenzcoin nötig.")

    usdt_set = base.binance_usdt_symbols()

    ref_bases: List[str] = []
    ref_pairs: Dict[str, str] = {}
    for raw in refs_raw:
        b, pair = base.resolve_pair(raw, usdt_set)
        if b not in ref_bases:
            ref_bases.append(b)
            ref_pairs[b] = pair

    start, end, window_label = resolve_event_window(
        from_s=args.from_date or FROM_DATE,
        to_s=args.to_date or TO_DATE,
        lookback_days=args.lookback,
        default_lookback=LOOKBACK_DAYS,
    )
    start_ms = base.utc_ms(start)
    end_ms = base.utc_ms(end)
    fetch_end_ms = base.utc_ms(end + pd.Timedelta(minutes=base.MAX_HOLD_MIN))

    print("\n" + "=" * 118)
    print("MULTI-REFERENCE BUY | GRAND über Top-Alts | beste Variante je Coin")
    print("=" * 118)
    print(f"Refs:       {', '.join(ref_bases)}")
    print(f"Events:     {', '.join(str(x) + 'm' for x in EVENT_WINDOWS_MIN)}")
    print(f"Thresholds: {', '.join(f'{x:g}%' for x in THRESHOLDS_PCT)}")
    print("Min Refs:   " + ", ".join(base.min_ref_label(x, len(ref_bases)) for x in MIN_REFS))
    print(f"Holds:      {', '.join(HOLD_HORIZONS.keys())}")
    n_grid = len(EVENT_WINDOWS_MIN) * len(THRESHOLDS_PCT) * len(MIN_REFS) * 2 * len(HOLD_HORIZONS)
    print(f"Varianten je Coin: {n_grid} | MIN_TRADES: {min_trades}")
    print(
        f"Zeitraum:   {start.strftime('%Y-%m-%d %H:%M %Z')} -> "
        f"{end.strftime('%Y-%m-%d %H:%M %Z')} ({window_label})"
    )
    print(f"Interval:   {KLINE_INTERVAL} | Fee: {FEE_BPS} bps/Seite | Slippage: {SLIPPAGE_BPS} bps")
    print("=" * 118)

    # --------------------------------------------------------
    # Target-Universum
    # --------------------------------------------------------

    skipped_universe: List[Tuple[str, str]] = []
    if args.coins:
        targets: List[Tuple[str, str]] = []
        for raw in args.coins:
            b, _ = base.normalize_symbol(raw)
            if b in ref_bases:
                skipped_universe.append((b, "ist Referenz"))
                continue
            pair = binance_pair_for(b, usdt_set)
            if pair is None:
                skipped_universe.append((b, "kein Binance-USDT-Pair"))
                continue
            targets.append((b, pair))
        source = "CLI --coins"
    else:
        print(f"\n1) Target-Universum (Top {top_n} Alts) ...", flush=True)
        targets, skipped_universe, source = build_universe(top_n, ref_bases, usdt_set)

    print(f"   Quelle: {source} | {len(targets)} Targets", flush=True)
    print("   " + ", ".join(b for b, _ in targets), flush=True)
    if skipped_universe:
        print(
            "   übersprungen: "
            + ", ".join(f"{s} ({why})" for s, why in skipped_universe),
            flush=True,
        )
    if not targets:
        raise SystemExit("Keine Targets.")

    # --------------------------------------------------------
    # Referenzdaten + Events (einmal für alle Targets)
    # --------------------------------------------------------

    print("\n2) Lade Referenzcoins:", flush=True)
    ref_data: Dict[str, pd.DataFrame] = {}
    for b in ref_bases:
        pair = ref_pairs[b]
        print(f"   {b:>8} ({pair}) ...", end="", flush=True)
        df = base.fetch_klines(pair, KLINE_INTERVAL, start_ms, end_ms)
        if df.empty:
            raise SystemExit(f"\nKeine Kerzen für Referenz {pair}.")
        ref_data[b] = df
        print(f" {len(df)} Kerzen", flush=True)

    period_from, period_to = effective_range_ms(
        start_ms, end_ms, *[d["open_time"].to_numpy() for d in ref_data.values()]
    )

    print("\n3) Erzeuge Multi-Reference Events (einmal für alle Coins) ...", flush=True)
    t_ev = time.time()
    all_events = base.detect_multi_ref_events(
        ref_data,
        ref_pairs,
        EVENT_WINDOWS_MIN,
        THRESHOLDS_PCT,
        MIN_REFS,
        start_ms,
        end_ms,
    )
    print(
        f"   {sum(len(v) for v in all_events.values())} Event-Instanzen über "
        f"{len(all_events)} Konfigurationen ({time.time() - t_ev:.0f}s)",
        flush=True,
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    print(f"\n4) Simulation je Coin ({len(targets)})\n", flush=True)
    rows: List[dict] = []
    failed: List[Tuple[str, str]] = []

    for i, (tbase, tpair) in enumerate(targets, 1):
        t_coin = time.time()
        prefix = f"[{i:>2}/{len(targets)}] {tpair:<12}"
        try:
            target_df = base.fetch_klines(tpair, KLINE_INTERVAL, start_ms, fetch_end_ms)
        except Exception as exc:  # noqa: BLE001
            print(f"{prefix} Fehler beim Laden: {exc}", flush=True)
            failed.append((tpair, f"Laden fehlgeschlagen: {str(exc)[:80]}"))
            continue
        if target_df.empty:
            print(f"{prefix} keine Kerzen", flush=True)
            failed.append((tpair, "keine Kerzen im Zeitraum"))
            continue

        bh = base.buy_and_hold(target_df, start_ms, end_ms)
        best, n_var, n_ok = best_variant_for_target(
            target_df, tpair, all_events, bh, min_trades
        )

        coin_from, _ = effective_range_ms(
            period_from, period_to, target_df["open_time"].to_numpy()
        )
        late_start = (
            berlin_date(coin_from)
            if berlin_date(coin_from) != berlin_date(period_from)
            else None
        )

        if best is None:
            print(
                f"{prefix} keine Variante mit n >= {min_trades} "
                f"({n_var} Varianten, {time.time() - t_coin:.0f}s)",
                flush=True,
            )
            failed.append((tpair, f"keine Variante mit n >= {min_trades}"))
            continue

        rows.append({
            "pair": tpair,
            "best": best,
            "bh_roi": bh["roi"] if bh else None,
            "late_start": late_start,
        })
        print(
            f"{prefix} best: {variant_label(best, len(ref_bases))} | "
            f"n={best['n']} | Cmp {base.fmt_pct(best['compound'])} | "
            f"vs B&H {base.fmt_pct(best['vs_bh'])} "
            f"({n_ok}/{n_var} Varianten qualifiziert, {time.time() - t_coin:.0f}s)",
            flush=True,
        )

    # --------------------------------------------------------
    # Zusammenfassung
    # --------------------------------------------------------

    if rows:
        print_summary_table(
            rows,
            len(ref_bases),
            ref_bases,
            block_range(period_from, period_to),
            min_trades,
        )
    else:
        print("\nKeine Coins mit qualifizierender Variante.")

    if failed:
        print("\nNicht in der Tabelle:")
        for pair, why in failed:
            print(f"   {pair:<12} {why}")

    print(f"\nFertig in {time.time() - t0:.0f}s. Nur Konsolen-Ausgabe, keine CSV. Keine Handelsempfehlung.")


if __name__ == "__main__":
    main()
