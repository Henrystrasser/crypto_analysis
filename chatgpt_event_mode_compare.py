#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chatgpt_event_mode_compare.py

EVENT-MODE-VERGLEICH: LEVEL vs CROSS
(gleiche Strategie wie chatgpt_multi_ref_single_target.py)

Die Multi-Ref-Strategie wird mit IDENTISCHEN Einstellungen zweimal gerechnet:

  level = jede (gemeinsame Ref-)Kerze, auf der die Bedingung gilt.
          Nach Hold-Ende wird erneut gekauft, falls sie weiter gilt.
          (= Verhalten von chatgpt_multi_ref_single_target.py)

  cross = nur Kerzen, auf denen die Bedingung NEU erfüllt ist
          (vorherige gemeinsame Kerze hat sie nicht erfüllt).

Bedingung (je Window / Threshold / MIN_REFS / Richtung):
  UP   = mindestens MIN_REFS Refs mit Move >= +Threshold über das Window
  DOWN = mindestens MIN_REFS Refs mit Move <= -Threshold über das Window

Alles andere ist identisch und kommt aus dem (gefixten) Basisscript:
Refs, Target, Windows, Thresholds, MIN_REFS, Interval, Holds, Fees,
Zeitraum, Non-overlap je Hold (Re-Entry exakt am Hold-Ende erlaubt),
exakte Entry-Kerze (fehlt sie, z.B. vor dem Listing, wird das Event
verworfen), exakte Exit-Kerze.

Signal = Ende der Referenz-Kerze, Kauf zum Close der Target-Kerze.
Alle Zeiten Europe/Berlin. Kein CSV.

Config unten in VS Code ändern; CLI-Flags sind nur optionale Overrides.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import chatgpt_multi_ref_single_target as base
from laggard_common import add_time_range_arguments, resolve_event_window


# ============================================================
# CONFIG (hier in VS Code ändern)
# ============================================================

REF_SYMBOLS: List[str] = ["BTC", "SOL", "ETH", "XRP", "BNB"]
TARGET_SYMBOL = "ENA"

EVENT_WINDOWS_MIN = [60, 180]
THRESHOLDS_PCT = [1.0, 1.5, 2.0, 2.5, 3.0]
MIN_REFS = [1, 2, 3, 4, 5]

# "UP" und/oder "DOWN" (wie im Basisscript beide).
DIRECTIONS = ["UP", "DOWN"]

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

# Zeitraum (Europe/Berlin). TO_DATE in der Zukunft -> letzte geschlossene Kerze.
# FROM_DATE=None -> LOOKBACK_DAYS rückwärts ab jetzt.
FROM_DATE: Optional[str] = "2024-01-01"
TO_DATE: Optional[str] = "2026-12-28"
LOOKBACK_DAYS = 400

KLINE_INTERVAL = "1h"

FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0

# Vergleichskriterium für "besser": "compound" oder "avg" (Ø ROI je Trade).
PRIMARY_METRIC = "compound"

# True = Level-Events zusätzlich mit der (langsamen) Original-Detektion
# des Basisscripts gegenprüfen.
VERIFY_DETECT = False


# ============================================================
# BASIS KONFIGURIEREN
# ============================================================

def apply_config_to_base() -> None:
    """Grid/Fees in das Basismodul schreiben (gleiche Mathematik wie dort)."""
    bar_min = base.interval_to_minutes(KLINE_INTERVAL)
    if any(w % bar_min != 0 for w in EVENT_WINDOWS_MIN):
        raise SystemExit("Alle EVENT_WINDOWS_MIN müssen durch KLINE_INTERVAL teilbar sein.")
    if any(h % bar_min != 0 for h in HOLD_HORIZONS.values()):
        raise SystemExit("Alle HOLD_HORIZONS müssen durch KLINE_INTERVAL teilbar sein.")
    if any(x <= 0 for x in THRESHOLDS_PCT):
        raise SystemExit("THRESHOLDS_PCT müssen > 0 sein.")
    if any(d not in ("UP", "DOWN") for d in DIRECTIONS):
        raise SystemExit("DIRECTIONS: nur 'UP' und/oder 'DOWN'.")

    base.KLINE_INTERVAL = KLINE_INTERVAL
    base.BAR_MIN = bar_min
    base.EVENT_WINDOWS_MIN = list(EVENT_WINDOWS_MIN)
    base.THRESHOLDS_PCT = list(THRESHOLDS_PCT)
    base.MIN_REFS = list(MIN_REFS)
    base.HOLD_HORIZONS = dict(HOLD_HORIZONS)
    base.WINDOW_BARS = {w: w // bar_min for w in EVENT_WINDOWS_MIN}
    base.HOLD_BARS = {h: m // bar_min for h, m in HOLD_HORIZONS.items()}
    base.MAX_HOLD_MIN = max(HOLD_HORIZONS.values())
    base.HORIZON_NAMES = list(HOLD_HORIZONS.keys())
    base.FEE_BPS = FEE_BPS
    base.SLIPPAGE_BPS = SLIPPAGE_BPS


# ============================================================
# EVENTS: LEVEL + CROSS
# ============================================================

Key = Tuple[str, int, float, int]  # direction, window, threshold, min_refs


def detect_level_and_cross(
    ref_data: Dict[str, pd.DataFrame],
    start_ms: int,
    end_ms: int,
) -> Tuple[Dict[Key, List[base.MultiRefEvent]], Dict[Key, set]]:
    """
    Level-Events identisch zu base.detect_multi_ref_events (vektorisiert),
    plus Menge der Cross-Zeitpunkte (Bedingung neu erfüllt).

    Zeitachse = gemeinsame Ref-Kerzen in [start, end] (wie im Basisscript).
    "Vorherige Kerze" = vorherige gemeinsame Kerze; ungültige Moves (NaN)
    zählen als "Bedingung nicht erfüllt".
    """
    common: Optional[set] = None
    for df in ref_data.values():
        t = set(int(x) for x in df["open_time"].tolist())
        common = t if common is None else common & t
    common_arr = np.array(
        sorted(t for t in (common or set()) if start_ms <= t <= end_ms),
        dtype=np.int64,
    )

    refs = list(ref_data.keys())
    level: Dict[Key, List[base.MultiRefEvent]] = {}
    cross: Dict[Key, set] = {}

    for window in EVENT_WINDOWS_MIN:
        rows = []
        for ref in refs:
            df = ref_data[ref]
            ser = base.build_reference_move_series(df, [window])[window]
            rows.append(
                pd.Series(ser.to_numpy(), index=df["open_time"].to_numpy())
                .reindex(common_arr)
                .to_numpy(dtype=np.float64)
            )
        M = np.vstack(rows) if rows else np.empty((0, 0))
        valid = ~np.isnan(M).any(axis=0)

        for threshold in THRESHOLDS_PCT:
            hits = {"UP": M >= threshold, "DOWN": M <= -threshold}
            for direction in DIRECTIONS:
                hit = hits[direction]
                count = hit.sum(axis=0)
                for min_refs in MIN_REFS:
                    cond = valid & (count >= min_refs)
                    prev = np.concatenate(([False], cond[:-1]))
                    is_cross = cond & ~prev
                    key = (direction, window, threshold, min_refs)
                    evs = []
                    for i in np.flatnonzero(cond):
                        evs.append(
                            base.MultiRefEvent(
                                end_ms=int(common_arr[i]),
                                direction=direction,
                                window_min=window,
                                threshold_pct=threshold,
                                min_refs=min_refs,
                                ref_moves={r: float(M[j, i]) for j, r in enumerate(refs)},
                                matched_refs=[r for j, r in enumerate(refs) if hit[j, i]],
                            )
                        )
                    level[key] = evs
                    cross[key] = set(int(t) for t in common_arr[is_cross])
    return level, cross


def verify_against_base(
    level: Dict[Key, List[base.MultiRefEvent]],
    ref_data: Dict[str, pd.DataFrame],
    ref_pairs: Dict[str, str],
    start_ms: int,
    end_ms: int,
) -> None:
    print("   Prüfe Level-Events gegen Original-Detektion (langsam) ...")
    orig = base.detect_multi_ref_events(
        ref_data, ref_pairs, EVENT_WINDOWS_MIN, THRESHOLDS_PCT, MIN_REFS, start_ms, end_ms,
    )
    bad = 0
    for (direction, window, threshold, min_refs), evs in level.items():
        a = [e.end_ms for e in evs]
        b = [e.end_ms for e in orig.get((window, threshold, min_refs, direction), [])]
        if a != b:
            bad += 1
    print(f"   Abweichende Konfigurationen: {bad}/{len(level)}")
    if bad:
        raise SystemExit("Level-Events weichen von der Original-Detektion ab.")


# ============================================================
# STATISTIK
# ============================================================

def stats_for(trades: List[dict], hold: str) -> dict:
    kept = base.filter_non_overlapping(trades, hold)
    vals = np.array([float(t[f"roi_{hold}"]) for t in kept], dtype=np.float64)
    if len(vals) == 0:
        return {"n": 0, "compound": None, "avg": None, "median": None, "hit": None}
    return {
        "n": int(len(vals)),
        "compound": (float(np.prod(1.0 + vals / 100.0)) - 1.0) * 100.0,
        "avg": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "hit": 100.0 * float(np.mean(vals > 0)),
    }


def f_pct(v: Optional[float], width: int) -> str:
    return f"{'n/a' if v is None else f'{v:+.2f}%':>{width}}"


def f_hit(v: Optional[float]) -> str:
    return f"{'n/a' if v is None else f'{v:.0f}%':>5}"


def diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


# ============================================================
# MAIN
# ============================================================

def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-Modus-Vergleich level vs cross (optionale Overrides)")
    p.add_argument("--refs", "--ref", nargs="+", default=None, metavar="COIN",
                   help=f"Referenzcoins (Default: {' '.join(REF_SYMBOLS)})")
    p.add_argument("--target", "--coin", default=None, metavar="COIN",
                   help=f"Ziel-Coin (Default: {TARGET_SYMBOL})")
    p.add_argument("--verify-detect", action="store_true", default=None,
                   help="Level-Events gegen Original-Detektion prüfen (langsam)")
    add_time_range_arguments(p)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    t_start = time.time()
    args = parse_cli_args(argv)
    apply_config_to_base()

    refs_raw = list(dict.fromkeys(str(x).strip().upper() for x in (args.refs or REF_SYMBOLS)))
    target_raw = args.target or TARGET_SYMBOL
    verify = VERIFY_DETECT if args.verify_detect is None else True

    usdt_set = base.binance_usdt_symbols()
    ref_bases: List[str] = []
    ref_pairs: Dict[str, str] = {}
    for raw in refs_raw:
        b, pair = base.resolve_pair(raw, usdt_set)
        if b not in ref_bases:
            ref_bases.append(b)
            ref_pairs[b] = pair
    target_base, target_pair = base.resolve_pair(target_raw, usdt_set)
    if target_base in ref_bases:
        raise SystemExit(f"Target {target_base} darf nicht gleichzeitig Referenz sein.")
    if max(MIN_REFS) > len(ref_bases):
        raise SystemExit("MIN_REFS größer als Anzahl Refs.")

    start, end, _ = resolve_event_window(
        from_s=args.from_date or FROM_DATE,
        to_s=args.to_date or TO_DATE,
        lookback_days=args.lookback,
        default_lookback=LOOKBACK_DAYS,
    )
    end = base.clamp_end_to_closed_candle(end)
    start_ms = base.utc_ms(start)
    end_ms = base.utc_ms(end)
    fetch_end_ms = base.utc_ms(end + pd.Timedelta(minutes=base.MAX_HOLD_MIN))

    print("\n" + "=" * 132)
    print("EVENT-MODUS-VERGLEICH | LEVEL vs CROSS | Multi-Ref -> Single Target")
    print("=" * 132)
    print(f"Refs:       {', '.join(ref_bases)}")
    print(f"Target:     {target_base} ({target_pair})")
    print(f"Windows:    {', '.join(str(w) + 'm' for w in EVENT_WINDOWS_MIN)}")
    print(f"Thresholds: {', '.join(f'{x:g}%' for x in THRESHOLDS_PCT)}")
    print("Min Refs:   " + ", ".join(base.min_ref_label(x, len(ref_bases)) for x in MIN_REFS))
    print(f"Richtungen: {', '.join(DIRECTIONS)} (beide = BUY Target)")
    print(f"Holds:      {', '.join(HOLD_HORIZONS)}")
    print(f"Zeitraum:   {start.strftime('%Y-%m-%d %H:%M %Z')} -> {end.strftime('%Y-%m-%d %H:%M %Z')} (Europe/Berlin)")
    print(f"Interval:   {KLINE_INTERVAL} | Fee: {FEE_BPS} bps/Seite | Slippage: {SLIPPAGE_BPS} bps")
    print("level = jede Kerze mit erfüllter Bedingung | cross = nur Neueintritt (vorherige Kerze nicht erfüllt)")
    print("Non-overlap je Hold, Re-Entry exakt am Hold-Ende erlaubt | exakte Entry-/Exit-Kerze")
    print("=" * 132)

    # ---------------- Daten ----------------
    print("\n1) Lade Referenzcoins:")
    ref_data: Dict[str, pd.DataFrame] = {}
    for b in ref_bases:
        pair = ref_pairs[b]
        print(f"   {b:>8} ({pair}) ...", end="", flush=True)
        df = base.fetch_klines(pair, KLINE_INTERVAL, start_ms, end_ms)
        if df.empty:
            raise SystemExit(f"\nKeine Kerzen für Referenz {pair}.")
        ref_data[b] = df
        print(f" {len(df)} Kerzen")

    print(f"\n2) Lade Target {target_pair} ...")
    target_df = base.fetch_klines(target_pair, KLINE_INTERVAL, start_ms, fetch_end_ms)
    if target_df.empty:
        raise SystemExit(f"Keine Kerzen für Target {target_pair}.")
    print(f"   {len(target_df)} Kerzen")

    bh = base.buy_and_hold(target_df, start_ms, end_ms)
    block_period = base._block_range(*base._effective_range_ms(
        start_ms,
        end_ms,
        target_df["open_time"].to_numpy(),
        *[d["open_time"].to_numpy() for d in ref_data.values()],
    ))

    # ---------------- Events ----------------
    print("\n3) Erzeuge Events (level + cross) ...")
    level, cross = detect_level_and_cross(ref_data, start_ms, end_ms)
    print(f"   level: {sum(len(v) for v in level.values())} Events | "
          f"cross: {sum(len(v) for v in cross.values())} Events | {len(level)} Konfigurationen")
    if verify:
        verify_against_base(level, ref_data, ref_pairs, start_ms, end_ms)

    # ---------------- Trades + Stats ----------------
    print("\n4) Simuliere Trades ...")
    results: Dict[Key, Dict[str, Tuple[dict, dict]]] = {}
    for key, evs in level.items():
        # Cross-Events sind eine Teilmenge der Level-Events: Trades je Event
        # einmal rechnen (identische Entry-/Exit-Logik), dann filtern.
        trades_level = base.simulate_events(target_df, evs, target_pair)
        cross_times = cross[key]
        trades_cross = [t for t in trades_level if int(t["event_time_ms"]) in cross_times]
        results[key] = {
            h: (stats_for(trades_level, h), stats_for(trades_cross, h))
            for h in HOLD_HORIZONS
        }

    # ---------------- Ausgabe je Variante ----------------
    print("\n5) Ergebnisse je Variante (L = level, C = cross, Δ = cross − level)")
    if bh:
        print(f"   B&H {target_base}: {base.fmt_pct(bh['roi'])} | {bh['buy_time']} -> {bh['sell_time']} (Kerzen-Close)")

    head = (
        f"{'Refs':>10} {'Hold':>5} | "
        f"{'n L':>5} {'Cmp L':>10} {'Ø L':>8} {'Med L':>8} {'Hit L':>5} | "
        f"{'n C':>5} {'Cmp C':>10} {'Ø C':>8} {'Med C':>8} {'Hit C':>5} | "
        f"{'ΔCmp':>10} {'ΔØ':>8} {'besser':>6}"
    )
    tally = {"total": 0, "level": 0, "cross": 0, "equal": 0, "no_trades": 0}
    tally_avg = {"level": 0, "cross": 0, "equal": 0}
    by_dir = {d: {"level": 0, "cross": 0, "equal": 0} for d in DIRECTIONS}

    for direction in DIRECTIONS:
        for window in EVENT_WINDOWS_MIN:
            for threshold in THRESHOLDS_PCT:
                print("\n" + "=" * 132)
                print(f"{target_pair} | {direction} | Event {window}m | Threshold {threshold:g}% | {block_period}")
                print("=" * 132)
                print(head)
                print("-" * 132)
                for min_refs in MIN_REFS:
                    key = (direction, window, threshold, min_refs)
                    label = base.min_ref_label(min_refs, len(ref_bases))
                    for h in HOLD_HORIZONS:
                        L, C = results[key][h]
                        tally["total"] += 1
                        better = "-"
                        if L["n"] == 0 or C["n"] == 0:
                            tally["no_trades"] += 1
                        else:
                            a, b = C[PRIMARY_METRIC], L[PRIMARY_METRIC]
                            if abs(a - b) < 1e-12:
                                better = "="
                                tally["equal"] += 1
                                by_dir[direction]["equal"] += 1
                            elif a > b:
                                better = "C"
                                tally["cross"] += 1
                                by_dir[direction]["cross"] += 1
                            else:
                                better = "L"
                                tally["level"] += 1
                                by_dir[direction]["level"] += 1
                            if abs(C["avg"] - L["avg"]) < 1e-12:
                                tally_avg["equal"] += 1
                            elif C["avg"] > L["avg"]:
                                tally_avg["cross"] += 1
                            else:
                                tally_avg["level"] += 1
                        print(
                            f"{label:>10} {h:>5} | "
                            f"{L['n']:>5} {f_pct(L['compound'], 10)} {f_pct(L['avg'], 8)} "
                            f"{f_pct(L['median'], 8)} {f_hit(L['hit'])} | "
                            f"{C['n']:>5} {f_pct(C['compound'], 10)} {f_pct(C['avg'], 8)} "
                            f"{f_pct(C['median'], 8)} {f_hit(C['hit'])} | "
                            f"{f_pct(diff(C['compound'], L['compound']), 10)} "
                            f"{f_pct(diff(C['avg'], L['avg']), 8)} {better:>6}"
                        )

    # ---------------- Zusammenfassung ----------------
    compared = tally["total"] - tally["no_trades"]
    print("\n" + "=" * 132)
    print(f"ZUSAMMENFASSUNG | {target_pair} | {block_period}")
    print("=" * 132)
    print(f"Varianten gesamt: {tally['total']} | verglichen (beide n>0): {compared} | "
          f"ohne Trades in mind. einem Modus: {tally['no_trades']}")
    print(f"Nach Compound ({'primär' if PRIMARY_METRIC == 'compound' else 'sekundär'}): "
          f"cross besser {tally['cross'] if PRIMARY_METRIC == 'compound' else '-'} | "
          f"level besser {tally['level'] if PRIMARY_METRIC == 'compound' else '-'} | "
          f"gleich {tally['equal'] if PRIMARY_METRIC == 'compound' else '-'}")
    print(f"Nach Ø ROI je Trade: cross besser {tally_avg['cross']} | level besser {tally_avg['level']} | "
          f"gleich {tally_avg['equal']}")
    for d in DIRECTIONS:
        x = by_dir[d]
        print(f"  {d:<4} ({PRIMARY_METRIC}): cross besser {x['cross']} | level besser {x['level']} | gleich {x['equal']}")
    print(f"Laufzeit: {(time.time() - t_start) / 60:.1f} min")
    print("=" * 132)


if __name__ == "__main__":
    main()
