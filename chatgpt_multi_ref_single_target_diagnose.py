#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose: Multi-Ref Preis-Regel, eine Zelle, jeder Trade
========================================================

Dieselbe Idee wie chatgpt_multi_ref_single_target.py, aber nur eine
Kombination. Mindestens MIN_REFS Refs laufen im Fenster über die Schwelle,
in der gewählten Richtung. Target wird zum Close gekauft, Hold fest.

Für jeden Trade: Zeit, ROI, welche Refs gestimmt haben und um wie viel.
Dazu Stunde, Wochentag, exakte Kombination und alle 2er/3er-Kombis.
"""

from __future__ import annotations

import argparse
import itertools
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


# ============================================================
# EINE ZELLE
# ============================================================

REF_SYMBOLS: List[str] = ["BTC", "ETH", "DOT"]
TARGET_SYMBOL = "PEPE"

EVENT_WINDOW_MIN = 60
THRESHOLD_PCT = 1
MIN_REFS = 3
DIRECTION = "UP"          # UP oder DOWN
HOLD = "60h"

FROM_DATE = "2024-01-01"
TO_DATE = "2026-12-28"
KLINE_INTERVAL = "1h"
FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0
OUT_CSV = "price_trades.csv"

# ============================================================

BINANCE = "https://data-api.binance.vision"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "price-diagnose/1.0"})
BERLIN_TZ = ZoneInfo("Europe/Berlin")
HOLD_MIN = {
    "3h": 180, "6h": 360, "12h": 720, "24h": 1440, "48h": 2880,
    "60h": 3600, "72h": 4320, "78h": 4680, "84h": 5040, "90h": 5400,
    "96h": 5760, "120h": 7200, "168h": 10080, "192h": 11520,
}
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


def interval_to_minutes(interval: str) -> int:
    unit, n = interval[-1], int(interval[:-1])
    if unit == "m":
        return n
    if unit == "h":
        return n * 60
    raise ValueError(interval)


BAR_MIN = interval_to_minutes(KLINE_INTERVAL)
BAR_MS = BAR_MIN * 60_000


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def berlin(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=BERLIN_TZ).strftime("%Y-%m-%d %H:%M")


def sleep_polite(seconds: float = 0.12) -> None:
    time.sleep(seconds)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose einer Multi-Ref-Zelle")
    p.add_argument("--ref", nargs="+", default=None)
    p.add_argument("--coin", dest="target", default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--min-refs", type=int, default=None)
    p.add_argument("--direction", choices=["UP", "DOWN"], default=None)
    p.add_argument("--hold", default=None)
    p.add_argument("--from", dest="from_date", default=None)
    p.add_argument("--to", dest="to_date", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def resolve_range(from_s: Optional[str], to_s: Optional[str]):
    end = datetime.now(timezone.utc)
    if to_s:
        end = datetime.fromisoformat(to_s).replace(tzinfo=timezone.utc)
    start = datetime.fromisoformat(from_s or FROM_DATE).replace(tzinfo=timezone.utc)
    return start, end


def clamp_end(end: datetime) -> datetime:
    now_ms = int(time.time() * 1000)
    last = (now_ms // BAR_MS) * BAR_MS - BAR_MS
    last_dt = pd.Timestamp(last, unit="ms", tz="UTC")
    if end.tzinfo is not None:
        last_dt = last_dt.tz_convert(end.tzinfo)
    return min(end, last_dt.to_pydatetime())


def binance_usdt() -> set[str]:
    r = SESSION.get(f"{BINANCE}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    return {s["symbol"] for s in r.json()["symbols"] if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"}


def pair_for(base: str, usdt: set[str]):
    base = base.strip().upper().removesuffix("USDT")
    aliases = {"RNDR": "RENDERUSDT", "RENDER": "RENDERUSDT", "MATIC": "MATICUSDT", "POL": "POLUSDT"}
    if base in aliases and aliases[base] in usdt:
        return base, aliases[base]
    pair = f"{base}USDT"
    if pair not in usdt:
        raise SystemExit(f"{pair} nicht auf Binance Spot USDT.")
    return base, pair


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        r = SESSION.get(
            f"{BINANCE}/api/v3/klines",
            params={"symbol": symbol, "interval": KLINE_INTERVAL, "startTime": cursor, "endTime": end_ms, "limit": 1000},
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
        nxt = data[-1][0] + 1
        if nxt <= cursor:
            break
        cursor = nxt
        sleep_polite()
        if len(data) < 1000:
            break
    cols = ["open_time", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols + ["ct", "q", "n", "tb", "tq", "i"])
    now_ms = int(time.time() * 1000)
    df = df[df["ct"].astype(np.int64) < now_ms][cols].copy()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype(np.int64)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def build(frames: Dict[str, pd.DataFrame], refs: List[str], target_df: pd.DataFrame, window_min: int):
    common = None
    for df in list(frames.values()) + [target_df]:
        t = df["open_time"].to_numpy(dtype=np.int64)
        common = t if common is None else np.intersect1d(common, t)
    bars = window_min // BAR_MIN
    cols = []
    for name in refs:
        df = frames[name]
        t = df["open_time"].to_numpy(dtype=np.int64)
        ret = (df["close"].pct_change(bars) * 100.0).to_numpy(dtype=np.float64)
        cols.append(ret[np.searchsorted(t, common)])
    moves = np.column_stack(cols)
    tt = target_df["open_time"].to_numpy(dtype=np.int64)
    closes = target_df["close"].to_numpy(dtype=np.float64)[np.searchsorted(tt, common)]
    ok = np.isfinite(moves).all(axis=1) & np.isfinite(closes) & (closes > 0)
    return common[ok], moves[ok], closes[ok]


def roi_of(entry: float, exit_: float) -> float:
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    return (exit_ * (1.0 - cost) / (entry * (1.0 + cost)) - 1.0) * 100.0


def fmt(v: float) -> str:
    return f"{v:+.2f}%"


def bucket(rows: List[dict], key) -> None:
    groups: Dict[str, List[float]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r["roi"])
    print(f"{'Gruppe':<16}{'n':>5}{'%pos':>7}{'Avg':>10}{'Summe':>10}")
    for name in groups:
        vals = groups[name]
        pos = 100.0 * sum(v > 0 for v in vals) / len(vals)
        print(f"{name:<16}{len(vals):>5}{pos:>6.0f}%{np.mean(vals):>10.2f}{np.sum(vals):>10.2f}")


def print_combos(alts: List[str], trades: List[dict], k: int) -> None:
    print(f"\nAlle {k}er-Kombinationen")
    print(f"{'Kombination':<28}{'n':>5}{'%pos':>7}{'Avg':>10}{'Summe':>10}{'Anteil':>8}")
    rows = []
    for combo in itertools.combinations(alts, k):
        vals = [t["roi"] for t in trades if set(combo) <= t["voted"]]
        if not vals:
            rows.append((",".join(combo), 0, None, None, None))
            continue
        pos = 100.0 * sum(v > 0 for v in vals) / len(vals)
        rows.append((",".join(combo), len(vals), pos, float(np.mean(vals)), float(np.sum(vals))))
    rows.sort(key=lambda r: (r[1], r[3] if r[3] is not None else -1e9), reverse=True)
    for name, n_c, pos, avg, total in rows:
        share = 100.0 * n_c / len(trades) if trades else 0
        if n_c == 0:
            print(f"{name:<28}{0:>5}{'—':>7}{'—':>10}{'—':>10}{share:>7.0f}%")
        else:
            print(f"{name:<28}{n_c:>5}{pos:>6.0f}%{avg:>10.2f}{total:>10.2f}{share:>7.0f}%")
    print("n = Trades, in denen alle Coins der Kombination gestimmt haben. Nicht 'genau nur diese'.")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    window = args.window or EVENT_WINDOW_MIN
    threshold = args.threshold if args.threshold is not None else THRESHOLD_PCT
    min_refs = args.min_refs if args.min_refs is not None else MIN_REFS
    direction = (args.direction or DIRECTION).upper()
    hold_name = args.hold or HOLD
    out_csv = args.out or OUT_CSV
    if hold_name not in HOLD_MIN:
        raise SystemExit(f"Hold {hold_name} nicht bekannt.")
    if window % BAR_MIN != 0:
        raise SystemExit("Fenster muss durch die Kerzengröße teilbar sein.")
    hold_bars = HOLD_MIN[hold_name] // BAR_MIN
    sign = 1.0 if direction == "UP" else -1.0

    usdt = binance_usdt()
    refs, pairs = [], {}
    for raw in (args.ref or REF_SYMBOLS):
        base, pair = pair_for(raw, usdt)
        if base not in refs:
            refs.append(base)
            pairs[base] = pair
    target, target_pair = pair_for(args.target or TARGET_SYMBOL, usdt)
    if target in refs:
        raise SystemExit("Target darf kein Ref sein.")
    if min_refs > len(refs):
        raise SystemExit("MIN_REFS größer als die Ref-Liste.")

    start, end = resolve_range(args.from_date, args.to_date)
    end = clamp_end(end)
    start_ms, end_ms = utc_ms(start), utc_ms(end)

    arrow = "+" if direction == "UP" else "-"
    print("\n" + "=" * 100)
    print("DIAGNOSE | MULTI-REF PREIS | EINE ZELLE")
    print("=" * 100)
    print(f"Refs:     {', '.join(refs)}")
    print(f"Regel:    {direction} | {window}m | {arrow}{threshold:g}% | {min_refs}/{len(refs)} | Hold {hold_name}")
    print(f"Target:   {target_pair}")
    print(f"Zeitraum: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}")
    print("=" * 100)

    frames = {}
    print("\n1) Lade Refs:")
    for base in refs:
        print(f"   {base:>6} ...", end="", flush=True)
        df = fetch_klines(pairs[base], start_ms, end_ms)
        df = df[df["open_time"] <= end_ms].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"Keine Kerzen für {pairs[base]}.")
        frames[base] = df
        print(f" {len(df)}", flush=True)
    print(f"\n2) Lade {target_pair} ...", flush=True)
    target_df = fetch_klines(target_pair, start_ms, end_ms)
    target_df = target_df[target_df["open_time"] <= end_ms].reset_index(drop=True)
    if target_df.empty:
        raise SystemExit("Keine Target-Kerzen.")
    print(f"   {len(target_df)}", flush=True)

    times, moves, closes = build(frames, refs, target_df, window)
    tm = (times >= start_ms) & (times <= end_ms)
    voted_mask = sign * moves >= threshold
    entry_idx = np.flatnonzero(tm & (voted_mask.sum(axis=1) >= min_refs))

    trades = []
    i = 0
    n = len(times)
    while i < len(entry_idx):
        e = int(entry_idx[i])
        x = e + hold_bars
        if x >= n or times[x] != times[e] + hold_bars * BAR_MS:
            i += 1
            continue
        roi = roi_of(float(closes[e]), float(closes[x]))
        voted = [refs[k] for k in range(len(refs)) if voted_mask[e, k]]
        moves_txt = " ".join(f"{refs[k]}:{moves[e, k]:+.1f}%" for k in range(len(refs)) if voted_mask[e, k])
        dt = datetime.fromtimestamp(int(times[e]) / 1000.0, tz=timezone.utc)
        trades.append({
            "entry": berlin(int(times[e]) + BAR_MS),
            "exit": berlin(int(times[x]) + BAR_MS),
            "roi": roi,
            "sign": "+" if roi > 0 else "-",
            "voted": set(voted),
            "alts": ",".join(voted),
            "moves": moves_txt,
            "n_refs": len(voted),
            "hour": dt.hour,
            "weekday": WEEKDAYS[dt.weekday()],
        })
        i = int(np.searchsorted(entry_idx, x, side="left"))

    print(f"\nSignale: {len(entry_idx):,} | Trades nach Hold: {len(trades)}")
    if not trades:
        print("Keine Trades.")
        return

    print(f"\n{'#':>3} {'Einstieg':<17} {'Ausstieg':<17} {'ROI':>9} {'Stimmen'}")
    print("-" * 90)
    for n_i, t in enumerate(trades, 1):
        print(f"{n_i:>3} {t['entry']:<17} {t['exit']:<17} {fmt(t['roi']):>9} {t['moves']}")

    rois = [t["roi"] for t in trades]
    wins = [v for v in rois if v > 0]
    losses = [v for v in rois if v <= 0]
    equity = 1.0
    for v in rois:
        equity *= 1.0 + v / 100.0
    print("\n" + "=" * 60)
    print("ZUSAMMEN")
    print("=" * 60)
    print(f"n            {len(rois)}")
    print(f"positiv      {len(wins)} ({100.0 * len(wins) / len(rois):.0f}%)")
    print(f"negativ      {len(losses)}")
    print(f"Avg          {np.mean(rois):+.2f}%")
    print(f"Avg Gewinn   {np.mean(wins):+.2f}%" if wins else "Avg Gewinn   n/a")
    print(f"Avg Verlust  {np.mean(losses):+.2f}%" if losses else "Avg Verlust  n/a")
    print(f"bester       {max(rois):+.2f}%")
    print(f"schlechtester{min(rois):+.2f}%")
    print(f"Compound     {(equity - 1) * 100:+.2f}%")
    if wins and losses:
        print(f"Payoff       {abs(np.mean(wins) / np.mean(losses)):.2f}x")

    mid = len(trades) // 2
    print("\nErste Hälfte gegen zweite Hälfte")
    for label, part in (("erste", trades[:mid]), ("zweite", trades[mid:])):
        vals = [t["roi"] for t in part]
        if vals:
            pos = 100.0 * sum(v > 0 for v in vals) / len(vals)
            print(f"  {label:<8} n={len(vals):<4} %pos={pos:.0f}%  Avg={np.mean(vals):+.2f}%")

    print("\nNach UTC-Stunde")
    bucket(trades, lambda t: f"{t['hour']:02d}:00")
    print("\nNach Wochentag (UTC)")
    bucket(trades, lambda t: t["weekday"])
    print("\nNach Anzahl Ref-Stimmen")
    bucket(trades, lambda t: f"{t['n_refs']} Refs")

    print("\nExakte Kombination (genau diese Refs, keine weiteren)")
    exact: Dict[str, List[float]] = {}
    for t in trades:
        exact.setdefault(t["alts"], []).append(t["roi"])
    print(f"{'Kombination':<28}{'n':>5}{'%pos':>7}{'Avg':>10}{'Summe':>10}")
    for name, vals in sorted(exact.items(), key=lambda kv: len(kv[1]), reverse=True):
        pos = 100.0 * sum(v > 0 for v in vals) / len(vals)
        print(f"{name:<28}{len(vals):>5}{pos:>6.0f}%{np.mean(vals):>10.2f}{np.sum(vals):>10.2f}")

    print("\nEinzelne Ref-Stimme, egal mit wem")
    alone: Dict[str, List[float]] = {}
    for t in trades:
        for name in t["voted"]:
            alone.setdefault(name, []).append(t["roi"])
    print(f"{'Ref':<10}{'n':>5}{'%pos':>7}{'Avg':>10}{'Summe':>10}")
    for name, vals in sorted(alone.items(), key=lambda kv: np.mean(kv[1]), reverse=True):
        pos = 100.0 * sum(v > 0 for v in vals) / len(vals)
        print(f"{name:<10}{len(vals):>5}{pos:>6.0f}%{np.mean(vals):>10.2f}{np.sum(vals):>10.2f}")

    if len(refs) >= 2:
        print_combos(refs, trades, 2)
    if len(refs) >= 3:
        print_combos(refs, trades, 3)
    if len(refs) >= 4:
        print_combos(refs, trades, 4)

    pd.DataFrame([{k: v for k, v in t.items() if k != "voted"} for t in trades]).to_csv(out_csv, index=False)
    print(f"\nCSV: {out_csv}")
    print("FERTIG")


if __name__ == "__main__":
    print(f"Start: {datetime.now(BERLIN_TZ):%Y-%m-%d %H:%M:%S %Z}", flush=True)
    main()
