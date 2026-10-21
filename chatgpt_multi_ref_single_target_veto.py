#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Ref Preis-Regel mit Veto-Kombinationen
============================================

Kauf, wenn mindestens MIN_REFS der Refs im Fenster über die Schwelle
laufen, UP oder DOWN.

Eine Veto-Kombination sperrt den Kauf, wenn alle ihre Coins in derselben
Richtung über dieselbe Schwelle laufen. Eine Kombination reicht.

  VETOS = [["BNB", "ETH"], ["DOGE"]]

OFFEN = ohne Veto.
VETO  = mit den Kombinationen.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


# ============================================================
# HIER ÄNDERN
# ============================================================

REF_SYMBOLS: List[str] = ["BTC", "ETH", "SOL", "XRP","DOGE","NEAR","ADA","BCH"]
TARGET_SYMBOL = "ENA"

# Eine innere Liste muss komplett in Signalrichtung über der Schwelle liegen.
VETOS: List[List[str]] = [["DOGE","NEAR","ADA","BCH"], ["SOL","DOGE","NEAR","BCH"],["NEAR"]]

EVENT_WINDOWS_MIN = [120]
THRESHOLDS_PCT = [1.5, 2.0, 2.5]
MIN_REFS = [1,2, 3, 4,5]

HOLD_HORIZONS = {
    "3h": 180
}

FROM_DATE = "2024-01-01"
TO_DATE = "2026-12-28"
KLINE_INTERVAL = "1h"
FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0

# ============================================================

BINANCE = "https://data-api.binance.vision"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "price-veto/1.0"})
BERLIN_TZ = ZoneInfo("Europe/Berlin")


def interval_to_minutes(interval: str) -> int:
    unit, n = interval[-1], int(interval[:-1])
    if unit == "m":
        return n
    if unit == "h":
        return n * 60
    raise ValueError(interval)


BAR_MIN = interval_to_minutes(KLINE_INTERVAL)
BAR_MS = BAR_MIN * 60_000
HOLD_BARS = {h: mins // BAR_MIN for h, mins in HOLD_HORIZONS.items()}
HORIZON_NAMES = list(HOLD_HORIZONS.keys())

if any(w % BAR_MIN != 0 for w in EVENT_WINDOWS_MIN):
    raise SystemExit("EVENT_WINDOWS_MIN muss durch die Kerzengröße teilbar sein.")


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def sleep_polite(seconds: float = 0.12) -> None:
    time.sleep(seconds)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Ref mit Veto-Kombinationen")
    p.add_argument("--ref", nargs="+", default=None)
    p.add_argument("--veto", nargs="+", action="append", default=None,
                   help="Eine Kombination: --veto BNB ETH --veto DOGE")
    p.add_argument("--coin", dest="target", default=None)
    p.add_argument("--from", dest="from_date", default=None)
    p.add_argument("--to", dest="to_date", default=None)
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


def build(frames, names, target_df, window_min):
    common = None
    for df in list(frames.values()) + [target_df]:
        t = df["open_time"].to_numpy(dtype=np.int64)
        common = t if common is None else np.intersect1d(common, t)
    bars = window_min // BAR_MIN
    cols = []
    for name in names:
        df = frames[name]
        t = df["open_time"].to_numpy(dtype=np.int64)
        ret = (df["close"].pct_change(bars) * 100.0).to_numpy(dtype=np.float64)
        cols.append(ret[np.searchsorted(t, common)])
    moves = np.column_stack(cols)
    tt = target_df["open_time"].to_numpy(dtype=np.int64)
    closes = target_df["close"].to_numpy(dtype=np.float64)[np.searchsorted(tt, common)]
    ok = np.isfinite(moves).all(axis=1) & np.isfinite(closes) & (closes > 0)
    return common[ok], moves[ok], closes[ok]


def stats_of(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0, "pct_pos": None, "avg": None, "compound": None}
    equity = 1.0
    for v in vals:
        equity *= 1.0 + v / 100.0
    n_pos = sum(v > 0 for v in vals)
    return {
        "n": len(vals),
        "pct_pos": 100.0 * n_pos / len(vals),
        "avg": float(np.mean(vals)),
        "compound": (equity - 1.0) * 100.0,
    }


def roi_of(entry: float, exit_: float) -> float:
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    return (exit_ * (1.0 - cost) / (entry * (1.0 + cost)) - 1.0) * 100.0


def run_entries(times: np.ndarray, closes: np.ndarray, entry_idx: np.ndarray) -> dict:
    out = {}
    n = len(times)
    for h, bars in HOLD_BARS.items():
        vals: List[float] = []
        i = 0
        while i < len(entry_idx):
            e = int(entry_idx[i])
            x = e + bars
            if x >= n or times[x] != times[e] + bars * BAR_MS:
                i += 1
                continue
            vals.append(roi_of(float(closes[e]), float(closes[x])))
            i = int(np.searchsorted(entry_idx, x, side="left"))
        out[h] = stats_of(vals)
    return out


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:+.2f}%"


def fmt_pos(v) -> str:
    return "n/a" if v is None else f"{v:.0f}%"


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
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

    raw_combos = args.veto if args.veto else VETOS
    veto_combos: List[List[str]] = []
    veto_names: List[str] = []
    for combo in raw_combos:
        resolved = []
        for raw in combo:
            base, pair = pair_for(raw, usdt)
            if base == target:
                print(f"Veto {base} übersprungen: ist Target.")
                continue
            if base not in pairs:
                pairs[base] = pair
            if base not in veto_names:
                veto_names.append(base)
            if base not in resolved:
                resolved.append(base)
        if resolved:
            veto_combos.append(resolved)
    if not veto_combos:
        raise SystemExit("Keine Veto-Kombinationen.")
    quorums = [q for q in MIN_REFS if 1 <= q <= len(refs)]

    start, end = resolve_range(args.from_date, args.to_date)
    end = clamp_end(end)
    start_ms, end_ms = utc_ms(start), utc_ms(end)

    print("\n" + "=" * 100)
    print("MULTI-REF PREIS MIT VETO")
    print("=" * 100)
    print(f"Refs:     {', '.join(refs)}")
    print("Veto:     " + " ODER ".join("[" + ",".join(c) + "]" for c in veto_combos))
    print("          alle in einer Klammer in Signalrichtung über der Schwelle -> kein Kauf")
    print(f"Target:   {target_pair}")
    print(f"Fenster:  {EVENT_WINDOWS_MIN} | Schwellen {THRESHOLDS_PCT} | Quoren {quorums}")
    print(f"Zeitraum: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}")
    print("=" * 100)

    load_names = list(dict.fromkeys(refs + veto_names))
    frames = {}
    print("\n1) Lade Coins:")
    for base in load_names:
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

    names = refs + [n for n in veto_names if n not in refs]
    ref_ix = [names.index(n) for n in refs]
    rows = []
    print("\n3) Ergebnisse\n", flush=True)
    for window in EVENT_WINDOWS_MIN:
        times, moves, closes = build(frames, names, target_df, window)
        tm = (times >= start_ms) & (times <= end_ms)
        ref_moves = moves[:, ref_ix]
        for direction, sign in (("UP", 1.0), ("DOWN", -1.0)):
            for threshold in THRESHOLDS_PCT:
                voted = sign * ref_moves >= threshold
                cnt = voted.sum(axis=1)
                blocked = np.zeros(len(times), dtype=bool)
                for combo in veto_combos:
                    hit = np.ones(len(times), dtype=bool)
                    for name in combo:
                        hit &= sign * moves[:, names.index(name)] >= threshold
                    blocked |= hit
                for q in quorums:
                    open_idx = np.flatnonzero(tm & (cnt >= q))
                    veto_idx = open_idx[~blocked[open_idx]]
                    open_stats = run_entries(times, closes, open_idx)
                    veto_stats = run_entries(times, closes, veto_idx)
                    arrow = "+" if direction == "UP" else "-"
                    print(
                        f"\n{direction} | {window}m | {arrow}{threshold:g}% | "
                        f"{q}/{len(refs)} | {target_pair}"
                    )
                    print(f"{'Arm':<10}{'Hold':>6}{'n':>6}{'%pos':>7}{'Avg':>10}{'Cmp':>11}")
                    for label, stats in (("OFFEN", open_stats), ("VETO", veto_stats)):
                        for h in HORIZON_NAMES:
                            s = stats[h]
                            print(
                                f"{label:<10}{h:>6}{s['n']:>6}{fmt_pos(s['pct_pos']):>7}"
                                f"{fmt_pct(s['avg']):>10}{fmt_pct(s['compound']):>11}"
                            )
                    rows.append({
                        "key": (window, threshold, q, direction),
                        "open": open_stats,
                        "veto": veto_stats,
                        "blocked": int(blocked[open_idx].sum()) if len(open_idx) else 0,
                    })

    print("\n" + "=" * 110)
    print("KOMPAKT | Hold 90h | VETO minus OFFEN")
    print("=" * 110)
    print(f"{'Win':>5} {'Thr':>7} {'Vote':>6} {'Dir':>5} {'n O':>5} {'Offen':>10} {'n V':>5} {'Veto':>10} {'Diff':>10} {'weg':>5}")
    print("-" * 110)
    for r in rows:
        window, threshold, q, direction = r["key"]
        o, v = r["open"]["90h"], r["veto"]["90h"]
        delta = v["compound"] - o["compound"] if v["compound"] is not None and o["compound"] is not None else None
        arrow = "+" if direction == "UP" else "-"
        print(
            f"{window:>4}m {arrow}{threshold:>5.1f}% {q}/{len(refs)} {direction:>5} "
            f"{o['n']:>5} {fmt_pct(o['compound']):>10} {v['n']:>5} {fmt_pct(v['compound']):>10} "
            f"{fmt_pct(delta):>10} {r['blocked']:>5}"
        )
    print("-" * 110)
    print("weg = Signale, die das Veto verworfen hat. Besser nur, wenn Diff positiv ist und n nicht zusammenbricht.")
    print("FERTIG")


if __name__ == "__main__":
    print(f"Start: {datetime.now(BERLIN_TZ):%Y-%m-%d %H:%M:%S %Z}", flush=True)
    main()
