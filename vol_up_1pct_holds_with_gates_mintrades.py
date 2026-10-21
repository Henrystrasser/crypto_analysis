#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC + extra Gates (e.g. ETH) with own thresholds, then voting alt group
=======================================================================

Buy only if ALL of the following are true:
  1. BTC volume >= its threshold AND BTC rise >= its rise threshold.
  2. EVERY coin in GATE_SYMBOLS (e.g. ETH) meets its OWN volume threshold
     (GATE_VOL_THRESHOLDS) AND its OWN rise threshold (GATE_RISE_PCTS).
  3. At least MIN_ALTS of the remaining voting group meet Alt-vol + Ref-rise.

Grid: window x BTC-rise x Gate-rise x Ref-rise x BTC-vol x Gate-vol x Alt-vol x quorum.

Entry = signal candle close. Fixed hold, non-overlapping. Fee per side.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


BTC = "BTC"
# Extra coins that must ALL pass volume + rise (hard gates, besides BTC)
GATE_SYMBOLS: List[str] = ["XRP"]

# Voting group (gates and BTC are automatically removed from this list)
ALT_SYMBOLS: List[str] = [ "DOT", "XRP", "TRX", "BNB","NEAR"]

TARGET_SYMBOL = "PEPE"

BTC_VOL_THRESHOLDS = [0.5,0.7,0.85]
GATE_VOL_THRESHOLDS = [1,1.5,2,2.5]   # eigene Volumen-Schwellen für die Gates
ALT_VOL_THRESHOLDS = [1, 1.5,2,2.5]
BTC_RISE_PCTS = [0.8, 1,1.2]
GATE_RISE_PCTS = [0.6,0.8, 1,1.2]       # eigene Rise-Schwellen für die Gates
REF_RISE_PCTS = [ 0.6,0.8,1,1.5,2, 2.5,3]
MIN_ALTS = [2,3]
MIN_TRADES = 70         # nur Varianten mit mindestens so vielen Trades (nach Non-Overlap) werden gesammelt
BASE_DAYS = 14
EVENT_WINDOW_MINS = [120]

HOLD_HORIZONS = {
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
}

FROM_DATE = "2024-01-01"
TO_DATE = "2026-12-28"
KLINE_INTERVAL = "1h"
FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0

BINANCE = "https://data-api.binance.vision"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "vol-btc-gates-holds/1.0"})
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

if any(mins % BAR_MIN != 0 for mins in HOLD_HORIZONS.values()):
    raise SystemExit("Every hold must be divisible by the candle size.")


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def sleep_polite(seconds: float = 0.12) -> None:
    time.sleep(seconds)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Volume + gates + voting group, full hold grid")
    p.add_argument("--alts", nargs="+", default=None, help="voting alt group (gates are removed)")
    p.add_argument("--gates", nargs="+", default=None, help="extra gate coins besides BTC (default: ETH)")
    p.add_argument("--coin", dest="target", default=None)
    p.add_argument("--from", dest="from_date", default=None)
    p.add_argument("--to", dest="to_date", default=None)
    p.add_argument("--btc-rise", nargs="+", type=float, default=None)
    p.add_argument("--gate-vol", nargs="+", type=float, default=None, help="overrides GATE_VOL_THRESHOLDS")
    p.add_argument("--gate-rise", nargs="+", type=float, default=None, help="overrides GATE_RISE_PCTS")
    p.add_argument("--window", nargs="+", type=int, default=None)
    p.add_argument("--ref-rise", nargs="+", type=float, default=None)
    return p.parse_args(argv)


def resolve_range(from_s: Optional[str], to_s: Optional[str]):
    end = datetime.now(timezone.utc)
    to_s = to_s or TO_DATE
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
    pair = f"{base}USDT"
    if pair not in usdt:
        raise SystemExit(f"{pair} is not a Binance spot USDT pair.")
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
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(
        rows,
        columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n", "tb", "tq", "i"],
    )
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"].astype(np.int64) < now_ms][cols].copy()
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype(np.int64)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def hour_median_ratio(times: np.ndarray, volume: np.ndarray) -> np.ndarray:
    hour = ((times // 3_600_000) % 24).astype(np.int64)
    day = (times // 86_400_000).astype(np.int64)
    ratio = np.full(len(times), np.nan)
    by_hour: Dict[int, List[tuple]] = {h: [] for h in range(24)}
    for i, (h, d, v) in enumerate(zip(hour, day, volume)):
        hist = [val for dd, val in by_hour[h] if 0 < d - dd <= BASE_DAYS and val > 0]
        if len(hist) >= 7 and v > 0:
            ratio[i] = v / float(np.median(hist))
        by_hour[h].append((int(d), float(v)))
    return ratio


def build(frames: Dict[str, pd.DataFrame], names: List[str], target_df: pd.DataFrame, window_bars: int):
    common = None
    for df in list(frames.values()) + [target_df]:
        t = df["open_time"].to_numpy(dtype=np.int64)
        common = t if common is None else np.intersect1d(common, t)
    vol_cols, ret_cols = [], []
    for name in names:
        df = frames[name]
        t = df["open_time"].to_numpy(dtype=np.int64)
        pos = np.searchsorted(t, common)
        vol_cols.append(hour_median_ratio(t, df["quote_volume"].to_numpy(dtype=np.float64))[pos])
        ret_cols.append((df["close"].pct_change(window_bars) * 100.0).to_numpy(dtype=np.float64)[pos])
    tt = target_df["open_time"].to_numpy(dtype=np.int64)
    closes = target_df["close"].to_numpy(dtype=np.float64)[np.searchsorted(tt, common)]
    vol = np.column_stack(vol_cols)
    moves = np.column_stack(ret_cols)
    ok = np.isfinite(vol).all(axis=1) & np.isfinite(moves).all(axis=1) & np.isfinite(closes) & (closes > 0)
    return common[ok], vol[ok], moves[ok], closes[ok]


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
    _, btc_pair = pair_for(BTC, usdt)

    # Resolve gates (extra hard requirements besides BTC)
    gates: List[str] = []
    for raw in (args.gates or GATE_SYMBOLS):
        base, _ = pair_for(raw, usdt)
        if base != BTC and base not in gates:
            gates.append(base)

    # Voting alts: remove BTC and any gates
    voting_alts: List[str] = []
    pairs: Dict[str, str] = {BTC: btc_pair}
    for raw in (args.alts or ALT_SYMBOLS):
        base, pair = pair_for(raw, usdt)
        if base == BTC or base in gates or base in voting_alts:
            continue
        voting_alts.append(base)
        pairs[base] = pair

    # Also load the gates
    for g in gates:
        _, pair = pair_for(g, usdt)
        pairs[g] = pair

    target, target_pair = pair_for(args.target or TARGET_SYMBOL, usdt)
    if target == BTC or target in gates or target in voting_alts:
        raise SystemExit("Target must not be BTC, a gate, or in the voting group.")

    btc_rises = list(args.btc_rise or BTC_RISE_PCTS)
    gate_rises = list(args.gate_rise or GATE_RISE_PCTS)
    ref_rises = list(args.ref_rise or REF_RISE_PCTS)
    gate_vols = list(args.gate_vol or GATE_VOL_THRESHOLDS)
    windows = list(args.window or EVENT_WINDOW_MINS)
    for w in windows:
        if w <= 0 or w % BAR_MIN:
            raise SystemExit(f"Window {w}m must be a positive multiple of {BAR_MIN}m.")
    quorums = [q for q in MIN_ALTS if 1 <= q <= len(voting_alts)]
    if not quorums and voting_alts:
        quorums = [1]

    start, end = resolve_range(args.from_date, args.to_date)
    end = clamp_end(end)
    start_ms, end_ms = utc_ms(start), utc_ms(end)
    fetch_start = start_ms - (BASE_DAYS + 2) * 86_400_000

    print("\n" + "=" * 100)
    print("VOLUME + GATES + VOTING GROUP | FULL HOLD GRID")
    print("=" * 100)
    print(f"BTC gate:    vol {BTC_VOL_THRESHOLDS} | rise {btc_rises}%")
    print(f"Extra gates: {', '.join(gates) or '—'}  | vol {gate_vols} | rise {gate_rises}%  (ALL must pass)")
    print(f"Voting group:{', '.join(voting_alts)} | vol {ALT_VOL_THRESHOLDS} | rise {ref_rises}% | quorum {quorums}")
    print(f"Windows:     {windows} min")
    print(f"Target:      {target_pair}")
    print(f"Holds:       {', '.join(HORIZON_NAMES)}")
    print(f"Range:       {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')}")
    print(f"Fee:         {FEE_BPS} bps/side")
    print(f"Min Trades:  {MIN_TRADES} (nur Holds mit mind. so vielen Trades werden bewertet und in Top-Blöcken gezeigt)")
    print("=" * 100)

    # Load all needed coins: BTC + gates + voting alts
    load_names = [BTC] + gates + voting_alts
    frames = {}
    print("\n1) Load coins:")
    for base in load_names:
        print(f"   {base:>6} ...", end="", flush=True)
        df = fetch_klines(pairs[base], fetch_start, end_ms)
        df = df[df["open_time"] <= end_ms].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"No candles for {pairs[base]}.")
        frames[base] = df
        print(f" {len(df)}", flush=True)
    print(f"\n2) Load {target_pair} ...", flush=True)
    target_df = fetch_klines(target_pair, start_ms, end_ms)
    target_df = target_df[target_df["open_time"] <= end_ms].reset_index(drop=True)
    if target_df.empty:
        raise SystemExit("No target candles.")
    print(f"   {len(target_df)}", flush=True)

    names = load_names  # order: BTC, gates..., voting_alts...
    gate_start = 1
    gate_end = 1 + len(gates)
    vote_start = gate_end

    rows = []
    blocks = []
    print("\n3) Holds\n", flush=True)
    for win in windows:
        times, vol, moves, closes = build(frames, names, target_df, win // BAR_MIN)
        tm = (times >= start_ms) & (times <= end_ms)
        btc_vol = vol[:, 0]
        btc_ret = moves[:, 0]
        gate_vol = vol[:, gate_start:gate_end] if gates else np.empty((len(times), 0))
        gate_ret = moves[:, gate_start:gate_end] if gates else np.empty((len(times), 0))
        vote_vol = vol[:, vote_start:]
        vote_ret = moves[:, vote_start:]

        for btc_rise in btc_rises:
            for gate_rise in gate_rises:
                for ref_rise in ref_rises:
                    for btc_thr in BTC_VOL_THRESHOLDS:
                        btc_ok = (btc_vol >= btc_thr) & (btc_ret >= btc_rise)
                        for gate_thr in gate_vols:
                            # Gates: ALL must pass their own gate_thr + gate_rise
                            if gates:
                                gate_ok = np.all(
                                    (gate_vol >= gate_thr) & (gate_ret >= gate_rise),
                                    axis=1,
                                )
                            else:
                                gate_ok = np.ones(len(times), dtype=bool)

                            for alt_thr in ALT_VOL_THRESHOLDS:
                                # Voting group
                                if voting_alts:
                                    vote_ok = (vote_vol >= alt_thr) & (vote_ret >= ref_rise)
                                    vote_cnt = vote_ok.sum(axis=1)
                                else:
                                    vote_cnt = np.zeros(len(times), dtype=int)

                                for q in quorums:
                                    idx = np.flatnonzero(tm & btc_ok & gate_ok & (vote_cnt >= q))
                                    stats = run_entries(times, closes, idx)
                                    # Nur Holds mit genug Trades berücksichtigen
                                    valid_holds = {h: s for h, s in stats.items() if s["n"] >= MIN_TRADES and s["compound"] is not None}
                                    if not valid_holds:
                                        continue
                                    gate_label = "+".join(gates) if gates else "—"
                                    blk = [
                                        f"\nVOL+GATES | win {win}m | BTC +{btc_rise:g}% & >= {btc_thr:g}x | "
                                        f"Gates({gate_label}) +{gate_rise:g}% & >= {gate_thr:g}x | "
                                        f"Vote +{ref_rise:g}% & >= {alt_thr:g}x {q}/{len(voting_alts)} | "
                                        f"{target_pair} | signals {len(idx):,}",
                                        f"{'Hold':>7}{'n':>6}{'%pos':>7}{'Avg':>10}{'Compound':>12}",
                                        "-" * 42,
                                    ]
                                    bb = None
                                    for h in HORIZON_NAMES:
                                        s = stats[h]
                                        # Nur Holds mit genug Trades in den Block schreiben
                                        if s["n"] < MIN_TRADES:
                                            continue
                                        blk.append(
                                            f"{h:>7}{s['n']:>6}{fmt_pos(s['pct_pos']):>7}"
                                            f"{fmt_pct(s['avg']):>10}{fmt_pct(s['compound']):>12}"
                                        )
                                        if s["compound"] is not None and (bb is None or s["compound"] > bb[0]):
                                            bb = (s["compound"], h)
                                    if bb is None:
                                        continue
                                    for ln in blk:
                                        print(ln)
                                    blocks.append((bb[0], bb[1], blk))
                                    rows.append({
                                        "key": (win, btc_rise, gate_rise, ref_rise, btc_thr, gate_thr, alt_thr, q),
                                        "stats": stats,
                                        "signals": len(idx),
                                    })

    print("\n" + "=" * 130)
    print(f"COMPACT | VOL+GATES | {target_pair} | compound by hold")
    print("=" * 130)
    head = (
        f"{'Win':>5} {'BTC%':>5} {'G%':>5} {'Ref%':>5} {'BTC':>6} {'Gate':>6} {'Alt':>6} {'Vote':>6} {'sig':>6}"
        + "".join(f"{h:>9}" for h in HORIZON_NAMES)
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        win, btc_rise, gate_rise, ref_rise, btc_thr, gate_thr, alt_thr, q = r["key"]
        line = (
            f"{win:>4}m {btc_rise:>4.1f}% {gate_rise:>4.1f}% {ref_rise:>4.1f}% "
            f"{btc_thr:>5.1f}x {gate_thr:>5.1f}x {alt_thr:>5.1f}x "
            f"{q}/{len(voting_alts)} {r['signals']:>6}"
        )
        for h in HORIZON_NAMES:
            s = r["stats"][h]
            if s["n"] >= MIN_TRADES:
                line += f"{fmt_pct(s['compound']):>9}"
            else:
                line += f"{'—':>9}"
        print(line)
    print("-" * len(head))
    print("A hold is the operating hold only if its neighbors have the same sign. The single peak is not.")
    top = sorted(blocks, key=lambda t: t[0], reverse=True)[:2]
    print("\n" + "=" * 88)
    print(f"TOP 2 COMPOUND BLOCKS" + ("" if top else " | none (no trades)"))
    print("=" * 88)
    for rank, (comp, hold, blk) in enumerate(top, 1):
        print(f"\n#{rank} | Compound {fmt_pct(comp)} at hold {hold}")
        for ln in blk:
            print(ln)
    print("DONE")


if __name__ == "__main__":
    print(f"Start: {datetime.now(BERLIN_TZ):%Y-%m-%d %H:%M:%S %Z}", flush=True)
    main()
