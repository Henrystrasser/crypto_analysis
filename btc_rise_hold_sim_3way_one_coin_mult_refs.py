#!/usr/bin/env python3
"""
Simulation: Ref-Move-Event (UP/DOWN) → Ziel-Coin kaufen → multi-Hold → ROI

Erweiterungen gegenüber der Basisvariante:
  1) Mehrere Refs getrennt (nicht gemischt): BTC, ETH, SOL …
     Gleiches Ziel, gleiche Holds — Vergleich, ob der Drift Ref-übergreifend ist.
  2) Zwei unabhängige Event-Specs (Default: 2%/6h und 3%/12h).
     Am Ende: Vorzeichen-Agreement je Horizont/Richtung.
  3) Event-Cluster + Block-Bootstrap:
     benachbarte Events (Lücke < CLUSTER_GAP_HOURS) = 1 Block.
     Zeigt, ob Ø ROI nur in wenigen Wochen steckt; 90%-CI per Cluster-Resample.

Trigger je Spec:
  UP:   Ref ≥ +threshold in window_min
  DOWN: Ref ≤ −threshold in window_min

Einstieg: Close Ziel am Fensterende. Ausstieg: Close nach Hold.
Jeder Horizont = eigene 1-Slot-Strategie (Non-Overlap = Hold-Länge).
Fees: FEE_BPS je Seite.

Alle Zeiten: Europe/Berlin.
Keine Handelsempfehlung.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from laggard_common import (
    ms_to_berlin_str,
    resolve_event_window,
)

import numpy as np
import pandas as pd
import requests

# --- Coins ---
REF_SYMBOLS: List[str] = ["BTC", "ETH", "SOL"]
TARGET_SYMBOL = "VIRTUAL"
FROM_DATE: Optional[str] = "2024-01-01"
TO_DATE: Optional[str] = "2026-12-28"

# Unabhängige Event-Definitionen (nicht mischen — getrennt rechnen, dann Agreement)
EVENT_SPECS: List[dict] = [
    {"name": "2pct_6h", "window_min": 360, "threshold_pct": 2.0},
    {"name": "3pct_12h", "window_min": 720, "threshold_pct": 3.0},
]

USE_THRESHOLD_CROSSING = True
RANDOM_SEED = 42
N_RANDOM_TRADES: Optional[int] = None
RUN_RANDOM = False

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
    "192h": 11520,
}

LOOKBACK_DAYS = 400
KLINE_INTERVAL = "1h"
FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0
SHOW_TRADES = False

# Cluster: Events näher als diese Lücke (Stunden) gehören zum selben Block
CLUSTER_GAP_HOURS = 48
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

BINANCE = "https://data-api.binance.vision"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-rise-hold-sim/0.7-multi"})


def normalize_ref_symbol(raw: str) -> tuple[str, str]:
    s = (raw or "BTC").strip().upper()
    if s.endswith("USDT") and len(s) > 4:
        base, pair = s[:-4], s
    else:
        base, pair = s, f"{s}USDT"
    if not base:
        raise SystemExit(f"Ungültiger Coin: {raw!r}")
    return base, pair


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-Hold-Sim — Parameter oben in der Datei ändern")
    return p.parse_args(argv)


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
HOLD_BARS: Dict[str, int] = {}
for _name, _mins in HOLD_HORIZONS.items():
    if _mins % BAR_MIN != 0:
        raise SystemExit(f"Hold {_name} nicht durch {KLINE_INTERVAL} teilbar")
    HOLD_BARS[_name] = _mins // BAR_MIN
MAX_HOLD_MIN = max(HOLD_HORIZONS.values())
HORIZON_NAMES = list(HOLD_HORIZONS.keys())


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def sleep_polite(seconds: float = 0.3) -> None:
    time.sleep(seconds)


def binance_usdt_symbols() -> set:
    r = SESSION.get(f"{BINANCE}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    return {
        s["symbol"]
        for s in r.json()["symbols"]
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
    }


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
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
        sleep_polite(0.15)
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
class MoveEvent:
    start_ms: int
    end_ms: int
    ref_move_pct: float
    direction: str


def detect_move_events(
    ref_df: pd.DataFrame,
    window_bars: int,
    threshold_pct: float,
) -> Tuple[List[MoveEvent], List[MoveEvent]]:
    df = ref_df.copy()
    df["move_pct"] = df["close"].pct_change(window_bars) * 100.0
    ups: List[MoveEvent] = []
    downs: List[MoveEvent] = []
    for i in range(len(df)):
        ret = df.at[i, "move_pct"]
        if pd.isna(ret):
            continue
        prev = df.at[i - 1, "move_pct"] if i > 0 else np.nan
        start_i = i - window_bars
        if start_i < 0:
            continue
        if ret >= threshold_pct:
            if USE_THRESHOLD_CROSSING and pd.notna(prev) and prev >= threshold_pct:
                pass
            else:
                ups.append(
                    MoveEvent(
                        start_ms=int(df.at[start_i, "open_time"]),
                        end_ms=int(df.at[i, "open_time"]),
                        ref_move_pct=float(ret),
                        direction="UP",
                    )
                )
        if ret <= -threshold_pct:
            if USE_THRESHOLD_CROSSING and pd.notna(prev) and prev <= -threshold_pct:
                pass
            else:
                downs.append(
                    MoveEvent(
                        start_ms=int(df.at[start_i, "open_time"]),
                        end_ms=int(df.at[i, "open_time"]),
                        ref_move_pct=float(ret),
                        direction="DOWN",
                    )
                )
    return ups, downs


def index_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("open_time", drop=False)


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


def apply_costs(entry: float, exit_: float) -> Tuple[float, float]:
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    return entry * (1.0 + cost), exit_ * (1.0 - cost)


def buy_and_hold_roi(df: pd.DataFrame, period_start_ms: int, period_end_ms: int) -> Optional[dict]:
    if df.empty:
        return None
    sub = df[(df["open_time"] >= period_start_ms) & (df["open_time"] <= period_end_ms)]
    if len(sub) < 2:
        sub = df[df["open_time"] <= period_end_ms]
        if len(sub) < 2:
            return None
    entry_raw = float(sub.iloc[0]["close"])
    exit_raw = float(sub.iloc[-1]["close"])
    if entry_raw <= 0:
        return None
    buy, sell = apply_costs(entry_raw, exit_raw)
    return {
        "bh_buy_time": ms_to_berlin_str(int(sub.iloc[0]["open_time"])),
        "bh_sell_time": ms_to_berlin_str(int(sub.iloc[-1]["open_time"])),
        "buy_hold_roi_pct": (sell / buy - 1.0) * 100.0,
    }


def roi_for_hold(idx: pd.DataFrame, entry_pos: int, hold_bars: int) -> Optional[float]:
    exit_pos = entry_pos + hold_bars
    if exit_pos >= len(idx):
        return None
    entry_raw = float(idx.iloc[entry_pos]["close"])
    exit_raw = float(idx.iloc[exit_pos]["close"])
    if entry_raw <= 0:
        return None
    buy, sell = apply_costs(entry_raw, exit_raw)
    return (sell / buy - 1.0) * 100.0


def simulate_coin(coin: str, df: pd.DataFrame, events: List[MoveEvent]) -> List[dict]:
    idx = index_by_time(df)
    trades: List[dict] = []
    for e in events:
        entry_pos = loc_at(idx, e.end_ms)
        if entry_pos is None:
            continue
        entry_raw = float(idx.iloc[entry_pos]["close"])
        if entry_raw <= 0:
            continue
        buy_time_ms = int(idx.iloc[entry_pos]["open_time"])
        row: dict = {
            "coin": coin,
            "direction": e.direction,
            "ref_move_pct": e.ref_move_pct,
            "buy_time": ms_to_berlin_str(buy_time_ms),
            "buy_time_ms": buy_time_ms,
        }
        any_ok = False
        for name, bars in HOLD_BARS.items():
            r = roi_for_hold(idx, entry_pos, bars)
            row[f"roi_{name}"] = r
            if r is not None:
                any_ok = True
        if any_ok:
            trades.append(row)
    return trades


def _random_candidate_indices(
    df: pd.DataFrame, period_start_ms: int, period_end_ms: int, need_bars: int
) -> List[int]:
    out: List[int] = []
    for i in range(len(df)):
        t = int(df.at[i, "open_time"])
        if t < period_start_ms or t > period_end_ms:
            continue
        if i + need_bars >= len(df):
            continue
        if float(df.at[i, "close"]) <= 0:
            continue
        out.append(i)
    return out


def _sample_nonoverlapping_positions(
    df: pd.DataFrame,
    candidates: List[int],
    n_target: int,
    hold_ms: int,
    rng: np.random.Generator,
) -> List[int]:
    if n_target <= 0 or not candidates:
        return []

    def ok(positions: List[int]) -> bool:
        ordered = sorted(positions, key=lambda p: int(df.at[p, "open_time"]))
        last = None
        for p in ordered:
            t = int(df.at[p, "open_time"])
            if last is not None and t <= last + hold_ms:
                return False
            last = t
        return True

    accepted: List[int] = []
    order = list(candidates)
    rng.shuffle(order)
    for pos in order:
        trial = accepted + [pos]
        if ok(trial):
            accepted.append(pos)
            if len(accepted) >= n_target:
                break
    if len(accepted) >= n_target:
        return sorted(accepted[:n_target], key=lambda p: int(df.at[p, "open_time"]))
    ordered_all = sorted(candidates, key=lambda p: int(df.at[p, "open_time"]))
    packed: List[int] = []
    last = None
    for pos in ordered_all:
        t = int(df.at[pos, "open_time"])
        if last is not None and t <= last + hold_ms:
            continue
        packed.append(pos)
        last = t
    if len(packed) <= n_target:
        return packed
    pick = sorted(rng.choice(len(packed), size=n_target, replace=False))
    return [packed[i] for i in pick]


def simulate_random_matched_horizons(
    coin: str,
    df: pd.DataFrame,
    target_n_by_h: Dict[str, int],
    period_start_ms: int,
    period_end_ms: int,
    rng: np.random.Generator,
) -> Tuple[List[dict], Dict[str, dict]]:
    idx = index_by_time(df)
    all_trades: List[dict] = []
    by_h: Dict[str, dict] = {}
    for name in HORIZON_NAMES:
        n_target = int(target_n_by_h.get(name, 0) or 0)
        hold_ms = HOLD_HORIZONS[name] * 60_000
        need_bars = HOLD_BARS[name]
        cands = _random_candidate_indices(df, period_start_ms, period_end_ms, need_bars)
        positions = _sample_nonoverlapping_positions(df, cands, n_target, hold_ms, rng)
        trades_h: List[dict] = []
        for entry_pos in positions:
            t_ms = int(df.at[entry_pos, "open_time"])
            pos = loc_at(idx, t_ms)
            if pos is None:
                continue
            entry_raw = float(idx.iloc[pos]["close"])
            if entry_raw <= 0:
                continue
            row: dict = {
                "coin": coin,
                "direction": "RANDOM",
                "ref_move_pct": None,
                "buy_time": ms_to_berlin_str(t_ms),
                "buy_time_ms": t_ms,
            }
            for hname in HORIZON_NAMES:
                row[f"roi_{hname}"] = None
            r = roi_for_hold(idx, pos, need_bars)
            row[f"roi_{name}"] = r
            if r is not None:
                trades_h.append(row)
                all_trades.append(row)
        st = column_stats(trades_h, f"roi_{name}")
        st["n_raw"] = n_target
        st["n_used"] = st["n_trades"]
        by_h[name] = st
    return all_trades, by_h


def fmt_pct(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   n/a"
    return f"{v:+.2f}%"


def column_stats(rows: List[dict], col: str) -> dict:
    vals: List[float] = []
    n_pos = n_neg = n_na = 0
    for row in rows:
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            n_na += 1
            continue
        fv = float(v)
        vals.append(fv)
        if fv > 0:
            n_pos += 1
        elif fv < 0:
            n_neg += 1
    denom = n_pos + n_neg
    pct_pos = (100.0 * n_pos / denom) if denom > 0 else None
    avg = float(np.mean(vals)) if vals else None
    if vals:
        equity = 1.0
        for r in vals:
            equity *= 1.0 + r / 100.0
        compound = (equity - 1.0) * 100.0
    else:
        compound = None
    return {
        "n_trades": len(vals),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_na": n_na,
        "pct_positive": pct_pos,
        "avg_roi_pct": avg,
        "compound_roi_pct": compound,
        "vals": vals,
    }


def filter_non_overlapping(trades: List[dict], horizon: str) -> List[dict]:
    hold_ms = HOLD_HORIZONS[horizon] * 60_000
    col = f"roi_{horizon}"
    ordered = sorted(trades, key=lambda t: int(t["buy_time_ms"]))
    kept: List[dict] = []
    last = None
    for t in ordered:
        v = t.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        buy_ms = int(t["buy_time_ms"])
        if last is not None and buy_ms <= last + hold_ms:
            continue
        kept.append(t)
        last = buy_ms
    return kept


def assign_clusters(trades: List[dict], gap_ms: int) -> List[List[dict]]:
    ordered = sorted(trades, key=lambda t: int(t["buy_time_ms"]))
    clusters: List[List[dict]] = []
    cur: List[dict] = []
    last_ms: Optional[int] = None
    for t in ordered:
        tms = int(t["buy_time_ms"])
        if last_ms is None or tms - last_ms > gap_ms:
            if cur:
                clusters.append(cur)
            cur = [t]
        else:
            cur.append(t)
        last_ms = tms
    if cur:
        clusters.append(cur)
    return clusters


def cluster_diagnostics(
    filtered: List[dict], horizon: str, gap_ms: int, rng: np.random.Generator, n_boot: int
) -> dict:
    col = f"roi_{horizon}"
    clusters = assign_clusters(filtered, gap_ms)
    cluster_means: List[float] = []
    cluster_sizes: List[int] = []
    cluster_sums: List[float] = []
    cluster_first: List[str] = []
    for cl in clusters:
        vals = [float(t[col]) for t in cl if t.get(col) is not None]
        if not vals:
            continue
        cluster_means.append(float(np.mean(vals)))
        cluster_sizes.append(len(vals))
        cluster_sums.append(float(np.sum(vals)))
        cluster_first.append(cl[0]["buy_time"])
    n_cl = len(cluster_means)
    eq_mean = float(np.mean(cluster_means)) if cluster_means else None
    tw_mean = (
        float(np.mean([float(t[col]) for t in filtered if t.get(col) is not None]))
        if filtered
        else None
    )
    top_share = None
    top_note = ""
    if cluster_sums:
        total = sum(abs(x) for x in cluster_sums) or 1.0
        ranked = sorted(
            zip(cluster_sums, cluster_sizes, cluster_first, cluster_means),
            key=lambda z: abs(z[0]),
            reverse=True,
        )
        k = min(3, len(ranked))
        top_share = sum(abs(ranked[i][0]) for i in range(k)) / total * 100.0
        bits = [
            f"{ranked[i][2][:16]} n={ranked[i][1]} Ø={ranked[i][3]:+.1f}%"
            for i in range(k)
        ]
        top_note = "; ".join(bits)

    ci_lo = ci_hi = None
    if n_cl >= 2 and n_boot > 0:
        boot = []
        idx = np.arange(n_cl)
        for _ in range(n_boot):
            draw = rng.choice(idx, size=n_cl, replace=True)
            boot.append(float(np.mean([cluster_means[int(j)] for j in draw])))
        ci_lo = float(np.percentile(boot, 5))
        ci_hi = float(np.percentile(boot, 95))

    return {
        "n_clusters": n_cl,
        "cluster_eq_mean": eq_mean,
        "trade_mean": tw_mean,
        "top3_abs_share_pct": top_share,
        "top3_note": top_note,
        "ci90_lo": ci_lo,
        "ci90_hi": ci_hi,
        "ci_excludes_zero": (
            ci_lo is not None and ci_hi is not None and (ci_hi < 0 or ci_lo > 0)
        ),
    }


def summarize_by_horizon(
    trades: List[dict], gap_ms: int, rng: np.random.Generator, n_boot: int
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for h in HORIZON_NAMES:
        col = f"roi_{h}"
        n_raw = sum(
            1
            for t in trades
            if t.get(col) is not None and not (isinstance(t.get(col), float) and np.isnan(t[col]))
        )
        filtered = filter_non_overlapping(trades, h)
        st = column_stats(filtered, col)
        st["n_raw"] = n_raw
        st["n_used"] = st["n_trades"]
        st["cluster"] = cluster_diagnostics(filtered, h, gap_ms, rng, n_boot)
        out[h] = st
    return out


def fmt_pos(stats: dict) -> str:
    p = stats.get("pct_positive")
    if p is None:
        return "   n/a"
    return f"{p:5.0f}%"


def vs_bh_for_stats(stats: dict, bh: Optional[dict]) -> Optional[float]:
    if bh is None or stats.get("compound_roi_pct") is None:
        return None
    return stats["compound_roi_pct"] - bh["buy_hold_roi_pct"]


def fmt_vs_bh(v: Optional[float]) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "–"
    return fmt_pct(v)


def print_block(ref: str, spec: dict, direction: str, pair: str, trades: List[dict], by_h: Dict[str, dict], bh: Optional[dict]) -> None:
    print()
    print(
        f"{direction} | Ref={ref} | {spec['threshold_pct']:g}% / {spec['window_min'] // 60}h | Ziel={pair}"
    )
    print(f"{'Hold':<8} {'n':>6} {'Cmp':>10} {'vsBH':>10}")
    for h in HORIZON_NAMES:
        st = by_h[h]
        n_used = st.get("n_used", st["n_trades"])
        print(
            f"{h:<8} {n_used:>6} {fmt_pct(st['compound_roi_pct']):>10} "
            f"{fmt_vs_bh(vs_bh_for_stats(st, bh)):>10}"
        )
    if bh is not None:
        print(f"Buy&Hold {fmt_pct(bh['buy_hold_roi_pct'])}")
    print(flush=True)


def sign_of(v: Optional[float]) -> Optional[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def print_agreement(store: List[dict], target: str) -> None:
    specs = [s["name"] for s in EVENT_SPECS]
    if len(specs) < 2:
        return
    print()
    print(f"AGREEMENT | Ziel={target} | Cmp je Spec")
    refs = []
    for row in store:
        if row["ref"] not in refs:
            refs.append(row["ref"])
    by = {}
    for row in store:
        by[(row["ref"], row["spec"], row["direction"], row["horizon"])] = row
    for ref in refs:
        for direction in ("UP", "DOWN"):
            print(f"\n{ref} {direction}")
            print(f"{'Hold':<8} " + "".join(f"{sp+' n':>10} {sp+' Cmp':>12} {sp+' vsBH':>12}" for sp in specs))
            for h in HORIZON_NAMES:
                line = f"{h:<8}"
                for sp in specs:
                    row = by.get((ref, sp, direction, h))
                    if row is None:
                        line += f"{'–':>10} {'–':>12} {'–':>12}"
                    else:
                        line += (
                            f"{row.get('n_used', 0):>10} "
                            f"{fmt_pct(row.get('cmp')):>12} "
                            f"{fmt_vs_bh(row.get('vs_bh')):>12}"
                        )
                print(line)
    print()


def print_ref_consistency(store: List[dict], target: str) -> None:
    print(f"REF-VERGLEICH | Ziel={target} | Cmp je Ref")
    specs = [s["name"] for s in EVENT_SPECS]
    refs = []
    for row in store:
        if row["ref"] not in refs:
            refs.append(row["ref"])
    by = {}
    for row in store:
        by[(row["ref"], row["spec"], row["direction"], row["horizon"])] = row
    for spec in specs:
        for direction in ("UP", "DOWN"):
            print(f"\n{spec} {direction}")
            print(f"{'Hold':<8} " + "".join(f"{r+' n':>8} {r+' Cmp':>10} {r+' vsBH':>10}" for r in refs))
            for h in HORIZON_NAMES:
                line = f"{h:<8}"
                for ref in refs:
                    row = by.get((ref, spec, direction, h))
                    if row is None:
                        line += f"{'–':>8} {'–':>10} {'–':>10}"
                    else:
                        line += (
                            f"{row.get('n_used', 0):>8} "
                            f"{fmt_pct(row.get('cmp')):>10} "
                            f"{fmt_vs_bh(row.get('vs_bh')):>10}"
                        )
                print(line)
    print()


def resolve_target_pair(target_raw: str, ref_pairs: List[str]) -> str:
    _base, pair = normalize_ref_symbol(target_raw)
    if pair in ref_pairs:
        raise SystemExit("Ziel-Coin darf nicht eine der Refs sein.")
    usdt_set = binance_usdt_symbols()
    if pair not in usdt_set:
        raise SystemExit(f"Ziel-Pair {pair} nicht auf Binance Spot USDT.")
    return pair


def main(argv: Optional[List[str]] = None) -> None:
    parse_cli_args(argv)
    run_random = RUN_RANDOM
    refs = [normalize_ref_symbol(x) for x in REF_SYMBOLS]
    ref_pairs = [p for _, p in refs]
    target_base, _ = normalize_ref_symbol(TARGET_SYMBOL)
    target_pair = resolve_target_pair(TARGET_SYMBOL, ref_pairs)
    gap_h = float(CLUSTER_GAP_HOURS)
    gap_ms = int(gap_h * 3600 * 1000)
    n_boot = int(N_BOOTSTRAP)

    start, end, window_label = resolve_event_window(
        from_s=FROM_DATE,
        to_s=TO_DATE,
        lookback_days=None,
        default_lookback=LOOKBACK_DAYS,
    )
    fetch_end = end + pd.Timedelta(minutes=MAX_HOLD_MIN)
    start_ms, end_ms = utc_ms(start), utc_ms(end)
    fetch_end_ms = utc_ms(fetch_end)

    print("=== Multi-Ref × Multi-Spec Event-Hold ===\n", flush=True)
    print(f"Refs (getrennt): {[b for b, _ in refs]}", flush=True)
    print(f"Ziel: {target_base} ({target_pair})", flush=True)
    print(
        "Specs: "
        + ", ".join(f"{s['name']}={s['threshold_pct']:g}%/{s['window_min']}m" for s in EVENT_SPECS),
        flush=True,
    )
    print(
        f"Zeitraum: {start.strftime('%Y-%m-%d %H:%M %Z')} → {end.strftime('%Y-%m-%d %H:%M %Z')} | {window_label}",
        flush=True,
    )
    print(
        f"Cluster-Lücke: {gap_h:g}h | Holds {HORIZON_NAMES}",
        flush=True,
    )
    print(
        f"Fee {FEE_BPS} bps/Seite | Crossing={USE_THRESHOLD_CROSSING} | RANDOM={run_random}\n",
        flush=True,
    )

    print("Lade Ziel-Kerzen …", flush=True)
    target_df = fetch_klines(target_pair, KLINE_INTERVAL, start_ms, fetch_end_ms)
    if target_df.empty:
        raise SystemExit(f"Keine Kerzen für {target_pair}")
    bh = buy_and_hold_roi(target_df, start_ms, end_ms)
    print(f"  {len(target_df)} Kerzen Ziel\n", flush=True)

    store: List[dict] = []
    rng_boot = np.random.default_rng(BOOTSTRAP_SEED)

    for ref_base, ref_pair in refs:
        print(f"Lade Ref {ref_base} ({ref_pair}) …", flush=True)
        ref_df = fetch_klines(ref_pair, KLINE_INTERVAL, start_ms, end_ms)
        if ref_df.empty:
            print(f"  keine Kerzen — skip {ref_base}\n", flush=True)
            continue
        print(f"  {len(ref_df)} Kerzen\n", flush=True)

        for spec in EVENT_SPECS:
            if spec["window_min"] % BAR_MIN != 0:
                raise SystemExit(f"Spec {spec['name']}: window nicht durch Intervall teilbar")
            window_bars = spec["window_min"] // BAR_MIN
            ups, downs = detect_move_events(ref_df, window_bars, spec["threshold_pct"])
            print(
                f"  {ref_base} {spec['name']}: {len(ups)} UP / {len(downs)} DOWN",
                flush=True,
            )

            dir_by_h: Dict[str, Dict[str, dict]] = {}
            for direction, events in (("UP", ups), ("DOWN", downs)):
                if not events:
                    continue
                trades = simulate_coin(target_pair, target_df, events)
                by_h = summarize_by_horizon(trades, gap_ms, rng_boot, n_boot)
                dir_by_h[direction] = by_h
                print_block(ref_base, spec, direction, target_pair, trades, by_h, bh)
                for h, st in by_h.items():
                    store.append(
                        {
                            "ref": ref_base,
                            "spec": spec["name"],
                            "direction": direction,
                            "horizon": h,
                            "n_used": st.get("n_used"),
                            "cmp": st["compound_roi_pct"],
                            "vs_bh": vs_bh_for_stats(st, bh),
                        }
                    )

            if run_random:
                rng = np.random.default_rng(RANDOM_SEED + sum(ord(c) for c in ref_pair + spec["name"]))
                if N_RANDOM_TRADES is not None:
                    targets = {h: int(N_RANDOM_TRADES) for h in HORIZON_NAMES}
                else:
                    targets = {}
                    for h in HORIZON_NAMES:
                        n_up = int(dir_by_h.get("UP", {}).get(h, {}).get("n_used", 0) or 0)
                        n_dn = int(dir_by_h.get("DOWN", {}).get(h, {}).get("n_used", 0) or 0)
                        targets[h] = max(n_up, n_dn)
                rand_trades, rand_by_h = simulate_random_matched_horizons(
                    target_pair, target_df, targets, start_ms, end_ms, rng
                )
                print_block(ref_base, spec, "RANDOM", target_pair, rand_trades, rand_by_h, bh)

    if store:
        print_agreement(store, target_pair)
        print_ref_consistency(store, target_pair)

    print(
        "n = Trades nach Non-Overlap. Cmp = reinvestierter ROI. vsBH = Cmp − Buy&Hold.\n"
        "Keine Handelsempfehlung.\n",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
