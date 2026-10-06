#!/usr/bin/env python3
"""
Gemeinsame Ref-Bewegung (BTC/ETH/SOL) → Alt kaufen → Hold → ROI.

Signale (UP und DOWN), nicht gemischt:
  VOTE 1/3, 2/3, 3/3  — so viele Refs überschreiten die Schwelle
  AVG                 — Durchschnitt der Ref-Moves ≥ Schwelle
  MEDIAN              — Median der Ref-Moves ≥ Schwelle
SUM wird nur mitausgegeben (Skala hängt von der Ref-Anzahl ab, keine eigene Schwelle).

Fenster: 60 / 180 / 360 Minuten. Schwellen: 1 % / 2 %.
Jeder Hold = eigener 1-Slot-Bot (Non-Overlap = Hold-Länge).
Fees je Seite. Ausgabe: Hold, n, Compound, vs Buy&Hold.

Parameter oben ändern. Keine Handelsempfehlung.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from laggard_common import ms_to_berlin_str, resolve_event_window

# --- hier in VS Code ändern ---
REF_SYMBOLS: List[str] = ["BTC", "ETH", "SOL"]
TARGET_SYMBOL = "UNI"
FROM_DATE: Optional[str] = "2024-01-01"
TO_DATE: Optional[str] = "2026-12-28"
LOOKBACK_DAYS = 400

WINDOWS_MIN: List[int] = [60, 180, 360]
THRESHOLDS_PCT: List[float] = [1.0, 2.0]
USE_THRESHOLD_CROSSING = True

HOLD_HORIZONS: Dict[str, int] = {
    "3h": 180,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
    "48h": 2880,
    "72h": 4320,
    "96h": 5760,
    "120h": 7200,
    "192h": 11520,
}

KLINE_INTERVAL = "1h"
FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0

BINANCE = "https://data-api.binance.vision"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "joint-ref-hold/0.1"})


def interval_to_minutes(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    if unit == "m":
        return n
    if unit == "h":
        return n * 60
    if unit == "d":
        return n * 60 * 24
    raise ValueError(interval)


BAR_MIN = interval_to_minutes(KLINE_INTERVAL)
for _w in WINDOWS_MIN:
    if _w % BAR_MIN != 0:
        raise SystemExit(f"Fenster {_w} min nicht durch {KLINE_INTERVAL} teilbar")
HOLD_BARS = {}
for _name, _mins in HOLD_HORIZONS.items():
    if _mins % BAR_MIN != 0:
        raise SystemExit(f"Hold {_name} nicht durch {KLINE_INTERVAL} teilbar")
    HOLD_BARS[_name] = _mins // BAR_MIN
HORIZON_NAMES = list(HOLD_HORIZONS.keys())
MAX_HOLD_MIN = max(HOLD_HORIZONS.values())
N_REFS = len(REF_SYMBOLS)


def to_pair(raw: str) -> Tuple[str, str]:
    s = raw.strip().upper()
    if s.endswith("USDT") and len(s) > 4:
        return s[:-4], s
    return s, f"{s}USDT"


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def sleep_polite(seconds: float = 0.15) -> None:
    time.sleep(seconds)


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        r = SESSION.get(
            f"{BINANCE}/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": KLINE_INTERVAL,
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
        nxt = data[-1][0] + 1
        if nxt <= cursor:
            break
        cursor = nxt
        sleep_polite(0.12)
        if len(data) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["open_time", "close"])
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore",
    ])
    df = df[["open_time", "close"]].copy()
    df["close"] = df["close"].astype(float)
    df["open_time"] = df["open_time"].astype(np.int64)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def align_refs(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = None
    for name, df in frames.items():
        part = df[["open_time", "close"]].rename(columns={"close": name})
        base = part if base is None else base.merge(part, on="open_time", how="inner")
    if base is None or base.empty:
        raise SystemExit("Keine gemeinsamen Ref-Kerzen.")
    return base.sort_values("open_time").reset_index(drop=True)


def panel_moves(panel: pd.DataFrame, window_bars: int) -> pd.DataFrame:
    out = panel[["open_time"]].copy()
    cols = []
    for name in REF_SYMBOLS:
        col = f"m_{name}"
        out[col] = panel[name].pct_change(window_bars) * 100.0
        cols.append(col)
    mat = out[cols]
    out["avg"] = mat.mean(axis=1)
    out["median"] = mat.median(axis=1)
    out["sum"] = mat.sum(axis=1)
    out["n_up"] = (mat >= 0).sum(axis=1)  # Platzhalter, echte Vote-Zählung pro Schwelle
    return out


def crossing_mask(series: pd.Series, predicate) -> pd.Series:
    hit = predicate(series)
    if not USE_THRESHOLD_CROSSING:
        return hit.fillna(False)
    prev = hit.shift(1)
    return (hit & ~prev.fillna(False)).fillna(False)


def events_for(moves: pd.DataFrame, threshold: float) -> Dict[str, pd.DataFrame]:
    """Jedes Signal: DataFrame mit open_time, avg, median, sum, n_vote."""
    cols = [f"m_{n}" for n in REF_SYMBOLS]
    mat = moves[cols]
    n_up = (mat >= threshold).sum(axis=1)
    n_dn = (mat <= -threshold).sum(axis=1)
    out: Dict[str, pd.DataFrame] = {}

    def pack(mask: pd.Series, direction: str, n_vote: pd.Series) -> pd.DataFrame:
        sub = moves.loc[mask, ["open_time", "avg", "median", "sum"]].copy()
        sub["direction"] = direction
        sub["n_vote"] = n_vote.loc[mask].astype(int).values
        return sub.reset_index(drop=True)

    for k in range(1, N_REFS + 1):
        up = crossing_mask(n_up, lambda s, kk=k: s >= kk)
        dn = crossing_mask(n_dn, lambda s, kk=k: s >= kk)
        out[f"VOTE_{k}/{N_REFS} UP"] = pack(up, "UP", n_up)
        out[f"VOTE_{k}/{N_REFS} DOWN"] = pack(dn, "DOWN", n_dn)
    for kind, series in (("AVG", moves["avg"]), ("MEDIAN", moves["median"])):
        up = crossing_mask(series, lambda s, th=threshold: s >= th)
        dn = crossing_mask(series, lambda s, th=threshold: s <= -th)
        out[f"{kind} UP"] = pack(up, "UP", n_up)
        out[f"{kind} DOWN"] = pack(dn, "DOWN", n_dn)
    return out


def apply_costs(entry: float, exit_: float) -> Tuple[float, float]:
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    return entry * (1.0 + cost), exit_ * (1.0 - cost)


def buy_and_hold(df: pd.DataFrame, start_ms: int, end_ms: int) -> Optional[float]:
    sub = df[(df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)]
    if len(sub) < 2:
        return None
    e, x = float(sub.iloc[0]["close"]), float(sub.iloc[-1]["close"])
    if e <= 0:
        return None
    buy, sell = apply_costs(e, x)
    return (sell / buy - 1.0) * 100.0


def simulate(target: pd.DataFrame, events: pd.DataFrame) -> List[dict]:
    if events.empty:
        return []
    idx = target.set_index("open_time", drop=False)
    times = idx.index
    trades = []
    for row in events.itertuples(index=False):
        t = int(row.open_time)
        if t in times:
            pos = times.get_loc(t)
        else:
            later = times[times >= t]
            if len(later) == 0:
                continue
            pos = times.get_loc(later[0])
        if isinstance(pos, slice):
            pos = pos.start
        if isinstance(pos, np.ndarray):
            pos = int(pos[0])
        pos = int(pos)
        entry = float(idx.iloc[pos]["close"])
        if entry <= 0:
            continue
        trade = {
            "buy_time_ms": int(idx.iloc[pos]["open_time"]),
            "avg": float(row.avg),
            "median": float(row.median),
            "sum": float(row.sum),
        }
        ok = False
        for name, bars in HOLD_BARS.items():
            exit_pos = pos + bars
            if exit_pos >= len(idx):
                trade[f"roi_{name}"] = None
                continue
            exit_ = float(idx.iloc[exit_pos]["close"])
            buy, sell = apply_costs(entry, exit_)
            trade[f"roi_{name}"] = (sell / buy - 1.0) * 100.0
            ok = True
        if ok:
            trades.append(trade)
    return trades


def non_overlap(trades: List[dict], horizon: str) -> List[float]:
    hold_ms = HOLD_HORIZONS[horizon] * 60_000
    col = f"roi_{horizon}"
    last = None
    vals = []
    for t in sorted(trades, key=lambda x: x["buy_time_ms"]):
        v = t.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        b = int(t["buy_time_ms"])
        if last is not None and b <= last + hold_ms:
            continue
        vals.append(float(v))
        last = b
    return vals


def compound(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    eq = 1.0
    for r in vals:
        eq *= 1.0 + r / 100.0
    return (eq - 1.0) * 100.0


def fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "     –"
    return f"{v:+.2f}%"


# ------------------------------------------------------------
# Block-Header: effektiver Zeitraum (Europe/Berlin)
# ------------------------------------------------------------

def _effective_range_ms(start_ms, end_ms, *time_arrays):
    """Effektiver Zeitraum: [start, end] begrenzt auf die tatsaechlich vorhandenen Kerzen
    (spaeteres Listing / frueheres Datenende)."""
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


def _block_range(from_ms, to_ms) -> str:
    """'from YYYY-MM-DD to YYYY-MM-DD' (Europe/Berlin) fuer Block-Header."""
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo as _ZoneInfo
    _tz = _ZoneInfo("Europe/Berlin")
    a = _dt.fromtimestamp(from_ms / 1000.0, tz=_tz).strftime("%Y-%m-%d")
    b = _dt.fromtimestamp(to_ms / 1000.0, tz=_tz).strftime("%Y-%m-%d")
    return f"from {a} to {b}"


def print_table(title: str, trades: List[dict], bh: Optional[float], events: pd.DataFrame) -> None:
    sum_note = ""
    if not events.empty:
        sum_note = (
            f"  | Events n={len(events)}  "
            f"ØAVG={events['avg'].mean():+.2f}%  "
            f"ØMED={events['median'].mean():+.2f}%  "
            f"ØSUM={events['sum'].mean():+.2f}%"
        )
    print()
    print(title + sum_note)
    print(f"{'Hold':<8} {'n':>6} {'Cmp':>10} {'vsBH':>10}")
    for h in HORIZON_NAMES:
        vals = non_overlap(trades, h)
        cmp = compound(vals)
        vs = None if cmp is None or bh is None else cmp - bh
        print(f"{h:<8} {len(vals):>6} {fmt(cmp):>10} {fmt(vs):>10}")
    print(f"Buy&Hold {fmt(bh)}")
    print(flush=True)


def main() -> None:
    refs = [to_pair(x) for x in REF_SYMBOLS]
    target_base, target_pair = to_pair(TARGET_SYMBOL)
    if target_pair in {p for _, p in refs}:
        raise SystemExit("Ziel darf keine Ref sein.")

    start, end, label = resolve_event_window(FROM_DATE, TO_DATE, None, LOOKBACK_DAYS)
    start_ms, end_ms = utc_ms(start), utc_ms(end)
    fetch_end_ms = utc_ms(end + pd.Timedelta(minutes=MAX_HOLD_MIN))

    print("=== Gemeinsame Refs → Alt-Hold ===", flush=True)
    print(f"Refs: {[b for b, _ in refs]}  Ziel: {target_base}", flush=True)
    print(f"Fenster {WINDOWS_MIN} min | Schwellen {THRESHOLDS_PCT}% | {label}", flush=True)
    print(f"Fee {FEE_BPS} bps/Seite | Crossing={USE_THRESHOLD_CROSSING}", flush=True)
    print("VOTE = wie viele Refs die Schwelle reißen. AVG/MEDIAN = Stärke, gleiche Schwelle.", flush=True)
    print("SUM nur Info (nicht als Signal). n = nach Non-Overlap. vsBH = Cmp − Buy&Hold.\n", flush=True)

    frames = {}
    for base, pair in refs:
        print(f"Lade {pair} …", flush=True)
        df = fetch_klines(pair, start_ms, end_ms)
        if df.empty:
            raise SystemExit(f"Keine Kerzen {pair}")
        frames[base] = df
    panel = align_refs(frames)

    print(f"Lade {target_pair} …", flush=True)
    target = fetch_klines(target_pair, start_ms, fetch_end_ms)
    if target.empty:
        raise SystemExit(f"Keine Kerzen {target_pair}")
    bh = buy_and_hold(target, start_ms, end_ms)
    block_period = _block_range(*_effective_range_ms(
        start_ms, end_ms, panel["open_time"].to_numpy(), target["open_time"].to_numpy()
    ))

    for window_min in WINDOWS_MIN:
        bars = window_min // BAR_MIN
        moves = panel_moves(panel, bars)
        for th in THRESHOLDS_PCT:
            signals = events_for(moves, th)
            for name, ev in signals.items():
                trades = simulate(target, ev)
                print_table(
                    f"{name} | {th:g}% / {window_min // 60}h | {target_base} | {block_period}",
                    trades,
                    bh,
                    ev,
                )

    print("\nFertig. Keine Handelsempfehlung.", flush=True)


if __name__ == "__main__":
    main()
