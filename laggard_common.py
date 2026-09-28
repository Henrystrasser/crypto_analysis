#!/usr/bin/env python3
"""
Gemeinsame Bausteine für Laggard-/Event-Hold-Simulationen.

Extrahiert aus btc_rise_hold_sim.py (v0.7), damit ETF-/TradFi-/Derivatives-
Skripte dieselbe Methodik nutzen (Horizonte, Fees, Non-Overlap, RANDOM, B&H).

Enthält auch CLI-Helfer für Event-Zeitfenster (--from/--to oder --lookback).

Keine Handelsempfehlung — reine Research-Backtests.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Defaults (Skripte dürfen Module-Attribute überschreiben oder Parameter setzen)
# ---------------------------------------------------------------------------

HOLD_HORIZONS: Dict[str, int] = {
    "24h": 1440,
    "48h": 2880,
    "96h": 5760,
    "192h": 11520,
}

FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0
KLINE_INTERVAL = "1h"
LOOKBACK_DAYS = 90
TOP_N = 100
RANDOM_SEED = 42
SHOW_TRADES = False

STABLE_SYMBOLS = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "USDD",
    "FRAX", "PYUSD", "EURC", "EURT", "GUSD", "LUSD", "SUSD", "USDP",
    "USD1", "USDY", "RLUSD", "CRVUSD", "GHO", "USDS",
}


EXCLUDE_SYMBOLS = {
    "HYPE",
    "ZEC",
}

COINGECKO = "https://api.coingecko.com/api/v3"
BINANCE_BASES = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
]
BINANCE = BINANCE_BASES[0]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "laggard-common/0.1"})


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """Exogenes Event; Coin-ROI wird erst NACH end_ms gemessen."""

    start_ms: int
    end_ms: int  # Einstiegs-Referenzzeit (erster Coin-Bar at/after)
    value: float
    direction: str  # z.B. INFLOW, OUTFLOW, RISK_ON, FUNDING_POS, …
    label: str = ""  # optionaler Anzeige-Name
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.direction


# ---------------------------------------------------------------------------
# Zeit / Intervalle
# ---------------------------------------------------------------------------


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


def hold_bars_map(
    kline_interval: str = KLINE_INTERVAL,
    hold_horizons: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    horizons = hold_horizons or HOLD_HORIZONS
    bar_min = interval_to_minutes(kline_interval)
    out: Dict[str, int] = {}
    for name, mins in horizons.items():
        if mins % bar_min != 0:
            raise ValueError(
                f"Hold {name} ({mins} Min) muss durch KLINE_INTERVAL "
                f"({bar_min} Min) teilbar sein."
            )
        out[name] = mins // bar_min
    return out


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_utc_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


# ---------------------------------------------------------------------------
# CLI: Event-Zeitfenster (--from/--to oder --lookback)
# ---------------------------------------------------------------------------

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_TIME_FMTS = (
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
)


def parse_utc_date(s: str, *, end_of_day: bool = False) -> datetime:
    """
    Parse a UTC date/datetime string.

    Accepted formats:
      YYYY-MM-DD
      YYYY-MM-DDTHH:MM  /  YYYY-MM-DD HH:MM
      YYYY-MM-DDTHH:MM:SS[.fff]  /  YYYY-MM-DD HH:MM:SS[.fff]

    Date-only:
      --from: 00:00:00.000 UTC on that day
      --to (end_of_day=True): 23:59:59.999 UTC on that day (inclusive calendar day)

    Datetime with time: used as-is (UTC); end_of_day is ignored.
    """
    raw = (s or "").strip()
    if not raw:
        raise SystemExit("Fehler: leeres Datumsargument.")

    if _DATE_ONLY_RE.match(raw):
        y, m, d = (int(x) for x in raw.split("-"))
        if end_of_day:
            return datetime(y, m, d, 23, 59, 59, 999000, tzinfo=timezone.utc)
        return datetime(y, m, d, 0, 0, 0, 0, tzinfo=timezone.utc)

    for fmt in _DATE_TIME_FMTS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    raise SystemExit(
        f"Fehler: Ungültiges Datum {s!r}. Erwartet YYYY-MM-DD oder "
        "YYYY-MM-DDTHH:MM / YYYY-MM-DD HH:MM (UTC)."
    )


def resolve_event_window(
    *,
    from_s: Optional[str],
    to_s: Optional[str],
    lookback_days: Optional[int],
    default_lookback: int,
) -> Tuple[datetime, datetime, str]:
    """
    Resolve (start, end, mode_label) for the event window (UTC).

    Rules:
      - --to without --from → SystemExit (German)
      - --from and --lookback together → SystemExit
      - --from set, --to omitted → end = now UTC
      - neither --from nor --lookback → lookback = default_lookback, end = now
      - start must be < end
    """
    if to_s and not from_s:
        raise SystemExit(
            "Fehler: --to/--end bzw. TO_DATE ohne FROM_DATE/--from/--start ist ungültig. "
            "Bitte FROM_DATE oder einen Startzeitpunkt mit --from/--start angeben."
        )
    if from_s and lookback_days is not None:
        raise SystemExit(
            "Fehler: FROM_DATE/--from und LOOKBACK/--lookback schließen sich aus. "
            "Bitte nur einen Modus aktivieren."
        )

    if from_s:
        start = parse_utc_date(from_s, end_of_day=False)
        if to_s:
            end = parse_utc_date(to_s, end_of_day=True)
        else:
            end = datetime.now(tz=timezone.utc)
        if start >= end:
            raise SystemExit(
                f"Fehler: Start ({start.strftime('%Y-%m-%d %H:%M:%S')} UTC) "
                f"muss vor Ende ({end.strftime('%Y-%m-%d %H:%M:%S')} UTC) liegen."
            )
        label = (
            f"Zeitraum: {start.strftime('%Y-%m-%d %H:%M')} → "
            f"{end.strftime('%Y-%m-%d %H:%M')} UTC (CLI)"
        )
        return start, end, label

    days = int(lookback_days) if lookback_days is not None else int(default_lookback)
    if days <= 0:
        raise SystemExit("Fehler: --lookback muss > 0 sein.")
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    label = f"Lookback: {days} Tage"
    return start, end, label


def add_time_range_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --from/--start, --to/--end, --lookback to an ArgumentParser."""
    parser.add_argument(
        "--from",
        "--start",
        dest="from_date",
        default=None,
        metavar="DATE",
        help=(
            "Start des Event-Zeitraums (UTC). "
            "Formate: YYYY-MM-DD (=00:00:00 UTC) oder "
            "YYYY-MM-DDTHH:MM / YYYY-MM-DD HH:MM. "
            "Nicht zusammen mit --lookback."
        ),
    )
    parser.add_argument(
        "--to",
        "--end",
        dest="to_date",
        default=None,
        metavar="DATE",
        help=(
            "Ende des Event-Zeitraums (UTC). "
            "Date-only = inklusiv bis 23:59:59.999 UTC an diesem Tag. "
            "Default: jetzt (UTC), wenn --from gesetzt ist."
        ),
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        metavar="DAYS",
        help=(
            "Lookback in Tagen bis jetzt (UTC). "
            "Nicht zusammen mit --from/--to. "
            "Default: Config FROM_DATE/TO_DATE bzw. LOOKBACK_DAYS; CLI überschreibt Config FROM_DATE/TO_DATE bzw. LOOKBACK_DAYS."
        ),
    )


def sleep_polite(seconds: float = 0.3) -> None:
    time.sleep(seconds)


def entry_ms_on_flow_day(flow_date: pd.Timestamp, entry_hour_utc: int = 20) -> int:
    """UTC-Zeitstempel ENTRY_HOUR_UTC am Flow-/Event-Kalendertag."""
    d = pd.Timestamp(flow_date).tz_localize(None)
    dt = datetime(
        int(d.year), int(d.month), int(d.day),
        int(entry_hour_utc), 0, 0, tzinfo=timezone.utc,
    )
    return utc_ms(dt)


# ---------------------------------------------------------------------------
# HTTP / Binance Spot
# ---------------------------------------------------------------------------


def _binance_get(
    path: str,
    params: Optional[dict] = None,
    timeout: int = 30,
    bases: Optional[Sequence[str]] = None,
):
    """GET gegen BINANCE_BASES mit Failover bei 418/429/451."""
    global BINANCE
    last_exc: Optional[Exception] = None
    pool = list(bases) if bases else list(BINANCE_BASES)
    ordered = [BINANCE] + [b for b in pool if b != BINANCE] if BINANCE in pool else pool
    for base in ordered:
        try:
            r = SESSION.get(f"{base}{path}", params=params, timeout=timeout)
            if r.status_code in (418, 429, 451):
                last_exc = requests.HTTPError(f"{r.status_code} {base}", response=r)
                sleep_polite(1.0)
                continue
            r.raise_for_status()
            BINANCE = base
            return r
        except requests.RequestException as exc:
            last_exc = exc
            sleep_polite(0.5)
    if last_exc:
        raise last_exc
    raise RuntimeError("Binance GET fehlgeschlagen")


def fetch_top_coins_no_stables(n: int = TOP_N) -> List[dict]:
    out: List[dict] = []
    page = 1
    while len(out) < n:
        for attempt in range(8):
            r = SESSION.get(
                f"{COINGECKO}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 100,
                    "page": page,
                    "sparkline": "false",
                },
                timeout=30,
            )
            if r.status_code == 429:
                wait = 10 + attempt * 5
                print(f"   CoinGecko 429 — warte {wait}s …", flush=True)
                sleep_polite(wait)
                continue
            r.raise_for_status()
            break
        else:
            r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for c in batch:
            sym = (c.get("symbol") or "").upper()
            if sym in STABLE_SYMBOLS or sym in EXCLUDE_SYMBOLS or sym == "BTC":
                continue
            out.append({"symbol": sym})
            if len(out) >= n:
                break
        page += 1
        sleep_polite(1.5)
    return out[:n]


def binance_usdt_symbols() -> set:
    r = _binance_get("/api/v3/exchangeInfo")
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


def resolve_universe(
    coin_pairs: Optional[List[str]] = None,
    top_n: int = TOP_N,
) -> List[str]:
    """Top-N Non-Stable Non-BTC → Binance Spot USDT, oder feste COIN_PAIRS."""
    usdt_set = binance_usdt_symbols()
    if coin_pairs:
        pairs = []
        for p in coin_pairs:
            p = p.upper()
            if not p.endswith("USDT"):
                p = f"{p}USDT"
            if p in usdt_set:
                pairs.append(p)
            else:
                print(f"   überspringe {p} (nicht auf Binance Spot USDT)", flush=True)
        return pairs

    coins = fetch_top_coins_no_stables(top_n)
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


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    rows: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        try:
            r = _binance_get(
                "/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
        except requests.HTTPError as exc:
            resp = getattr(exc, "response", None)
            if resp is not None and resp.status_code == 429:
                sleep_polite(5)
                continue
            raise
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
        return pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close", "volume"]
        )

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
    return (
        df.drop_duplicates("open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Kosten / ROI / Simulation
# ---------------------------------------------------------------------------


def apply_costs(
    entry: float,
    exit_: float,
    fee_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
) -> Tuple[float, float]:
    fee = FEE_BPS if fee_bps is None else fee_bps
    slip = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    cost = (fee + slip) / 10_000.0
    buy = entry * (1.0 + cost)
    sell = exit_ * (1.0 - cost)
    return buy, sell


def buy_and_hold_roi(
    df: pd.DataFrame,
    period_start_ms: int,
    period_end_ms: int,
    fee_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
) -> Optional[dict]:
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
    buy, sell = apply_costs(entry_raw, exit_raw, fee_bps, slippage_bps)
    roi_pct = (sell / buy - 1.0) * 100.0
    return {
        "bh_buy_time": ms_to_utc_str(int(sub.iloc[0]["open_time"])),
        "bh_sell_time": ms_to_utc_str(int(sub.iloc[-1]["open_time"])),
        "bh_buy_price": buy,
        "bh_sell_price": sell,
        "buy_hold_roi_pct": roi_pct,
    }


def index_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("open_time", drop=False)


def loc_at(df_idx: pd.DataFrame, t_ms: int) -> Optional[int]:
    """Erste Bar mit open_time >= t_ms (oder exakt)."""
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


def roi_for_hold(
    idx: pd.DataFrame,
    entry_pos: int,
    hold_bars: int,
    fee_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
) -> Optional[float]:
    exit_pos = entry_pos + hold_bars
    if exit_pos >= len(idx):
        return None
    entry_raw = float(idx.iloc[entry_pos]["close"])
    exit_raw = float(idx.iloc[exit_pos]["close"])
    if entry_raw <= 0:
        return None
    buy, sell = apply_costs(entry_raw, exit_raw, fee_bps, slippage_bps)
    return (sell / buy - 1.0) * 100.0


def simulate_from_events(
    coin: str,
    df: pd.DataFrame,
    events: List[Event],
    strategy_label: str,
    hold_bars: Optional[Dict[str, int]] = None,
    fee_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
    value_col: str = "event_value",
) -> List[dict]:
    """
    Pro Event: Kauf am Close der ersten Coin-Bar at/after event.end_ms;
    ROI für jeden Hold-Horizont separat.
    """
    hb = hold_bars or hold_bars_map()
    fee = FEE_BPS if fee_bps is None else fee_bps
    slip = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
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
            "strategy": strategy_label,
            "direction": e.direction,
            "event_start": ms_to_utc_str(e.start_ms),
            "event_end": ms_to_utc_str(e.end_ms),
            value_col: e.value,
            "buy_time": ms_to_utc_str(buy_time_ms),
            "buy_time_ms": buy_time_ms,
            "buy_price": entry_raw * (1.0 + (fee + slip) / 10_000.0),
        }
        for k, v in (e.meta or {}).items():
            if k not in row:
                row[k] = v
        any_ok = False
        for name, bars in hb.items():
            r = roi_for_hold(idx, entry_pos, bars, fee, slip)
            row[f"roi_{name}"] = r
            if r is not None:
                any_ok = True
        if any_ok:
            trades.append(row)
    return trades


def simulate_random(
    coin: str,
    df: pd.DataFrame,
    n_trades: int,
    period_start_ms: int,
    period_end_ms: int,
    rng: np.random.Generator,
    hold_bars: Optional[Dict[str, int]] = None,
    fee_bps: Optional[float] = None,
    slippage_bps: Optional[float] = None,
    value_col: str = "event_value",
) -> List[dict]:
    hb = hold_bars or hold_bars_map()
    fee = FEE_BPS if fee_bps is None else fee_bps
    slip = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    if n_trades <= 0 or df.empty:
        return []
    max_bars = max(hb.values())
    candidates: List[int] = []
    for i in range(len(df)):
        t = int(df.at[i, "open_time"])
        if t < period_start_ms or t > period_end_ms:
            continue
        if i + max_bars >= len(df):
            continue
        if float(df.at[i, "close"]) <= 0:
            continue
        candidates.append(i)
    if not candidates:
        return []

    n = min(n_trades, len(candidates))
    chosen = rng.choice(candidates, size=n, replace=False)
    chosen = sorted(int(x) for x in chosen)

    idx = index_by_time(df)
    trades: List[dict] = []
    for entry_pos in chosen:
        t_ms = int(df.at[entry_pos, "open_time"])
        pos = loc_at(idx, t_ms)
        if pos is None:
            continue
        entry_raw = float(idx.iloc[pos]["close"])
        if entry_raw <= 0:
            continue
        row: dict = {
            "coin": coin,
            "strategy": "RANDOM",
            "direction": "RANDOM",
            "event_start": "",
            "event_end": "",
            value_col: None,
            "buy_time": ms_to_utc_str(t_ms),
            "buy_time_ms": t_ms,
            "buy_price": entry_raw * (1.0 + (fee + slip) / 10_000.0),
        }
        any_ok = False
        for name, bars in hb.items():
            r = roi_for_hold(idx, pos, bars, fee, slip)
            row[f"roi_{name}"] = r
            if r is not None:
                any_ok = True
        if any_ok:
            trades.append(row)
    return trades


# ---------------------------------------------------------------------------
# Stats / Non-Overlap / Summary
# ---------------------------------------------------------------------------


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
    additive = float(np.sum(vals)) if vals else None
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
        "additive_roi_pct": additive,
        "compound_roi_pct": compound,
    }


def filter_non_overlapping(
    trades: List[dict],
    horizon: str,
    hold_horizons: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """Cooldown = Hold-Länge; buy muss STRICTLY after last_kept + H."""
    horizons = hold_horizons or HOLD_HORIZONS
    hold_ms = horizons[horizon] * 60_000
    col = f"roi_{horizon}"
    ordered = sorted(trades, key=lambda t: int(t["buy_time_ms"]))
    kept: List[dict] = []
    last_kept_buy_ms: Optional[int] = None
    for t in ordered:
        v = t.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        buy_ms = int(t["buy_time_ms"])
        if last_kept_buy_ms is not None and buy_ms <= last_kept_buy_ms + hold_ms:
            continue
        kept.append(t)
        last_kept_buy_ms = buy_ms
    return kept


def summarize_by_horizon(
    trades: List[dict],
    horizon_names: Optional[Sequence[str]] = None,
    hold_horizons: Optional[Dict[str, int]] = None,
) -> Dict[str, dict]:
    names = list(horizon_names) if horizon_names else list(HOLD_HORIZONS.keys())
    out: Dict[str, dict] = {}
    for h in names:
        col = f"roi_{h}"
        n_raw = 0
        for t in trades:
            v = t.get(col)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                n_raw += 1
        filtered = filter_non_overlapping(trades, h, hold_horizons)
        st = column_stats(filtered, col)
        st["n_raw"] = n_raw
        st["n_used"] = st["n_trades"]
        out[h] = st
    return out


def fmt_pos(stats: dict) -> str:
    p = stats.get("pct_positive")
    if p is None:
        return "   n/a"
    return f"{p:5.0f}%"


def print_coin_result(
    coin: str,
    strategy: str,
    trades: List[dict],
    by_h: Dict[str, dict],
    bh: Optional[dict],
    trigger: str,
    horizon_names: Optional[Sequence[str]] = None,
    show_trades: bool = False,
    value_col: str = "event_value",
    value_header: str = "Val",
) -> None:
    names = list(horizon_names) if horizon_names else list(HOLD_HORIZONS.keys())
    print()
    print("=" * 120)
    print(
        f"COIN {coin} | {strategy} | {trigger} | "
        f"Hold {', '.join(names)} | {len(trades)} Trades"
    )
    if bh is not None:
        print(
            f"Buy&Hold über Zeitraum: {bh['bh_buy_time']} → {bh['bh_sell_time']} | "
            f"ROI {fmt_pct(bh['buy_hold_roi_pct'])}"
        )
    else:
        print("Buy&Hold über Zeitraum: n/a")
    print("=" * 120)

    if show_trades and trades:
        header = (
            f"{'Buy UTC':<22} {'Dir':<12} {value_header:>10}"
            + "".join(f"{('ROI ' + h):>9}" for h in names)
        )
        print(header)
        print("-" * len(header))
        for t in trades:
            line = (
                f"{t['buy_time']:<22} {str(t['direction']):<12} "
                f"{fmt_pct(t.get(value_col)) if isinstance(t.get(value_col), float) else str(t.get(value_col) or ''):>10}"
            )
            for h in names:
                line += f"{fmt_pct(t.get(f'roi_{h}')):>9}"
            print(line)
        print("-" * len(header))
    else:
        for h in names:
            st = by_h[h]
            n_used = st.get("n_used", st["n_trades"])
            n_raw = st.get("n_raw", st["n_trades"])
            print(
                f"  {h:>4}: n={n_used}/{n_raw} (used/raw)  "
                f"%pos={fmt_pos(st)}  Ø={fmt_pct(st['avg_roi_pct'])}  "
                f"Compound={fmt_pct(st['compound_roi_pct'])}"
            )

    if bh is not None:
        print(f"Vergleich Buy&Hold (einfach halten): {fmt_pct(bh['buy_hold_roi_pct'])}")
    print(flush=True)
    sys.stdout.flush()


def append_csv(path: Optional[str], rows: List[dict], write_header: bool) -> None:
    """No-op if path is None/leer (WRITE_CSV=False in den Sims)."""
    if not path or not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=write_header, index=False)


def make_summary_row(
    pair: str,
    strategy: str,
    trades: List[dict],
    by_h: Dict[str, dict],
    bh: Optional[dict],
    coin_seed: int,
    rand_by_h: Dict[str, dict],
    extra: Optional[dict] = None,
) -> dict:
    row: dict = {
        "coin": pair,
        "strategy": strategy,
        "n_events": len(trades),
        "fee_bps": FEE_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "random_seed": coin_seed if strategy == "RANDOM" else None,
        "buy_hold_roi_pct": None if bh is None else bh["buy_hold_roi_pct"],
        "bh_buy_time": None if bh is None else bh["bh_buy_time"],
        "bh_sell_time": None if bh is None else bh["bh_sell_time"],
    }
    if extra:
        row.update(extra)
    for h, st in by_h.items():
        row[f"n_{h}"] = st["n_trades"]
        row[f"n_used_{h}"] = st.get("n_used", st["n_trades"])
        row[f"n_raw_{h}"] = st.get("n_raw", st["n_trades"])
        row[f"pct_pos_{h}"] = st["pct_positive"]
        row[f"avg_roi_{h}"] = st["avg_roi_pct"]
        row[f"compound_roi_{h}"] = st["compound_roi_pct"]
        row[f"additive_roi_{h}"] = st["additive_roi_pct"]
        if bh is not None and st["compound_roi_pct"] is not None:
            row[f"vs_bh_{h}"] = st["compound_roi_pct"] - bh["buy_hold_roi_pct"]
        else:
            row[f"vs_bh_{h}"] = None
        r_cmp = rand_by_h[h]["compound_roi_pct"]
        if (
            strategy != "RANDOM"
            and st["compound_roi_pct"] is not None
            and r_cmp is not None
        ):
            row[f"vs_rand_{h}"] = st["compound_roi_pct"] - r_cmp
        else:
            row[f"vs_rand_{h}"] = None
    return row


def print_rankings(
    summary_rows: List[dict],
    strategies: Sequence[str],
    horizon_names: Optional[Sequence[str]] = None,
) -> None:
    names = list(horizon_names) if horizon_names else list(HOLD_HORIZONS.keys())
    for strategy in strategies:
        subset = [s for s in summary_rows if s["strategy"] == strategy]
        if not subset:
            continue
        for h in names:
            print("\n" + "=" * 100)
            print(
                f"ÜBERSICHT {strategy} | Hold={h} "
                f"(sortiert nach Compound-ROI)"
            )
            print("=" * 100)
            ranked = sorted(
                subset,
                key=lambda s: (
                    s[f"compound_roi_{h}"]
                    if s[f"compound_roi_{h}"] is not None
                    else float("-inf")
                ),
                reverse=True,
            )
            hdr = (
                f"{'Coin':<14} {'Strat':<10} {'n_u':>4} {'n_r':>4} {'%pos':>6} "
                f"{'Ø ROI':>9} {'Compound':>10} {'Buy&Hold':>10} "
                f"{'vs BH':>10} {'vs Rand':>10}"
            )
            print(hdr)
            print("-" * len(hdr))
            for s in ranked:
                pos = (
                    f"{s[f'pct_pos_{h}']:.0f}%"
                    if s[f"pct_pos_{h}"] is not None
                    else "n/a"
                )
                n_u = s.get(f"n_used_{h}", s[f"n_{h}"])
                n_r = s.get(f"n_raw_{h}", s[f"n_{h}"])
                print(
                    f"{s['coin']:<14} {s['strategy']:<10} {n_u:>4} {n_r:>4} "
                    f"{pos:>6} "
                    f"{fmt_pct(s[f'avg_roi_{h}']):>9} "
                    f"{fmt_pct(s[f'compound_roi_{h}']):>10} "
                    f"{fmt_pct(s.get('buy_hold_roi_pct')):>10} "
                    f"{fmt_pct(s.get(f'vs_bh_{h}')):>10} "
                    f"{fmt_pct(s.get(f'vs_rand_{h}')):>10}"
                )


def coin_seed_for(pair: str, base_seed: int = RANDOM_SEED) -> int:
    return base_seed + (sum(ord(c) for c in pair) % 10_000)


def reset_csv_files(*paths) -> None:
    for p in paths:
        if not p:
            continue
        path = Path(p)
        if path.exists():
            path.unlink()


def threshold_crossing_mask(
    series: pd.Series,
    thr: float,
    side: str = "above",
) -> pd.Series:
    """
    True nur beim Übergang über/unter die Schwelle (nicht jede Bar solange true).
    side: 'above' → series >= thr; 'below' → series <= thr (thr negativ oder positiv).
    """
    if side == "above":
        cond = series >= thr
    elif side == "below":
        cond = series <= thr
    else:
        raise ValueError("side must be 'above' or 'below'")
    prev = cond.shift(1, fill_value=False)
    return cond & ~prev


def cost_note(fee_bps: Optional[float] = None, slippage_bps: Optional[float] = None) -> str:
    fee = FEE_BPS if fee_bps is None else fee_bps
    slip = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    fee_pct = fee / 100.0
    if fee or slip:
        note = f"Fee {fee} bps ({fee_pct:g}%) je Seite Kauf+Verkauf"
        if slip:
            note += f" + Slippage {slip} bps"
        return note
    return "ohne Fees/Slippage"
