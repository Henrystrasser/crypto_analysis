#!/usr/bin/env python3
"""
Simulation: BTC steigt ODER fällt → Altcoin kaufen → multi-Hold → ROI

Eine Strategie (BOTH):
  Kauf, wenn BTC in window_min Minuten um ≥ +threshold steigt (UP)
  ODER um ≤ −threshold fällt (DOWN).
  Einstieg: Close des Coins am Fensterende.
  Ausstieg: Close nach dem jeweiligen Hold in HOLD_HORIZONS.

Wichtig — Hold-Horizonte sind getrennte Bot-Strategien (Absicht):
  Jeder Horizont wird separat ausgewertet (Korrelationen nach BTC-Moves).
  Kein globales Ein-Slot-Portfolio über alle Horizonte.

Zusätzlich:
  RANDOM — pro Hold-Fenster n_used = Rise/BOTH n_used (non-overlapping)
  Buy&Hold — einmal kaufen, bis Periodenende halten

Fees: Binance Spot 0,075% je Seite (FEE_BPS=7.5).

Non-Overlap (1 Slot, Cooldown = Hold-Länge), separat je Horizont:
  buy_time_i > last_kept_buy + Dauer(H). Angezeigtes n = n_used;
  n_raw höchstens kurz in Klammern.

Ausgabe: nur Konsole (keine CSV).
Universum: Top-N ohne Stables/BTC, ohne EXCLUDE_SYMBOLS, Binance Spot USDT.
Zeitraum: primär Config FROM_DATE/TO_DATE oder LOOKBACK_DAYS (in VS Code ändern); CLI --from/--to/--lookback als optionaler Override (deutsche Zeit, Europe/Berlin).

vs_bh = Compound − Buy&Hold (Experiment-Metrik neben vs RANDOM).

RUN_RANDOM steuert die optionale RANDOM-Baseline (Config; CLI --no-random).

Alle Uhrzeiten/Datumsgrenzen: deutsche Zeit (Europe/Berlin, CET/CEST).

Keine Handelsempfehlung — Backtest-Skript.
Abhängigkeiten: pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import sys

from laggard_common import (
    add_time_range_arguments,
    fetch_top_coins_robust,
    ms_to_berlin_str,
    resolve_event_window,
)
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# --- Strategie / Event ---
# Referenz-Coin für UP/DOWN-Events — hier in VS Code ändern; CLI --ref ist ein optionaler Override.
REF_SYMBOL = "BCH"
BTC_MOVE_WINDOW_MIN = 60
BTC_MOVE_THRESHOLD_PCT = 1.0
USE_THRESHOLD_CROSSING = True
RANDOM_SEED = 42
# Pro Horizont: RANDOM-n_used = Rise-n_used (BOTH). Override: feste Anzahl je Horizont
N_RANDOM_TRADES: Optional[int] = None
# RANDOM-Baseline: True = mitlaufen (vs Rand); False = überspringen (schneller)
RUN_RANDOM = True

# Optional: UP und DOWN zusätzlich getrennt ausgeben (neben BOTH)
SHOW_SEPARATE_DIRECTIONS = False

# Hold-Horizonte ab Kauf (Fensterende)
HOLD_HORIZONS: Dict[str, int] = {
    "24h": 1440,
    "48h": 2880,    # 2 Tage
    "96h": 5760,    # 4 Tage
    "192h": 11520,  # 8 Tage
}

# --- Zeitraum (deutsche Zeit, Europe/Berlin inkl. Sommer-/Winterzeit) — hier in VS Code ändern ---
# Wenn FROM_DATE gesetzt: fester Zeitraum nutzen.
#   TO_DATE=None → Ende = jetzt.
#   Date-only: FROM = 00:00:00 Berlin, TO = inklusiv bis 23:59:59.999 Berlin.
# Wenn FROM_DATE=None: LOOKBACK_DAYS rückwärts ab jetzt.
# FROM_DATE und LOOKBACK_DAYS nicht gleichzeitig "aktiv" (FROM hat Vorrang).
FROM_DATE: Optional[str] = None  # z.B. "2026-09-01" oder "2026-09-01 12:00"
TO_DATE: Optional[str] = None    # z.B. "2026-09-02"
LOOKBACK_DAYS = 290
KLINE_INTERVAL = "1h"
TOP_N = 100
COIN_PAIRS: List[str] = []

FEE_BPS = 7.5  # 0,075% Binance Spot je Seite (Kauf+Verkauf)
SLIPPAGE_BPS = 0.0

SHOW_TRADES = False  # nur Endergebnisse, keine Event-/Trade-Zeilen

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
# Primär Vision-Data-API; Fallback wenn Rate-Limit/Ban (418/-1003)
BINANCE_BASES = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
]
BINANCE = BINANCE_BASES[0]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-rise-hold-sim/0.7"})


def _binance_get(path: str, params: Optional[dict] = None, timeout: int = 30):
    """GET gegen BINANCE_BASES mit einfachem Failover bei 418/429/451."""
    global BINANCE
    last_exc: Optional[Exception] = None
    bases = [BINANCE] + [b for b in BINANCE_BASES if b != BINANCE]
    for base in bases:
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
if BTC_MOVE_WINDOW_MIN % BAR_MIN != 0:
    raise SystemExit(
        f"BTC_MOVE_WINDOW_MIN ({BTC_MOVE_WINDOW_MIN}) muss durch "
        f"KLINE_INTERVAL ({KLINE_INTERVAL} = {BAR_MIN} Min) teilbar sein."
    )
WINDOW_BARS = BTC_MOVE_WINDOW_MIN // BAR_MIN

HOLD_BARS: Dict[str, int] = {}
for _name, _mins in HOLD_HORIZONS.items():
    if _mins % BAR_MIN != 0:
        raise SystemExit(
            f"Hold {_name} ({_mins} Min) muss durch KLINE_INTERVAL "
            f"({BAR_MIN} Min) teilbar sein."
        )
    HOLD_BARS[_name] = _mins // BAR_MIN

MAX_HOLD_MIN = max(HOLD_HORIZONS.values())
HORIZON_NAMES = list(HOLD_HORIZONS.keys())


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# Anzeige aller Zeiten in deutscher Zeit (Europe/Berlin, CET/CEST) über
# laggard_common.ms_to_berlin_str. Die Event-Erkennung (±X% im rollierenden
# Fenster) arbeitet auf absoluten Zeitstempeln und ist zeitzonenunabhängig.


def sleep_polite(seconds: float = 0.3) -> None:
    time.sleep(seconds)


def normalize_ref_symbol(raw: str) -> tuple[str, str]:
    """Base + USDT-Pair aus 'BTC', 'bnb' oder 'BNBUSDT'."""
    s = (raw or "BTC").strip().upper()
    if s.endswith("USDT") and len(s) > 4:
        base, pair = s[:-4], s
    else:
        base, pair = s, f"{s}USDT"
    if not base:
        raise SystemExit(f"Ungültiger Referenz-Coin: {raw!r}")
    return base, pair


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BOTH Hold-Sim: Ref UP/DOWN → Altcoin-Holds + RANDOM"
    )
    p.add_argument(
        "--ref",
        "--ref-symbol",
        dest="ref",
        default=None,
        metavar="COIN",
        help=f"Referenz-Coin (Default: Config REF_SYMBOL={REF_SYMBOL!r})",
    )
    p.add_argument(
        "--no-random",
        dest="run_random",
        action="store_false",
        default=None,
        help="RANDOM-Baseline für diesen Lauf überspringen",
    )
    add_time_range_arguments(p)
    return p.parse_args(argv)

def fetch_top_coins_no_stables(n: int = TOP_N, exclude_base: str = "BTC") -> List[dict]:
    """CoinGecko → Cache → Binance-Volumen (Fallback); siehe laggard_common.fetch_top_coins_robust."""
    return fetch_top_coins_robust(
        n,
        exclude_base=exclude_base,
        stable_symbols=STABLE_SYMBOLS,
        exclude_symbols=EXCLUDE_SYMBOLS,
        session=SESSION,
        coingecko=COINGECKO,
    )


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


@dataclass
class BtcMoveEvent:
    start_ms: int
    end_ms: int
    btc_move_pct: float
    direction: str  # "UP" | "DOWN"


def detect_btc_move_events(
    btc: pd.DataFrame,
) -> Tuple[List[BtcMoveEvent], List[BtcMoveEvent]]:
    df = btc.copy()
    df["move_pct"] = df["close"].pct_change(WINDOW_BARS) * 100.0
    ups: List[BtcMoveEvent] = []
    downs: List[BtcMoveEvent] = []

    for i in range(len(df)):
        ret = df.at[i, "move_pct"]
        if pd.isna(ret):
            continue
        prev = df.at[i - 1, "move_pct"] if i > 0 else np.nan
        start_i = i - WINDOW_BARS
        if start_i < 0:
            continue

        if ret >= BTC_MOVE_THRESHOLD_PCT:
            if (
                USE_THRESHOLD_CROSSING
                and pd.notna(prev)
                and prev >= BTC_MOVE_THRESHOLD_PCT
            ):
                pass
            else:
                ups.append(
                    BtcMoveEvent(
                        start_ms=int(df.at[start_i, "open_time"]),
                        end_ms=int(df.at[i, "open_time"]),
                        btc_move_pct=float(ret),
                        direction="UP",
                    )
                )

        if ret <= -BTC_MOVE_THRESHOLD_PCT:
            if (
                USE_THRESHOLD_CROSSING
                and pd.notna(prev)
                and prev <= -BTC_MOVE_THRESHOLD_PCT
            ):
                pass
            else:
                downs.append(
                    BtcMoveEvent(
                        start_ms=int(df.at[start_i, "open_time"]),
                        end_ms=int(df.at[i, "open_time"]),
                        btc_move_pct=float(ret),
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
    buy = entry * (1.0 + cost)
    sell = exit_ * (1.0 - cost)
    return buy, sell


def buy_and_hold_roi(
    df: pd.DataFrame, period_start_ms: int, period_end_ms: int
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
    buy, sell = apply_costs(entry_raw, exit_raw)
    roi_pct = (sell / buy - 1.0) * 100.0
    return {
        "bh_buy_time": ms_to_berlin_str(int(sub.iloc[0]["open_time"])),
        "bh_sell_time": ms_to_berlin_str(int(sub.iloc[-1]["open_time"])),
        "bh_buy_price": buy,
        "bh_sell_price": sell,
        "buy_hold_roi_pct": roi_pct,
    }


def roi_for_hold(
    idx: pd.DataFrame, entry_pos: int, hold_bars: int
) -> Optional[float]:
    exit_pos = entry_pos + hold_bars
    if exit_pos >= len(idx):
        return None
    entry_raw = float(idx.iloc[entry_pos]["close"])
    exit_raw = float(idx.iloc[exit_pos]["close"])
    if entry_raw <= 0:
        return None
    buy, sell = apply_costs(entry_raw, exit_raw)
    return (sell / buy - 1.0) * 100.0


def simulate_coin(
    coin: str,
    df: pd.DataFrame,
    events: List[BtcMoveEvent],
    strategy_label: str,
) -> List[dict]:
    """
    Pro Event: Kauf am Fensterende, ROI für jeden Hold-Horizont (Verkauf danach).
    strategy_label: BOTH | UP | DOWN — Tag der Strategie-Spalte.
    direction im Trade bleibt UP/DOWN vom Event (bei BOTH gemischt).
    """
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
            "btc_move_start": ms_to_berlin_str(e.start_ms),
            "btc_move_end": ms_to_berlin_str(e.end_ms),
            "btc_move_pct": e.btc_move_pct,
            "buy_time": ms_to_berlin_str(buy_time_ms),
            "buy_time_ms": buy_time_ms,
            "buy_price": entry_raw * (1.0 + (FEE_BPS + SLIPPAGE_BPS) / 10_000.0),
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
    df: pd.DataFrame,
    period_start_ms: int,
    period_end_ms: int,
    need_bars: int,
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
    """Zufällige Entry-Positionen mit Abstand > hold_ms (wie Non-Overlap-Filter)."""
    if n_target <= 0 or not candidates:
        return []

    def ok(positions: List[int]) -> bool:
        ordered = sorted(positions, key=lambda p: int(df.at[p, "open_time"]))
        last: Optional[int] = None
        for p in ordered:
            t = int(df.at[p, "open_time"])
            if last is not None and t <= last + hold_ms:
                return False
            last = t
        return True

    cands = list(candidates)
    accepted: List[int] = []
    order = list(cands)
    rng.shuffle(order)
    for pos in order:
        trial = accepted + [pos]
        if ok(trial):
            accepted.append(pos)
            if len(accepted) >= n_target:
                break

    if len(accepted) >= n_target:
        return sorted(accepted[:n_target], key=lambda p: int(df.at[p, "open_time"]))

    # Fallback: chronologische Maximalpackung, dann Zufalls-Teilmenge
    ordered_all = sorted(cands, key=lambda p: int(df.at[p, "open_time"]))
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
    """
    Pro Hold-Horizont: genau so viele non-overlapping Zufalls-Käufe
    wie target_n_by_h[h] (typisch = n_used der Rise-Strategie für dieses Fenster).
    """
    idx = index_by_time(df)
    all_trades: List[dict] = []
    by_h: Dict[str, dict] = {}

    for name in HORIZON_NAMES:
        n_target = int(target_n_by_h.get(name, 0) or 0)
        hold_ms = HOLD_HORIZONS[name] * 60_000
        need_bars = HOLD_BARS[name]
        cands = _random_candidate_indices(
            df, period_start_ms, period_end_ms, need_bars
        )
        positions = _sample_nonoverlapping_positions(
            df, cands, n_target, hold_ms, rng
        )
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
                "strategy": "RANDOM",
                "direction": "RANDOM",
                "btc_move_start": "",
                "btc_move_end": "",
                "btc_move_pct": None,
                "buy_time": ms_to_berlin_str(t_ms),
                "buy_price": entry_raw * (1.0 + (FEE_BPS + SLIPPAGE_BPS) / 10_000.0),
                "buy_time_ms": t_ms,
                "random_horizon": name,
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



def simulate_random_coin(
    coin: str,
    df: pd.DataFrame,
    n_trades: int,
    period_start_ms: int,
    period_end_ms: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Legacy: flache Zufalls-Käufe (ohne Match pro Horizont)."""
    if n_trades <= 0 or df.empty:
        return []
    max_bars = max(HOLD_BARS.values())
    candidates = _random_candidate_indices(
        df, period_start_ms, period_end_ms, max_bars
    )
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
            "btc_move_start": "",
            "btc_move_end": "",
            "btc_move_pct": None,
            "buy_time": ms_to_berlin_str(t_ms),
            "buy_time_ms": t_ms,
            "buy_price": entry_raw * (1.0 + (FEE_BPS + SLIPPAGE_BPS) / 10_000.0),
        }
        any_ok = False
        for name, bars in HOLD_BARS.items():
            r = roi_for_hold(idx, pos, bars)
            row[f"roi_{name}"] = r
            if r is not None:
                any_ok = True
        if any_ok:
            trades.append(row)
    return trades


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


def filter_non_overlapping(trades: List[dict], horizon: str) -> List[dict]:
    hold_ms = HOLD_HORIZONS[horizon] * 60_000
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


def summarize_by_horizon(trades: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for h in HORIZON_NAMES:
        col = f"roi_{h}"
        n_raw = 0
        for t in trades:
            v = t.get(col)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                n_raw += 1
        filtered = filter_non_overlapping(trades, h)
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


def print_coin_result(
    coin: str,
    strategy: str,
    trades: List[dict],
    by_h: Dict[str, dict],
    bh: Optional[dict],
    period: Optional[str] = None,
) -> None:
    if strategy == "BOTH":
        trigger = (
            f"Buy nach BTC |move| ≥ {BTC_MOVE_THRESHOLD_PCT}% / "
            f"{BTC_MOVE_WINDOW_MIN}m (UP und DOWN)"
        )
    elif strategy == "UP":
        trigger = (
            f"Buy nach BTC ≥ +{BTC_MOVE_THRESHOLD_PCT}% / {BTC_MOVE_WINDOW_MIN}m"
        )
    elif strategy == "DOWN":
        trigger = (
            f"Buy nach BTC ≤ −{BTC_MOVE_THRESHOLD_PCT}% / {BTC_MOVE_WINDOW_MIN}m"
        )
    else:
        trigger = "Zufalls-Käufe (unabhängig von BTC)"
    print()
    print("=" * 120)
    print(
        f"COIN {coin} | {strategy} | {trigger} | "
        f"Hold {', '.join(HORIZON_NAMES)} | {len(trades)} Trades"
        + (f" | {period}" if period else "")
    )
    if bh is not None:
        print(
            f"Buy&Hold über Zeitraum: {bh['bh_buy_time']} → {bh['bh_sell_time']} | "
            f"ROI {fmt_pct(bh['buy_hold_roi_pct'])}"
        )
    else:
        print("Buy&Hold über Zeitraum: n/a")
    print("=" * 120)

    if SHOW_TRADES and trades:
        header = (
            f"{'Buy (Berlin)':<24} {'Dir':<5} {'BTC%':>7}"
            + "".join(f"{('ROI ' + h):>9}" for h in HORIZON_NAMES)
        )
        print(header)
        print("-" * len(header))
        for t in trades:
            line = (
                f"{t['buy_time']:<24} {t['direction']:<5} "
                f"{fmt_pct(t['btc_move_pct']):>7}"
            )
            for h in HORIZON_NAMES:
                line += f"{fmt_pct(t.get(f'roi_{h}')):>9}"
            print(line)
        print("-" * len(header))

        foot_pos = f"{'% positiv':<24} {'':<5} {'':>7}"
        foot_avg = f"{'Ø ROI':<24} {'':<5} {'':>7}"
        foot_cmp = f"{'Compound':<24} {'':<5} {'':>7}"
        for h in HORIZON_NAMES:
            st = by_h[h]
            foot_pos += f"{fmt_pos(st):>9}"
            foot_avg += f"{fmt_pct(st['avg_roi_pct']):>9}"
            foot_cmp += f"{fmt_pct(st['compound_roi_pct']):>9}"
        print(foot_pos)
        print(foot_avg)
        print(foot_cmp)
    else:
        for h in HORIZON_NAMES:
            st = by_h[h]
            n_used = st.get("n_used", st["n_trades"])
            n_raw = st.get("n_raw", st["n_trades"])
            n_txt = f"n={n_used}"
            if n_raw is not None and n_raw != n_used:
                n_txt += f" (raw {n_raw})"
            print(
                f"  {h:>4}: {n_txt}  "
                f"%pos={fmt_pos(st)}  Ø={fmt_pct(st['avg_roi_pct'])}  "
                f"Compound={fmt_pct(st['compound_roi_pct'])}"
            )

    if bh is not None:
        print(f"Vergleich Buy&Hold (einfach halten): {fmt_pct(bh['buy_hold_roi_pct'])}")

    print(flush=True)
    sys.stdout.flush()


def resolve_universe(
    exclude_base: str = "BTC",
    exclude_pair: Optional[str] = None,
) -> List[str]:
    usdt_set = binance_usdt_symbols()
    skip_pair = exclude_pair or f"{exclude_base}USDT"
    if COIN_PAIRS:
        pairs = []
        for p in COIN_PAIRS:
            p = p.upper()
            if not p.endswith("USDT"):
                p = f"{p}USDT"
            if p == skip_pair:
                print(f"   überspringe {p} (Referenz-Coin)", flush=True)
                continue
            if p in usdt_set:
                pairs.append(p)
            else:
                print(f"   überspringe {p} (nicht auf Binance Spot USDT)", flush=True)
        return pairs

    coins = fetch_top_coins_no_stables(TOP_N, exclude_base=exclude_base)
    pairs = []
    skipped = []
    for c in coins:
        pair = map_to_binance_pair(c["symbol"], usdt_set)
        if pair and pair != skip_pair:
            pairs.append(pair)
        elif pair == skip_pair:
            skipped.append(c["symbol"] + " (ref)")
        else:
            skipped.append(c["symbol"])
    if skipped:
        print(f"   ohne Binance-Pair: {skipped}", flush=True)
    return pairs




def make_summary_row(
    pair: str,
    strategy: str,
    trades: List[dict],
    by_h: Dict[str, dict],
    bh: Optional[dict],
    coin_seed: int,
    rand_by_h: Optional[Dict[str, dict]],
) -> dict:
    row: dict = {
        "coin": pair,
        "strategy": strategy,
        "n_events": len(trades),
        "btc_move_window_min": BTC_MOVE_WINDOW_MIN,
        "btc_move_threshold_pct": BTC_MOVE_THRESHOLD_PCT,
        "fee_bps": FEE_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "random_seed": coin_seed if strategy == "RANDOM" else None,
        "buy_hold_roi_pct": None if bh is None else bh["buy_hold_roi_pct"],
        "bh_buy_time": None if bh is None else bh["bh_buy_time"],
        "bh_sell_time": None if bh is None else bh["bh_sell_time"],
    }
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
        if rand_by_h is not None:
            r_cmp = rand_by_h[h]["compound_roi_pct"]
            if (
                strategy != "RANDOM"
                and st["compound_roi_pct"] is not None
                and r_cmp is not None
            ):
                row[f"vs_rand_{h}"] = st["compound_roi_pct"] - r_cmp
            else:
                row[f"vs_rand_{h}"] = None
        else:
            row[f"vs_rand_{h}"] = None
    return row


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_cli_args(argv)
    run_random = RUN_RANDOM if args.run_random is None else args.run_random
    ref_base, ref_pair = normalize_ref_symbol(args.ref or REF_SYMBOL)
    start, end, window_label = resolve_event_window(
        from_s=args.from_date or FROM_DATE,
        to_s=args.to_date or TO_DATE,
        lookback_days=args.lookback,  # only CLI; None → use LOOKBACK when no FROM
        default_lookback=LOOKBACK_DAYS,
    )

    print(
        f"=== Simulation: {ref_base} UP oder DOWN → Coin kaufen → multi-Hold ROI ===\n",
        flush=True,
    )
    fetch_end = end + pd.Timedelta(minutes=MAX_HOLD_MIN)
    start_ms, end_ms = utc_ms(start), utc_ms(end)
    fetch_end_ms = utc_ms(fetch_end)

    fee_pct = FEE_BPS / 100.0
    if FEE_BPS or SLIPPAGE_BPS:
        cost_note = f"Fee {FEE_BPS} bps ({fee_pct:g}%) je Seite Kauf+Verkauf"
        if SLIPPAGE_BPS:
            cost_note += f" + Slippage {SLIPPAGE_BPS} bps"
    else:
        cost_note = "ohne Fees/Slippage"

    print(
        f"Zeitraum Events: {start.strftime('%Y-%m-%d %H:%M %Z')} → {end.strftime('%Y-%m-%d %H:%M %Z')} (Europe/Berlin)",
        flush=True,
    )
    print(
        f"Referenz: {ref_base} ({ref_pair})",
        flush=True,
    )
    print(
        f"BOTH: Kauf wenn {ref_base} ≥ +{BTC_MOVE_THRESHOLD_PCT}% ODER "
        f"≤ −{BTC_MOVE_THRESHOLD_PCT}% in {BTC_MOVE_WINDOW_MIN} Min "
        f"({WINDOW_BARS} × {KLINE_INTERVAL})",
        flush=True,
    )
    if run_random:
        print(
            f"RANDOM: Zufalls-Käufe, Seed={RANDOM_SEED} "
            f"(pro Horizont = n_used Rise/BOTH bzw. N_RANDOM_TRADES)",
            flush=True,
        )
    else:
        print("RANDOM: aus (RUN_RANDOM=False)", flush=True)
    print(
        f"{window_label} | Intervall: {KLINE_INTERVAL} | "
        f"Fenster: {BTC_MOVE_WINDOW_MIN} Min ({WINDOW_BARS} Bars)",
        flush=True,
    )
    print(
        f"Hold-Horizonte (jeweils eigene Bot-Strategie, kein globales Portfolio): "
        f"{HORIZON_NAMES}",
        flush=True,
    )
    print(
        f"Exclude: {sorted(EXCLUDE_SYMBOLS)} | Top-N: {TOP_N} "
        f"(ohne Stables, ohne Ref)",
        flush=True,
    )
    print(f"Kosten: {cost_note}", flush=True)
    print(
        "Non-overlap: 1 Slot je Horizont, Cooldown = Hold-Länge "
        "(n = n_used nach Filter)\n",
        flush=True,
    )

    print("1) Coin-Universum…", flush=True)
    universe = resolve_universe(exclude_base=ref_base, exclude_pair=ref_pair)
    print(f"   {len(universe)} Paare\n", flush=True)

    print(f"2) {ref_base}-Kerzen (Event-Erkennung, {ref_pair})…", flush=True)
    ref_df = fetch_klines(ref_pair, KLINE_INTERVAL, start_ms, end_ms)
    if ref_df.empty:
        raise SystemExit(f"Keine Kerzen für Referenz {ref_pair}.")
    ups, downs = detect_btc_move_events(ref_df)
    both = sorted(ups + downs, key=lambda e: e.end_ms)
    print(
        f"   {len(ref_df)} Kerzen | {len(ups)} UP | {len(downs)} DOWN | "
        f"{len(both)} BOTH",
        flush=True,
    )
    if not both:
        raise SystemExit("Keine Events — Schwelle/Fenster/LOOKBACK anpassen.")

    print(flush=True)

    summary_rows: List[dict] = []

    print(f"3) Simulation Coin für Coin ({len(universe)})\n", flush=True)
    for i, pair in enumerate(universe, 1):
        print(f"--- [{i}/{len(universe)}] lade {pair} …", flush=True)
        try:
            df = fetch_klines(pair, KLINE_INTERVAL, start_ms, fetch_end_ms)
        except Exception as exc:  # noqa: BLE001
            print(f"    Fehler: {exc}\n", flush=True)
            continue
        if df.empty:
            print("    keine Kerzen\n", flush=True)
            continue

        bh = buy_and_hold_roi(df, start_ms, end_ms)
        # Effektiver Zeitraum je Coin fuer die Block-Header (spaeteres Listing beachten).
        coin_period = _block_range(*_effective_range_ms(
            start_ms, end_ms, ref_df["open_time"].to_numpy(), df["open_time"].to_numpy()
        ))
        rand_period = _block_range(*_effective_range_ms(start_ms, end_ms, df["open_time"].to_numpy()))

        coin_seed = RANDOM_SEED + (sum(ord(c) for c in pair) % 10_000)

        batch: List[Tuple[str, List[dict], Dict[str, dict]]] = []

        # Hauptstrategie: UP und DOWN zusammen
        both_trades = simulate_coin(pair, df, both, "BOTH")
        both_by_h = summarize_by_horizon(both_trades)
        batch.append(("BOTH", both_trades, both_by_h))

        if SHOW_SEPARATE_DIRECTIONS:
            if ups:
                up_trades = simulate_coin(pair, df, ups, "UP")
                batch.append(("UP", up_trades, summarize_by_horizon(up_trades)))
            if downs:
                dn_trades = simulate_coin(pair, df, downs, "DOWN")
                batch.append(("DOWN", dn_trades, summarize_by_horizon(dn_trades)))

        rand_by_h: Optional[Dict[str, dict]] = None
        if run_random:
            rng = np.random.default_rng(coin_seed)
            if N_RANDOM_TRADES is not None:
                targets = {h: int(N_RANDOM_TRADES) for h in HORIZON_NAMES}
            else:
                targets = {h: int(both_by_h[h]["n_used"]) for h in HORIZON_NAMES}
            rand_trades, rand_by_h = simulate_random_matched_horizons(
                pair, df, targets, start_ms, end_ms, rng
            )
            batch.append(("RANDOM", rand_trades, rand_by_h))

        for strategy, trades, by_h in batch:
            print_coin_result(
                pair, strategy, trades, by_h, bh,
                period=rand_period if strategy == "RANDOM" else coin_period,
            )
            summary_rows.append(
                make_summary_row(
                    pair, strategy, trades, by_h, bh, coin_seed,
                    rand_by_h if run_random else None,
                )
            )

        del df

    if summary_rows:
        strategies = ["BOTH"]
        if SHOW_SEPARATE_DIRECTIONS:
            strategies.extend(["UP", "DOWN"])
        if run_random:
            strategies.append("RANDOM")

        for strategy in strategies:
            subset = [s for s in summary_rows if s["strategy"] == strategy]
            if not subset:
                continue
            for h in HORIZON_NAMES:
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
                    f"{'Coin':<14} {'Strat':<6} {'n':>10} {'%pos':>6} "
                    f"{'Ø ROI':>9} {'Compound':>10} {'Buy&Hold':>10} "
                    f"{'vs BH':>10}"
                )
                if run_random:
                    hdr += f" {'vs Rand':>10}"
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
                    if n_r is not None and n_r != n_u:
                        n_txt = f"{n_u}({n_r})"
                    else:
                        n_txt = str(n_u)
                    line = (
                        f"{s['coin']:<14} {s['strategy']:<6} {n_txt:>10} "
                        f"{pos:>6} "
                        f"{fmt_pct(s[f'avg_roi_{h}']):>9} "
                        f"{fmt_pct(s[f'compound_roi_{h}']):>10} "
                        f"{fmt_pct(s.get('buy_hold_roi_pct')):>10} "
                        f"{fmt_pct(s.get(f'vs_bh_{h}')):>10}"
                    )
                    if run_random:
                        line += f" {fmt_pct(s.get(f'vs_rand_{h}')):>10}"
                    print(line)

    footer = (
        "\nFertig. Jeder Hold-Horizont = eigene Bot-Strategie "
        f"({', '.join(HORIZON_NAMES)}) — kein globales Portfolio.\n"
        "BOTH: Kauf am Ende jedes Ref-UP- oder Ref-DOWN-Fensters, "
        "Verkauf nach dem jeweiligen Hold.\n"
        "n = n_used nach Non-Overlap; n_raw nur in Klammern wenn abweichend.\n"
        "Compound je Horizont = Produkt (1+roi) − 1 über die gefilterten Events.\n"
    )
    if run_random:
        footer += (
            "RANDOM = pro Horizont gleiche n_used wie Rise; "
            "Buy&Hold = einmal kaufen bis Periodenende.\n"
            "vs BH = Compound − Buy&Hold (Experiment-Metrik); "
            "vs Rand = Compound − RANDOM.\n"
        )
    else:
        footer += "Buy&Hold = einmal kaufen bis Periodenende.\n"
        footer += "vs BH = Compound − Buy&Hold (Experiment-Metrik).\n"
    footer += (
        "SHOW_SEPARATE_DIRECTIONS=True zeigt zusätzlich reine UP-/DOWN-Läufe.\n"
        "Nur Konsolen-Ausgabe — keine CSV.\n"
        "Keine Handelsempfehlung — reine Simulation.\n"
    )
    print(footer, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
