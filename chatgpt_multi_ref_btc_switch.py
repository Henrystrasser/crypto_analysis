#!/usr/bin/env python3

"""

Multi-Reference Event -> Target-Coin Buy Backtest



Idee:

 Mehrere Referenz-Coins werden gleichzeitig beobachtet.

 Wenn mindestens MIN_REFS der Referenzen innerhalb eines Fensters

 gemeinsam stark steigen ODER fallen, wird der TARGET_SYMBOL gekauft.



Beispiel:

 REF_SYMBOLS = ["BTC", "ETH", "SOL"]

 window = 3h

 threshold = 2%



 BTC +2.4%, ETH +2.1%, SOL +0.3%

 -> 2/3 UP

 -> Signal, wenn MIN_REFS=2



UP und DOWN sind beide Kaufsignale:

 UP   = mindestens N Referenzen steigen >= threshold

 DOWN = mindestens N Referenzen fallen <= -threshold



Das Script testet:

 - mehrere Zeitfenster

 - mehrere Schwellen

 - mehrere Mindestanzahlen bestätigender Referenzen

 - UP und DOWN

 - mehrere Hold-Horizonte

 - Non-overlap je Strategie/Hold

 - Gebühren + Slippage

 - Compound

 - % positive Trades

 - Buy & Hold

 - Signal Compound minus Buy & Hold

 - optional RANDOM-Baseline



Wichtig:

 Das Signal wird am Ende der Referenz-Kerze ausgelöst.

 Der Target-Coin wird zum Close der Target-Kerze an diesem Zeitpunkt gekauft.

 Fehlt die Target-Kerze exakt zu diesem Zeitpunkt (z.B. vor dem Listing),

 wird das Event verworfen (kein Ausweichen auf eine spätere Kerze).

 Re-Entry exakt am Hold-Ende des vorherigen Trades ist erlaubt.

 Es wird keine zukünftige Information verwendet.



CLI:

 python multi_ref_buy.py

 python multi_ref_buy.py --ref BTC ETH SOL --coin VIRTUAL

 python multi_ref_buy.py --ref BTC ETH SOL --coin VIRTUAL --no-random

 python multi_ref_buy.py --from 2025-01-01 --to 2026-09-30



Abhängigkeiten:

 pip install numpy pandas requests

 plus laggard_common.py aus deinem bestehenden Projekt.

"""



from __future__ import annotations



import argparse

import sys

import time

from dataclasses import dataclass

from datetime import datetime

from typing import Dict, List, Optional, Tuple



import numpy as np

import pandas as pd

import requests



from laggard_common import add_time_range_arguments, ms_to_berlin_str, resolve_event_window





# ============================================================

# CONFIG

# ============================================================



REF_SYMBOLS: List[str] = ["BTC", "SOL", "ETH","XRP","BNB"]

TARGET_SYMBOL = "PEPE"



# Mehrere Event-Fenster testen.

EVENT_WINDOWS_MIN = [60,120,180]



# Schwellen separat testen, damit Nachbarn gut sichtbar sind.

THRESHOLDS_PCT = [1.0, 1.5,2.0,3.0]



# Wie viele Referenzcoins müssen gleichzeitig in dieselbe Richtung laufen?

# Bei 3 Refs:

#   1 = ANY

#   2 = 2/3

#   3 = ALL

MIN_REFS = [1, 2, 3,4,5]



# Holds sind jeweils eigenständige Strategien.

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



FROM_DATE: Optional[str] = "2025-01-01"

TO_DATE: Optional[str] = "2026-12-28"

LOOKBACK_DAYS = 400



# 1h ist für den ersten Test bewusst gewählt:

# schnell, wenig API-Daten und passend zu den aktuellen Fenstern.

KLINE_INTERVAL = "1h"



FEE_BPS = 7.5

SLIPPAGE_BPS = 0.0



RUN_RANDOM = False

RANDOM_SEED = 46

N_RANDOM_TRADES: Optional[int] = None



# Keine künstliche Top-N-Begrenzung.

SHOW_TRADES = False



STABLE_SYMBOLS = {

   "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "USDD",

   "FRAX", "PYUSD", "EURC", "EURT", "GUSD", "LUSD", "SUSD", "USDP",

   "USD1", "USDY", "RLUSD", "CRVUSD", "GHO", "USDS",

}



EXCLUDE_SYMBOLS = {"HYPE", "ZEC"}



BINANCE = "https://data-api.binance.vision"



SESSION = requests.Session()

SESSION.headers.update({"User-Agent": "multi-ref-buy/1.0"})





# ============================================================

# HELPERS

# ============================================================



def normalize_symbol(raw: str) -> tuple[str, str]:

   s = (raw or "").strip().upper()

   if s.endswith("USDT") and len(s) > 4:

       base = s[:-4]

       return base, s

   if not s:

       raise ValueError("Leerer Coin.")

   return s, f"{s}USDT"





def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:

   p = argparse.ArgumentParser(

       description="Multi-Reference UP/DOWN -> Target-Coin Buy Backtest"

   )

   p.add_argument(

       "--ref",

       "--refs",

       nargs="+",

       default=None,

       metavar="COIN",

       help=f"Referenzcoins (Default: {' '.join(REF_SYMBOLS)})",

   )

   p.add_argument(

       "--coin",

       "--target",

       dest="target",

       default=None,

       metavar="COIN",

       help=f"Ziel-Coin (Default: {TARGET_SYMBOL})",

   )

   p.add_argument(

       "--no-random",

       dest="run_random",

       action="store_false",

       default=None,

       help="RANDOM-Baseline überspringen",

   )

   add_time_range_arguments(p)

   return p.parse_args(argv)





def min_ref_label(min_refs: int, total_refs: int) -> str:
    """Human-readable label for the quorum, e.g. 3/5."""
    return f"{min_refs}/{total_refs}"


def interval_to_minutes(interval: str) -> int:

   unit = interval[-1]

   n = int(interval[:-1])

   if unit == "m":

       return n

   if unit == "h":

       return n * 60

   if unit == "d":

       return n * 1440

   raise ValueError(f"Unsupported interval: {interval}")





BAR_MIN = interval_to_minutes(KLINE_INTERVAL)



if any(w % BAR_MIN != 0 for w in EVENT_WINDOWS_MIN):

   raise SystemExit("Alle EVENT_WINDOWS_MIN müssen durch KLINE_INTERVAL teilbar sein.")



if any(h % BAR_MIN != 0 for h in HOLD_HORIZONS.values()):

   raise SystemExit("Alle HOLD_HORIZONS müssen durch KLINE_INTERVAL teilbar sein.")



WINDOW_BARS = {w: w // BAR_MIN for w in EVENT_WINDOWS_MIN}

HOLD_BARS = {h: mins // BAR_MIN for h, mins in HOLD_HORIZONS.items()}

MAX_HOLD_MIN = max(HOLD_HORIZONS.values())

HORIZON_NAMES = list(HOLD_HORIZONS.keys())





def utc_ms(dt: datetime) -> int:

   return int(dt.timestamp() * 1000)





def sleep_polite(seconds: float = 0.2) -> None:

   time.sleep(seconds)





def binance_usdt_symbols() -> set[str]:

   r = SESSION.get(f"{BINANCE}/api/v3/exchangeInfo", timeout=30)

   r.raise_for_status()

   return {

       s["symbol"]

       for s in r.json()["symbols"]

       if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"

   }





def resolve_pair(raw: str, usdt_set: set[str]) -> tuple[str, str]:

   base, pair = normalize_symbol(raw)

   aliases = {

       "RNDR": "RENDERUSDT",

       "RENDER": "RENDERUSDT",

       "MATIC": "MATICUSDT",

       "POL": "POLUSDT",

   }

   if base in aliases and aliases[base] in usdt_set:

       return base, aliases[base]

   if pair not in usdt_set:

       raise SystemExit(f"Pair {pair} nicht auf Binance Spot USDT.")

   return base, pair





def fetch_klines(

   symbol: str,

   interval: str,

   start_ms: int,

   end_ms: int,

) -> pd.DataFrame:

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



   # Nur ABGESCHLOSSENE Kerzen (keine laufende Kerze).

   now_ms = int(time.time() * 1000)

   df = df[df["close_time"].astype(np.int64) < now_ms]



   df = df[

       ["open_time", "open", "high", "low", "close", "volume"]

   ].copy()



   for col in ("open", "high", "low", "close", "volume"):

       df[col] = df[col].astype(float)



   df["open_time"] = df["open_time"].astype(np.int64)



   return (

       df.drop_duplicates("open_time")

       .sort_values("open_time")

       .reset_index(drop=True)

   )





# ============================================================

# EVENT MODEL

# ============================================================



@dataclass

class MultiRefEvent:

   end_ms: int

   direction: str

   window_min: int

   threshold_pct: float

   min_refs: int

   ref_moves: Dict[str, float]

   matched_refs: List[str]



   @property

   def score(self) -> int:

       return len(self.matched_refs)





def build_reference_move_series(

   df: pd.DataFrame,

   windows: List[int],

) -> Dict[int, pd.Series]:

   result: Dict[int, pd.Series] = {}



   for window_min in windows:

       bars = WINDOW_BARS[window_min]

       result[window_min] = df["close"].pct_change(bars) * 100.0



   return result





def detect_multi_ref_events(

   ref_data: Dict[str, pd.DataFrame],

   ref_pairs: Dict[str, str],

   windows: List[int],

   thresholds: List[float],

   min_refs_list: List[int],

   start_ms: int,

   end_ms: int,

) -> Dict[Tuple[int, float, int, str], List[MultiRefEvent]]:

   """

   Erstellt Events getrennt nach:

     window / threshold / min_refs / direction.



   Ein Event entsteht am gemeinsamen Timestamp, an dem die Referenzkerzen

   verfügbar sind. Für jedes Ref wird die Veränderung über genau das

   konfigurierte Fenster berechnet.

   """



   # Alle Referenzen müssen gemeinsame Kerzenzeiten besitzen.

   common_times: Optional[set[int]] = None

   for df in ref_data.values():

       times = set(int(x) for x in df["open_time"].tolist())

       common_times = times if common_times is None else common_times & times



   if not common_times:

       return {}



   common_times = {

       t for t in common_times

       if start_ms <= t <= end_ms

   }



   move_by_ref: Dict[str, Dict[int, pd.Series]] = {}

   for ref, df in ref_data.items():

       move_by_ref[ref] = build_reference_move_series(df, windows)



   events: Dict[Tuple[int, float, int, str], List[MultiRefEvent]] = {}



   for window in windows:

       for threshold in thresholds:

           for min_refs in min_refs_list:

               for direction in ("UP", "DOWN"):

                   events[(window, threshold, min_refs, direction)] = []



               for t in sorted(common_times):

                   moves: Dict[str, float] = {}



                   for ref in ref_data:

                       df = ref_data[ref]

                       # Index lookup via search is avoided by creating maps.

                       # Data is 1h and common timestamps, so exact match is expected.

                       row_idx = df.index[df["open_time"] == t]

                       if len(row_idx) == 0:

                           continue



                       pos = int(row_idx[0])

                       value = move_by_ref[ref][window].iloc[pos]



                       if pd.notna(value):

                           moves[ref] = float(value)



                   if len(moves) != len(ref_data):

                       continue



                   up_refs = [

                       ref for ref, move in moves.items()

                       if move >= threshold

                   ]

                   down_refs = [

                       ref for ref, move in moves.items()

                       if move <= -threshold

                   ]



                   if len(up_refs) >= min_refs:

                       events[

                           (window, threshold, min_refs, "UP")

                       ].append(

                           MultiRefEvent(

                               end_ms=t,

                               direction="UP",

                               window_min=window,

                               threshold_pct=threshold,

                               min_refs=min_refs,

                               ref_moves=moves,

                               matched_refs=up_refs,

                           )

                       )



                   if len(down_refs) >= min_refs:

                       events[

                           (window, threshold, min_refs, "DOWN")

                       ].append(

                           MultiRefEvent(

                               end_ms=t,

                               direction="DOWN",

                               window_min=window,

                               threshold_pct=threshold,

                               min_refs=min_refs,

                               ref_moves=moves,

                               matched_refs=down_refs,

                           )

                       )



   return events





# ============================================================

# TARGET / TRADE CALCULATION

# ============================================================



def apply_costs(entry: float, exit_: float) -> Tuple[float, float]:

   cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0

   return entry * (1.0 + cost), exit_ * (1.0 - cost)





def exact_position(df: pd.DataFrame, t_ms: int) -> Optional[int]:

   """

   Position der Kerze mit open_time == t_ms, sonst None.



   KEIN Vorwärtssuchen: fehlt die Kerze (z.B. vor dem Listing oder

   Datenlücke), wird das Event verworfen statt auf eine spätere

   Kerze verschoben.

   """

   times = df["open_time"].to_numpy(dtype=np.int64)

   i = int(np.searchsorted(times, int(t_ms)))

   if i < len(times) and int(times[i]) == int(t_ms):

       return i

   return None





def roi_for_hold_stable(
    target_df: pd.DataFrame,
    entry_pos: int,
    hold_bars: int,
) -> Optional[float]:
    """Stablecoin-denominierte Rendite: STABLE -> TARGET -> STABLE."""
    entry_time = int(target_df["open_time"].iat[entry_pos])
    exit_time = entry_time + int(hold_bars) * int(BAR_MIN) * 60_000
    exit_pos = exact_position(target_df, exit_time)
    if exit_pos is None:
        return None
    target_entry = float(target_df["close"].iat[entry_pos])
    target_exit = float(target_df["close"].iat[exit_pos])
    if min(target_entry, target_exit) <= 0:
        return None
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    factor = (1.0 - cost) * (target_exit / target_entry) * (1.0 - cost)
    return (factor - 1.0) * 100.0


def roi_for_hold_btc(
    target_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    entry_pos: int,
    hold_bars: int,
) -> Optional[float]:
    """BTC-denominierte Rendite eines vollständigen BTC <-> TARGET Zyklus."""
    entry_time = int(target_df["open_time"].iat[entry_pos])
    exit_time = entry_time + int(hold_bars) * int(BAR_MIN) * 60_000
    exit_pos = exact_position(target_df, exit_time)
    btc_entry_pos = exact_position(btc_df, entry_time)
    btc_exit_pos = exact_position(btc_df, exit_time)
    if exit_pos is None or btc_entry_pos is None or btc_exit_pos is None:
        return None
    target_entry = float(target_df["close"].iat[entry_pos])
    target_exit = float(target_df["close"].iat[exit_pos])
    btc_entry = float(btc_df["close"].iat[btc_entry_pos])
    btc_exit = float(btc_df["close"].iat[btc_exit_pos])
    if min(target_entry, target_exit, btc_entry, btc_exit) <= 0:
        return None
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    btc_factor = (1.0 - cost) * (btc_entry / target_entry) * (target_exit / btc_exit) * (1.0 - cost)
    return (btc_factor - 1.0) * 100.0


def simulate_events(
    target_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    events: List[MultiRefEvent],
    target_pair: str,
) -> List[dict]:
    """Erzeugt Trades als vollständige BTC <-> TARGET Umschichtungen."""
    trades: List[dict] = []
    for event in events:
        entry_pos = exact_position(target_df, event.end_ms)
        btc_entry_pos = exact_position(btc_df, event.end_ms)
        if entry_pos is None or btc_entry_pos is None:
            continue
        entry_time = int(target_df.iloc[entry_pos]["open_time"])
        entry_price = float(target_df.iloc[entry_pos]["close"])
        btc_entry_price = float(btc_df.iloc[btc_entry_pos]["close"])
        if entry_price <= 0 or btc_entry_price <= 0:
            continue
        row = {
            "coin": target_pair, "direction": event.direction,
            "window_min": event.window_min, "threshold_pct": event.threshold_pct,
            "min_refs": event.min_refs, "score": event.score,
            "matched_refs": ",".join(event.matched_refs), "event_time_ms": event.end_ms,
            "buy_time_ms": entry_time,
            "buy_time": ms_to_berlin_str(entry_time + int(BAR_MIN) * 60_000),
            "ref_moves": event.ref_moves, "btc_entry_price": btc_entry_price,
            "target_entry_price": entry_price, "position": "BTC -> TARGET -> BTC",
        }
        any_ok = False
        for hname, bars in HOLD_BARS.items():
            roi_btc = roi_for_hold_btc(target_df, btc_df, entry_pos, bars)
            roi_stable = roi_for_hold_stable(target_df, entry_pos, bars)
            row[f"roi_{hname}"] = roi_btc
            row[f"roi_stable_{hname}"] = roi_stable
            if roi_btc is not None:
                any_ok = True
        if any_ok:
            trades.append(row)
    return trades


def filter_non_overlapping(

   trades: List[dict],

   horizon: str,

) -> List[dict]:

   """

   Pro Hold-Horizont:

     1 Slot

     Cooldown = Hold-Länge



   Trade bei exakt last_buy + hold ist erlaubt (vorheriger Trade

   ist dann gerade verkauft) - einheitlich mit den anderen Scripts.

   """



   hold_ms = HOLD_HORIZONS[horizon] * 60_000

   col = f"roi_{horizon}"



   ordered = sorted(

       trades,

       key=lambda x: int(x["buy_time_ms"]),

   )



   kept: List[dict] = []

   last_buy_ms: Optional[int] = None



   for trade in ordered:

       roi = trade.get(col)



       if roi is None:

           continue



       buy_ms = int(trade["buy_time_ms"])



       if (

           last_buy_ms is not None

           and buy_ms < last_buy_ms + hold_ms

       ):

           continue



       kept.append(trade)

       last_buy_ms = buy_ms



   return kept





def column_stats(

   trades: List[dict],

   col: str,

) -> dict:

   vals = [

       float(t[col])

       for t in trades

       if t.get(col) is not None

       and not (

           isinstance(t.get(col), float)

           and np.isnan(t.get(col))

       )

   ]



   if not vals:

       return {

           "n": 0,

           "n_pos": 0,

           "n_neg": 0,

           "pct_pos": None,

           "avg": None,

           "compound": None,

       }



   n_pos = sum(v > 0 for v in vals)

   n_neg = sum(v < 0 for v in vals)



   equity = 1.0

   for v in vals:

       equity *= 1.0 + v / 100.0



   return {

       "n": len(vals),

       "n_pos": n_pos,

       "n_neg": n_neg,

       "pct_pos": (

           100.0 * n_pos / (n_pos + n_neg)

           if n_pos + n_neg

           else None

       ),

       "avg": float(np.mean(vals)),

       "compound": (equity - 1.0) * 100.0,

   }





def summarize(

   trades: List[dict],

) -> Dict[str, dict]:

   result = {}



   for h in HORIZON_NAMES:

       filtered = filter_non_overlapping(trades, h)

       result[h] = column_stats(filtered, f"roi_{h}")



   return result





def random_positions(

   df: pd.DataFrame,

   start_ms: int,

   end_ms: int,

   hold_bars: int,

   n_target: int,

   rng: np.random.Generator,

) -> List[int]:

   hold_ms = hold_bars * BAR_MIN * 60_000

   times = df["open_time"].to_numpy(dtype=np.int64)

   closes = df["close"].to_numpy(dtype=np.float64)



   # Kandidat nur, wenn die exakte Exit-Kerze existiert (wie bei der

   # Strategie) -> jeder akzeptierte Kandidat liefert einen Trade.

   exit_exists = np.isin(times + hold_ms, times)



   candidates = [

       int(i)

       for i in np.flatnonzero(

           (times >= start_ms)

           & (times <= end_ms)

           & exit_exists

           & (closes > 0)

       )

   ]



   if not candidates or n_target <= 0:

       return []



   # Exakt n_target nicht-überlappende Zufallstrades (Abstand >= Hold),

   # gleichverteilt über alle zulässigen Anordnungen:

   # Slots im Kerzenraster, n Positionen aus (Slots - (n-1)*hold_bars)

   # ziehen und je Trade um i*hold_bars verschieben. Liegt eine Position

   # auf einer Lücke (keine Kerze/kein Exit), wird neu gezogen.

   bar_ms = BAR_MIN * 60_000

   cand_arr = np.array(candidates, dtype=np.int64)

   t0 = int(times[cand_arr[0]])

   slot_of = {int((int(times[i]) - t0) // bar_ms): int(i) for i in cand_arr}

   n_slots = max(slot_of) + 1

   free = n_slots - (n_target - 1) * hold_bars



   if free >= n_target:

       offsets = np.arange(n_target, dtype=np.int64) * hold_bars

       for _ in range(500):

           picks = np.sort(rng.choice(free, size=n_target, replace=False)) + offsets

           if all(int(k) in slot_of for k in picks):

               return [slot_of[int(k)] for k in picks]



   # Fallback (sehr lückige Daten): greedy in zufälliger Reihenfolge.

   order = candidates.copy()

   rng.shuffle(order)



   accepted: List[int] = []



   for pos in order:

       t = int(times[pos])



       # Non-overlap wie Strategie: Abstand >= Hold ist erlaubt.

       if all(

           abs(t - int(times[p])) >= hold_ms

           for p in accepted

       ):

           accepted.append(pos)



       if len(accepted) >= n_target:

           break



   if len(accepted) < n_target:

       print(

           f"   Hinweis RANDOM: nur {len(accepted)}/{n_target} "

           f"nicht-überlappende Trades möglich."

       )



   return sorted(accepted, key=lambda p: int(times[p]))





def random_matched_stats(
    target_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    target_n: Dict[str, int],
    start_ms: int,
    end_ms: int,
    rng: np.random.Generator,
) -> Dict[str, dict]:
    result = {}
    for h in HORIZON_NAMES:
        positions = random_positions(target_df, start_ms, end_ms, HOLD_BARS[h], target_n.get(h, 0), rng)
        vals = []
        for pos in positions:
            roi = roi_for_hold_btc(target_df, btc_df, pos, HOLD_BARS[h])
            if roi is not None:
                vals.append(roi)
        if not vals:
            result[h] = {"n": 0, "pct_pos": None, "avg": None, "compound": None}
            continue
        equity = 1.0
        for v in vals:
            equity *= 1.0 + v / 100.0
        result[h] = {
            "n": len(vals),
            "pct_pos": 100.0 * sum(v > 0 for v in vals) / len(vals),
            "avg": float(np.mean(vals)),
            "compound": (equity - 1.0) * 100.0,
        }
    return result


def _fmt_ts(ts_ms: int) -> str:
    return pd.to_datetime(int(ts_ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M UTC")


def buy_and_hold_target_btc(
    target_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    start_ms: int,
    end_ms: int,
) -> dict:
    """Benchmark: BTC -> TARGET at the start, hold TARGET, then TARGET -> BTC at the end.

    Uses the first available target/BTC candle at or after start_ms and the last
    available target/BTC candle at or before end_ms. The same entry/exit cost
    model as the strategy is applied, so the comparison is apples-to-apples.
    """
    t = target_df[(target_df["open_time"] >= start_ms) & (target_df["open_time"] <= end_ms)].copy()
    b = btc_df[(btc_df["open_time"] >= start_ms) & (btc_df["open_time"] <= end_ms)].copy()

    if t.empty or b.empty:
        return {"roi": None, "buy_time": "", "sell_time": ""}

    common = sorted(set(t["open_time"].astype(np.int64)) & set(b["open_time"].astype(np.int64)))
    if not common:
        return {"roi": None, "buy_time": "", "sell_time": ""}

    entry_ts = common[0]
    exit_ts = common[-1]
    if entry_ts == exit_ts:
        return {"roi": None, "buy_time": "", "sell_time": ""}

    t_by = t.set_index("open_time")
    b_by = b.set_index("open_time")

    target_entry = float(t_by.loc[entry_ts, "close"])
    target_exit = float(t_by.loc[exit_ts, "close"])
    btc_entry = float(b_by.loc[entry_ts, "close"])
    btc_exit = float(b_by.loc[exit_ts, "close"])

    if min(target_entry, target_exit, btc_entry, btc_exit) <= 0:
        return {"roi": None, "buy_time": "", "sell_time": ""}

    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    btc_factor = (1.0 - cost) * (btc_entry / target_entry) * (target_exit / btc_exit) * (1.0 - cost)
    roi = (btc_factor - 1.0) * 100.0

    return {
        "roi": float(roi),
        "buy_time": _fmt_ts(entry_ts),
        "sell_time": _fmt_ts(exit_ts),
    }


def print_section(

   target_pair: str,

   direction: str,

   window: int,

   threshold: float,

   min_refs: int,

   trades: List[dict],

   stats: Dict[str, dict],

   bh: Optional[dict],

   total_refs: int,

   random_stats: Optional[Dict[str, dict]] = None,

   period: Optional[str] = None,

) -> None:

   label = min_ref_label(min_refs, total_refs)



   print()

   print("=" * 118)

   print(

       f"{direction:4} | "

       f"Event {window}m | "

       f"{'+' if direction == 'UP' else '-'}{threshold:g}% | "

       f"Refs {label} | "

       f"TARGET {target_pair} | Konto: BTC"

       + (f" | {period}" if period else "")

   )

   print("=" * 118)



   header = (

       f"{'Hold':>7} "

       f"{'n':>7} "

       f"{'%pos':>8} "

       f"{'Ø ROI':>11} "

       f"{'BTC Compound':>13} "
       f"{'Stable Compound':>15} "

       f"{'vs HODL Target':>15}"

   )



   if random_stats is not None:

       header += f" {'vs Rand':>11}"



   print(header)

   print("-" * len(header))



   bh_roi = bh["roi"] if bh else None



   for h in HORIZON_NAMES:

       s = stats[h]



       vs_bh = (

           s["compound"] - bh_roi

           if s["compound"] is not None and bh_roi is not None

           else None

       )



       stable_vals = [
           float(t[f"roi_stable_{h}"])
           for t in trades
           if t.get(f"roi_stable_{h}") is not None
       ]
       stable_compound = None
       if stable_vals:
           stable_equity = 1.0
           for v in stable_vals:
               stable_equity *= 1.0 + v / 100.0
           stable_compound = (stable_equity - 1.0) * 100.0

       line = (
           f"{h:>7} "
           f"{s['n']:>7} "
           f"{fmt_pos(s['pct_pos']):>8} "
           f"{fmt_pct(s['avg']):>11} "
           f"{fmt_pct(s['compound']):>13} "
           f"{fmt_pct(stable_compound):>15} "
           f"{fmt_pct(vs_bh):>15}"
       )



       if random_stats is not None:

           r = random_stats[h]

           vs_rand = (

               s["compound"] - r["compound"]

               if s["compound"] is not None

               and r["compound"] is not None

               else None

           )

           line += f" {fmt_pct(vs_rand):>11}"



       print(line)



   if bh:

       print(

           f"\nHODL Target (BTC-denominiert): {fmt_pct(bh['roi'])} | "

           f"{bh['buy_time']} -> {bh['sell_time']} (Kerzen-Close)"

       )



   print(

       f"Events raw: {len(trades)} | "

       f"Non-overlap je Hold separat"

   )





def print_trade_examples(

   trades: List[dict],

   max_rows: int = 5,

) -> None:

   if not SHOW_TRADES or not trades:

       return



   print("\nBeispiel-Events:")



   for t in trades[:max_rows]:

       moves = " ".join(

           f"{k}:{v:+.2f}%"

           for k, v in t["ref_moves"].items()

       )



       print(

           f"  {t['buy_time']} | "

           f"{t['direction']} | "

           f"{moves} | "

           f"matched={t['matched_refs']}"

       )





def clamp_end_to_closed_candle(end: datetime) -> datetime:

   """

   TO_DATE in der Zukunft -> Ende = Open-Zeit der letzten

   ABGESCHLOSSENEN Kerze (keine laufende Kerze).

   """

   bar_ms = int(BAR_MIN) * 60_000

   now_ms = int(time.time() * 1000)

   last_closed_open_ms = (now_ms // bar_ms) * bar_ms - bar_ms

   last_closed = pd.Timestamp(last_closed_open_ms, unit="ms", tz="UTC")

   if end.tzinfo is not None:

       last_closed = last_closed.tz_convert(end.tzinfo)

   else:

       last_closed = last_closed.tz_localize(None)

   return min(end, last_closed.to_pydatetime())





# ============================================================

# MAIN

# ============================================================



def main(argv: Optional[List[str]] = None) -> None:

   args = parse_cli_args(argv)



   refs_raw = args.ref or REF_SYMBOLS

   target_raw = args.target or TARGET_SYMBOL

   run_random = RUN_RANDOM if args.run_random is None else args.run_random



   # Unique refs, preserving order.

   refs_raw = list(dict.fromkeys(

       str(x).strip().upper()

       for x in refs_raw

   ))



   if len(refs_raw) < 1:

       raise SystemExit("Mindestens ein Referenzcoin nötig.")

   if "BTC" not in refs_raw:

       raise SystemExit("BTC muss als Referenzcoin enthalten sein, weil das Konto in BTC geführt wird.")



   if any(x <= 0 for x in THRESHOLDS_PCT):

       raise SystemExit("THRESHOLDS_PCT müssen > 0 sein.")



   usdt_set = binance_usdt_symbols()



   ref_bases: List[str] = []

   ref_pairs: Dict[str, str] = {}



   for raw in refs_raw:

       base, pair = resolve_pair(raw, usdt_set)

       if base in ref_bases:

           continue

       ref_bases.append(base)

       ref_pairs[base] = pair



   target_base, target_pair = resolve_pair(target_raw, usdt_set)



   if target_base in ref_bases:

       raise SystemExit(

           f"Target {target_base} darf nicht gleichzeitig Referenz sein."

       )



   start, end, window_label = resolve_event_window(

       from_s=args.from_date or FROM_DATE,

       to_s=args.to_date or TO_DATE,

       lookback_days=args.lookback,

       default_lookback=LOOKBACK_DAYS,

   )

   end = clamp_end_to_closed_candle(end)



   start_ms = utc_ms(start)

   end_ms = utc_ms(end)

   fetch_end_ms = utc_ms(

       end + pd.Timedelta(minutes=MAX_HOLD_MIN)

   )



   print("\n" + "=" * 118)

   print("MULTI-REFERENCE BUY BACKTEST")

   print("=" * 118)



   print(

       f"Refs:   {', '.join(ref_bases)}"

   )

   print(

       f"Target: {target_base} ({target_pair})"

   )

   print(

       f"Events: {', '.join(str(x) + 'm' for x in EVENT_WINDOWS_MIN)}"

   )

   print(

       f"Thresholds: {', '.join(f'{x:g}%' for x in THRESHOLDS_PCT)}"

   )

   print(

       "Min Refs: "

       + ", ".join(

           min_ref_label(x, len(ref_bases))

           for x in MIN_REFS

       )

   )

   print(

       f"Zeitraum: {start.strftime('%Y-%m-%d %H:%M %Z')} "

       f"-> {end.strftime('%Y-%m-%d %H:%M %Z')}"

   )

   print(

       f"Interval: {KLINE_INTERVAL} | "

       f"Fee: {FEE_BPS} bps/Seite | "

       f"Slippage: {SLIPPAGE_BPS} bps"

   )

   print(

       "UP = Referenzen steigen -> BTC -> TARGET | "

       "DOWN = Referenzen fallen -> BTC -> TARGET"

   )

   print(

       "Signalzeitpunkt = Ende der Referenz-Kerze; "

       "Target-Einstieg = Target-Close dieser Kerze; das Konto startet vollständig in BTC."

   )

   print("=" * 118)



   # --------------------------------------------------------

   # Load reference data

   # --------------------------------------------------------



   ref_data: Dict[str, pd.DataFrame] = {}



   print("\n1) Lade Referenzcoins:")



   for base in ref_bases:

       pair = ref_pairs[base]



       print(f"   {base:>8} ({pair}) ...", end="", flush=True)



       df = fetch_klines(

           pair,

           KLINE_INTERVAL,

           start_ms,

           fetch_end_ms,

       )



       if df.empty:

           raise SystemExit(

               f"\nKeine Kerzen für Referenz {pair}."

           )



       ref_data[base] = df



       print(f" {len(df)} Kerzen")



   # --------------------------------------------------------

   # Load target

   # --------------------------------------------------------



   print(f"\n2) Lade Target {target_pair} ...")



   target_df = fetch_klines(

       target_pair,

       KLINE_INTERVAL,

       start_ms,

       fetch_end_ms,

   )



   if target_df.empty:

       raise SystemExit(

           f"Keine Kerzen für Target {target_pair}."

       )



   print(f"   {len(target_df)} Kerzen")



   bh = buy_and_hold_target_btc(
       target_df,
       ref_data["BTC"],
       start_ms,
       end_ms,
   )



   # Effektiver Zeitraum fuer die Block-Header (Refs + Target vorhanden).

   block_period = _block_range(*_effective_range_ms(

       start_ms,

       end_ms,

       target_df["open_time"].to_numpy(),

       *[d["open_time"].to_numpy() for d in ref_data.values()],

   ))



   # --------------------------------------------------------

   # Detect events

   # --------------------------------------------------------



   print("\n3) Erzeuge Multi-Reference Events ...")



   all_events = detect_multi_ref_events(

       ref_data,

       ref_pairs,

       EVENT_WINDOWS_MIN,

       THRESHOLDS_PCT,

       MIN_REFS,

       start_ms,

       end_ms,

   )



   total_event_count = sum(

       len(events)

       for events in all_events.values()

   )



   print(

       f"   {total_event_count} Event-Instanzen über "

       f"{len(all_events)} Konfigurationen"

   )



   # --------------------------------------------------------

   # Process all configurations

   # --------------------------------------------------------



   print("\n4) Ergebnisse\n")



   result_cache: Dict[

       Tuple[int, float, int, str],

       Tuple[List[dict], Dict[str, dict]]

   ] = {}



   for window in EVENT_WINDOWS_MIN:

       for threshold in THRESHOLDS_PCT:

           for min_refs in MIN_REFS:

               for direction in ("UP", "DOWN"):

                   key = (

                       window,

                       threshold,

                       min_refs,

                       direction,

                   )



                   events = all_events.get(key, [])



                   trades = simulate_events(

                       target_df,

                       ref_data["BTC"],

                       events,

                       target_pair,

                   )



                   stats = summarize(trades)



                   result_cache[key] = (

                       trades,

                       stats,

                   )



   # --------------------------------------------------------

   # Random baseline

   #

   # Matched against each configuration separately.

   # This keeps the number of trades comparable.

   # --------------------------------------------------------



   random_cache: Dict[

       Tuple[int, float, int, str],

       Dict[str, dict]

   ] = {}



   if run_random:

       print("5) RANDOM-Baselines ...")



       for key, (trades, stats) in result_cache.items():

           target_n = {

               h: stats[h]["n"]

               for h in HORIZON_NAMES

           }



           seed = RANDOM_SEED + (

               key[0] * 1000

               + int(key[1] * 100)

               + key[2] * 10

               + (1 if key[3] == "DOWN" else 0)

           )



           rng = np.random.default_rng(seed)



           random_cache[key] = random_matched_stats(

               target_df,

               ref_data["BTC"],

               target_n,

               start_ms,

               end_ms,

               rng,

           )



   # --------------------------------------------------------

   # Print ordered output

   #

   # EXACTLY the requested structure:

   # window -> threshold -> refs -> UP/DOWN -> holds

   # --------------------------------------------------------



   for window in EVENT_WINDOWS_MIN:

       for threshold in THRESHOLDS_PCT:

           for min_refs in MIN_REFS:

               for direction in ("UP", "DOWN"):

                   key = (

                       window,

                       threshold,

                       min_refs,

                       direction,

                   )



                   trades, stats = result_cache[key]



                   random_stats = (

                       random_cache.get(key)

                       if run_random

                       else None

                   )



                   print_section(

                       target_pair,

                       direction,

                       window,

                       threshold,

                       min_refs,

                       trades,

                       stats,

                       bh,

                       len(ref_bases),

                       random_stats,

                       period=block_period,

                   )



                   print_trade_examples(trades)



   # --------------------------------------------------------

   # Compact overview

   # --------------------------------------------------------



   print("\n" + "=" * 118)

   print("KOMPAKTÜBERSICHT")

   print("=" * 118)



   print(

       f"{'Event':>8} "

       f"{'Thresh':>8} "

       f"{'Refs':>10} "

       f"{'Dir':>6} "

       f"{'Hold':>7} "

       f"{'n':>6} "

       f"{'Ø':>10} "

       f"{'Compound':>13} "

       f"{'vs HODL Target':>15}"

   )



   print("-" * 118)



   for window in EVENT_WINDOWS_MIN:

       for threshold in THRESHOLDS_PCT:

           for min_refs in MIN_REFS:

               for direction in ("UP", "DOWN"):

                   key = (

                       window,

                       threshold,

                       min_refs,

                       direction,

                   )



                   _, stats = result_cache[key]

                   label = min_ref_label(

                       min_refs,

                       len(ref_bases),

                   )



                   for h in HORIZON_NAMES:

                       s = stats[h]



                       vs_bh = (

                           s["compound"] - bh["roi"]

                           if bh

                           and s["compound"] is not None

                           else None

                       )



                       print(

                           f"{window:>7}m "

                           f"{threshold:>7.1f}% "

                           f"{label:>10} "

                           f"{direction:>6} "

                           f"{h:>7} "

                           f"{s['n']:>6} "

                           f"{fmt_pct(s['avg']):>10} "

                           f"{fmt_pct(s['compound']):>13} "

                           f"{fmt_pct(vs_bh):>13}"

                       )



   print("\n" + "=" * 118)

   print("Fertig.")

   print(

       "Compound = Produkt der einzelnen non-overlapping Trade-Returns."

   )

   print(

       "BTC Compound = dieselben Signale mit BTC -> Target -> BTC. Stable Compound = exakt dieselben Signale mit Stablecoin -> Target -> Stablecoin. HODL Target = Buy-and-Hold des Targets. HODL bedeutet: am Anfang komplett BTC -> Target, dann bis zum Ende halten und komplett Target -> BTC."

   )

   print(

       "UP und DOWN sind beide BTC -> TARGET Umschichtungssignale."

   )

   print(

       "Jeder Hold ist eine separate Strategie; bei jedem Trade werden 100% BTC -> TARGET -> BTC gewechselt."

   )

   print("=" * 118)





if __name__ == "__main__":

   main()
