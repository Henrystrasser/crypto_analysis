#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sector_rotation_sim.py – Krypto-SEKTOR-ROTATION Backtest (Binance Spot, 1h-Kerzen, USDT-Paare)

Idee: In einem Sektor (AI, L1, L2, Memes, DeFi, Gaming) läuft ein Coin (der "Leader") stark,
der Rest des Sektors hat (noch) nicht nachgezogen. Hypothese: die Nachzügler ("Laggards")
holen auf. Wir kaufen den Laggard (Varianten A/B/C) bzw. zum Vergleich den Leader (Variante D).

Ablauf:
  1. Daten laden (Binance 1h-Klines, lokal gecacht unter ./cache)
  2. Jede Stunde pro Sektor: Lookback-Rendite jedes Coins
     Signal = Leader-Rendite >= LEADER_THRESHOLD_PCT  UND  Median der übrigen <= LAG_MAX
     Nur die ERSTE Kerze, die die Bedingung erfüllt (Schwellen-Kreuzung, wie btc_rise_hold_sim)
  3. Einstieg = Close der Signal-Kerze, Ausstieg = Close nach HOLD Stunden, Gebühr pro Seite
  4. Kein Überlappen pro Sektor & Haltedauer (Cooldown = Haltedauer)
  5. Baselines: RANDOM (gleiche Anzahl Trades, zufällige Zeiten/Coins im Sektor, mehrere Seeds),
     Buy&Hold Sektor-Basket, und pro Trade: Überrendite vs. Sektor-Basket im selben Fenster.

Alle Zeiten in der Ausgabe: Europe/Berlin (CET/CEST). Kerzen-Index = SCHLUSSZEIT der Kerze.
Nur Konsolen-Ausgabe; Ergebnisse zusätzlich als JSON (results.json) neben dem Skript.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# =============================================================================
# KONFIGURATION (hier in VS Code anpassen)
# =============================================================================
START_LOCAL = "2024-01-01 00:00"      # Beginn (Europe/Berlin)
END_LOCAL = None                      # None = bis jetzt (letzte abgeschlossene Kerze)
TZ_LOCAL = "Europe/Berlin"
WARMUP_DAYS = 3                       # Daten vor START für Lookback/Volumen laden

# Sektoren: nur Coins mit Binance-USDT-Paar; keine Stables, kein BTC.
# Exchange-Sektor (BNB, CRO, OKB): CRO/OKB nicht auf Binance -> nur BNB -> Sektor entfällt.
SECTORS = {
    "AI":     ["FET", "RENDER", "TAO", "NEAR", "WLD", "AR", "GRT"],
    "L1":     ["ETH", "SOL", "AVAX", "ADA", "DOT", "SUI", "APT", "SEI"],
    "L2":     ["ARB", "OP", "POL", "STRK", "IMX", "MANTA"],
    "Memes":  ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF"],
    "DeFi":   ["UNI", "AAVE", "SKY", "LDO", "CRV", "PENDLE", "JUP"],
    "Gaming": ["SAND", "MANA", "AXS", "GALA", "ILV"],
}

# Umbenennungen / Ticker-Wechsel: Coin -> Liste von (Vorgänger-Symbol, Preisfaktor).
# Vorgänger-Preis * Faktor = Preis in Einheiten des neuen Coins. Die Serien werden aneinander-
# gehängt (neuer Ticker hat Vorrang, Lücke dazwischen bleibt NaN).
#   RNDR -> RENDER 1:1 (Juli 2024), MATIC -> POL 1:1 (Sept 2024), MKR -> SKY 1:24000 (Sept 2025)
ALIASES = {
    "RENDER": [("RNDR", 1.0)],
    "POL":    [("MATIC", 1.0)],
    "SKY":    [("MKR", 1.0 / 24000.0)],
}
QUOTE = "USDT"

# Strategie
LOOKBACK_CONFIGS = [(6, 5.0), (24, 10.0)]   # (Lookback in h, Leader-Schwelle in %)
LAG_MODE = "ratio"                    # "ratio": Median(Rest) <= Leader-Rendite * LAG_RATIO
LAG_RATIO = 1.0 / 3.0                 # "fixed": Median(Rest) <= LAG_MAX_PCT
LAG_MAX_PCT = 1.5
HOLDS_H = [6, 12, 24, 48, 72, 120]
VARIANTS = {
    "A": "Laggard (schwächster Coin)",
    "B": "Liquider Laggard (max. 24h-Volumen)",
    "C": "Basket aller Nicht-Leader",
    "D": "Leader-Momentum (Leader kaufen)",
}
FEE_PCT = 0.075                       # pro Seite (7.5 bps), kein Slippage
MIN_SECTOR_COINS = 3                  # Leader + mind. 2 andere für sinnvollen Median
LISTING_WARMUP_H = 24                 # neue Listings erst nach X h (zusätzlich zum Lookback)
FFILL_LIMIT_H = 120                   # Exit-Preis über Lücken (Rename-Pausen) max. X h fortschreiben

# Baselines / Statistik
N_RANDOM_SEEDS = 20
RANDOM_SEED = 42
MIN_TRADES_BEST = 30                  # beste Konfig nur mit mind. so vielen Trades
N_BOOTSTRAP = 10000
TOP_N = 5

# Daten
BASE_URLS = ["https://api.binance.com", "https://data-api.binance.vision"]
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"
RESULTS_JSON = SCRIPT_DIR / "results.json"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]

FEE = FEE_PCT / 100.0


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================
def fmt_ts(ts) -> str:
    """UTC-Timestamp -> 'YYYY-MM-DD HH:MM CET/CEST'."""
    return pd.Timestamp(ts).tz_convert(TZ_LOCAL).strftime("%Y-%m-%d %H:%M %Z")


def pct(x, d=2) -> str:
    return "   n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{d}f}%"


_active_base = None


def _get(path: str, params: dict):
    """GET mit automatischem Fallback (api.binance.com ist evtl. geo-geblockt -> data-api)."""
    global _active_base
    bases = [_active_base] if _active_base else BASE_URLS
    last_err = None
    for base in bases:
        for attempt in range(4):
            try:
                r = requests.get(base + path, params=params, timeout=20)
            except requests.RequestException as e:
                last_err = e
                time.sleep(1 + attempt)
                continue
            if r.status_code in (403, 451):        # geo-block -> nächste Basis-URL
                last_err = f"HTTP {r.status_code} @ {base}"
                break
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + 2 * attempt)
                continue
            _active_base = base
            return r
    raise RuntimeError(f"Binance nicht erreichbar: {last_err}")


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """1h-Klines laden; Cache unter cache/<SYMBOL>_1h.csv.gz, inkrementell ergänzt."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{symbol}_1h.csv.gz"
    df = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=KLINE_COLS[:8])
    have_from = int(df["open_time"].min()) if len(df) else None
    cursor = int(df["open_time"].max()) + 3_600_000 if len(df) else start_ms
    if have_from is not None and have_from > start_ms + 3_600_000 and not (CACHE_DIR / f"{symbol}.complete_from").exists():
        cursor = start_ms                       # Cache beginnt zu spät -> komplett neu
        df = df.iloc[0:0]
    rows = []
    while cursor < end_ms:
        r = _get("/api/v3/klines", {"symbol": symbol, "interval": "1h",
                                    "startTime": cursor, "endTime": end_ms, "limit": 1000})
        if r.status_code == 400:                # ungültiges Symbol
            return pd.DataFrame(columns=KLINE_COLS[:8])
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 3_600_000
        if len(batch) < 1000:
            break
        time.sleep(0.05)
    if rows:
        new = pd.DataFrame(rows, columns=KLINE_COLS).iloc[:, :8]
        df = pd.concat([df, new]).drop_duplicates("open_time").sort_values("open_time")
        df.to_csv(f, index=False)
        (CACHE_DIR / f"{symbol}.complete_from").write_text(str(start_ms))
    return df


def load_coin(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Close + Quote-Volumen je Coin, Vorgänger-Ticker (ALIASES) angehängt. Index = Kerzen-Schluss (UTC)."""
    parts = []
    for sym, factor in ALIASES.get(coin, []) + [(coin, 1.0)]:
        raw = fetch_klines(sym + QUOTE, start_ms, end_ms)
        if raw.empty:
            continue
        d = pd.DataFrame({
            "close": raw["close"].astype(float).values * factor,
            "qvol": raw["quote_volume"].astype(float).values,
            "src": sym,
        }, index=pd.to_datetime(raw["open_time"].astype("int64") + 3_600_000, unit="ms", utc=True))
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    out = parts[-1]                              # aktueller Ticker hat Vorrang
    for p in reversed(parts[:-1]):
        out = pd.concat([p[p.index < out.index.min()], out])
    return out.sort_index()


def summarize(net: np.ndarray, excess: np.ndarray) -> dict:
    n = len(net)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "pos": float((net > 0).mean()),
        "avg": float(net.mean()),
        "med": float(np.median(net)),
        "comp": float(np.prod(1 + net) - 1),
        "exc": float(np.nanmean(excess)),
    }


def tstat_bootstrap(x: np.ndarray, n_boot: int, seed: int):
    x = x[~np.isnan(x)]
    n = len(x)
    t = x.mean() / (x.std(ddof=1) / math.sqrt(n))
    p = math.erfc(abs(t) / math.sqrt(2))       # zweiseitig, Normal-Approximation
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return t, p, np.percentile(boots, [2.5, 97.5])


# =============================================================================
# KERN
# =============================================================================
class Market:
    """Hält Preis-/Volumen-Matrizen (Zeit x Coin) als numpy-Arrays."""

    def __init__(self, start_utc, end_utc):
        start_ms = int((start_utc - pd.Timedelta(days=WARMUP_DAYS)).timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)
        idx = pd.date_range(start_utc - pd.Timedelta(days=WARMUP_DAYS), end_utc, freq="1h")
        self.coins, closes, vols, self.coverage, self.missing = [], {}, {}, {}, []
        for sector, members in SECTORS.items():
            for coin in members:
                df = load_coin(coin, start_ms, end_ms - 3_600_000)
                df = df[(df.index >= idx[0]) & (df.index <= end_utc)]
                if df.empty:
                    self.missing.append(f"{sector}:{coin}")
                    continue
                self.coins.append(coin)
                closes[coin] = df["close"]
                vols[coin] = df["qvol"]
                in_period = df[df.index >= start_utc]
                expected = (df.index.max() - max(df.index.min(), start_utc)) / pd.Timedelta(hours=1) + 1
                self.coverage[coin] = {
                    "first": df.index.min(), "last": df.index.max(),
                    "srcs": "+".join(dict.fromkeys(df["src"])),
                    "bars": len(in_period), "gap_h": int(expected - len(in_period)),
                }
        self.index = idx
        self.C = pd.DataFrame(closes).reindex(idx)[self.coins]
        self.Cff = self.C.ffill(limit=FFILL_LIMIT_H)          # nur für Exit-Preise
        self.V24 = pd.DataFrame(vols).reindex(idx)[self.coins].rolling(24, min_periods=24).sum()
        self.start_i = int(np.searchsorted(idx, start_utc))
        self.T = len(idx)
        self.col = {c: i for i, c in enumerate(self.coins)}
        self.sector_cols = {s: [self.col[c] for c in m if c in self.col] for s, m in SECTORS.items()}
        first_valid = self.C.notna().values.argmax(axis=0)
        self.age = np.arange(self.T)[:, None] - first_valid[None, :]   # Stunden seit Listing
        self._fwd = {}

    def lookback_ret(self, L):
        c = self.C.values
        r = np.full_like(c, np.nan)
        r[L:] = c[L:] / c[:-L] - 1
        r[self.age < L + LISTING_WARMUP_H] = np.nan               # frische Listings ignorieren
        return r

    def fwd_ret(self, H):
        """Brutto-Rendite Einstieg Close[t] -> Ausstieg Close[t+H] (Exit-Preis ffill)."""
        if H not in self._fwd:
            c, cf = self.C.values, self.Cff.values
            r = np.full_like(c, np.nan)
            r[:-H] = cf[H:] / c[:-H] - 1
            self._fwd[H] = r
        return self._fwd[H]


def sector_signals(R: np.ndarray, thr: float, start_i: int):
    """Pro Stunde: Leader, Leader-Rendite, Median der Übrigen, Signal-Kreuzung."""
    n_av = np.sum(~np.isnan(R), axis=1)
    ok = n_av >= MIN_SECTOR_COINS
    Rf = np.where(np.isnan(R), -np.inf, R)
    leader = Rf.argmax(axis=1)
    lead_ret = np.where(ok, Rf[np.arange(len(R)), leader], np.nan)
    others = R.copy()
    others[np.arange(len(R)), leader] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med_oth = np.nanmedian(others, axis=1)
    lag_max = lead_ret * LAG_RATIO if LAG_MODE == "ratio" else LAG_MAX_PCT / 100
    with np.errstate(invalid="ignore"):
        cond = ok & (lead_ret >= thr) & (med_oth <= lag_max)
    prev = np.concatenate([[False], cond[:-1]])
    cross = cond & ~prev
    cross[:start_i] = False
    return np.flatnonzero(cross), leader, lead_ret, med_oth, lag_max, ok


def pick_positions(variant, t, R_t, lead, lag_max_t, V_t):
    """Welche Sektor-Spalten (lokal) werden gekauft?"""
    avail = np.flatnonzero(~np.isnan(R_t))
    others = avail[avail != lead]
    if variant == "A":
        return [others[np.argmin(R_t[others])]]
    if variant == "B":
        lagg = others[R_t[others] <= lag_max_t]
        lagg = lagg if len(lagg) else others
        v = np.nan_to_num(V_t[lagg], nan=-1)
        return [lagg[np.argmax(v)]]
    if variant == "C":
        return list(others)
    if variant == "D":
        return [lead]
    raise ValueError(variant)


def run_strategy(mk: Market, L: int, thr_pct: float):
    """Alle Trades für einen Lookback, alle Varianten & Haltedauern."""
    RL = mk.lookback_ret(L)
    V = mk.V24.values
    trades = []
    for sector, cols in mk.sector_cols.items():
        R = RL[:, cols]
        ev, leader, lead_ret, med_oth, lag_max, _ = sector_signals(R, thr_pct / 100, mk.start_i)
        lag_arr = np.broadcast_to(lag_max, (mk.T,)) if np.ndim(lag_max) else np.full(mk.T, lag_max)
        for H in HOLDS_H:
            F = mk.fwd_ret(H)[:, cols]
            next_free = -1
            for t in ev:
                if t < next_free or t + H > mk.T - 1:
                    continue
                next_free = t + H                       # Cooldown = Haltedauer
                avail = ~np.isnan(R[t])
                basket = np.nanmean(np.where(avail, F[t], np.nan))
                for var in VARIANTS:
                    pos = pick_positions(var, t, R[t], leader[t], lag_arr[t], V[t, cols])
                    gross = float(np.nanmean(F[t, pos]))
                    if np.isnan(gross):
                        continue
                    trades.append({
                        "variant": var, "L": L, "H": H, "sector": sector, "t": int(t),
                        "coin": "+".join(mk.coins[cols[p]] for p in pos) if var != "C" else f"basket({len(pos)})",
                        "leader": mk.coins[cols[leader[t]]], "lead_ret": float(lead_ret[t]),
                        "med_oth": float(med_oth[t]), "coin_ret": float(np.nanmean(R[t, pos])),
                        "gross": gross, "net": (1 + gross) * (1 - FEE) ** 2 - 1,
                        "basket": float(basket), "excess": gross - float(basket),
                    })
    return pd.DataFrame(trades), RL


def random_baseline(mk: Market, RL: np.ndarray, sector: str, H: int, n: int, seed: int):
    """n zufällige, nicht überlappende Trades (Zufalls-Coin des Sektors). Gibt (net, excess) zurück."""
    if n == 0:
        return np.array([]), np.array([])
    cols = mk.sector_cols[sector]
    R = RL[:, cols]
    F = mk.fwd_ret(H)[:, cols]
    avail = ~np.isnan(R) & ~np.isnan(F)
    valid = np.flatnonzero((np.arange(mk.T) >= mk.start_i) & (np.arange(mk.T) + H <= mk.T - 1)
                           & (avail.sum(axis=1) >= MIN_SECTOR_COINS))
    rng = np.random.default_rng(seed)
    slots = len(valid) - (n - 1) * H
    n = min(n, max(slots, 0))
    # Abstands-Trick: n sortierte Positionen + i*H -> garantiert >= H Stunden Abstand
    p = np.sort(rng.choice(slots, size=n, replace=False)) + np.arange(n) * H
    ts = valid[p]
    score = np.where(avail[ts], rng.random((n, len(cols))), -1)
    pick = score.argmax(axis=1)
    gross = F[ts, pick]
    basket = np.nanmean(np.where(avail[ts], F[ts], np.nan), axis=1)
    return (1 + gross) * (1 - FEE) ** 2 - 1, gross - basket


def random_for_config(mk, RL, df_cfg, H):
    """Pooled über Sektoren, gemittelt über Seeds."""
    avgs, excs, poss = [], [], []
    counts = df_cfg.groupby("sector").size()
    for s in range(N_RANDOM_SEEDS):
        nets, exs = [], []
        for k, (sector, n) in enumerate(counts.items()):
            a, b = random_baseline(mk, RL, sector, H, int(n), RANDOM_SEED + s + 1000 * k + 7 * H)
            nets.append(a)
            exs.append(b)
        nets, exs = np.concatenate(nets), np.concatenate(exs)
        avgs.append(nets.mean())
        excs.append(np.nanmean(exs))
        poss.append((nets > 0).mean())
    return {"avg": float(np.mean(avgs)), "sd": float(np.std(avgs)), "exc": float(np.mean(excs)),
            "pos": float(np.mean(poss)), "seed_avgs": [float(x) for x in avgs]}


# =============================================================================
# AUSGABE
# =============================================================================
HDR = f"{'Hold':>5} {'n':>5} {'%pos':>6} {'avgROI':>8} {'medROI':>8} {'compROI':>10} {'avgExc':>8} {'RND avg':>8} {'vsRND':>8} {'RND>=':>6}"


def row(label, s, rnd=None):
    if s.get("n", 0) == 0:
        return f"{label:>5} {0:>5}  (keine Trades)"
    r = f"{label:>5} {s['n']:>5} {s['pos'] * 100:5.1f}% {pct(s['avg']):>8} {pct(s['med']):>8} {pct(s['comp'], 1):>10} {pct(s['exc']):>8}"
    if rnd:
        beat = np.mean(np.array(rnd["seed_avgs"]) >= s["avg"])
        r += f" {pct(rnd['avg']):>8} {pct(s['avg'] - rnd['avg']):>8} {beat * 100:5.0f}%"
    return r


def print_coverage(mk: Market):
    print("\n=== DATENABDECKUNG (Zeiten Europe/Berlin, Kerzen-Schluss) ===")
    for sector, members in SECTORS.items():
        have = [c for c in members if c in mk.coverage]
        print(f"{sector:7s} {len(have)}/{len(members)} Coins")
        for c in have:
            cv = mk.coverage[c]
            print(f"   {c:7s} {cv['srcs']:12s} ab {fmt_ts(cv['first'])}  bis {fmt_ts(cv['last'])}"
                  f"  Kerzen={cv['bars']:6d}  Lücken={cv['gap_h']}h")
    if mk.missing:
        print("FEHLEND/übersprungen:", ", ".join(mk.missing))
    print("Exchange-Sektor entfällt: CRO/OKB haben kein Binance-USDT-Paar, BNB allein ergibt keinen Sektor.")


def print_buy_hold(mk: Market):
    print("\n=== BUY & HOLD Sektor-Basket (gleichgewichtet, START -> ENDE) ===")
    print(f"{'Sektor':7s} {'B&H Start-Coins':>16} {'#':>3} {'Index stündl. rebal.':>21} {'#Coins Ende':>12}")
    c, cf = mk.C.values, mk.Cff.values
    hr = np.full_like(c, np.nan)
    hr[1:] = c[1:] / c[:-1] - 1
    out = {}
    for sector, cols in mk.sector_cols.items():
        s0 = c[mk.start_i, cols]
        have = ~np.isnan(s0)
        bh = np.nanmean(cf[-1, cols][have] / s0[have]) * (1 - FEE) ** 2 - 1
        idx_ret = np.nanmean(hr[mk.start_i + 1:, cols], axis=1)
        idx_ret = np.nan_to_num(idx_ret)
        rebal = np.prod(1 + idx_ret) - 1
        out[sector] = {"bh": float(bh), "rebal_index": float(rebal)}
        print(f"{sector:7s} {pct(bh, 1):>16} {have.sum():>3} {pct(rebal, 1):>21} {np.sum(~np.isnan(c[-1, cols])):>12}")
    return out


def main():
    global LOOKBACK_CONFIGS, LAG_MODE
    ap = argparse.ArgumentParser(description="Krypto-Sektor-Rotation Backtest")
    ap.add_argument("--lookbacks", type=str, help="z.B. '6:5,24:10' (Lookback h : Schwelle %%)")
    ap.add_argument("--lag-mode", choices=["ratio", "fixed"])
    args = ap.parse_args()
    if args.lookbacks:
        LOOKBACK_CONFIGS = [(int(a), float(b)) for a, b in (x.split(":") for x in args.lookbacks.split(","))]
    if args.lag_mode:
        LAG_MODE = args.lag_mode

    t0 = time.time()
    start_utc = pd.Timestamp(START_LOCAL, tz=TZ_LOCAL).tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC").floor("1h")             # letzte abgeschlossene Kerze schließt hier
    end_utc = now if END_LOCAL is None else min(now, pd.Timestamp(END_LOCAL, tz=TZ_LOCAL).tz_convert("UTC"))

    print("=" * 100)
    print("KRYPTO SEKTOR-ROTATION BACKTEST – Binance Spot 1h")
    print(f"Zeitraum: {fmt_ts(start_utc)}  ->  {fmt_ts(end_utc)}   Gebühr {FEE_PCT}%/Seite, kein Slippage")
    lag_txt = f"Median(Rest) <= Leader * {LAG_RATIO:.3f}" if LAG_MODE == "ratio" else f"Median(Rest) <= {LAG_MAX_PCT}%"
    print(f"Signal: Leader >= Schwelle UND {lag_txt}; nur erste Kreuzung; Cooldown = Haltedauer pro Sektor")
    print("=" * 100)

    mk = Market(start_utc, end_utc)
    t_data = time.time() - t0
    print_coverage(mk)
    bh = print_buy_hold(mk)

    results, all_trades, rnd_cache = {}, [], {}
    for L, thr in LOOKBACK_CONFIGS:
        df, RL = run_strategy(mk, L, thr)
        all_trades.append(df)
        for var, vname in VARIANTS.items():
            print(f"\n=== Variante {var}: {vname} | Lookback {L}h, Leader >= {thr}% ===")
            print(HDR)
            for H in HOLDS_H:
                d = df[(df.variant == var) & (df.H == H)] if len(df) else df
                s = summarize(d["net"].values, d["excess"].values) if len(d) else {"n": 0}
                key = (L, H)                                    # RANDOM hängt nur von n/Sektor ab
                rnd = None
                if len(d):
                    if key not in rnd_cache:
                        rnd_cache[key] = random_for_config(mk, RL, d, H)
                    rnd = rnd_cache[key]
                results[f"{var}|L{L}|H{H}"] = {**s, "variant": var, "L": L, "thr": thr, "H": H, "random": rnd}
                print(row(f"{H}h", s, rnd))
    print("\nLegende: avgExc = Ø(Brutto-Trade-ROI − Brutto-Sektor-Basket-ROI im selben Fenster);"
          " RND = Zufalls-Trades (gleiche Anzahl/Sektor, Ø über", N_RANDOM_SEEDS, "Seeds);"
          " RND>= = Anteil Seeds mit Ø-ROI >= Strategie; compROI = Π(1+r)−1 über alle Trades (überlappend über Sektoren!)")

    trades = pd.concat(all_trades, ignore_index=True)
    trades["entry"] = mk.index[trades["t"].values]
    trades["exit"] = mk.index[(trades["t"] + trades["H"]).values]
    trades["year"] = trades["entry"].dt.tz_convert(TZ_LOCAL).dt.year

    # ---- beste Konfigurationen: gesamt und nur Laggard-Varianten (A/B/C) ----
    cand = {k: v for k, v in results.items() if v.get("n", 0) >= MIN_TRADES_BEST}
    best_key = max(cand, key=lambda k: cand[k]["exc"])
    best_lag_key = max((k for k in cand if cand[k]["variant"] != "D"), key=lambda k: cand[k]["exc"])
    best_roi_key = max(cand, key=lambda k: cand[k]["avg"])
    rb = results[best_roi_key]
    print(f"\n(Beste nach Ø ROI: Variante {rb['variant']}, L{rb['L']}, H{rb['H']}: "
          f"Ø ROI {pct(rb['avg'])}, Ø Exc {pct(rb['exc'])}, n={rb['n']})")
    best_stats = {}
    for title, key in (("BESTE KONFIG GESAMT", best_key), ("BESTE LAGGARD-KONFIG (A/B/C)", best_lag_key)):
        if title.startswith("BESTE LAGGARD") and key == best_key:
            continue
        best_stats[key] = report_config(mk, trades, results, key, title)

    print(f"Zeitzone-Check: START {START_LOCAL} Berlin = {start_utc} UTC; Ende {fmt_ts(end_utc)} = {end_utc} UTC")
    runtime = time.time() - t0
    print(f"\nLaufzeit: {runtime:.1f}s (davon Daten laden {t_data:.1f}s)")

    bt = trades[(trades.variant == results[best_key]["variant"]) & (trades.L == results[best_key]["L"])
                & (trades.H == results[best_key]["H"])]
    json_out = {
        "generated": fmt_ts(pd.Timestamp.now(tz="UTC")), "period": [fmt_ts(start_utc), fmt_ts(end_utc)],
        "config": {"lookbacks": LOOKBACK_CONFIGS, "lag_mode": LAG_MODE, "lag_ratio": LAG_RATIO,
                   "lag_max_pct": LAG_MAX_PCT, "holds": HOLDS_H, "fee_pct": FEE_PCT, "sectors": SECTORS},
        "coverage": {c: {k: (fmt_ts(v) if isinstance(v, pd.Timestamp) else v) for k, v in cv.items()}
                     for c, cv in mk.coverage.items()},
        "missing": mk.missing, "buy_hold": bh, "results": results, "best": best_key, "best_laggard": best_lag_key,
        "best_stats": best_stats,
        "best_trades": bt.assign(entry=bt.entry.map(fmt_ts), exit=bt.exit.map(fmt_ts)).to_dict("records"),
        "runtime_s": runtime,
    }
    RESULTS_JSON.write_text(json.dumps(json_out, indent=1, default=float))


def report_config(mk, trades, results, key, title):
    """Detail-Report für eine Konfig: Sektoren, Jahre, Signifikanz, Top/Flop, Sanity."""
    b = results[key]
    bt = trades[(trades.variant == b["variant"]) & (trades.L == b["L"]) & (trades.H == b["H"])].sort_values("t")
    print("\n" + "=" * 100)
    print(f"{title} (max Ø Überrendite, n>={MIN_TRADES_BEST}): Variante {b['variant']} "
          f"({VARIANTS[b['variant']]}), Lookback {b['L']}h/{b['thr']}%, Hold {b['H']}h")
    print("=" * 100)

    def block(sub, groupcol):
        print(f"\n--- {sub} ---")
        print(f"{'':>8} {'n':>5} {'%pos':>6} {'avgROI':>8} {'medROI':>8} {'compROI':>10} {'avgExc':>8}")
        for g, d in bt.groupby(groupcol):
            s = summarize(d["net"].values, d["excess"].values)
            print(f"{str(g):>8} {s['n']:>5} {s['pos'] * 100:5.1f}% {pct(s['avg']):>8} {pct(s['med']):>8} "
                  f"{pct(s['comp'], 1):>10} {pct(s['exc']):>8}")

    block("pro Sektor", "sector")
    block("pro Jahr (Einstieg, Berlin-Zeit)", "year")

    t, p, ci = tstat_bootstrap(bt["excess"].values, N_BOOTSTRAP, RANDOM_SEED)
    t2, p2, ci2 = tstat_bootstrap(bt["net"].values, N_BOOTSTRAP, RANDOM_SEED)
    # Cluster pro Einstiegstag: gleichzeitige Signale in mehreren Sektoren sind nicht unabhängig
    day = bt["entry"].dt.tz_convert(TZ_LOCAL).dt.date
    tc, pc, cic = tstat_bootstrap(bt.groupby(day)["excess"].mean().values, N_BOOTSTRAP, RANDOM_SEED)
    n_cfg = len(results)
    print("\n--- Signifikanz ---")
    print(f"Ø Überrendite vs Sektor: {pct(np.nanmean(bt['excess']))}  t={t:.2f}  p≈{p:.4f}  "
          f"Bootstrap-95%-KI [{pct(ci[0])}, {pct(ci[1])}]")
    print(f"  (Tages-Cluster, {day.nunique()} Tage): t={tc:.2f}  p≈{pc:.4f}  KI [{pct(cic[0])}, {pct(cic[1])}]")
    print(f"Ø Netto-ROI:             {pct(bt['net'].mean())}  t={t2:.2f}  p≈{p2:.4f}  "
          f"Bootstrap-95%-KI [{pct(ci2[0])}, {pct(ci2[1])}]")
    print(f"Multiple Testing: {n_cfg} Konfigurationen -> Bonferroni-Schwelle p < {0.05 / n_cfg:.5f}: "
          f"{'BESTANDEN' if p < 0.05 / n_cfg else 'NICHT bestanden'} (Überrendite).")

    cols = ["sector", "coin", "leader", "lead_ret", "coin_ret", "entry", "exit", "net", "basket", "excess"]
    for sub, d in (("TOP 5 beste Trades", bt.nlargest(TOP_N, "net")), ("TOP 5 schlechteste Trades", bt.nsmallest(TOP_N, "net"))):
        print(f"\n--- {sub} ---")
        for _, r in d[cols].iterrows():
            print(f"{r.sector:7s} {r.coin:12s} Leader {r.leader:6s}({pct(r.lead_ret, 1)}) Coin-LB {pct(r.coin_ret, 1):>7}"
                  f"  {fmt_ts(r.entry)} -> {fmt_ts(r.exit)}  ROI {pct(r.net):>8}  Basket {pct(r.basket):>8}  Exc {pct(r.excess):>8}")

    print("\n--- Sanity-Checks ---")
    r0 = bt.iloc[0]
    if b["variant"] != "C":
        e, x = mk.C[r0.coin].iloc[r0.t], mk.Cff[r0.coin].iloc[r0.t + b["H"]]
        print(f"Erster Trade {r0.coin}: Entry {e:.6g} @ {fmt_ts(r0.entry)}, Exit {x:.6g} @ {fmt_ts(r0.exit)} -> brutto "
              f"{pct(x / e - 1)} (Sim: {pct(r0.gross)})")
    overl = any((g["t"].diff().dropna() < b["H"]).any() for _, g in bt.groupby("sector"))
    print(f"Überlappung pro Sektor: {'JA (Fehler!)' if overl else 'keine'}")
    rnd_exc = [v["random"]["exc"] for v in results.values() if v.get("random")]
    print(f"RANDOM Ø Überrendite über alle Konfigs (sollte ~0 sein): {pct(np.mean(rnd_exc), 3)}")
    return {"t_exc": t, "p_exc": p, "ci_exc": list(ci), "t_exc_dayclust": tc, "p_exc_dayclust": pc,
            "t_net": t2, "p_net": p2, "ci_net": list(ci2)}



if __name__ == "__main__":
    main()
