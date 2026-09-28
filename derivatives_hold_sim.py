#!/usr/bin/env python3
"""
Simulation: BTC-Perp Derivatives-Stress → Altcoin Spot kaufen → multi-Hold ROI

Event (exogen zum Altcoin-Spot):
  MODE = "funding" (Default):
    FUNDING_POS: Funding-Rate ≥ +FUND_THR  (Default: nur „>“-Signal)
    FUNDING_NEG: Funding-Rate ≤ −FUND_THR
    BOTH: extreme Funding beider Seiten (optional)
    Entry = Funding-Timestamp (calcTime / fundingTime).
  MODE = "oi":
    OI_DROP:  OI-%-Änderung über OI_WINDOW_HOURS ≤ −OI_DROP_PCT (Stress)
    OI_SURGE: OI-% ≥ +OI_SURGE_PCT („>“-Seite; OI_SURGE_PCT > 0 nötig)
    Entry = Timestamp der OI-Bar am Fensterende.
    Bei MODE=oi DIRECTION auf OI_DROP oder OI_SURGE setzen (nicht FUNDING_*).

Datenquellen:
  1) Binance USD-M fapi (/fapi/v1/fundingRate, /futures/data/openInterestHist)
     — FAPI_BASES inkl. fapi.binance.com (kann geo-451 sein).
  2) Fallback data.binance.vision:
     - monthly fundingRate ZIPs
     - daily metrics ZIPs (sum_open_interest)

Coin-ROI weiterhin auf Spot-5m-Kerzen (laggard_common), Non-Overlap wie v0.7.
Liquidationen: bewusst weggelassen (Endpunkt instabil / geo-sensitiv).

Keine Handelsempfehlung — Backtest-Skript.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

import laggard_common as lc

# --- Strategie ---
MODE = "funding"  # "funding" | "oi"
FUTURES_SYMBOL = "BTCUSDT"
DIRECTION = "FUNDING_POS"  # funding: FUNDING_POS|FUNDING_NEG|BOTH; oi: OI_DROP|OI_SURGE (Default = nur >)
# funding is typically ~0.0001 = 0.01%; thr in absolute rate units
FUND_THR = 0.0001  # 0.01% (= typ. Binance-Cap/extrem); ggf. auf 0.00005 senken
OI_WINDOW_HOURS = 4
OI_DROP_PCT = 3.0  # OI fällt ≥ 3% in Fenster
OI_SURGE_PCT = 0.0  # 0 = Surge-Events aus
USE_THRESHOLD_CROSSING = True

SHOW_SEPARATE_DIRECTIONS = False
N_RANDOM_TRADES: Optional[int] = None
RANDOM_SEED = 42

LOOKBACK_DAYS = 390
KLINE_INTERVAL = "5m"
TOP_N = 50
COIN_PAIRS: List[str] = []

FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0
SHOW_TRADES = False

CSV_TRADES = "derivatives_hold_trades.csv"
CSV_SUMMARY = "derivatives_hold_summary.csv"
WRITE_CSV = False  # True → Trades/Summary-CSV schreiben (Platz!)

FAPI_BASES = [
    "https://fapi.binance.com",
    "https://api.binance.com",
]
VISION_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
VISION_DL = "https://data.binance.vision"


def _sync_common() -> Dict[str, int]:
    lc.FEE_BPS = FEE_BPS
    lc.SLIPPAGE_BPS = SLIPPAGE_BPS
    lc.KLINE_INTERVAL = KLINE_INTERVAL
    lc.LOOKBACK_DAYS = LOOKBACK_DAYS
    lc.TOP_N = TOP_N
    lc.RANDOM_SEED = RANDOM_SEED
    lc.SHOW_TRADES = SHOW_TRADES
    lc.SESSION.headers.update({"User-Agent": "derivatives-hold-sim/0.1"})
    return lc.hold_bars_map(KLINE_INTERVAL, lc.HOLD_HORIZONS)


def _fapi_get(path: str, params: Optional[dict] = None, timeout: int = 30):
    last_exc: Optional[Exception] = None
    for base in FAPI_BASES:
        try:
            r = lc.SESSION.get(f"{base}{path}", params=params, timeout=timeout)
            if r.status_code in (418, 429, 451):
                last_exc = requests.HTTPError(f"{r.status_code} {base}", response=r)
                lc.sleep_polite(0.5)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            last_exc = exc
            lc.sleep_polite(0.3)
    if last_exc:
        raise last_exc
    raise RuntimeError("fapi GET fehlgeschlagen")


# -------------------- Funding --------------------


def fetch_funding_fapi(symbol: str, start_ms: int, end_ms: int) -> Optional[pd.DataFrame]:
    rows: List[dict] = []
    cursor = start_ms
    try:
        while cursor < end_ms:
            r = _fapi_get(
                "/fapi/v1/fundingRate",
                params={
                    "symbol": symbol,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            data = r.json()
            if not data:
                break
            for d in data:
                rows.append(
                    {
                        "funding_time": int(d["fundingTime"]),
                        "funding_rate": float(d["fundingRate"]),
                    }
                )
            nxt = int(data[-1]["fundingTime"]) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            lc.sleep_polite(0.15)
            if len(data) < 1000:
                break
    except Exception as exc:  # noqa: BLE001
        print(f"   fapi fundingRate nicht verfügbar: {exc}", flush=True)
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("funding_time").sort_values("funding_time")
    print(f"   Funding via fapi: {len(df)} Punkte", flush=True)
    return df.reset_index(drop=True)


def _list_vision_keys(prefix: str) -> List[str]:
    keys: List[str] = []
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": prefix, "max-keys": 1000}
        if marker:
            params["marker"] = marker
        r = lc.SESSION.get(VISION_S3, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        batch = [
            el.text
            for el in root.findall("s3:Contents/s3:Key", ns)
            if el.text and el.text.endswith(".zip") and "CHECKSUM" not in el.text
        ]
        keys.extend(batch)
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=ns)
        if truncated.lower() != "true" or not batch:
            break
        marker = batch[-1]
    return keys


def fetch_funding_vision(symbol: str, start_ms: int, end_ms: int) -> Optional[pd.DataFrame]:
    """Monatliche Funding-ZIPs von data.binance.vision."""
    prefix = f"data/futures/um/monthly/fundingRate/{symbol}/"
    try:
        keys = _list_vision_keys(prefix)
    except Exception as exc:  # noqa: BLE001
        print(f"   Vision Funding-Liste fehlgeschlagen: {exc}", flush=True)
        return None
    start_ym = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m")
    end_ym = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime("%Y-%m")
    # ein Monat Puffer vor start_ym
    start_buf = (pd.Period(start_ym, freq="M") - 1).strftime("%Y-%m")
    frames = []
    for key in keys:
        # …/BTCUSDT-fundingRate-2026-08.zip
        base = key.rsplit("/", 1)[-1].replace(".zip", "")
        parts = base.split("-")
        # BTCUSDT-fundingRate-YYYY-MM
        if len(parts) < 4:
            continue
        stamp = f"{parts[-2]}-{parts[-1]}"
        if stamp < start_buf or stamp > end_ym:
            continue
        url = f"{VISION_DL}/{key}"
        try:
            r = lc.SESSION.get(url, timeout=60)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    part = pd.read_csv(f)
            frames.append(part)
            lc.sleep_polite(0.1)
        except Exception as exc:  # noqa: BLE001
            print(f"   Vision ZIP skip {key}: {exc}", flush=True)
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # Spalten: calc_time, funding_interval_hours, last_funding_rate
    tcol = "calc_time" if "calc_time" in df.columns else "fundingTime"
    rcol = "last_funding_rate" if "last_funding_rate" in df.columns else "fundingRate"
    out = pd.DataFrame(
        {
            "funding_time": df[tcol].astype(np.int64),
            "funding_rate": df[rcol].astype(float),
        }
    )
    out = (
        out[(out["funding_time"] >= start_ms) & (out["funding_time"] <= end_ms)]
        .drop_duplicates("funding_time")
        .sort_values("funding_time")
        .reset_index(drop=True)
    )
    print(f"   Funding via Vision ZIPs: {len(out)} Punkte", flush=True)
    return out if not out.empty else None


def load_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    df = fetch_funding_fapi(symbol, start_ms, end_ms)
    if df is None or df.empty:
        print("   Fallback: data.binance.vision Funding-Archive …", flush=True)
        df = fetch_funding_vision(symbol, start_ms, end_ms)
    if df is None or df.empty:
        raise SystemExit(
            "Keine Funding-Daten (fapi geo-blockiert und Vision leer). "
            "VPN/anderes Netz oder MODE='oi' versuchen."
        )
    return df


def detect_funding_events(
    df: pd.DataFrame, thr: float
) -> Tuple[List[lc.Event], List[lc.Event]]:
    pos: List[lc.Event] = []
    neg: List[lc.Event] = []
    prev_pos = False
    prev_neg = False
    for _, row in df.iterrows():
        rate = float(row["funding_rate"])
        t = int(row["funding_time"])
        is_pos = rate >= thr
        is_neg = rate <= -thr
        fire_pos = is_pos if not USE_THRESHOLD_CROSSING else (is_pos and not prev_pos)
        fire_neg = is_neg if not USE_THRESHOLD_CROSSING else (is_neg and not prev_neg)
        if fire_pos:
            pos.append(
                lc.Event(
                    start_ms=t,
                    end_ms=t,
                    value=rate,
                    direction="FUNDING_POS",
                )
            )
        if fire_neg:
            neg.append(
                lc.Event(
                    start_ms=t,
                    end_ms=t,
                    value=rate,
                    direction="FUNDING_NEG",
                )
            )
        prev_pos = is_pos
        prev_neg = is_neg
    return pos, neg


# -------------------- Open Interest --------------------


def fetch_oi_fapi(symbol: str, start_ms: int, end_ms: int, period: str = "1h") -> Optional[pd.DataFrame]:
    rows: List[dict] = []
    cursor = start_ms
    try:
        while cursor < end_ms:
            r = _fapi_get(
                "/futures/data/openInterestHist",
                params={
                    "symbol": symbol,
                    "period": period,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 500,
                },
            )
            data = r.json()
            if not data:
                break
            for d in data:
                rows.append(
                    {
                        "ts": int(d["timestamp"]),
                        "oi": float(d["sumOpenInterest"]),
                    }
                )
            nxt = int(data[-1]["timestamp"]) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            lc.sleep_polite(0.15)
            if len(data) < 500:
                break
    except Exception as exc:  # noqa: BLE001
        print(f"   fapi openInterestHist nicht verfügbar: {exc}", flush=True)
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    print(f"   OI via fapi: {len(df)} Punkte", flush=True)
    return df


def fetch_oi_vision(symbol: str, start_ms: int, end_ms: int) -> Optional[pd.DataFrame]:
    """Tägliche metrics-ZIPs → sum_open_interest (~15m)."""
    start_d = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()
    end_d = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).date()
    days = pd.date_range(start=start_d, end=end_d, freq="D")
    frames = []
    misses = 0
    for day in days:
        ds = day.strftime("%Y-%m-%d")
        key = f"data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{ds}.zip"
        url = f"{VISION_DL}/{key}"
        try:
            r = lc.SESSION.get(url, timeout=30)
            if r.status_code != 200:
                misses += 1
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    part = pd.read_csv(f)
            frames.append(part)
            lc.sleep_polite(0.05)
        except Exception:  # noqa: BLE001
            misses += 1
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # create_time like '2026-09-20 00:30:00' (UTC)
    ts = pd.to_datetime(df["create_time"], utc=True)
    # robust ms (unabhängig von datetime64[ns]/[us])
    ts_ms = (
        (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    ).astype(np.int64)
    out = pd.DataFrame(
        {
            "ts": ts_ms,
            "oi": df["sum_open_interest"].astype(float),
        }
    )
    out = (
        out[(out["ts"] >= start_ms) & (out["ts"] <= end_ms)]
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    print(
        f"   OI via Vision metrics: {len(out)} Punkte ({misses} Tage fehlend)",
        flush=True,
    )
    return out if not out.empty else None


def load_oi(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    # period für Fenster: 1h wenn window>=1h
    period = "1h" if OI_WINDOW_HOURS >= 1 else "15m"
    df = fetch_oi_fapi(symbol, start_ms, end_ms, period=period)
    if df is None or df.empty:
        print("   Fallback: Vision daily metrics (OI) …", flush=True)
        df = fetch_oi_vision(symbol, start_ms, end_ms)
    if df is None or df.empty:
        raise SystemExit("Keine OI-Daten verfügbar.")
    return df


def detect_oi_events(df: pd.DataFrame) -> Tuple[List[lc.Event], List[lc.Event]]:
    """OI %-Change über OI_WINDOW_HOURS (approx via time delta)."""
    window_ms = int(OI_WINDOW_HOURS * 3_600_000)
    drops: List[lc.Event] = []
    surges: List[lc.Event] = []
    ts = df["ts"].to_numpy(dtype=np.int64)
    oi = df["oi"].to_numpy(dtype=float)
    prev_drop = False
    prev_surge = False
    j = 0
    for i in range(len(df)):
        target = ts[i] - window_ms
        while j < i and ts[j] < target:
            j += 1
        # j ist erste Bar >= target; nimm j-1 wenn näher, sonst j
        if j == 0:
            continue
        # wähle Index k mit ts[k] möglichst nahe target, k < i
        k = j - 1
        if k < 0 or oi[k] <= 0:
            continue
        chg = (oi[i] / oi[k] - 1.0) * 100.0
        is_drop = chg <= -OI_DROP_PCT
        is_surge = OI_SURGE_PCT > 0 and chg >= OI_SURGE_PCT
        fire_drop = is_drop if not USE_THRESHOLD_CROSSING else (is_drop and not prev_drop)
        fire_surge = is_surge if not USE_THRESHOLD_CROSSING else (is_surge and not prev_surge)
        t = int(ts[i])
        if fire_drop:
            drops.append(
                lc.Event(
                    start_ms=int(ts[k]),
                    end_ms=t,
                    value=float(chg),
                    direction="OI_DROP",
                )
            )
        if fire_surge:
            surges.append(
                lc.Event(
                    start_ms=int(ts[k]),
                    end_ms=t,
                    value=float(chg),
                    direction="OI_SURGE",
                )
            )
        prev_drop = is_drop
        prev_surge = is_surge
    return drops, surges


def main() -> None:
    hold_bars = _sync_common()
    horizon_names = list(lc.HOLD_HORIZONS.keys())
    max_hold_min = max(lc.HOLD_HORIZONS.values())

    print(
        "=== Simulation: Derivatives-Stress (BTC Perp) → Alt Spot → multi-Hold ===\n",
        flush=True,
    )
    end = datetime.now(tz=timezone.utc)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    fetch_end = end + pd.Timedelta(minutes=max_hold_min)
    start_ms, end_ms = lc.utc_ms(start), lc.utc_ms(end)
    fetch_end_ms = lc.utc_ms(fetch_end)

    print(
        f"Zeitraum: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} UTC",
        flush=True,
    )
    print(f"MODE={MODE} | SYMBOL={FUTURES_SYMBOL} | DIRECTION={DIRECTION}", flush=True)
    print(f"Kosten: {lc.cost_note()}\n", flush=True)

    print("1) Derivatives-Events …", flush=True)
    if MODE == "funding":
        fdf = load_funding(FUTURES_SYMBOL, start_ms, end_ms)
        a, b = detect_funding_events(fdf, FUND_THR)
        label_a, label_b = "FUNDING_POS", "FUNDING_NEG"
        value_col = "funding_rate"
        thr_note = f"Funding |rate| ≥ {FUND_THR}"
        print(f"   {len(a)} POS | {len(b)} NEG", flush=True)
    elif MODE == "oi":
        odf = load_oi(FUTURES_SYMBOL, start_ms, end_ms)
        a, b = detect_oi_events(odf)
        label_a, label_b = "OI_DROP", "OI_SURGE"
        value_col = "oi_chg_pct"
        thr_note = f"OI {OI_WINDOW_HOURS}h drop ≤ −{OI_DROP_PCT}%"
        print(f"   {len(a)} DROP | {len(b)} SURGE", flush=True)
    else:
        raise SystemExit("MODE muss 'funding' oder 'oi' sein.")

    both = sorted(a + b, key=lambda e: e.end_ms)
    if DIRECTION == label_a:
        primary, primary_label = a, label_a
    elif DIRECTION == label_b:
        primary, primary_label = b, label_b
    elif DIRECTION == "BOTH":
        primary, primary_label = both, "BOTH"
    else:
        raise SystemExit(
            f"DIRECTION={DIRECTION!r} ungültig für MODE={MODE} — "
            f"erwartet {label_a}|{label_b}|BOTH"
        )

    if not primary:
        raise SystemExit("Keine Derivatives-Events — Schwellen lockern.")

    print("2) Coin-Universum …", flush=True)
    universe = lc.resolve_universe(COIN_PAIRS or None, TOP_N)
    print(f"   {len(universe)} Paare\n", flush=True)

    lc.reset_csv_files(*( (CSV_TRADES, CSV_SUMMARY) if WRITE_CSV else () ))
    header_pending = True
    summary_rows: List[dict] = []

    print(f"3) Simulation ({len(universe)} Coins)\n", flush=True)
    for i, pair in enumerate(universe, 1):
        print(f"--- [{i}/{len(universe)}] lade {pair} …", flush=True)
        try:
            df = lc.fetch_klines(pair, KLINE_INTERVAL, start_ms, fetch_end_ms)
        except Exception as exc:  # noqa: BLE001
            print(f"    Fehler: {exc}\n", flush=True)
            continue
        if df.empty:
            print("    keine Kerzen\n", flush=True)
            continue

        bh = lc.buy_and_hold_roi(df, start_ms, end_ms)
        coin_seed = lc.coin_seed_for(pair, RANDOM_SEED)
        rng = np.random.default_rng(coin_seed)

        batch: List[Tuple[str, List[dict], Dict[str, dict]]] = []
        pt = lc.simulate_from_events(
            pair, df, primary, primary_label, hold_bars, value_col=value_col
        )
        pbh = lc.summarize_by_horizon(pt)
        batch.append((primary_label, pt, pbh))

        if SHOW_SEPARATE_DIRECTIONS and primary_label == "BOTH":
            if a:
                t = lc.simulate_from_events(
                    pair, df, a, label_a, hold_bars, value_col=value_col
                )
                batch.append((label_a, t, lc.summarize_by_horizon(t)))
            if b:
                t = lc.simulate_from_events(
                    pair, df, b, label_b, hold_bars, value_col=value_col
                )
                batch.append((label_b, t, lc.summarize_by_horizon(t)))

        n_rand = N_RANDOM_TRADES if N_RANDOM_TRADES is not None else len(primary)
        rand_trades = lc.simulate_random(
            pair, df, n_rand, start_ms, end_ms, rng, hold_bars, value_col=value_col
        )
        rand_by_h = lc.summarize_by_horizon(rand_trades)
        batch.append(("RANDOM", rand_trades, rand_by_h))

        for strategy, trades, by_h in batch:
            lc.print_coin_result(
                pair, strategy, trades, by_h, bh,
                thr_note if strategy != "RANDOM" else "Zufalls-Käufe",
                horizon_names, SHOW_TRADES,
                value_col=value_col, value_header="Val",
            )
            if WRITE_CSV:
                lc.append_csv(CSV_TRADES if WRITE_CSV else None, trades, write_header=header_pending)
                header_pending = False
            summary_rows.append(
                lc.make_summary_row(
                    pair, strategy, trades, by_h, bh, coin_seed, rand_by_h,
                    extra={
                        "mode": MODE,
                        "futures_symbol": FUTURES_SYMBOL,
                        "fund_thr": FUND_THR,
                        "oi_window_hours": OI_WINDOW_HOURS,
                        "oi_drop_pct": OI_DROP_PCT,
                    },
                )
            )
        del df

    if summary_rows:
        if WRITE_CSV:
            pd.DataFrame(summary_rows).to_csv(CSV_SUMMARY, index=False)
        strategies = [primary_label, "RANDOM"]
        if SHOW_SEPARATE_DIRECTIONS and primary_label == "BOTH":
            strategies = ["BOTH", label_a, label_b, "RANDOM"]
        lc.print_rankings(summary_rows, strategies, horizon_names)
        if WRITE_CSV:
            print(f"\nTrades-CSV:   {CSV_TRADES}")
            print(f"Summary-CSV:  {CSV_SUMMARY}")
        else:
            print("\nKeine CSV geschrieben (WRITE_CSV=False).", flush=True)

    print(
        "\nFertig. Event = BTC-Perp Funding/OI (exogen zu Alt-Spot).\n"
        "Entry = Funding-/OI-Timestamp; Spot-ROI danach gemessen.\n"
        "Häufige Events → Non-Overlap kritisch für ehrliche Compounds.\n"
        "Keine Handelsempfehlung.\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
