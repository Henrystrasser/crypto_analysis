#!/usr/bin/env python3
"""
Simulation: US Spot BTC ETF Netto-Flow (Tag) → Altcoin kaufen → multi-Hold ROI

Event (exogen zum Altcoin-Preis):
  INFLOW:  Tages-Nettoflow ≥ +FLOW_THRESHOLD_USD  (Default: nur „>“-Signal)
  OUTFLOW: Tages-Nettoflow ≤ −FLOW_THRESHOLD_USD
  BOTH:    eines von beiden (optional)

Einstiegsuhrzeit (dokumentiert, fest):
  FLOW_DATE = Kalendertag des Flows (nicht Publikationsmorgen).
  Entry = erste 5m-Spot-Kerze at/after ENTRY_HOUR_UTC am FLOW_DATE.
  Default ENTRY_HOUR_UTC = 20  (≈ US Cash Close EST/EDT-nah; DST-neutraler
  Kompromiss 20:00 UTC). Holds laufen in Coin-5m-Kerzen ab dieser Bar.

Zeitzone: Die Flow-Daten sind externe Tagesdaten (US-Handelstag, UTC-/US-
  basiert). Deshalb bleiben Tageszuordnung und Einstieg bewusst in UTC
  (ENTRY_HOUR_UTC, sonst Lookahead/Verschiebung gegenüber US-Close). Nur die
  ANZEIGE aller Zeiten ist deutsche Zeit (Europe/Berlin, CET/CEST):
  20:00 UTC = 22:00 CEST (Sommer) bzw. 21:00 CET (Winter).

Datenquellen (Reihenfolge):
  1) Lokale CSV `etf_btc_spot_flows.csv` (Spalten: date, net_flow_usd;
     date = YYYY-MM-DD; net_flow_usd in USD, z.B. 5e8 für +500 Mio).
  2) Fetch Farside (bitcoin-etf-flow-all-data) — Total in Mio. USD → *1e6.
  3) Wenn beides fehlt und ALLOW_SYNTHETIC_FLOWS=True: synthetische Demo-Daten.
     Default False → klarer Exit mit Hinweis.

Optional BTC_CONFIRM_FILTER (Default False): zusätzlich |BTC daily ret| Filter —
  absichtlich AUS, damit das Signal rein flow-basiert bleibt.

Methodik wie btc_rise_hold_sim.py v0.7 (laggard_common): gleiche Horizonte,
Fees, Non-Overlap, RANDOM, Buy&Hold.

Keine Handelsempfehlung — Backtest-Skript.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import laggard_common as lc

# --- Strategie ---
DIRECTION = "INFLOW"  # INFLOW | OUTFLOW | BOTH  (Default = nur >)
FLOW_THRESHOLD_USD = 500_000_000.0  # ±500 Mio. USD
USE_PERCENTILE_THRESHOLD = False
PERCENTILE = 90.0  # nur wenn USE_PERCENTILE_THRESHOLD
ENTRY_HOUR_UTC = 20  # bewusst UTC (US-Flow-Tag); Anzeige in Berliner Zeit
ALLOW_SYNTHETIC_FLOWS = False
BTC_CONFIRM_FILTER = False  # optional, Default aus
BTC_CONFIRM_PCT = 0.5  # |BTC 1d return| ≥ x% am Flow-Tag

SHOW_SEPARATE_DIRECTIONS = False
N_RANDOM_TRADES: Optional[int] = None
RANDOM_SEED = 42

LOOKBACK_DAYS = 90
KLINE_INTERVAL = "5m"
TOP_N = 50
COIN_PAIRS: List[str] = []

FEE_BPS = 7.5
SLIPPAGE_BPS = 0.0
SHOW_TRADES = False

CSV_FLOWS = "etf_btc_spot_flows.csv"  # Eingabe (lesen OK)
SAVE_FLOW_CACHE = False  # True → Fetch nach CSV_FLOWS speichern
CSV_TRADES = "etf_flow_hold_trades.csv"
CSV_SUMMARY = "etf_flow_hold_summary.csv"
WRITE_CSV = False  # True → Trades/Summary-CSV schreiben (Platz!)

FARSIDE_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"


def _entry_berlin_note() -> str:
    """ENTRY_HOUR_UTC in deutscher Zeit (Sommer/Winter) — nur Anzeige."""
    summer = lc.fmt_berlin(datetime(2026, 7, 1, ENTRY_HOUR_UTC, tzinfo=timezone.utc), "%H:%M %Z")
    winter = lc.fmt_berlin(datetime(2026, 1, 15, ENTRY_HOUR_UTC, tzinfo=timezone.utc), "%H:%M %Z")
    return f"= {summer} / {winter}"


def _sync_common() -> Dict[str, int]:
    lc.FEE_BPS = FEE_BPS
    lc.SLIPPAGE_BPS = SLIPPAGE_BPS
    lc.KLINE_INTERVAL = KLINE_INTERVAL
    lc.LOOKBACK_DAYS = LOOKBACK_DAYS
    lc.TOP_N = TOP_N
    lc.RANDOM_SEED = RANDOM_SEED
    lc.SHOW_TRADES = SHOW_TRADES
    lc.SESSION.headers.update({"User-Agent": "etf-flow-hold-sim/0.1"})
    return lc.hold_bars_map(KLINE_INTERVAL, lc.HOLD_HORIZONS)


def write_template_csv(path: str = CSV_FLOWS) -> None:
    """Schreibt eine leere Template-CSV mit Header + Kommentar-Hinweise."""
    sample = (
        "# US Spot BTC ETF daily NET flow (USD). Replace with Farside/SoSoValue.\n"
        "# date = FLOW calendar day (YYYY-MM-DD); net_flow_usd = sum of all spot ETFs.\n"
        "date,net_flow_usd\n"
        "2026-01-02,250000000\n"
        "2026-01-03,-180000000\n"
    )
    Path(path).write_text(sample, encoding="utf-8")
    print(f"Template geschrieben: {path}", flush=True)


def _parse_flow_number(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().replace(",", "")
    if s in ("", "-", "—", "nan", "None"):
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def load_flows_csv(path: str = CSV_FLOWS) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = cols.get("date")
    flow_col = cols.get("net_flow_usd") or cols.get("netflow") or cols.get("flow")
    if not date_col or not flow_col:
        raise SystemExit(
            f"{path}: braucht Spalten date, net_flow_usd (gefunden: {list(df.columns)})"
        )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], utc=False).dt.normalize(),
            "net_flow_usd": df[flow_col].map(_parse_flow_number),
        }
    ).dropna()
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    print(f"   Flows aus CSV: {len(out)} Tage ({path})", flush=True)
    return out


def fetch_farside_flows() -> Optional[pd.DataFrame]:
    """Farside Total-Spalte: Millionen USD → USD."""
    try:
        r = lc.SESSION.get(FARSIDE_URL, timeout=60)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
    except Exception as exc:  # noqa: BLE001
        print(f"   Farside-Fetch fehlgeschlagen: {exc}", flush=True)
        return None
    if not tables:
        return None
    t = tables[0]
    # Erste + letzte Spalte (Date, Total)
    date_s = t.iloc[:, 0]
    total_s = t.iloc[:, -1]
    rows = []
    for d, v in zip(date_s, total_s):
        if pd.isna(d) or str(d).strip().lower() in ("total", "nan", ""):
            continue
        try:
            dt = pd.to_datetime(str(d), dayfirst=True)
        except Exception:  # noqa: BLE001
            continue
        num = _parse_flow_number(v)
        if num is None:
            continue
        rows.append({"date": dt.normalize(), "net_flow_usd": num * 1_000_000.0})
    if not rows:
        return None
    out = (
        pd.DataFrame(rows)
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    print(f"   Flows von Farside: {len(out)} Tage", flush=True)
    # Cache lokal
    try:
        if SAVE_FLOW_CACHE:
            out.assign(date=out["date"].dt.strftime("%Y-%m-%d")).to_csv(
                CSV_FLOWS, index=False
            )
            print(f"   gespeichert → {CSV_FLOWS}", flush=True)
        else:
            print("   Flow-Cache nicht geschrieben (SAVE_FLOW_CACHE=False).", flush=True)
    except OSError as exc:
        print(f"   CSV-Cache übersprungen: {exc}", flush=True)
    return out


def synthetic_flows(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Nur für Strukturtests — KEINE echten Flows."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range(start=start, end=end, freq="B")
    flows = rng.normal(loc=50e6, scale=400e6, size=len(days))
    # ein paar klare Extremtage
    if len(days) > 10:
        flows[5] = 800e6
        flows[12] = -700e6
        flows[-3] = 600e6
    print(
        "   WARNUNG: SYNTHETISCHE Flows (ALLOW_SYNTHETIC_FLOWS=True) — "
        "nur Demo, durch echte Farside/SoSoValue-Daten ersetzen!",
        flush=True,
    )
    return pd.DataFrame({"date": days.normalize(), "net_flow_usd": flows})


def load_or_fetch_flows(start: datetime, end: datetime) -> pd.DataFrame:
    flows = load_flows_csv(CSV_FLOWS)
    if flows is None:
        print("   keine lokale CSV — versuche Farside …", flush=True)
        flows = fetch_farside_flows()
    if flows is None:
        if ALLOW_SYNTHETIC_FLOWS:
            flows = synthetic_flows(pd.Timestamp(start.date()), pd.Timestamp(end.date()))
        else:
            write_template_csv(CSV_FLOWS)
            raise SystemExit(
                "Keine ETF-Flow-Daten.\n"
                f"  • Lege {CSV_FLOWS} an (date,net_flow_usd) — Template geschrieben,\n"
                "  • oder stelle Netz-Zugriff auf Farside sicher,\n"
                "  • oder setze ALLOW_SYNTHETIC_FLOWS=True (nur Demo)."
            )
    # start/end sind UTC-aware; Flow-Tage sind externe UTC-/US-Tage ->
    # Filter bewusst über das UTC-Datum (nicht auf Berliner Tage umstellen).
    mask = (flows["date"] >= pd.Timestamp(start.date())) & (
        flows["date"] <= pd.Timestamp(end.date())
    )
    return flows.loc[mask].reset_index(drop=True)


def resolve_threshold(flows: pd.DataFrame) -> float:
    if not USE_PERCENTILE_THRESHOLD:
        return float(FLOW_THRESHOLD_USD)
    abs_f = flows["net_flow_usd"].abs()
    thr = float(np.nanpercentile(abs_f, PERCENTILE))
    print(
        f"   Perzentil-Schwelle P{PERCENTILE:g}(|flow|) = {thr/1e6:.1f} Mio USD",
        flush=True,
    )
    return thr


def detect_flow_events(
    flows: pd.DataFrame,
    thr: float,
) -> Tuple[List[lc.Event], List[lc.Event]]:
    inflows: List[lc.Event] = []
    outflows: List[lc.Event] = []
    # Threshold-Crossing: nicht jeden Tag feuern solange |flow| über Schwelle,
    # sondern nur wenn Vortag die Bedingung NICHT erfüllte.
    prev_in = False
    prev_out = False
    for _, row in flows.iterrows():
        v = float(row["net_flow_usd"])
        d = row["date"]
        # Externe Tagesdaten (US-Flow-Tag): Event-Zuordnung bleibt UTC,
        # nur die Anzeige ist Berliner Zeit.
        entry = lc.entry_ms_on_flow_day(d, ENTRY_HOUR_UTC, tz=timezone.utc)
        day_start = lc.entry_ms_on_flow_day(d, 0, tz=timezone.utc)

        is_in = v >= thr
        is_out = v <= -thr
        if is_in and not prev_in:
            inflows.append(
                lc.Event(
                    start_ms=day_start,
                    end_ms=entry,
                    value=v,
                    direction="INFLOW",
                    meta={"flow_date": pd.Timestamp(d).strftime("%Y-%m-%d")},
                )
            )
        if is_out and not prev_out:
            outflows.append(
                lc.Event(
                    start_ms=day_start,
                    end_ms=entry,
                    value=v,
                    direction="OUTFLOW",
                    meta={"flow_date": pd.Timestamp(d).strftime("%Y-%m-%d")},
                )
            )
        prev_in = is_in
        prev_out = is_out
    return inflows, outflows


def optional_btc_confirm(
    events: List[lc.Event], start_ms: int, end_ms: int
) -> List[lc.Event]:
    if not BTC_CONFIRM_FILTER or not events:
        return events
    btc = lc.fetch_klines("BTCUSDT", "1d", start_ms - 2 * 86_400_000, end_ms)
    if btc.empty:
        print("   BTC-Confirm übersprungen (keine 1d-Daten)", flush=True)
        return events
    btc["ret"] = btc["close"].pct_change() * 100.0
    # BTC-1d-Kerzen (Binance: UTC-Tage) passen zum UTC-Flow-Tag -> bewusst UTC.
    btc["day"] = pd.to_datetime(btc["open_time"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    by_day = dict(zip(btc["day"], btc["ret"]))
    kept = []
    for e in events:
        day = (e.meta or {}).get("flow_date")
        ret = by_day.get(day)
        if ret is None or abs(float(ret)) < BTC_CONFIRM_PCT:
            continue
        kept.append(e)
    print(
        f"   BTC-Confirm |ret|≥{BTC_CONFIRM_PCT}%: {len(kept)}/{len(events)} Events",
        flush=True,
    )
    return kept


def main() -> None:
    hold_bars = _sync_common()
    horizon_names = list(lc.HOLD_HORIZONS.keys())
    max_hold_min = max(lc.HOLD_HORIZONS.values())

    print(
        "=== Simulation: BTC Spot-ETF Flow → Coin kaufen → multi-Hold ROI ===\n",
        flush=True,
    )
    end = datetime.now(tz=timezone.utc)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    fetch_end = end + pd.Timedelta(minutes=max_hold_min)
    start_ms, end_ms = lc.utc_ms(start), lc.utc_ms(end)
    fetch_end_ms = lc.utc_ms(fetch_end)

    print(
        f"Zeitraum: {lc.fmt_berlin(start)} → {lc.fmt_berlin(end)} (Europe/Berlin)",
        flush=True,
    )
    print(
        f"Entry: {ENTRY_HOUR_UTC}:00 UTC ({_entry_berlin_note()}) am Flow-Kalendertag | "
        f"Richtung={DIRECTION} | Schwelle=±{FLOW_THRESHOLD_USD/1e6:.0f} Mio USD",
        flush=True,
    )
    print(f"Kosten: {lc.cost_note()}", flush=True)
    print(
        "Non-overlap: 1 Slot, Cooldown = Hold-Länge\n",
        flush=True,
    )

    print("1) ETF-Flows laden …", flush=True)
    flows = load_or_fetch_flows(start, end)
    if flows.empty:
        raise SystemExit("Keine Flow-Zeilen im LOOKBACK-Fenster.")
    thr = resolve_threshold(flows)
    inflows, outflows = detect_flow_events(flows, thr)
    inflows = optional_btc_confirm(inflows, start_ms, end_ms)
    outflows = optional_btc_confirm(outflows, start_ms, end_ms)
    both = sorted(inflows + outflows, key=lambda e: e.end_ms)
    print(
        f"   {len(flows)} Flow-Tage | {len(inflows)} INFLOW | "
        f"{len(outflows)} OUTFLOW | {len(both)} BOTH",
        flush=True,
    )
    if DIRECTION == "INFLOW":
        primary = inflows
        primary_label = "INFLOW"
    elif DIRECTION == "OUTFLOW":
        primary = outflows
        primary_label = "OUTFLOW"
    elif DIRECTION == "BOTH":
        primary = both
        primary_label = "BOTH"
    else:
        raise SystemExit(
            f"DIRECTION={DIRECTION!r} ungültig — erwartet INFLOW|OUTFLOW|BOTH"
        )
    if not primary:
        raise SystemExit("Keine Events — Schwelle/LOOKBACK/CSV prüfen.")

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
        primary_trades = lc.simulate_from_events(
            pair, df, primary, primary_label, hold_bars, value_col="net_flow_usd"
        )
        primary_by_h = lc.summarize_by_horizon(primary_trades)
        batch.append((primary_label, primary_trades, primary_by_h))

        if SHOW_SEPARATE_DIRECTIONS and DIRECTION == "BOTH":
            if inflows:
                t = lc.simulate_from_events(
                    pair, df, inflows, "INFLOW", hold_bars, value_col="net_flow_usd"
                )
                batch.append(("INFLOW", t, lc.summarize_by_horizon(t)))
            if outflows:
                t = lc.simulate_from_events(
                    pair, df, outflows, "OUTFLOW", hold_bars, value_col="net_flow_usd"
                )
                batch.append(("OUTFLOW", t, lc.summarize_by_horizon(t)))

        n_rand = N_RANDOM_TRADES if N_RANDOM_TRADES is not None else len(primary)
        rand_trades = lc.simulate_random(
            pair, df, n_rand, start_ms, end_ms, rng, hold_bars, value_col="net_flow_usd"
        )
        rand_by_h = lc.summarize_by_horizon(rand_trades)
        batch.append(("RANDOM", rand_trades, rand_by_h))

        trigger = (
            f"ETF net flow |≥| {thr/1e6:.0f}M USD @ {ENTRY_HOUR_UTC}:00 UTC "
            f"({_entry_berlin_note()})"
        )
        for strategy, trades, by_h in batch:
            lc.print_coin_result(
                pair, strategy, trades, by_h, bh,
                trigger if strategy != "RANDOM" else "Zufalls-Käufe",
                horizon_names, SHOW_TRADES,
                value_col="net_flow_usd", value_header="Flow$",
            )
            if WRITE_CSV:
                lc.append_csv(CSV_TRADES if WRITE_CSV else None, trades, write_header=header_pending)
                header_pending = False
            summary_rows.append(
                lc.make_summary_row(
                    pair, strategy, trades, by_h, bh, coin_seed, rand_by_h,
                    extra={
                        "flow_threshold_usd": thr,
                        "entry_hour_utc": ENTRY_HOUR_UTC,
                        "direction_mode": DIRECTION,
                    },
                )
            )
        del df

    if summary_rows:
        if WRITE_CSV:
            pd.DataFrame(summary_rows).to_csv(CSV_SUMMARY, index=False)
        strategies = [primary_label, "RANDOM"]
        if SHOW_SEPARATE_DIRECTIONS and DIRECTION == "BOTH":
            strategies = ["BOTH", "INFLOW", "OUTFLOW", "RANDOM"]
        lc.print_rankings(summary_rows, strategies, horizon_names)
        if WRITE_CSV:
            print(f"\nTrades-CSV:   {CSV_TRADES}")
            print(f"Summary-CSV:  {CSV_SUMMARY}")
        else:
            print("\nKeine CSV geschrieben (WRITE_CSV=False).", flush=True)

    print(
        "\nFertig. Event = US Spot BTC ETF Tages-Nettoflow (exogen).\n"
        f"Entry = erste 5m-Bar ≥ {ENTRY_HOUR_UTC}:00 UTC ({_entry_berlin_note()}) am Flow-Tag.\n"
        "Alle angezeigten Zeiten: deutsche Zeit (Europe/Berlin, CET/CEST).\n"
        "vs Rand = Compound − RANDOM; vs BH = Compound − Buy&Hold.\n"
        "Keine Handelsempfehlung — reine Simulation.\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
