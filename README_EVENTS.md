# Event-Hold-Simulationen (Laggard-Research)

Keine Handelsempfehlung — reine Backtests. Methodik angelehnt an `btc_rise_hold_sim.py` v0.7.

Gemeinsames Modul: `laggard_common.py` (Horizonte, Fees, Spot-Klines, Non-Overlap, RANDOM, Buy&Hold, Stats).

## Horizonte & Kosten (alle Skripte)

| Hold | Dauer |
|------|-------|
| 5m … 192h | identisch zu BTC-Skript |

- `FEE_BPS=7.5` je Seite (Kauf+Verkauf), optional `SLIPPAGE_BPS`
- Non-Overlap: Cooldown = Hold-Länge; Kauf muss **strictly after** `last_kept + H`
- Baseline: RANDOM (`n = n_events`, Seed 42 + Coin-Salt), Buy&Hold über LOOKBACK
- Stats: `n_raw/n_used`, `%pos`, Ø, Compound, `vs_bh`, `vs_rand`

**vs_rand lesen:** Compound der Event-Strategie minus Compound gleich vieler Zufalls-Käufe. Positiv = besser als Zufall bei gleicher Trade-Anzahl (nach Non-Overlap). Kein Signifikanztest.

## 1) BTC-Move (bestehend)

`btc_rise_hold_sim.py` — Event aus BTC-Spot-Fenster; Entry am Fensterende.

## 2) ETF-Flows — `etf_flow_hold_sim.py`

| | |
|--|--|
| **Event** | US Spot BTC ETF Tages-Nettoflow ≥ +Thr (INFLOW) oder ≤ −Thr (OUTFLOW) |
| **Default Thr** | `FLOW_THRESHOLD_USD = 500e6` (oder Perzentil) |
| **Entry** | Erste 5m-Kerze ≥ **`ENTRY_HOUR_UTC=20`** am **Flow-Kalendertag** (≈ US Cash Close; DST-neutraler Kompromiss) |
| **Warum clean** | Flow ist institutionelles Spot-ETF-Signal, **nicht** aus dem Altcoin-Preis abgeleitet |

**Daten:**

1. Lokal `etf_btc_spot_flows.csv` (`date,net_flow_usd`)
2. Fetch Farside (`bitcoin-etf-flow-all-data`, Total Mio. → USD)
3. `ALLOW_SYNTHETIC_FLOWS=True` nur Demo (Default False)

Helper: `python scripts/fetch_etf_flows_example.py`

## 3) TradFi — `tradfi_hold_sim.py`

| | |
|--|--|
| **MODE=equity** | `SYMBOL=QQQ` (oder SPY): Tagesreturn ≥ +`EQ_THRESHOLD_PCT` (RISK_ON) / ≤ −Thr (RISK_OFF); Default 1.5% |
| **MODE=yield** | `^TNX` Tagesänderung ≥ `YIELD_THRESHOLD_BPS` (Default 8 bps) |
| **Entry** | Erste 5m-Kerze ≥ `ENTRY_HOUR_UTC=20` am Event-Tag |
| **Warum clean** | Equities/Yields sind TradFi-Risikoproxies, unabhängig vom Alt-Pfad |

Daten: `yfinance` (in `requirements.txt`).

## 4) Derivatives — `derivatives_hold_sim.py`

| | |
|--|--|
| **MODE=funding** | BTCUSDT Perp Funding ≥ +`FUND_THR` / ≤ −Thr; Entry = Funding-Timestamp |
| **MODE=oi** | OI-% über `OI_WINDOW_HOURS` ≤ −`OI_DROP_PCT` (Stress); optional Surge |
| **Warum clean** | Futures-Stress auf **BTC-Perp**; ROI auf **Alt-Spot** danach — kein Lookahead im Coin |

Daten: fapi → Fallback `data.binance.vision` (Funding-Monats-ZIPs / daily metrics OI). Liquidationen bewusst weggelassen.

## Knobs zuerst tunen

1. **Schwellen:** `FLOW_THRESHOLD_USD`, `EQ_THRESHOLD_PCT`, `FUND_THR` (Default 0.0001) / `OI_DROP_PCT` (zu viele Events → Non-Overlap schneidet hart)
2. **`ENTRY_HOUR_UTC`** (ETF/TradFi): 20 vs 21 vs 0 (nächster UTC-Tag)
3. **`LOOKBACK_DAYS`**, **`TOP_N`** / `COIN_PAIRS=['ETHUSDT']` für Smoke-Tests
4. **`DIRECTION`**: Default nur „>“-Seite (`INFLOW` / `RISK_ON` / `FUNDING_POS`); `BOTH` / Gegenrichtung optional; `SHOW_SEPARATE_DIRECTIONS`
5. Derivatives: bei fapi-451 Vision-Fallback; aktueller Monat Funding ggf. unvollständig bis Monats-ZIP da ist

## Start

```bash
cd /path/to/crypto-laggard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional Flows cachen
python scripts/fetch_etf_flows_example.py

# Smoke (1 Coin)
# In jedem Skript kurz setzen: COIN_PAIRS = ['ETHUSDT']
python etf_flow_hold_sim.py
python tradfi_hold_sim.py
python derivatives_hold_sim.py
```

Ausgabe standardmäßig nur Konsole (`WRITE_CSV=False`).
