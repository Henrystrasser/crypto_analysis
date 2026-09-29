from zoneinfo import ZoneInfo

import ccxt
import pandas as pd


# Alle Uhrzeiten/Offsets/Tage in diesem Skript sind deutsche Zeit (Europe/Berlin,
# Sommer-/Winterzeit automatisch), konsistent mit simulation_offset_compute.py.
# ccxt liefert UTC; umgerechnet wird genau einmal beim Laden der Daten.
BERLIN = ZoneInfo("Europe/Berlin")


def berlin_wallclock(day, hour):
  """
  Berliner Ortszeit `hour`:00 am Berliner Kalendertag `day` als tz-aware
  Timestamp (oder None, wenn es diese Uhrzeit an dem Tag nicht gibt).
  DST wie simulation_offset_compute.py: Frühjahr 02:00 existiert nicht ->
  None (kein Trade für diese Stunde an dem Tag); Herbst 02:00 doppelt -> nur
  das erste Auftreten (CEST). Dauern danach = echte Stunden.
  """
  naive = pd.to_datetime(day).normalize() + pd.Timedelta(hours=hour)
  local = naive.tz_localize(BERLIN, ambiguous=True, nonexistent="NaT")
  if pd.isna(local):
    return None
  return local


def fetch_all_ohlcv(exchange, symbol, timeframe, since_timestamp):
  all_ohlcv = []
  limit = 1000
  current_since = since_timestamp

  while True:
    try:
      ohlcv = exchange.fetch_ohlcv(
          symbol, timeframe=timeframe, since=current_since, limit=limit
      )
      if not ohlcv:
        break

      all_ohlcv.extend(ohlcv)
      next_since = ohlcv[-1][0] + 1

      if next_since <= current_since:
        break
      current_since = next_since

      if len(ohlcv) < limit:
        break
    except Exception as e:
      print(f"Fehler beim Laden: {e}")
      break

  return all_ohlcv


# ==========================================
# ANPASSBARE HANDELS-LOGIK FUNKTIONEN
# ==========================================
def custom_buy_condition(row, portfolio_state):
  """Hier kannst du definieren, wann der Bot im Intervall 1 (Morgen) kaufen

  soll. (z.B. row['Pattern_Match'] == True)
  """
  return True


def custom_sell_condition(row, portfolio_state):
  """Hier kannst du definieren, wann der Bot im Intervall 2 (Abend) verkaufen

  soll.
  """
  return True


# ==========================================
# SIMULATIONS-ENGINE (Morgens kaufen, Abends verkaufen)
# ==========================================
def run_simulation(results_df, fee_rate=0.0, initial_cash=10000.0):
  cash = initial_cash
  crypto = 0.0
  position_open = False

  trade_log = []

  # Wir iterieren Tag für Tag (innerhalb desselben Tages findet Kauf und Verkauf statt)
  for i in range(len(results_df)):
    row = results_df.iloc[i]

    # SCHRITT 1: Intervall 1 (Kauf-Zeitpunkt am Morgen, z.B. 06:00 Uhr)
    price_buy = row["Price_1"]
    time_buy = f"{row['Date']} {row['Time_1']}"

    if not position_open and custom_buy_condition(row, {"cash": cash}):
      effective_cash = cash * (1 - fee_rate)  # Gebühr beim Kauf abgezogen
      crypto = effective_cash / price_buy
      cash = 0.0
      position_open = True
      trade_log.append({
          "Time": time_buy,
          "Action": "BUY",
          "Price": price_buy,
          "Cash": cash,
          "Crypto": crypto,
      })

    # SCHRITT 2: Intervall 2 (Verkaufs-Zeitpunkt am Abend, z.B. 18:00 Uhr desselben Tages)
    price_sell = row["Price_2"]
    time_sell = f"{row['Date']} {row['Time_2']}"

    if position_open and custom_sell_condition(row, {"crypto": crypto}):
      gross_cash = crypto * price_sell
      cash = gross_cash * (1 - fee_rate)  # Gebühr beim Verkauf abgezogen
      crypto = 0.0
      position_open = False
      trade_log.append({
          "Time": time_sell,
          "Action": "SELL",
          "Price": price_sell,
          "Cash": cash,
          "Crypto": crypto,
      })

  # Endwert berechnen (falls zum Ende noch position_open)
  final_value = cash
  if position_open:
    final_value = crypto * results_df.iloc[-1]["Price_2"]

  return final_value, pd.DataFrame(trade_log)


# ==========================================
# HAUPTPROGRAMM & KONFIGURATION
# ==========================================
def main():
  exchange = ccxt.binance()
  temp_symbol = "ZEC/USDT"

  # --- HIER ZEITRAUM ANPASSEN ---
  start_date_str = "2022-01-01"  # Start des Backtests (YYYY-MM-DD)
  end_date_str = "2026-12-31"  # Ende des Backtests (YYYY-MM-DD)
  offset_hours = 6
  interval_hours = 12

  # Start-Timestamp für den API-Abruf berechnen
  since_timestamp = (
      int(pd.Timestamp(start_date_str).tz_localize(BERLIN).timestamp() * 1000)
      - 48 * 60 * 60 * 1000
  )

  print(
      f"Lade historische Daten für {temp_symbol} ab dem"
      f" {start_date_str}..."
  )
  ohlcv = fetch_all_ohlcv(exchange, temp_symbol, "1h", since_timestamp)

  df = pd.DataFrame(
      ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
  )
  # Einmalige Umrechnung: ccxt UTC-ms -> tz-aware UTC -> Europe/Berlin.
  df["datetime"] = pd.to_datetime(
      df["timestamp"], unit="ms", utc=True
  ).dt.tz_convert(BERLIN)
  df.set_index("datetime", inplace=True)
  df = df[~df.index.duplicated(keep="first")]
  # Handelstag = Berliner Kalenderdatum (Tagesgrenze = Mitternacht Berlin)
  df["date"] = df.index.date

  results = []
  for date, group in df.groupby("date"):
    t1 = offset_hours % 24
    t2 = (offset_hours + interval_hours) % 24
    # Berliner Wanduhr-Stunden dieses Berliner Tages (DST: Frühjahr 02:00
    # fehlt -> Tag übersprungen; Herbst 02:00 doppelt -> nur erstes Auftreten)
    wanted = [berlin_wallclock(date, h) for h in dict.fromkeys([t1, t2])]
    hour_data = group.loc[[x for x in wanted if x is not None and x in group.index]]

    if len(hour_data) >= 2:
      hour_data = hour_data.sort_index()
      results.append({
          "Date": date,
          "Time_1": hour_data.index[0].strftime("%H:%M %Z"),
          "Price_1": hour_data.iloc[0]["close"],
          "Time_2": hour_data.index[1].strftime("%H:%M %Z"),
          "Price_2": hour_data.iloc[1]["close"],
      })

  result_df = pd.DataFrame(results)

  # --- FILTRIERUNG NACH GEWÜNSCHTEM DATUMSBEREICH ---
  if start_date_str:
    result_df = result_df[
        result_df["Date"] >= pd.to_datetime(start_date_str).date()
    ]
  if end_date_str:
    result_df = result_df[
        result_df["Date"] <= pd.to_datetime(end_date_str).date()
    ]

  if result_df.empty:
    print(
        "Fehler: Keine Daten im gewählten Datumsbereich gefunden oder Bereich"
        " liegt in der Zukunft."
    )
    return

  initial_capital = 10000.0

  print(
      f"\n--- SIMULATION STARTET (Kauf morgens, Verkauf abends; Europe/Berlin) ---"
      f" ({start_date_str} bis {end_date_str})"
  )
  print(f"Startkapital: {initial_capital:,.2f} USDT")
  print(f"Anzahl Handels-Tage im Test: {len(result_df)}")

  # 1. Simulation OHNE Gebühren
  final_val_no_fee, log_no_fee = run_simulation(
      result_df, fee_rate=0.0, initial_cash=initial_capital
  )
  profit_no_fee = final_val_no_fee - initial_capital
  print(f"\n[1] Ohne Gebühren:")
  print(f"Endkapital: {final_val_no_fee:,.2f} USDT")
  print(
      f"Gewinn/Verlust: {profit_no_fee:+,.2f} USDT"
      f" ({(profit_no_fee/initial_capital)*100:.2f}%)"
  )

  # 2. Simulation MIT 0.1% Gebühr pro Trade
  final_val_with_fee, log_with_fee = run_simulation(
      result_df, fee_rate=0.001, initial_cash=initial_capital
  )
  profit_with_fee = final_val_with_fee - initial_capital
  print(f"\n[2] Mit Gebühr (0.1% pro Trade):")
  print(f"Endkapital: {final_val_with_fee:,.2f} USDT")
  print(
      f"Gewinn/Verlust: {profit_with_fee:+,.2f} USDT"
      f" ({(profit_with_fee/initial_capital)*100:.2f}%)"
  )
  print(
      f"Gebühren-Fress-Effekt: {final_val_no_fee - final_val_with_fee:,.2f}"
      " USDT Unterschied"
  )


if __name__ == "__main__":
  main()