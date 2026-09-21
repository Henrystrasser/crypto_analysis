import ccxt
import pandas as pd


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
  """Hier kannst du definieren, wann der Bot kaufen soll."""
  return True


def custom_sell_condition(row, portfolio_state):
  """Hier kannst du definieren, wann der Bot verkaufen soll."""
  return True


# ==========================================
# SIMULATIONS-ENGINE (Angepasst auf chronologischen Buy -> Sell Ablauf)
# ==========================================
def run_simulation(results_df, fee_rate=0.0, initial_cash=10000.0):
  cash = initial_cash
  crypto = 0.0
  position_open = False

  trade_log = []

  for i in range(len(results_df)):
    row = results_df.iloc[i]

    # SCHRITT 1: Kauf zum Start-Zeitpunkt (Time_Buy)
    price_buy = row["Price_Buy"]
    time_buy = f"{row['Date_Buy']} {row['Time_Buy']}"

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

    # SCHRITT 2: Verkauf zum Ziel-Zeitpunkt 12h später (Time_Sell)
    price_sell = row["Price_Sell"]
    time_sell = f"{row['Date_Sell']} {row['Time_Sell']}"

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
    final_value = crypto * results_df.iloc[-1]["Price_Sell"]

  return final_value, pd.DataFrame(trade_log)


def generate_result_df(df, offset_hours, interval_hours, start_date_str, end_date_str):
  results = []
  
  # Wir iterieren durch alle eindeutigen Tage im DataFrame
  dates = df["date"].unique()
  
  for d in dates:
    # Datum in Timestamp umwandeln für präzise Stundenabfragen
    current_dt = pd.to_datetime(d) + pd.Timedelta(hours=offset_hours)
    target_sell_dt = current_dt + pd.Timedelta(hours=interval_hours)

    # Prüfen, ob beide Zeitpunkte (Kauf und Verkauf) in unserem DataFrame existieren
    if current_dt in df.index and target_sell_dt in df.index:
      price_buy = df.loc[current_dt, "close"]
      price_sell = df.loc[target_sell_dt, "close"]

      results.append({
          "Date_Buy": current_dt.strftime("%Y-%m-%d"),
          "Time_Buy": current_dt.strftime("%H:%M"),
          "Price_Buy": price_buy,
          "Date_Sell": target_sell_dt.strftime("%Y-%m-%d"),
          "Time_Sell": target_sell_dt.strftime("%H:%M"),
          "Price_Sell": price_sell,
      })

  result_df = pd.DataFrame(results)

  if start_date_str and not result_df.empty:
    result_df = result_df[
        result_df["Date_Buy"] >= start_date_str
    ]
  if end_date_str and not result_df.empty:
    result_df = result_df[
        result_df["Date_Buy"] <= end_date_str
    ]

  return result_df


# ==========================================
# HAUPTPROGRAMM & KONFIGURATION
# ==========================================
def main():
  exchange = ccxt.binance()
  temp_symbol = "SOL/USDT"

  # --- HIER ZEITRAUM ANPASSEN ---
  start_date_str = "2026-09-15"  # Start des Backtests (YYYY-MM-DD)
  end_date_str = "2026-09-30"    # Ende des Backtests (YYYY-MM-DD)
  interval_hours = 12
  fee_rate = 0.0005              # 0.02% Gebühr (z.B. Futures Maker)
  initial_capital = 10000.0

  # --- STEUERUNG FÜR OFFSET-VERGLEICH ---
  COMPARE_OFFSETS = True          # Auf True stellen, um mehrere Offsets zu testen
  offsets_to_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

  # Start-Timestamp für den API-Abruf berechnen (etwas Puffer nach vorne)
  since_timestamp = (
      int(pd.Timestamp(start_date_str).timestamp() * 1000)
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
  df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
  df.set_index("datetime", inplace=True)
  df = df[~df.index.duplicated(keep="first")]
  df["date"] = df.index.date

  if COMPARE_OFFSETS:
    print(f"\n==========================================")
    print(f" STARTE OFFSET-VERGLEICH FÜR: {offsets_to_test}")
    print(f"==========================================")
    
    comparison_results = []

    for offset in offsets_to_test:
      result_df = generate_result_df(df, offset, interval_hours, start_date_str, end_date_str)
      if result_df.empty:
        continue

      final_val, _ = run_simulation(result_df, fee_rate=fee_rate, initial_cash=initial_capital)
      profit = final_val - initial_capital
      roi = (profit / initial_capital) * 100

      sell_hour = (offset + interval_hours) % 24
      comparison_results.append({
          "Offset (Stunde)": offset,
          "Kauf-Uhrzeit": f"{offset:02d}:00",
          "Verkauf-Uhrzeit": f"{sell_hour:02d}:00",
          "Endkapital (USDT)": final_val,
          "Gewinn/Verlust": profit,
          "ROI (%)": roi
      })

    comp_df = pd.DataFrame(comparison_results)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print("\nERGEBNIS-ÜBERSICHT:")
    print(comp_df.to_string(index=False))
    print("==========================================\n")

  else:
    # Einzelner Standard-Lauf mit einem festen Offset
    offset_hours = 0
    result_df = generate_result_df(df, offset_hours, interval_hours, start_date_str, end_date_str)

    if result_df.empty:
      print("Fehler: Keine Daten im gewählten Datumsbereich gefunden.")
      return

    print(f"\n--- SIMULATION STARTET (Offset: {offset_hours}h) ({start_date_str} bis {end_date_str}) ---")
    print(f"Startkapital: {initial_capital:,.2f} USDT")
    
    final_val_fee, _ = run_simulation(result_df, fee_rate=fee_rate, initial_cash=initial_capital)
    profit_fee = final_val_fee - initial_capital
    print(f"Endkapital (mit {fee_rate*100}% Gebühr): {final_val_fee:,.2f} USDT")
    print(f"Gewinn/Verlust: {profit_fee:+,.2f} USDT ({(profit_fee/initial_capital)*100:.2f}%)")


if __name__ == "__main__":
  main()