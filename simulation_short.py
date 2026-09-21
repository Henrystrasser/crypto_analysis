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
def custom_short_condition(row, portfolio_state):
  """Bedingung für den Short-Trade (Tag)."""
  return True


def custom_long_condition(row, portfolio_state):
  """Bedingung für den Long-Trade (Nacht)."""
  return True


# ==========================================
# SIMULATIONS-ENGINE (24/7: Wechsel zwischen Short & Long)
# ==========================================
def run_simulation(results_df, fee_rate=0.0002, initial_cash=10000.0):
  cash = initial_cash
  trade_log = []

  for i in range(len(results_df)):
    row = results_df.iloc[i]

    # ----------------------------------------------------
    # LEG 1: TAG (Price_1 bis Price_2) -> SHORT
    # ----------------------------------------------------
    p_in_short = row["Price_1"]
    time_in_short = f"{row['Date']} {row['Time_1']}"
    p_out_short = row["Price_2"]
    time_out_short = f"{row['Date']} {row['Time_2']}"

    if custom_short_condition(row, {"cash": cash}):
      # Gebühr beim Short-Einstieg
      cash = cash * (1 - fee_rate)
      # Short-Multiplikator: Gewinn wenn Kurs sinkt (P_in > P_out)
      short_multiplier = 1.0 + (p_in_short - p_out_short) / p_in_short
      cash = cash * short_multiplier
      # Gebühr beim Short-Ausstieg
      cash = cash * (1 - fee_rate)

      trade_log.append({
          "Time": time_in_short,
          "Action": "SHORT_OPEN",
          "Price": p_in_short,
          "Cash": cash,
      })
      trade_log.append({
          "Time": time_out_short,
          "Action": "SHORT_CLOSE",
          "Price": p_out_short,
          "Cash": cash,
      })

    # ----------------------------------------------------
    # LEG 2: NACHT (Price_2 bis Price_1 des Folgetages) -> LONG
    # ----------------------------------------------------
    if i + 1 < len(results_df):
      next_row = results_df.iloc[i + 1]
      p_in_long = p_out_short  # Startet da, wo Short endete
      time_in_long = time_out_short
      p_out_long = next_row["Price_1"]
      time_out_long = f"{next_row['Date']} {next_row['Time_1']}"

      if custom_long_condition(next_row, {"cash": cash}):
        # Gebühr beim Long-Einstieg
        cash = cash * (1 - fee_rate)
        # Long-Multiplikator: Gewinn wenn Kurs steigt
        long_multiplier = p_out_long / p_in_long
        cash = cash * long_multiplier
        # Gebühr beim Long-Ausstieg
        cash = cash * (1 - fee_rate)

        trade_log.append({
            "Time": time_in_long,
            "Action": "LONG_OPEN",
            "Price": p_in_long,
            "Cash": cash,
        })
        trade_log.append({
            "Time": time_out_long,
            "Action": "LONG_CLOSE",
            "Price": p_out_long,
            "Cash": cash,
        })

  return cash, pd.DataFrame(trade_log)


def generate_result_df(
    df, offset_hours, interval_hours, start_date_str, end_date_str
):
  results = []
  for date, group in df.groupby("date"):
    t1 = offset_hours % 24
    t2 = (offset_hours + interval_hours) % 24
    hour_data = group[group.index.hour.isin([t1, t2])]

    if len(hour_data) >= 2:
      hour_data = hour_data.sort_index()
      results.append({
          "Date": date,
          "Time_1": hour_data.index[0].strftime("%H:%M"),
          "Price_1": hour_data.iloc[0]["close"],
          "Time_2": hour_data.index[1].strftime("%H:%M"),
          "Price_2": hour_data.iloc[1]["close"],
      })

  result_df = pd.DataFrame(results)

  if start_date_str and not result_df.empty:
    result_df = result_df[
        result_df["Date"] >= pd.to_datetime(start_date_str).date()
    ]
  if end_date_str and not result_df.empty:
    result_df = result_df[
        result_df["Date"] <= pd.to_datetime(end_date_str).date()
    ]

  return result_df


# ==========================================
# HAUPTPROGRAMM & KONFIGURATION
# ==========================================
def main():
  exchange = ccxt.binance()
  temp_symbol = "UNI/USDT"

  # --- HIER ZEITRAUM ANPASSEN ---
  start_date_str = "2022-06-30"  # Start des Backtests (YYYY-MM-DD)
  end_date_str = "2027-06-30"  # Ende des Backtests (YYYY-MM-DD)
  interval_hours = 12
  fee_rate = 0.0002  # 0.02% Futures Maker-Gebühr
  initial_capital = 10000.0

  # Offset-Vergleich (Testet verschiedene Startzeiten für den Wechsel)
  COMPARE_OFFSETS = True
  offsets_to_test = [0, 3, 6, 8, 10, 13, 16, 19, 22]

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
    print(
        f" STARTE 24/7 (SHORT/LONG) OFFSET-VERGLEICH FÜR: {offsets_to_test}"
    )
    print(f"==========================================")

    comparison_results = []

    for offset in offsets_to_test:
      result_df = generate_result_df(
          df, offset, interval_hours, start_date_str, end_date_str
      )
      if result_df.empty:
        continue

      final_val, _ = run_simulation(
          result_df, fee_rate=fee_rate, initial_cash=initial_capital
      )
      profit = final_val - initial_capital
      roi = (profit / initial_capital) * 100

      comparison_results.append({
          "Offset (Stunde)": offset,
          "Uhrzeit 1": f"{(offset)%24:02d}:00",
          "Uhrzeit 2": f"{(offset + interval_hours)%24:02d}:00",
          "Endkapital (USDT)": final_val,
          "Gewinn/Verlust": profit,
          "ROI (%)": roi,
      })

    comp_df = pd.DataFrame(comparison_results)
    pd.set_option("display.float_format", lambda x: "%.2f" % x)
    print("\nERGEBNIS-ÜBERSICHT (24/7 Strategie):")
    print(comp_df.to_string(index=False))
    print("==========================================\n")

  else:
    offset_hours = 7
    result_df = generate_result_df(
        df, offset_hours, interval_hours, start_date_str, end_date_str
    )

    if result_df.empty:
      print("Fehler: Keine Daten im gewählten Datumsbereich gefunden.")
      return

    print(
        f"\n--- SIMULATION STARTET (Offset: {offset_hours}h) ({start_date_str}"
        f" bis {end_date_str}) ---"
    )
    print(f"Startkapital: {initial_capital:,.2f} USDT")

    final_val_fee, _ = run_simulation(
        result_df, fee_rate=fee_rate, initial_cash=initial_capital
    )
    profit_fee = final_val_fee - initial_capital
    print(
        f"Endkapital (mit {fee_rate*100}% Gebühr):"
        f" {final_val_fee:,.2f} USDT"
    )
    print(
        f"Gewinn/Verlust: {profit_fee:+,.2f} USDT"
        f" ({(profit_fee/initial_capital)*100:.2f}%)"
    )


if __name__ == "__main__":
  main()