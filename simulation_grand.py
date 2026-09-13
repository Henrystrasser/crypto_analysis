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
      break

  return all_ohlcv


# ==========================================
# ANPASSBARE HANDELS-LOGIK FUNKTIONEN
# ==========================================
def custom_buy_condition(row, portfolio_state):
  return True


def custom_sell_condition(row, portfolio_state):
  return True


# ==========================================
# SIMULATIONS-ENGINE
# ==========================================
def run_simulation(results_df, fee_rate=0.0, initial_cash=10000.0):
  cash = initial_cash
  crypto = 0.0
  position_open = False

  trade_log = []

  for i in range(len(results_df)):
    row = results_df.iloc[i]

    # Kauf-Zeitpunkt (Intervall 2 / Abend)
    price_buy = row["Price_2"]
    time_buy = f"{row['Date']} {row['Time_2']}"

    if not position_open and custom_buy_condition(row, {"cash": cash}):
      effective_cash = cash * (1 - fee_rate)
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

    # Verkaufs-Zeitpunkt (Intervall 1 des Folgetages / Morgen)
    if i + 1 < len(results_df):
      next_row = results_df.iloc[i + 1]
      price_sell = next_row["Price_1"]
      time_sell = f"{next_row['Date']} {next_row['Time_1']}"

      if position_open and custom_sell_condition(next_row, {"crypto": crypto}):
        gross_cash = crypto * price_sell
        cash = gross_cash * (1 - fee_rate)
        crypto = 0.0
        position_open = False
        trade_log.append({
            "Time": time_sell,
            "Action": "SELL",
            "Price": price_sell,
            "Cash": cash,
            "Crypto": crypto,
        })

  final_value = cash
  if position_open:
    final_value = crypto * results_df.iloc[-1]["Price_1"]

  return final_value, pd.DataFrame(trade_log)


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

  top_50_symbols = [
      "BTC/USDT",
      "ETH/USDT",
      "SOL/USDT",
      "XRP/USDT",
      "BNB/USDT",
      "DOGE/USDT",
      "ADA/USDT",
      "AVAX/USDT",
      "LINK/USDT",
      "SUI/USDT",
      "DOT/USDT",
      "NEAR/USDT",
      "UNI/USDT",
      "POL/USDT",
      "LTC/USDT",
      "BCH/USDT",
      "APT/USDT",
      "ICP/USDT",
      "RENDER/USDT",
      "FET/USDT",
      "ARB/USDT",
      "INJ/USDT",
      "OP/USDT",
      "ATOM/USDT",
      "SEI/USDT",
      "TIA/USDT",
      "PEPE/USDT",
      "SHIB/USDT",
      "ETC/USDT",
      "FIL/USDT",
      "STX/USDT",
      "IMX/USDT",
      "AR/USDT",
      "FTM/USDT",
      "GRT/USDT",
      "RUNE/USDT",
      "ALGO/USDT",
      "XLM/USDT",
      "HBAR/USDT",
      "VET/USDT",
      "THETA/USDT",
      "JUP/USDT",
      "PENDLE/USDT",
      "WIF/USDT",
      "BONK/USDT",
      "FLOKI/USDT",
      "KAS/USDT",
      "AKT/USDT",
      "TON/USDT",
      "EOS/USDT",
  ]

  start_date_str = "2022-01-01"
  end_date_str = "2026-06-30"
  interval_hours = 12
  fee_rate = 0.0002  # 0.02% Futures Maker-Gebühr
  initial_capital = 10000.0

  # Offsets, die für jeden Coin durchprobiert werden sollen
  offsets_to_test = [0, 3, 6, 8, 10, 12, 15, 18, 21]

  since_timestamp = (
      int(pd.Timestamp(start_date_str).timestamp() * 1000)
      - 48 * 60 * 60 * 1000
  )

  portfolio_summary = []

  print(
      f"\nStarte Optimierung für {len(top_50_symbols)} Coins (teste"
      f" {len(offsets_to_test)} Offsets pro Coin)...\n"
  )

  for symbol in top_50_symbols:
    print(f"Teste: {symbol:<10} ...", end=" ")
    ohlcv = fetch_all_ohlcv(exchange, symbol, "1h", since_timestamp)

    if len(ohlcv) < 100:
      print("Übersprungen (zu wenig Historie).")
      continue

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df["date"] = df.index.date

    best_offset = None
    best_final_val = -1.0
    best_profit = 0.0
    best_roi = -999.0

    # Finde den besten Offset für DIESEN Coin
    for offset in offsets_to_test:
      result_df = generate_result_df(
          df, offset, interval_hours, start_date_str, end_date_str
      )
      if result_df.empty:
        continue

      final_val, _ = run_simulation(
          result_df, fee_rate=fee_rate, initial_cash=initial_capital
      )

      if final_val > best_final_val:
        best_final_val = final_val
        best_offset = offset
        best_profit = final_val - initial_capital
        best_roi = (best_profit / initial_capital) * 100

    if best_offset is not None:
      portfolio_summary.append({
          "Symbol": symbol,
          "Bester Offset": f"{best_offset}h",
          "Endkapital": best_final_val,
          "Gewinn/Verlust": best_profit,
          "ROI (%)": best_roi,
      })
      print(
          f"Bester Offset: {best_offset}h | ROI: {best_roi:+.2f}%"
      )
    else:
      print("Keine gültigen Daten in den Offsets.")

  # Finale Auswertung sortiert nach bestem ROI
  if portfolio_summary:
    summary_df = pd.DataFrame(portfolio_summary)
    summary_df = summary_df.sort_values(by="ROI (%)", ascending=False)
    pd.set_option("display.float_format", lambda x: "%.2f" % x)

    print("\n" + "=" * 70)
    print(" TOP 50 OPTIMIERUNGS-ERGEBNIS (Mit jeweils bestem Offset)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
  main()