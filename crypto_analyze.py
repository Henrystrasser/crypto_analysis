import ccxt
import pandas as pd


def fetch_all_ohlcv(exchange, symbol, timeframe, since_timestamp):
  all_ohlcv = []
  limit = 1000  # Maximales Limit pro Anfrage bei Binance
  current_since = since_timestamp

  while True:
    try:
      # Lade einen Block von Daten
      ohlcv = exchange.fetch_ohlcv(
          symbol, timeframe=timeframe, since=current_since, limit=limit
      )
      if not ohlcv:
        break

      all_ohlcv.extend(ohlcv)

      # Der nächste Block beginnt nach dem Zeitstempel der letzten Kerze
      next_since = ohlcv[-1][0] + 1

      # Wenn wir beim aktuellen Zeitpunkt angekommen sind, abbrechen
      if next_since <= current_since:
        break
      current_since = next_since

      # Wenn weniger als das Limit zurückgegeben wurde, sind wir am Ende angelangt
      if len(ohlcv) < limit:
        break
    except Exception as e:
      print(f"Fehler beim Laden: {e}")
      break

  return all_ohlcv


def analyze_crypto_intraday_long(
    symbol="BTC/USDT", days=180, offset_hours=0, interval_hours=12
):
  exchange = ccxt.binance()

  # Berechne den Start-Zeitpunkt in Millisekunden (jetzt minus X Tage)
  since_timestamp = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)

  print(
      f"Lade historische 1h-Daten für {symbol} für die letzten {days} Tage"
      " (dies kann einen Moment dauern)..."
  )
  ohlcv = fetch_all_ohlcv(exchange, symbol, "1h", since_timestamp)

  df = pd.DataFrame(
      ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
  )
  df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
  df.set_index("datetime", inplace=True)

  # Duplikate entfernen (falls welche entstehen)
  df = df[~df.index.duplicated(keep="first")]
  df["date"] = df.index.date

  results = []

  for date, group in df.groupby("date"):
    target_hour_1 = offset_hours % 24
    target_hour_2 = (offset_hours + interval_hours) % 24

    hour_data = group[group.index.hour.isin([target_hour_1, target_hour_2])]

    if len(hour_data) >= 2:
      hour_data = hour_data.sort_index()

      price_t1 = hour_data.iloc[0]["close"]
      time_t1 = hour_data.index[0]

      price_t2 = hour_data.iloc[1]["close"]
      time_t2 = hour_data.index[1]

      diff = price_t2 - price_t1
      percent_diff = (diff / price_t1) * 100
      cheaper_in_morning = price_t1 < price_t2

      results.append({
          "Date": date,
          f"Time_1 ({time_t1.strftime('%H:%M')})": price_t1,
          f"Time_2 ({time_t2.strftime('%H:%M')})": price_t2,
          "Diff (%)": round(percent_diff, 2),
          "Früh billiger?": cheaper_in_morning,
      })

  result_df = pd.DataFrame(results)

  if result_df.empty:
    print("Nicht genügend Daten für diesen Offset gefunden.")
    return

  total_days = len(result_df)
  cheaper_count = result_df["Früh billiger?"].sum()
  win_rate = (cheaper_count / total_days) * 100

  print("\n--- ERGEBNISSE (Langzeittest) ---")
  print(
      f"Getestetes Intervall: Start um Stunde {target_hour_1}:00, "
      f"Ende um Stunde {target_hour_2}:00 (Offset: {offset_hours}h)"
  )
  print(f"Anzahl analysierter Tage: {total_days}")
  print(
      f"Tage, an denen es zum Startzeitpunkt billiger war: {cheaper_count}"
      f" ({win_rate:.1f}%)"
  )
  print(f"Durchschnittliche prozentuale Änderung: {result_df['Diff (%)'].mean():.2f}%")

  # Zeige die letzten 60 Tage im Detail
  print(f"\Letzten {days} Tage im Detail:")
  print(result_df.tail(days).to_string(index=False))

  return result_df


# --- HIER ANPASSEN ---
# Jetzt kannst du problemlos z.B. 180 oder 365 Tage abrufen
result_df = analyze_crypto_intraday_long(
    symbol="ZEC/USDT",
    days=380,  # Anzahl der Tage (z.B. ein halbes Jahr)
    offset_hours=10,  # Dein gewünschter Offset
    interval_hours=12,
)