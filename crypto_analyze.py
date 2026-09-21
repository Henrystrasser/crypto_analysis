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


def analyze_crypto_intraday_long(
    symbol="UNI/USDT", days=180, offset_hours=0, interval_hours=12
):
    exchange = ccxt.binance()

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

    df = df[~df.index.duplicated(keep="first")]
    df["date"] = df.index.date

    results = []
    target_hour_1 = offset_hours % 24
    target_hour_2 = (offset_hours + interval_hours) % 24

    for date, group in df.groupby("date"):
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
                "Time_1": time_t1.strftime("%H:%M"),
                "Price_1": price_t1,
                "Time_2": time_t2.strftime("%H:%M"),
                "Price_2": price_t2,
                "Price_Diff": price_t2 - price_t1,
                "Diff (%)": round(percent_diff, 2),
                "Früh billiger?": cheaper_in_morning,
            })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print("Nicht genügend Daten für diesen Offset gefunden.")
        return

    # Vortag: Abendpreis und ob der Vortag gestiegen/gefallen ist
    result_df["Prev_Price_2"] = result_df["Price_2"].shift(1)
    result_df["Prev_Rose"] = result_df["Früh billiger?"].shift(1)

    def check_condition(row):
        if pd.isna(row["Prev_Price_2"]) or pd.isna(row["Prev_Rose"]):
            return False

        p1_today = row["Price_1"]
        p2_yest = row["Prev_Price_2"]
        yesterday_rose = bool(row["Prev_Rose"])

        # Gestern Anstieg → heutiger Morgen niedriger als gestriger Abend
        if yesterday_rose and p1_today < p2_yest:
            return True

        # Gestern Abstieg → heutiger Morgen höher als gestriger Abend
        if (not yesterday_rose) and p1_today > p2_yest:
            return True

        return False

    result_df["Pattern_Match"] = result_df.apply(check_condition, axis=1)
    result_df = result_df.drop(columns=["Prev_Price_2", "Prev_Rose"])

    total_days = len(result_df)
    cheaper_count = result_df["Früh billiger?"].sum()
    win_rate = (cheaper_count / total_days) * 100

    match_count = result_df["Pattern_Match"].sum()
    match_rate = (match_count / total_days) * 100

    total_price_diff_sum = result_df["Price_Diff"].sum()
    total_percent_sum = result_df["Diff (%)"].sum()

    print("\n--- ERGEBNISSE (Langzeittest mit Muster-Check & Summen) ---")
    print(
        f"Getestetes Intervall: Start um Stunde {target_hour_1}:00, "
        f"Ende um Stunde {target_hour_2}:00 (Offset: {offset_hours}h)"
    )
    print(f"Anzahl analysierter Tage: {total_days}")
    print(
        f"Tage, an denen es früh billiger war: {cheaper_count} ({win_rate:.1f}%)"
    )
    print(
        f"Tage mit positivem Pattern-Match: {match_count}"
        f" ({match_rate:.1f}%)"
    )
    print(
        "Pattern: gestern Anstieg → heute Morgen < gestern Abend; "
        "gestern Abstieg → heute Morgen > gestern Abend"
    )
    print(
        f"Summe der absoluten Preisunterschiede (Price_2 - Price_1):"
        f" {total_price_diff_sum:,.2f} USD"
    )
    print(f"Kumulierte prozentuale Tagesschwankungen: {total_percent_sum:,.2f}%")
    print(
        f"Durchschnittliche prozentuale Änderung pro Tag: "
        f"{result_df['Diff (%)'].mean():.2f}%"
    )

    print("\nAlle Tage im Detail:")
    #print(result_df.to_string(index=False))

    return result_df



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

temp_symbol = "JUP/USDT"
result_df = analyze_crypto_intraday_long(
    symbol=temp_symbol, days=300, offset_hours=3, interval_hours=12
)