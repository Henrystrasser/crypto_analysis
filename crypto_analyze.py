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
    start_hour = offset_hours % 24
    end_hour = (offset_hours + interval_hours) % 24

    # Ein Fenster pro Kalendertag: Start = Tag + offset, Ende = Start + interval
    # (Ende darf auf den Folgetag fallen — genau der Unterschied 02:00 vs 14:00)
    unique_dates = sorted(pd.unique(df["date"]))
    for date in unique_dates:
        t1 = pd.Timestamp(date) + pd.Timedelta(hours=start_hour)
        t2 = t1 + pd.Timedelta(hours=interval_hours)

        if t1 not in df.index or t2 not in df.index:
            continue

        price_t1 = df.loc[t1, "close"]
        price_t2 = df.loc[t2, "close"]
        diff = price_t2 - price_t1
        percent_diff = (diff / price_t1) * 100
        cheaper_at_start = price_t1 < price_t2

        results.append(
            {
                "Date": date,
                "Time_1": t1.strftime("%Y-%m-%d %H:%M"),
                "Price_1": price_t1,
                "Time_2": t2.strftime("%Y-%m-%d %H:%M"),
                "Price_2": price_t2,
                "Price_Diff": diff,
                "Diff (%)": round(percent_diff, 2),
                "Start billiger?": cheaper_at_start,
            }
        )

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print("Nicht genügend Daten für diesen Offset gefunden.")
        return

    result_df["Prev_Price_2"] = result_df["Price_2"].shift(1)
    result_df["Prev_Rose"] = result_df["Start billiger?"].shift(1)

    def check_condition(row):
        if pd.isna(row["Prev_Price_2"]) or pd.isna(row["Prev_Rose"]):
            return False

        p1_today = row["Price_1"]
        p2_yest = row["Prev_Price_2"]
        yesterday_rose = bool(row["Prev_Rose"])

        if yesterday_rose and p1_today < p2_yest:
            return True
        if (not yesterday_rose) and p1_today > p2_yest:
            return True
        return False

    result_df["Pattern_Match"] = result_df.apply(check_condition, axis=1)
    result_df = result_df.drop(columns=["Prev_Price_2", "Prev_Rose"])

    total_days = len(result_df)
    cheaper_count = result_df["Start billiger?"].sum()
    win_rate = (cheaper_count / total_days) * 100

    match_count = result_df["Pattern_Match"].sum()
    match_rate = (match_count / total_days) * 100

    total_price_diff_sum = result_df["Price_Diff"].sum()
    total_percent_sum = result_df["Diff (%)"].sum()

    print("\n--- ERGEBNISSE (Langzeittest mit Muster-Check & Summen) ---")
    print(
        f"Getestetes Intervall: Start {start_hour:02d}:00, "
        f"Ende {end_hour:02d}:00 "
        f"(Offset {offset_hours}h, Dauer {interval_hours}h"
        f"{', über Mitternacht' if end_hour < start_hour else ''})"
    )
    print(f"Anzahl analysierter Fenster: {total_days}")
    print(
        f"Fenster, in denen der Start günstiger war: {cheaper_count} ({win_rate:.1f}%)"
    )
    print(
        f"Tage mit positivem Pattern-Match: {match_count} ({match_rate:.1f}%)"
    )
    print(
        "Pattern: gestern Anstieg → heutiger Start < gestriges Ende; "
        "gestern Abstieg → heutiger Start > gestriges Ende"
    )
    print(
        f"Summe der Preisunterschiede (Price_2 - Price_1): "
        f"{total_price_diff_sum:,.2f} USD"
    )
    print(f"Kumulierte prozentuale Änderungen: {total_percent_sum:,.2f}%")
    print(
        f"Durchschnittliche prozentuale Änderung pro Fenster: "
        f"{result_df['Diff (%)'].mean():.2f}%"
    )

    print("\nAlle Fenster im Detail:")
    #print(result_df.to_string(index=False))

    return result_df


if __name__ == "__main__":
    temp_symbol = "TNSR/USDT"
    result_df = analyze_crypto_intraday_long(
        symbol=temp_symbol, days=100, offset_hours=2, interval_hours=12
    )