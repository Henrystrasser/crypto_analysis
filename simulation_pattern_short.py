import ccxt
import pandas as pd


def fetch_all_ohlcv(exchange, symbol, timeframe, since_timestamp):
    all_ohlcv = []
    limit = 1000
    current_since = since_timestamp

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit,
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


def build_day_table(df, offset_hours, interval_hours):
    morning_hour = offset_hours % 24
    evening_hour = (offset_hours + interval_hours) % 24
    rows = []

    for date, group in df.groupby("date"):
        hour_data = group[group.index.hour.isin([morning_hour, evening_hour])]
        if len(hour_data) < 2:
            continue

        hour_data = hour_data.sort_index()
        t1 = hour_data.index[0]
        t2 = hour_data.index[1]
        price_morning = float(hour_data.iloc[0]["close"])
        price_evening = float(hour_data.iloc[1]["close"])

        rows.append({
            "Date": date,
            "Time_Morning": t1.strftime("%H:%M"),
            "Price_Morning": price_morning,
            "Time_Evening": t2.strftime("%H:%M"),
            "Price_Evening": price_evening,
            "Day_Change_%": (price_evening / price_morning - 1.0) * 100.0,
            "Up_Day": price_evening > price_morning,
            "Down_Day": price_evening < price_morning,
        })

    day_df = pd.DataFrame(rows)
    if day_df.empty:
        return day_df, morning_hour, evening_hour

    day_df["Prev_Up"] = day_df["Up_Day"].shift(1)
    day_df["Prev_Evening"] = day_df["Price_Evening"].shift(1)

    def pattern_hit(row):
        if pd.isna(row["Prev_Up"]) or pd.isna(row["Prev_Evening"]):
            return False
        if bool(row["Prev_Up"]):
            return row["Price_Morning"] < row["Prev_Evening"]
        return row["Price_Morning"] > row["Prev_Evening"]

    day_df["Pattern_Hit"] = day_df.apply(pattern_hit, axis=1)
    return day_df, morning_hour, evening_hour


def run_pattern_bot(
    day_df,
    trade_long=True,
    trade_short=True,
    initial_cash=10000.0,
    fee_rate=0.001,
):
    """
    Abstiegstag → Long: Abend kaufen, nächsten Morgen verkaufen
      (Pattern: Morgen soll über dem Abend liegen)

    Anstiegstag → Short: Abend shorten, nächsten Morgen schließen
      (Pattern: Morgen soll unter dem Abend liegen)
    """
    cash = initial_cash
    crypto = 0.0
    position = None  # None | "long" | "short"
    entry_price = 0.0
    trades = []

    for i in range(len(day_df)):
        row = day_df.iloc[i]

        # Position vom Vortag am Morgen schließen
        if position == "long":
            sell_price = row["Price_Morning"]
            cash = (crypto * sell_price) * (1.0 - fee_rate)
            trades.append({
                "Date": row["Date"],
                "Time": row["Time_Morning"],
                "Action": "SELL",
                "Side": "LONG",
                "Price": sell_price,
                "Cash": cash,
                "Pattern_Hit": row["Pattern_Hit"],
                "Reason": "Long schließen nächster Morgen",
            })
            crypto = 0.0
            position = None

        elif position == "short":
            cover_price = row["Price_Morning"]
            # Short: Gewinn wenn cover < entry
            cash = cash * (entry_price / cover_price) * (1.0 - fee_rate)
            trades.append({
                "Date": row["Date"],
                "Time": row["Time_Morning"],
                "Action": "COVER",
                "Side": "SHORT",
                "Price": cover_price,
                "Cash": cash,
                "Pattern_Hit": row["Pattern_Hit"],
                "Reason": "Short schließen nächster Morgen",
            })
            position = None
            entry_price = 0.0

        # Neue Position am Abend
        if position is None and trade_long and bool(row["Down_Day"]):
            buy_price = row["Price_Evening"]
            crypto = (cash * (1.0 - fee_rate)) / buy_price
            cash = 0.0
            position = "long"
            entry_price = buy_price
            trades.append({
                "Date": row["Date"],
                "Time": row["Time_Evening"],
                "Action": "BUY",
                "Side": "LONG",
                "Price": buy_price,
                "Cash": cash,
                "Pattern_Hit": None,
                "Reason": "Abstiegstag: Long am Abend",
            })

        elif position is None and trade_short and bool(row["Up_Day"]):
            short_price = row["Price_Evening"]
            cash = cash * (1.0 - fee_rate)
            position = "short"
            entry_price = short_price
            trades.append({
                "Date": row["Date"],
                "Time": row["Time_Evening"],
                "Action": "SHORT",
                "Side": "SHORT",
                "Price": short_price,
                "Cash": cash,
                "Pattern_Hit": None,
                "Reason": "Anstiegstag: Short am Abend",
            })

    final_value = cash
    if position == "long":
        final_value = crypto * day_df.iloc[-1]["Price_Evening"]
    elif position == "short":
        last = day_df.iloc[-1]["Price_Evening"]
        final_value = cash * (entry_price / last)

    return final_value, pd.DataFrame(trades)


def main():
    symbol = "BCH/USDT"
    days = 200
    offset_hours =6
    interval_hours = 12
    initial_cash = 10000.0
    fee_rate = 0.001

    TRADE_LONG = True    # Abstiegstag → Long über Nacht
    TRADE_SHORT = True   # Anstiegstag → Short über Nacht

    exchange = ccxt.binance()
    since_timestamp = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)

    print(f"Lade {days} Tage 1h-Daten für {symbol}...")
    ohlcv = fetch_all_ohlcv(exchange, symbol, "1h", since_timestamp)

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df["date"] = df.index.date

    day_df, morning_hour, evening_hour = build_day_table(
        df, offset_hours, interval_hours
    )
    if day_df.empty:
        print("Keine Tagespaare gefunden.")
        return

    final_value, trade_df = run_pattern_bot(
        day_df,
        trade_long=TRADE_LONG,
        trade_short=TRADE_SHORT,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
    )

    profit = final_value - initial_cash
    roi = (profit / initial_cash) * 100.0
    usable = day_df.dropna(subset=["Prev_Up"])
    hit_rate = usable["Pattern_Hit"].mean() * 100.0 if not usable.empty else 0.0

    n_long = int((trade_df["Action"] == "BUY").sum()) if not trade_df.empty else 0
    n_short = int((trade_df["Action"] == "SHORT").sum()) if not trade_df.empty else 0

    print("\n==========================================")
    print(" PATTERN-BOT SIMULATION (LONG + SHORT)")
    print("==========================================")
    print(f"Coin            : {symbol}")
    print(f"Zeitraum        : letzte {days} Tage")
    print(f"Morgen / Abend  : {morning_hour:02d}:00 / {evening_hour:02d}:00  (UTC)")
    print("Long            : Abstiegstag Abend kaufen, Morgen verkaufen")
    print("Short           : Anstiegstag Abend shorten, Morgen closen")
    print(f"Long aktiv      : {TRADE_LONG}")
    print(f"Short aktiv     : {TRADE_SHORT}")
    print(f"Pattern-Treffer : {hit_rate:.1f}% der Folgemorgen")
    print(f"Gebühr / Seite  : {fee_rate * 100:.3f}%")
    print(f"Longs / Shorts  : {n_long} / {n_short}")
    print("------------------------------------------")
    print(f"Startkapital    : {initial_cash:,.2f}")
    print(f"Endkapital      : {final_value:,.2f}")
    print(f"Gewinn/Verlust  : {profit:+,.2f} ({roi:+.2f}%)")
    print("==========================================\n")

    if not trade_df.empty:
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print("Trades:")
        print(trade_df.to_string(index=False))


if __name__ == "__main__":
    main()