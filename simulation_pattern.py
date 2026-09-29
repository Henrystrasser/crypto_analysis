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
        # Berliner Wanduhr-Stunden dieses Berliner Tages (DST: Frühjahr 02:00
        # fehlt -> Tag übersprungen; Herbst 02:00 doppelt -> nur erstes Auftreten)
        wanted = [berlin_wallclock(date, h) for h in dict.fromkeys([morning_hour, evening_hour])]
        hour_data = group.loc[[x for x in wanted if x is not None and x in group.index]]
        if len(hour_data) < 2:
            continue

        hour_data = hour_data.sort_index()
        t1 = hour_data.index[0]
        t2 = hour_data.index[1]

        price_morning = float(hour_data.iloc[0]["close"])
        price_evening = float(hour_data.iloc[1]["close"])

        rows.append({
            "Date": date,
            "Time_Morning": t1.strftime("%H:%M %Z"),
            "Price_Morning": price_morning,
            "Time_Evening": t2.strftime("%H:%M %Z"),
            "Price_Evening": price_evening,
            "Day_Change_%": (price_evening / price_morning - 1.0) * 100.0,
            "Down_Day": price_evening < price_morning,
            "Up_Day": price_evening > price_morning,
        })

    return pd.DataFrame(rows), morning_hour, evening_hour


def run_evening_buy_next_morning_sell(
    day_df,
    buy_on="down",
    initial_cash=10000.0,
    fee_rate=0.001,
):
    """
    buy_on:
      "down" -> Abstiegstag: Abend kaufen, nächsten Morgen verkaufen
      "up"   -> Anstiegstag: Abend kaufen, nächsten Morgen verkaufen
    """
    if buy_on not in ("down", "up"):
        raise ValueError('buy_on muss "down" oder "up" sein.')

    cash = initial_cash
    crypto = 0.0
    position_open = False
    trades = []

    signal_col = "Down_Day" if buy_on == "down" else "Up_Day"
    reason_buy = (
        "Abstiegstag, Kauf am Abend"
        if buy_on == "down"
        else "Anstiegstag, Kauf am Abend"
    )
    reason_sell = (
        "Nächster Morgen nach Abstiegstag"
        if buy_on == "down"
        else "Nächster Morgen nach Anstiegstag"
    )

    for i in range(len(day_df)):
        row = day_df.iloc[i]

        if position_open:
            sell_price = row["Price_Morning"]
            gross = crypto * sell_price
            cash = gross * (1.0 - fee_rate)
            trades.append({
                "Date": row["Date"],
                "Time": row["Time_Morning"],
                "Action": "SELL",
                "Price": sell_price,
                "Cash": cash,
                "Crypto": 0.0,
                "Reason": reason_sell,
            })
            crypto = 0.0
            position_open = False

        if (not position_open) and bool(row[signal_col]):
            buy_price = row["Price_Evening"]
            effective_cash = cash * (1.0 - fee_rate)
            crypto = effective_cash / buy_price
            cash = 0.0
            position_open = True
            trades.append({
                "Date": row["Date"],
                "Time": row["Time_Evening"],
                "Action": "BUY",
                "Price": buy_price,
                "Cash": cash,
                "Crypto": crypto,
                "Reason": reason_buy,
            })

    final_value = cash
    if position_open:
        last_price = day_df.iloc[-1]["Price_Evening"]
        final_value = crypto * last_price

    return final_value, pd.DataFrame(trades)


def main():
    symbol = "POL/USDT"
    days =300
    offset_hours = 20
    interval_hours = 12
    initial_cash = 10000.0
    fee_rate = 0.00075

    # "down" = Abstiegstag am Abend kaufen
    # "up"   = Anstiegstag am Abend kaufen
    BUY_ON = "down"

    exchange = ccxt.binance()
    since_timestamp = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)

    print(f"Lade {days} Tage 1h-Daten für {symbol}...")
    ohlcv = fetch_all_ohlcv(exchange, symbol, "1h", since_timestamp)

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

    day_df, morning_hour, evening_hour = build_day_table(
        df, offset_hours, interval_hours
    )

    if day_df.empty:
        print("Keine verwertbaren Tagespaare gefunden.")
        return

    final_value, trade_df = run_evening_buy_next_morning_sell(
        day_df,
        buy_on=BUY_ON,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
    )

    profit = final_value - initial_cash
    roi = (profit / initial_cash) * 100.0
    signal_days = int(day_df["Down_Day"].sum() if BUY_ON == "down" else day_df["Up_Day"].sum())
    n_buys = int((trade_df["Action"] == "BUY").sum()) if not trade_df.empty else 0
    n_sells = int((trade_df["Action"] == "SELL").sum()) if not trade_df.empty else 0
    mode_label = "Abstiegstag" if BUY_ON == "down" else "Anstiegstag"

    print("\n==========================================")
    print(" PATTERN-SIMULATION")
    print("==========================================")
    print(f"Coin              : {symbol}")
    print(f"Zeitraum          : letzte {days} Tage")
    print(f"Morgen / Abend    : {morning_hour:02d}:00 / {evening_hour:02d}:00  (Europe/Berlin, CET/CEST)")
    print(f"Modus             : {mode_label} → Abend kaufen, nächsten Morgen verkaufen")
    print(f"Gebühr je Seite   : {fee_rate * 100:.3f} %")
    print(f"Analysierte Tage  : {len(day_df)}")
    print(f"Signaltage        : {signal_days}")
    print(f"Buys / Sells      : {n_buys} / {n_sells}")
    print("------------------------------------------")
    print(f"Startkapital      : {initial_cash:,.2f} EUR")
    print(f"Endkapital        : {final_value:,.2f} EUR")
    print(f"Gewinn/Verlust    : {profit:+,.2f} EUR ({roi:+.2f} %)")
    print("==========================================\n")

    #if not trade_df.empty:
    #    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    #    print("Trades:")
    #    print(trade_df.to_string(index=False))


if __name__ == "__main__":
    main()