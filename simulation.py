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


def custom_buy_condition(row, portfolio_state):
    return True


def custom_sell_condition(row, portfolio_state):
    return True


def run_simulation(results_df, fee_rate=0.0, initial_cash=10000.0):
    cash = initial_cash
    crypto = 0.0
    position_open = False
    trade_log = []

    for i in range(len(results_df)):
        row = results_df.iloc[i]

        price_buy = row["Price_Buy"]
        time_buy = f"{row['Date_Buy']} {row['Time_Buy']}"

        if not position_open and custom_buy_condition(row, {"cash": cash}):
            effective_cash = cash * (1 - fee_rate)
            crypto = effective_cash / price_buy
            cash = 0.0
            position_open = True
            trade_log.append(
                {
                    "Time": time_buy,
                    "Action": "BUY",
                    "Price": price_buy,
                    "Cash": cash,
                    "Crypto": crypto,
                }
            )

        price_sell = row["Price_Sell"]
        time_sell = f"{row['Date_Sell']} {row['Time_Sell']}"

        if position_open and custom_sell_condition(row, {"crypto": crypto}):
            gross_cash = crypto * price_sell
            cash = gross_cash * (1 - fee_rate)
            crypto = 0.0
            position_open = False
            trade_log.append(
                {
                    "Time": time_sell,
                    "Action": "SELL",
                    "Price": price_sell,
                    "Cash": cash,
                    "Crypto": crypto,
                }
            )

    final_value = cash
    if position_open:
        final_value = crypto * results_df.iloc[-1]["Price_Sell"]

    return final_value, pd.DataFrame(trade_log)


def buy_and_hold_roi(df, start_date_str, end_date_str, fee_rate):
    window = df.copy()
    if start_date_str:
        window = window[window.index >= pd.Timestamp(start_date_str).tz_localize(BERLIN)]
    if end_date_str:
        window = window[
            window.index
            <= (pd.Timestamp(end_date_str) + pd.Timedelta(days=1)).tz_localize(BERLIN)
        ]
    if window.empty:
        return None

    first = float(window.iloc[0]["close"])
    last = float(window.iloc[-1]["close"])
    # gleiche Gebühr wie Strategie: einmal kaufen, einmal verkaufen
    return ((last / first) * (1 - fee_rate) ** 2 - 1.0) * 100


def generate_result_df(df, offset_hours, interval_hours, start_date_str, end_date_str):
    results = []
    dates = df["date"].unique()
    for d in dates:
        # Kauf = Berliner Ortszeit offset_hours:00 am Berliner Tag d (DST siehe
        # berlin_wallclock); Verkauf = Kauf + interval_hours echte Stunden.
        current_dt = berlin_wallclock(d, offset_hours)
        if current_dt is None:
            continue
        target_sell_dt = current_dt + pd.Timedelta(hours=interval_hours)

        if current_dt in df.index and target_sell_dt in df.index:
            results.append(
                {
                    "Date_Buy": current_dt.strftime("%Y-%m-%d"),
                    "Time_Buy": current_dt.strftime("%H:%M %Z"),
                    "Price_Buy": df.loc[current_dt, "close"],
                    "Date_Sell": target_sell_dt.strftime("%Y-%m-%d"),
                    "Time_Sell": target_sell_dt.strftime("%H:%M %Z"),
                    "Price_Sell": df.loc[target_sell_dt, "close"],
                }
            )

    result_df = pd.DataFrame(results)
    if start_date_str and not result_df.empty:
        result_df = result_df[result_df["Date_Buy"] >= start_date_str]
    if end_date_str and not result_df.empty:
        result_df = result_df[result_df["Date_Buy"] <= end_date_str]
    return result_df


def main():
    exchange = ccxt.binance()
    temp_symbol = "NEXO/USDT"

    start_date_str = "2024-03-20"
    end_date_str = "2026-09-15"
    interval_hours = 12
    fee_rate = 0.00075
    initial_capital = 10000.0

    COMPARE_OFFSETS = True
    offsets_to_test = list(range(24))

    since_timestamp = (
        int(pd.Timestamp(start_date_str).tz_localize(BERLIN).timestamp() * 1000) - 48 * 60 * 60 * 1000
    )

    print(f"Lade {temp_symbol} ab {start_date_str}...")
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

    hold_roi = buy_and_hold_roi(df, start_date_str, end_date_str, fee_rate)
    if hold_roi is None:
        print("Keine Daten für Buy-and-Hold.")
        return

    if COMPARE_OFFSETS:
        print(
            f"\n{temp_symbol}  {start_date_str} → {end_date_str}  Hold ROI: {hold_roi:+.2f}%"
            "  | Kauf/Verkauf = deutsche Zeit (Europe/Berlin, CET/CEST)\n"
        )
        rows = []

        for offset in offsets_to_test:
            result_df = generate_result_df(
                df, offset, interval_hours, start_date_str, end_date_str
            )
            if result_df.empty:
                continue

            final_val, _ = run_simulation(
                result_df, fee_rate=fee_rate, initial_cash=initial_capital
            )
            strat_roi = (final_val / initial_capital - 1.0) * 100
            sell_hour = (offset + interval_hours) % 24
            rows.append(
                {
                    "Kauf": f"{offset:02d}:00",
                    "Verkauf": f"{sell_hour:02d}:00",
                    "ROI %": strat_roi,
                    "Hold ROI %": hold_roi,
                    "vs Hold": strat_roi - hold_roi,
                }
            )

        comp_df = pd.DataFrame(rows)
        pd.set_option("display.float_format", lambda x: "%.2f" % x)
        print(comp_df.to_string(index=False))
        print()
    else:
        offset_hours = 0
        result_df = generate_result_df(
            df, offset_hours, interval_hours, start_date_str, end_date_str
        )
        if result_df.empty:
            print("Keine Daten im gewählten Datumsbereich.")
            return

        final_val, _ = run_simulation(
            result_df, fee_rate=fee_rate, initial_cash=initial_capital
        )
        strat_roi = (final_val / initial_capital - 1.0) * 100
        print(
            f"{temp_symbol} Offset {offset_hours:02d}:00  "
            f"ROI {strat_roi:+.2f}%  Hold {hold_roi:+.2f}%  "
            f"vs Hold {strat_roi - hold_roi:+.2f}%"
        )


if __name__ == "__main__":
    main()
