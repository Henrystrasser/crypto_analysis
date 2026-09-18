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
            trade_log.append({
                "Time": time_buy,
                "Action": "BUY",
                "Price": price_buy,
                "Cash": cash,
                "Crypto": crypto,
            })

        price_sell = row["Price_Sell"]
        time_sell = f"{row['Date_Sell']} {row['Time_Sell']}"

        if position_open and custom_sell_condition(row, {"crypto": crypto}):
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
        final_value = crypto * results_df.iloc[-1]["Price_Sell"]

    return final_value, pd.DataFrame(trade_log)


def generate_result_df(df, offset_hours, interval_hours, start_date_str, end_date_str):
    results = []
    dates = df["date"].unique()

    for d in dates:
        current_dt = pd.to_datetime(d) + pd.Timedelta(hours=offset_hours)
        target_sell_dt = current_dt + pd.Timedelta(hours=interval_hours)

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
        result_df = result_df[result_df["Date_Buy"] >= start_date_str]
    if end_date_str and not result_df.empty:
        result_df = result_df[result_df["Date_Buy"] <= end_date_str]

    return result_df


def simulate_offset_interval(
    df, offset, interval_hours, start_date_str, end_date_str, fee_rate, initial_capital
):
    result_df = generate_result_df(
        df, offset, interval_hours, start_date_str, end_date_str
    )
    if result_df.empty:
        return None

    final_val, _ = run_simulation(
        result_df, fee_rate=fee_rate, initial_cash=initial_capital
    )
    profit = final_val - initial_capital
    roi = (profit / initial_capital) * 100
    sell_hour = (offset + interval_hours) % 24

    return {
        "Offset (Stunde)": offset,
        "Intervall (Stunden)": interval_hours,
        "Kauf-Uhrzeit": f"{offset:02d}:00",
        "Verkauf-Uhrzeit": f"{sell_hour:02d}:00",
        "Trades": len(result_df),
        "Endkapital (USDT)": final_val,
        "Gewinn/Verlust": profit,
        "ROI (%)": roi,
    }


def main():
    exchange = ccxt.binance()
    temp_symbol = "JUP/USDT"

    start_date_str = "2026-07-01"
    end_date_str = "2026-10-01"
    interval_hours = 12
    fee_rate = 0.001
    initial_capital = 10000.0

    COMPARE_OFFSETS = True
    offsets_to_test = list(range(24))

    COMPARE_INTERVALS = True
    intervals_to_test = list(range(1, 24))

    since_timestamp = (
        int(pd.Timestamp(start_date_str).timestamp() * 1000)
        - 48 * 60 * 60 * 1000
    )

    print(f"Lade historische Daten für {temp_symbol} ab dem {start_date_str}...")
    ohlcv = fetch_all_ohlcv(exchange, temp_symbol, "1h", since_timestamp)

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df["date"] = df.index.date

    pd.set_option("display.float_format", lambda x: "%.2f" % x)

    best_offset = None

    if COMPARE_OFFSETS:
        print("\n==========================================")
        print(f" OFFSET-VERGLEICH  (Intervall fest: {interval_hours}h)")
        print("==========================================")
        comparison_results = []

        for offset in offsets_to_test:
            row = simulate_offset_interval(
                df,
                offset,
                interval_hours,
                start_date_str,
                end_date_str,
                fee_rate,
                initial_capital,
            )
            if row:
                comparison_results.append(row)

        if comparison_results:
            comp_df = pd.DataFrame(comparison_results).sort_values(
                "Offset (Stunde)", ascending=True
            )
            print("\nERGEBNIS-ÜBERSICHT OFFSETS:")
            print(comp_df.to_string(index=False))

            best = comp_df.loc[comp_df["ROI (%)"].idxmax()]
            best_offset = int(best["Offset (Stunde)"])
            print("\nBester Offset:")
            print(
                f"  Kauf {best['Kauf-Uhrzeit']} / "
                f"Verkauf {best['Verkauf-Uhrzeit']} / "
                f"ROI {best['ROI (%)']:.2f}%"
            )
        else:
            print("Keine Offset-Ergebnisse.")
        print("==========================================\n")

    else:
        offset_hours = 0
        result_df = generate_result_df(
            df, offset_hours, interval_hours, start_date_str, end_date_str
        )
        if result_df.empty:
            print("Fehler: Keine Daten im gewählten Datumsbereich gefunden.")
            return

        print(
            f"\n--- SIMULATION STARTET (Offset: {offset_hours}h) "
            f"({start_date_str} bis {end_date_str}) ---"
        )
        print(f"Startkapital: {initial_capital:,.2f} USDT")
        final_val_fee, _ = run_simulation(
            result_df, fee_rate=fee_rate, initial_cash=initial_capital
        )
        profit_fee = final_val_fee - initial_capital
        print(
            f"Endkapital (mit {fee_rate * 100}% Gebühr): "
            f"{final_val_fee:,.2f} USDT"
        )
        print(
            f"Gewinn/Verlust: {profit_fee:+,.2f} USDT "
            f"({(profit_fee / initial_capital) * 100:.2f}%)"
        )
        best_offset = offset_hours

    if COMPARE_INTERVALS:
        if best_offset is None:
            best_offset = 0
            print(
                "Hinweis: Kein Offset-Vergleich gelaufen. "
                f"Intervall-Test nutzt Offset {best_offset}."
            )

        print("\n==========================================")
        print(f" INTERVALL-VERGLEICH  (Offset fest: {best_offset:02d}:00)")
        print("==========================================")

        interval_results = []
        for interval in intervals_to_test:
            row = simulate_offset_interval(
                df,
                best_offset,
                interval,
                start_date_str,
                end_date_str,
                fee_rate,
                initial_capital,
            )
            if row:
                interval_results.append(row)

        if interval_results:
            int_df = pd.DataFrame(interval_results).sort_values(
                "Intervall (Stunden)", ascending=True
            )
            print("\nERGEBNIS-ÜBERSICHT INTERVALLE:")
            print(int_df.to_string(index=False))

            best_int = int_df.loc[int_df["ROI (%)"].idxmax()]
            print("\nBestes Intervall beim besten Offset:")
            print(
                f"  Offset {int(best_int['Offset (Stunde)']):02d}:00 / "
                f"halten {int(best_int['Intervall (Stunden)'])}h / "
                f"Verkauf {best_int['Verkauf-Uhrzeit']} / "
                f"ROI {best_int['ROI (%)']:.2f}%"
            )
        else:
            print("Keine Intervall-Ergebnisse.")
        print("==========================================\n")


if __name__ == "__main__":
    main()