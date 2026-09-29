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
        except Exception:
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


def compute_buy_and_hold(df, start_date_str, end_date_str, fee_rate, initial_cash):
    """Kauf zum ersten Close im Fenster, Verkauf zum letzten Close."""
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

    first_price = float(window.iloc[0]["close"])
    last_price = float(window.iloc[-1]["close"])
    first_time = window.index[0]
    last_time = window.index[-1]

    crypto = (initial_cash * (1 - fee_rate)) / first_price
    final_value = crypto * last_price * (1 - fee_rate)
    profit = final_value - initial_cash
    roi = (profit / initial_cash) * 100
    raw_move = ((last_price / first_price) - 1.0) * 100

    return {
        "Hold Start": first_time.strftime("%Y-%m-%d %H:%M %Z"),
        "Hold Ende": last_time.strftime("%Y-%m-%d %H:%M %Z"),
        "Hold Startpreis": first_price,
        "Hold Endpreis": last_price,
        "Hold Endkapital": final_value,
        "Hold Gewinn/Verlust": profit,
        "Hold ROI (%)": roi,
        "Hold Kursbewegung (%)": raw_move,
    }


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
            price_buy = df.loc[current_dt, "close"]
            price_sell = df.loc[target_sell_dt, "close"]

            results.append(
                {

                    "Date_Buy": current_dt.strftime("%Y-%m-%d"),
                    "Time_Buy": current_dt.strftime("%H:%M %Z"),
                    "Price_Buy": price_buy,
                    "Date_Sell": target_sell_dt.strftime("%Y-%m-%d"),
                    "Time_Sell": target_sell_dt.strftime("%H:%M %Z"),
                    "Price_Sell": price_sell,
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

    top_200_symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "XRP/USDT",
        "SOL/USDT",
        "TRX/USDT",
        "ZEC/USDT",
        "HYPE/USDT",
        "DOGE/USDT",
        "XMR/USDT",
        "WBT/USDT",
        "ADA/USDT",
        "LINK/USDT",
        "RAIN/USDT",
        "LEO/USDT",
        "XLM/USDT",
        "BCH/USDT",
        "UNI/USDT",
        "NEAR/USDT",
        "LTC/USDT",
        "AVAX/USDT",
        "CC/USDT",
        "HBAR/USDT",
        "SUI/USDT",
        "TON/USDT",
        "SHIB/USDT",
        "CRO/USDT",
        "TAO/USDT",
        "M/USDT",
        "OKB/USDT",
        "AAVE/USDT",
        "MNT/USDT",
        "ENA/USDT",
        "ONDO/USDT",
        "BTW/USDT",
        "PEPE/USDT",
        "PUMP/USDT",
        "WLFI/USDT",
        "SKY/USDT",
        "ICP/USDT",
        "WLD/USDT",
        "ARB/USDT",
        "VVV/USDT",
        "ETC/USDT",
        "BGB/USDT",
        "GT/USDT",
        "LIT/USDT",
        "ASTER/USDT",
        "POL/USDT",
        "KAS/USDT",
        "JST/USDT",
        "AKE/USDT",
        "PI/USDT",
        "ALGO/USDT",
        "KCS/USDT",
        "JUP/USDT",
        "RENDER/USDT",
        "QNT/USDT",
        "FIL/USDT",
        "CAKE/USDT",
        "VET/USDT",
        "DASH/USDT",
        "INJ/USDT",
        "APT/USDT",
        "ATOM/USDT",
        "AERO/USDT",
        "ETHFI/USDT",
        "PENGU/USDT",
        "FLR/USDT",
        "MORPHO/USDT",
        "BDX/USDT",
        "STX/USDT",
        "NEXO/USDT",
        "CRV/USDT",
        "FET/USDT",
        "PYTH/USDT",
        "TRUMP/USDT",
        "RAY/USDT",
        "SEI/USDT",
        "OP/USDT",
        "TIA/USDT",
        "LDO/USDT",
        "IMX/USDT",
        "SAND/USDT",
        "XTZ/USDT",
        "IOTA/USDT",
        "GALA/USDT",
        "FLOW/USDT",
        "XDC/USDT",
        "PENDLE/USDT",
        "STRK/USDT",
        "VIRTUAL/USDT",
        "SPX/USDT",
        "BONK/USDT",
        "W/USDT",
        "ZK/USDT",
        "EIGEN/USDT",
        "MOVE/USDT",
        "BERA/USDT",
        "S/USDT",
        "AR/USDT",
        "FLOKI/USDT",
        "MON/USDT",
        "CFX/USDT",
        "GRT/USDT",
        "SYRUP/USDT",
        "ENS/USDT",
        "WIF/USDT",
        "KITE/USDT",
        "TWT/USDT",
        "THETA/USDT",
        "JASMY/USDT",
        "AKT/USDT",
        "KAIA/USDT",
        "COMP/USDT",
        "RUNE/USDT",
        "CVX/USDT",
        "FARTCOIN/USDT",
        "AXS/USDT",
        "MINA/USDT",
        "XEC/USDT",
        "NEO/USDT",
        "MX/USDT",
        "CHZ/USDT",
        "XCN/USDT",
        "TEL/USDT",
        "MANA/USDT",
        "ZBCN/USDT",
        "AIOZ/USDT",
        "TRAC/USDT",
        "ZRO/USDT",
        "GOMINING/USDT",
        "SFP/USDT",
        "JTO/USDT",
        "1INCH/USDT",
        "MKR/USDT",
        "BAT/USDT",
        "EGLD/USDT",
        "ZEN/USDT",
        "GLM/USDT",
        "FTM/USDT",
        "STG/USDT",
        "APE/USDT",
        "ATH/USDT",
        "ZANO/USDT",
        "FORM/USDT",
        "DOT/USDT",
        "HNT/USDT",
        "KAVA/USDT",
        "ROSE/USDT",
        "ZIL/USDT",
        "CELO/USDT",
        "KSM/USDT",
        "WAVES/USDT",
        "QTUM/USDT",
        "ICX/USDT",
        "ONT/USDT",
        "IOST/USDT",
        "ANKR/USDT",
        "HOT/USDT",
        "ZRX/USDT",
        "SNX/USDT",
        "YFI/USDT",
        "SUSHI/USDT",
        "CRV/USDT",
        "LRC/USDT",
        "ENJ/USDT",
        "STORJ/USDT",
        "SKL/USDT",
        "CTSI/USDT",
        "BAND/USDT",
        "RLC/USDT",
        "OCEAN/USDT",
        "RSR/USDT",
        "CKB/USDT",
        "ONE/USDT",
        "GLMR/USDT",
        "MOVR/USDT",
        "ASTR/USDT",
        "BLUR/USDT",
        "MASK/USDT",
        "GMX/USDT",
        "DYDX/USDT",
        "ORDI/USDT",
        "SATS/USDT",
        "RPL/USDT",
        "SSV/USDT",
        "TNSR/USDT",
        "ALT/USDT",
        "MANTA/USDT",
        "PIXEL/USDT",
        "PORTAL/USDT",
        "ACE/USDT",
        "NFP/USDT",
        "AI/USDT",
        "XAI/USDT",
        "MAVIA/USDT",
        "OMNI/USDT",
        "REZ/USDT",
        "BB/USDT",
        "NOT/USDT",
        "DOGS/USDT",
        "CATI/USDT",
        "ENA/USDT",
        "HMSTR/USDT",
        "SCR/USDT",
        "KAITO/USDT",
        "IP/USDT",
        "INIT/USDT",
    ]

    start_date_str = "2026-05-10"
    end_date_str = "2026-10-01"
    interval_hours = 12
    fee_rate = 0.00075  # 0.10% Spot-Taker; nicht 0.02%
    initial_capital = 10000.0
    offsets_to_test = list(range(24))

    since_timestamp = (
        int(pd.Timestamp(start_date_str).tz_localize(BERLIN).timestamp() * 1000) - 48 * 60 * 60 * 1000
    )

    portfolio_summary = []

    print(
        f"\nStarte Optimierung für {len(top_50_symbols)} Coins "
        f"(teste {len(offsets_to_test)} Offsets von 0-23 Berliner Zeit pro Coin)...\n"
    )

    for symbol in top_200_symbols:
        print(f"Teste: {symbol:<10} ...", end=" ")
        ohlcv = fetch_all_ohlcv(exchange, symbol, "1h", since_timestamp)

        if len(ohlcv) < 100:
            print("Übersprungen (zu wenig Historie).")
            continue

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

        data_start = df.index.min().strftime("%Y-%m-%d")
        data_end = df.index.max().strftime("%Y-%m-%d")

        hold = compute_buy_and_hold(
            df, start_date_str, end_date_str, fee_rate, initial_capital
        )
        if hold is None:
            print("Keine Hold-Daten.")
            continue

        best_offset = None
        best_final_val = -1.0
        best_profit = 0.0
        best_roi = -999.0

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
            sell_hour = (best_offset + interval_hours) % 24
            portfolio_summary.append(
                {
                    "Coin": f"{symbol}",
                    "Bester Offset": f"{best_offset:02d}:00",
                    "Daten von": f"{data_start} bis {data_end}",
                    "Strategie ROI (%)": best_roi,
                    "Hold ROI (%)": hold["Hold ROI (%)"],
                    "Hold Kurs (%)": hold["Hold Kursbewegung (%)"],
                    "Edge vs Hold (%)": best_roi - hold["Hold ROI (%)"]
                }
            )
            print(
                f"Start {best_offset:02d}:00 | Strat {best_roi:+.2f}% | "
                f"Hold {hold['Hold ROI (%)']:+.2f}% | "
                f"Edge {best_roi - hold['Hold ROI (%)']:+.2f}%"
            )
        else:
            print("Keine gültigen Daten in den Offsets.")

    if portfolio_summary:
        summary_df = pd.DataFrame(portfolio_summary)
        summary_df = summary_df.sort_values(by="Strategie ROI (%)", ascending=False)
        pd.set_option("display.float_format", lambda x: "%.2f" % x)

        print("\n" + "=" * 120)
        print(" OPTIMIERUNG inkl. Buy-and-Hold über denselben Zeitraum | alle Zeiten Europe/Berlin")
        print("=" * 120)
        print(summary_df.to_string(index=False))
        print("=" * 120)
        print(
            "Hinweis: Hold kauft zum ersten Close im Fenster und verkauft zum letzten, "
            "mit derselben fee_rate auf beiden Seiten."
        )


if __name__ == "__main__":
    main()
