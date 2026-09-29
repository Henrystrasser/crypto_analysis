from zoneinfo import ZoneInfo

import ccxt
import pandas as pd

# Alle Uhrzeiten/Offsets in diesem Skript sind deutsche Zeit (Europe/Berlin,
# Sommer-/Winterzeit automatisch). ccxt liefert UTC; umgerechnet wird genau
# einmal beim Laden der Daten (siehe main()).
BERLIN = ZoneInfo("Europe/Berlin")


def berlin_wallclock(day, hour):
    """
    Berliner Ortszeit `hour`:00 am Berliner Kalendertag `day` als tz-aware
    Timestamp (oder None, wenn es diese Uhrzeit an dem Tag nicht gibt).

    DST-Regeln (so gewählt, dass jeder Offset pro Tag höchstens EINEN Trade hat
    und kein Offset einen anderen dupliziert):
      - Frühjahr (23h-Tag, z.B. 2025-03-30): 02:00 existiert nicht (Uhr springt
        02:00 CET -> 03:00 CEST). -> None: Offset 2 hat an diesem Tag keine
        Kerze und wird übersprungen (wie eine fehlende Kerze). Bewusst KEIN
        Verschieben auf 03:00, sonst wären Offset 2 und 3 an dem Tag identisch
        (Doppelzählung im 5er-Fenster).
      - Herbst (25h-Tag, z.B. 2025-10-26): 02:00 gibt es zweimal
        (02:00 CEST = 00:00 UTC und 02:00 CET = 01:00 UTC). Es zählt nur das
        ERSTE Auftreten (CEST, 00:00 UTC). Die zweite 02:00-Stunde ist an diesem
        Tag für keinen Offset Kaufzeitpunkt -> keine Doppelzählung.
    Alle anderen Stunden sind eindeutig. Das naive "Tag + hour" hier ist nur das
    Etikett der Wanduhrzeit; alle Dauern danach (Hold-Fenster) werden mit
    tz-aware Timestamps gerechnet und sind damit echte vergangene Zeit.
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
    """Kauf zum ersten Close im Fenster, Verkauf zum letzten Close (Berlin-Tage)."""
    window = df.copy()
    if start_date_str:
        # Mitternacht Berlin des Starttags (Mitternacht ist in Berlin nie DST-mehrdeutig)
        start_ts = pd.Timestamp(start_date_str).tz_localize(BERLIN)
        window = window[window.index >= start_ts]
    if end_date_str:
        # Mitternacht Berlin des Folgetags (Kalendertag +1, nicht +24h)
        end_ts = (pd.Timestamp(end_date_str) + pd.Timedelta(days=1)).tz_localize(BERLIN)
        window = window[window.index <= end_ts]

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

    # Vorfilter auf das Zeitfenster: Kauf-Datum ist immer der Berliner Tag d
    # (Offset < 24), daher identisch zum Nachfilter unten, nur schneller.
    if start_date_str:
        dates = [d for d in dates if str(d) >= start_date_str]
    if end_date_str:
        dates = [d for d in dates if str(d) <= end_date_str]

    for d in dates:
        # Kauf = Berliner Ortszeit offset_hours:00 am Tag d (DST-Regeln siehe
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


def find_best_offset(
    df, interval_hours, start_date_str, end_date_str, offsets_to_test, fee_rate, initial_cash
):
    """
    Besten Offset (Berliner Stunde 0–23) im Zeitfenster finden.

    Score eines Offsets o = Summe der ROIs des 5er-Fensters
    (o-2, o-1, o, o+1, o+2) mod 24 — also inkl. des Offsets selbst,
    zyklisch über 0–23. Gewählt wird der Offset mit der höchsten
    Fenster-Summe, nicht der einzelne Peak — das dämpft stark springende
    Offsets. Ein Offset ist nur wählbar, wenn alle 5 Fenster-Stunden ein
    Ergebnis haben.

    Fallback (kein Offset mit vollständigem 5er-Fenster): der einzelne
    Offset mit dem höchsten Endwert (Fenster nur aus dem Offset selbst).
    """
    roi_by_offset = {}
    val_by_offset = {}

    for offset in offsets_to_test:
        result_df = generate_result_df(
            df, offset, interval_hours, start_date_str, end_date_str
        )
        if result_df.empty:
            continue

        final_val, _ = run_simulation(
            result_df, fee_rate=fee_rate, initial_cash=initial_cash
        )
        val_by_offset[offset] = final_val
        roi_by_offset[offset] = (final_val / initial_cash - 1.0) * 100

    if not roi_by_offset:
        return None, -1.0

    window_deltas = (-2, -1, 0, 1, 2)
    best_offset = None
    best_window_sum = None

    for offset in roi_by_offset:
        window_rois = []
        for d in window_deltas:
            nb = (offset + d) % 24
            if nb not in roi_by_offset:
                window_rois = None
                break
            window_rois.append(roi_by_offset[nb])
        if window_rois is None:
            continue
        window_sum = sum(window_rois)
        if best_window_sum is None or window_sum > best_window_sum:
            best_window_sum = window_sum
            best_offset = offset

    # Fallback: einzelner Peak, falls kein 5er-Fenster vollständig ist
    if best_offset is None:
        best_offset = max(val_by_offset, key=val_by_offset.get)

    return best_offset, val_by_offset[best_offset]


def generate_single_day_row(df, day, offset_hours, interval_hours):
    # Kauf = Berliner Ortszeit offset_hours:00 am Tag `day` (DST-Regeln siehe
    # berlin_wallclock); Verkauf = Kauf + interval_hours echte Stunden.
    current_dt = berlin_wallclock(day, offset_hours)
    if current_dt is None:
        return None
    target_sell_dt = current_dt + pd.Timedelta(hours=interval_hours)

    if current_dt not in df.index or target_sell_dt not in df.index:
        return None

    return {
        "Date_Buy": current_dt.strftime("%Y-%m-%d"),
        "Time_Buy": current_dt.strftime("%H:%M %Z"),
        "Price_Buy": df.loc[current_dt, "close"],
        "Date_Sell": target_sell_dt.strftime("%Y-%m-%d"),
        "Time_Sell": target_sell_dt.strftime("%H:%M %Z"),
        "Price_Sell": df.loc[target_sell_dt, "close"],
        "Offset": offset_hours,
    }


def run_daily_walk_forward(
    df,
    start_date_str,
    end_date_str,
    lookback_days,
    interval_hours,
    offsets_to_test,
    fee_rate,
    initial_capital,
):
    """
    Für jeden Handelstag (Berliner Kalendertag, Grenze = Mitternacht Berlin):
      1) besten Offset (5er-Fenster-Regel) auf den letzten lookback_days Tagen
      2) an diesem Tag mit genau diesem Offset handeln (Long interval_hours)
    Kapital wird über die Tage fortgeschrieben.
    Rückgabe: (Endwert, benutzte Offsets, Tages-Logs, Trade-Log-DataFrame)
    """
    start = pd.Timestamp(start_date_str).normalize()
    end = pd.Timestamp(end_date_str).normalize()

    # Tage sind naive Kalenderdaten (Berliner Datum); Tagesarithmetik unten ist
    # reine Kalenderarithmetik und damit DST-sicher.
    trade_days = sorted(pd.to_datetime(pd.unique(df["date"])))
    trade_days = [d.normalize() for d in trade_days if start <= d.normalize() <= end]

    rows = []
    offsets_used = []
    day_logs = []

    for day in trade_days:
        prev_start = (day - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        prev_end = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        best_offset, _ = find_best_offset(
            df,
            interval_hours,
            prev_start,
            prev_end,
            offsets_to_test,
            fee_rate,
            initial_capital,
        )

        if best_offset is None:
            day_logs.append(
                {
                    "Tag": day.strftime("%Y-%m-%d"),
                    "Offset": None,
                    "Hinweis": "kein Offset aus Lookback",
                }
            )
            continue

        row = generate_single_day_row(df, day, best_offset, interval_hours)
        if row is None:
            day_logs.append(
                {
                    "Tag": day.strftime("%Y-%m-%d"),
                    "Offset": best_offset,
                    "Hinweis": "keine Kerzen für Buy/Sell",
                }
            )
            continue

        rows.append(row)
        offsets_used.append(int(best_offset))
        day_logs.append(
            {
                "Tag": day.strftime("%Y-%m-%d"),
                "Offset": best_offset,
                "Hinweis": "",
            }
        )

    if not rows:
        return initial_capital, offsets_used, day_logs, pd.DataFrame()

    result_df = pd.DataFrame(rows)
    final_val, trade_log = run_simulation(
        result_df, fee_rate=fee_rate, initial_cash=initial_capital
    )
    return final_val, offsets_used, day_logs, trade_log


def main():
    exchange = ccxt.binance()

    top_200_symbols = [
        "NEXO/USDT"
    ]

    start_date_str = "2025-05-15"
    end_date_str = "2026-10-01"
    print(start_date_str," - ",end_date_str)
    interval_hours = 12
    lookback_days = 100  # Offset jeden Tag neu aus den letzten N Tagen
    fee_rate = 0.00075  # 0.075% Spot-Taker
    initial_capital = 10000.0
    offsets_to_test = list(range(24))

    # Extra Historie: Lookback vor Start + 48h Puffer (Start = Mitternacht Berlin)
    lookback_ms = (lookback_days * 24 + 48) * 60 * 60 * 1000
    since_timestamp = (
        int(pd.Timestamp(start_date_str).tz_localize(BERLIN).timestamp() * 1000)
        - lookback_ms
    )

    portfolio_summary = []

    print(
        f"\nTägliches Walk-Forward für {len(top_200_symbols)} Coins | "
        f"Lookback={lookback_days} Tage | Offsets 0-23 (Berlin-Zeit) | "
        f"Hold-Fenster {interval_hours}h | 5er-Fenster-Regel (o-2..o+2)\n"
    )

    for symbol in top_200_symbols:
        print(f"Teste: {symbol:<14} ...", end=" ", flush=True)
        ohlcv = fetch_all_ohlcv(exchange, symbol, "1h", since_timestamp)

        if len(ohlcv) < 100:
            print("Übersprungen (zu wenig Historie).")
            continue

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Einmalige Umrechnung: ccxt UTC-ms -> tz-aware UTC -> Europe/Berlin.
        # Index bleibt tz-aware (eindeutige Zeitpunkte, auch am 25h-Tag).
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

        final_val, offsets_used, day_logs, trade_log = run_daily_walk_forward(
            df,
            start_date_str,
            end_date_str,
            lookback_days,
            interval_hours,
            offsets_to_test,
            fee_rate,
            initial_capital,
        )

        if not offsets_used:
            print("Kein gültiger Walk-Forward-Offset.")
            continue

        profit = final_val - initial_capital
        roi = (profit / initial_capital) * 100
        last_offset = offsets_used[-1]
        offsets_list_str = str(offsets_used)

        portfolio_summary.append(
            {
                "Coin": symbol,
                "Letzter Offset": f"{last_offset:02d}:00 Berlin",
                "Beste Offsets": offsets_list_str,
                "Tage": len(offsets_used),
                "Daten von": f"{data_start} bis {data_end}",
                "Strategie ROI (%)": roi,
                "Hold ROI (%)": hold["Hold ROI (%)"],
                "Hold Kurs (%)": hold["Hold Kursbewegung (%)"],
                "Edge vs Hold (%)": roi - hold["Hold ROI (%)"],
            }
        )
        print(
            f"letzter Offset {last_offset:02d}:00 Berlin | Strat {roi:+.2f}% | "
            f"Hold {hold['Hold ROI (%)']:+.2f}% | "
            f"Edge {roi - hold['Hold ROI (%)']:+.2f}% | "
            f"Offsets {offsets_list_str}"
        )
        if not trade_log.empty:
            first_t = trade_log.iloc[0]
            last_t = trade_log.iloc[-1]
            print(
                f"    Erster Trade: {first_t['Action']} {first_t['Time']} "
                f"@ {first_t['Price']:.6g} | Letzter Trade: {last_t['Action']} "
                f"{last_t['Time']} @ {last_t['Price']:.6g} | "
                f"Hold {hold['Hold Start']} -> {hold['Hold Ende']}"
            )

    if portfolio_summary:
        summary_df = pd.DataFrame(portfolio_summary)
        summary_df = summary_df.sort_values(by="Strategie ROI (%)", ascending=False)
        pd.set_option("display.float_format", lambda x: "%.2f" % x)
        pd.set_option("display.max_colwidth", 200)
        pd.set_option("display.width", 200)

        print("\n" + "=" * 140)
        print(
            f" TÄGLICHES WALK-FORWARD (Lookback {lookback_days}d, 5er-Fenster-Regel) "
            "| inkl. Buy-and-Hold | alle Zeiten Europe/Berlin"
        )
        print("=" * 140)
        print(summary_df.to_string(index=False))
        print("=" * 140)
        print("\nBester Offset je Handelstag (Berliner Stunde, Reihenfolge = Zeitverlauf):\n")
        for _, row in summary_df.iterrows():
            print(f"{row['Coin']:<16} {row['Beste Offsets']}")
        print("=" * 140)
        print(
            "Hinweis: Für jeden Handelstag (Berliner Kalendertag) wird der beste "
            f"Offset aus den letzten {lookback_days} Tagen bestimmt (Score = Summe "
            "der ROIs des 5er-Fensters Offset-2 .. Offset+2 inkl. Offset selbst, "
            "zyklisch) und nur an diesem Tag gehandelt. "
            "Alle Uhrzeiten/Offsets in deutscher Zeit (Europe/Berlin, inkl. "
            "Sommer-/Winterzeit). "
            "Kapital wird über die Tage fortgeschrieben. "
            "Hold kauft zum ersten Close im Gesamtfenster und verkauft zum letzten, "
            "mit derselben fee_rate auf beiden Seiten."
        )


if __name__ == "__main__":
    main()
