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


def generate_result_df(df, offset_hours, interval_hours, start_ts, end_ts):
    results = []
    dates = sorted(set(df.loc[start_ts:end_ts].index.date)) if not df.empty else []

    for d in dates:
        buy_dt = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=offset_hours)
        sell_dt = buy_dt + pd.Timedelta(hours=interval_hours)

        if buy_dt < start_ts or buy_dt >= end_ts:
            continue
        if buy_dt not in df.index or sell_dt not in df.index:
            continue

        results.append({
            "Date_Buy": buy_dt.strftime("%Y-%m-%d"),
            "Time_Buy": buy_dt.strftime("%H:%M"),
            "Price_Buy": float(df.loc[buy_dt, "close"]),
            "Date_Sell": sell_dt.strftime("%Y-%m-%d"),
            "Time_Sell": sell_dt.strftime("%H:%M"),
            "Price_Sell": float(df.loc[sell_dt, "close"]),
        })

    return pd.DataFrame(results)


def run_simulation(results_df, fee_rate=0.0, initial_cash=10000.0):
    cash = initial_cash
    crypto = 0.0
    position_open = False

    for _, row in results_df.iterrows():
        if not position_open:
            price_buy = row["Price_Buy"]
            crypto = (cash * (1.0 - fee_rate)) / price_buy
            cash = 0.0
            position_open = True

        if position_open:
            price_sell = row["Price_Sell"]
            cash = (crypto * price_sell) * (1.0 - fee_rate)
            crypto = 0.0
            position_open = False

    if position_open and not results_df.empty:
        cash = crypto * results_df.iloc[-1]["Price_Sell"]

    return cash


def pick_robust_offset(rows, keep_ratio=0.75):
    """
    Wählt nicht das nackte Maximum, sondern das Offset
    in der Hochzone mit dem besten 3-Stunden-Umfeld.
    """
    roi = {int(r["offset"]): float(r["roi"]) for r in rows}
    peak = max(rows, key=lambda x: x["roi"])
    cutoff = keep_ratio * peak["roi"]

    best_row = None
    best_score = None

    for row in rows:
        if row["roi"] < cutoff:
            continue
        prev_h = (row["offset"] - 1) % 24
        next_h = (row["offset"] + 1) % 24
        if prev_h not in roi or next_h not in roi:
            continue
        score = (roi[prev_h] + row["roi"] + roi[next_h]) / 3.0
        if best_score is None or score > best_score:
            best_score = score
            best_row = dict(row)
            best_row["neighbor_score"] = score
            best_row["cutoff"] = cutoff
            best_row["peak_offset"] = peak["offset"]
            best_row["peak_roi"] = peak["roi"]

    if best_row is None:
        best_row = dict(peak)
        best_row["neighbor_score"] = peak["roi"]
        best_row["cutoff"] = cutoff
        best_row["peak_offset"] = peak["offset"]
        best_row["peak_roi"] = peak["roi"]

    return best_row


def find_best_offset(
    df, start_ts, end_ts, interval_hours, fee_rate, initial_cash, keep_ratio=0.75
):
    rows = []

    for offset in range(24):
        result_df = generate_result_df(df, offset, interval_hours, start_ts, end_ts)
        if result_df.empty:
            continue

        final_val = run_simulation(
            result_df, fee_rate=fee_rate, initial_cash=initial_cash
        )
        roi = (final_val / initial_cash - 1.0) * 100.0
        rows.append({
            "offset": offset,
            "trades": len(result_df),
            "final": final_val,
            "roi": roi,
        })

    if not rows:
        return None

    return pick_robust_offset(rows, keep_ratio=keep_ratio)


def main():
    symbol = "UNI/USDT"
    lookback_months = 12
    sim_months = 34
    interval_hours = 12
    fee_rate = 0.0001
    initial_cash = 10000.0
    KEEP_RATIO = 0.75

    # True  = auch handeln, wenn das gewählte Trainings-Offset negativ war
    # False = bei negativem Trainings-ROI das nächste Fenster aussetzen
    TRADE_IF_NEGATIVE_OFFSET = False
    TRADE_IF_NEGATIVE_OFFSET = True

    exchange = ccxt.binance()
    now = pd.Timestamp.now(tz="UTC").floor("h")
    trade_start = now - pd.DateOffset(months=sim_months)
    data_start = trade_start - pd.DateOffset(months=lookback_months)
    since_timestamp = int(data_start.timestamp() * 1000)

    print(f"Lade Daten für {symbol} ab {data_start.date()} ...")
    ohlcv = fetch_all_ohlcv(exchange, symbol, "1h", since_timestamp)

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    cash = initial_cash
    window_logs = []
    window_start = trade_start

    print("\n==========================================")
    print(" WALK-FORWARD SIMULATION")
    print("==========================================")
    print(f"Coin                 : {symbol}")
    print(f"Handelszeitraum      : {trade_start.date()} bis {now.date()}")
    print(f"Fenster              : {lookback_months} Monate")
    print(f"Haltedauer           : {interval_hours} Stunden")
    print(f"Gebühr je Seite      : {fee_rate * 100:.3f} %")
    print(f"Offset-Wahl          : robust (Hochzone + 3h-Schnitt, ratio={KEEP_RATIO})")
    print(f"Negatives Offset     : {'handeln' if TRADE_IF_NEGATIVE_OFFSET else 'aussetzen'}")
    print(f"Startkapital         : {initial_cash:,.2f}")
    print("==========================================\n")

    while window_start < now:
        window_end = min(window_start + pd.DateOffset(months=lookback_months), now)
        train_start = window_start - pd.DateOffset(months=lookback_months)
        train_end = window_start

        best = find_best_offset(
            df,
            train_start,
            train_end,
            interval_hours,
            fee_rate,
            cash,
            keep_ratio=KEEP_RATIO,
        )

        used_offset = None
        train_roi = None
        start_cash = cash
        trades = 0

        if best is None:
            action = "Kein Trainingsergebnis – kein Handel"
        elif best["roi"] <= 0 and not TRADE_IF_NEGATIVE_OFFSET:
            action = "Robustes Offset negativ – kein Handel"
            used_offset = best["offset"]
            train_roi = best["roi"]
        else:
            used_offset = best["offset"]
            train_roi = best["roi"]
            result_df = generate_result_df(
                df, used_offset, interval_hours, window_start, window_end
            )
            trades = len(result_df)
            if result_df.empty:
                action = "Offset gewählt, aber keine Trades im Fenster"
            else:
                cash = run_simulation(
                    result_df, fee_rate=fee_rate, initial_cash=cash
                )
                if best["roi"] <= 0:
                    action = "Gehandelt (negatives Offset erlaubt)"
                else:
                    action = "Gehandelt"

        window_roi = (cash / start_cash - 1.0) * 100.0 if start_cash else 0.0
        sell_hour = None if used_offset is None else (used_offset + interval_hours) % 24

        window_logs.append({
            "Fenster Start": window_start.strftime("%Y-%m-%d"),
            "Fenster Ende": window_end.strftime("%Y-%m-%d"),
            "Train ROI %": train_roi,
            "Offset": used_offset,
            "Peak Offset": None if best is None else best.get("peak_offset"),
            "Kauf": None if used_offset is None else f"{used_offset:02d}:00",
            "Verkauf": None if sell_hour is None else f"{sell_hour:02d}:00",
            "Aktion": action,
            "Trades": trades,
            "Start": start_cash,
            "Ende": cash,
            "Fenster %": window_roi,
        })

        peak_txt = ""
        if best is not None:
            peak_txt = f" | Peak-Offset: {best.get('peak_offset')}"

        print(
            f"{window_start.date()} → {window_end.date()} | "
            f"Train-ROI: {train_roi if train_roi is not None else float('nan'):.2f}% | "
            f"Offset: {used_offset}{peak_txt} | {action} | "
            f"{start_cash:,.2f} → {cash:,.2f}"
        )

        window_start = window_end

    profit = cash - initial_cash
    roi = (profit / initial_cash) * 100.0
    log_df = pd.DataFrame(window_logs)

    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print("\nFENSTER-ÜBERSICHT:")
    print(log_df.to_string(index=False))
    print("\n==========================================")
    print(f"Endkapital      : {cash:,.2f}")
    print(f"Gewinn/Verlust  : {profit:+,.2f} ({roi:+.2f} %)")
    print("==========================================")


if __name__ == "__main__":
    main()