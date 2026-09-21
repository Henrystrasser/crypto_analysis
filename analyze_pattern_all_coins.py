import time
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
            print(f"Fehler beim Laden von {symbol}: {e}")
            break

    return all_ohlcv


def pattern_rate_for_offset(df, offset_hours, interval_hours=12):
    rows = []
    dates = sorted(df["date"].unique())

    for d in dates:
        buy_dt = pd.Timestamp(d) + pd.Timedelta(hours=offset_hours)
        sell_dt = buy_dt + pd.Timedelta(hours=interval_hours)
        if buy_dt not in df.index or sell_dt not in df.index:
            continue

        price_buy = float(df.loc[buy_dt, "close"])
        price_sell = float(df.loc[sell_dt, "close"])
        rows.append({
            "Buy_Time": buy_dt,
            "Sell_Time": sell_dt,
            "Price_1": price_buy,
            "Price_2": price_sell,
            "Rose": price_buy < price_sell,
        })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return None

    result_df["Prev_Price_2"] = result_df["Price_2"].shift(1)
    result_df["Prev_Rose"] = result_df["Rose"].shift(1)
    usable = result_df.dropna(subset=["Prev_Price_2", "Prev_Rose"])
    if usable.empty:
        return None

    def check_condition(row):
        if bool(row["Prev_Rose"]):
            return row["Price_1"] < row["Prev_Price_2"]
        return row["Price_1"] > row["Prev_Price_2"]

    hits = usable.apply(check_condition, axis=1)
    return {
        "Tage": int(len(usable)),
        "Hits": int(hits.sum()),
        "Pattern %": float(hits.mean() * 100.0),
    }


def main():
    days = 300
    interval_hours = 12
    offsets = list(range(24))

    top_50_symbols = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "SUI/USDT",
        "DOT/USDT", "NEAR/USDT", "UNI/USDT", "POL/USDT", "LTC/USDT",
        "BCH/USDT", "APT/USDT", "ICP/USDT", "RENDER/USDT", "FET/USDT",
        "ARB/USDT", "INJ/USDT", "OP/USDT", "ATOM/USDT", "SEI/USDT",
        "TIA/USDT", "PEPE/USDT", "SHIB/USDT", "ETC/USDT", "FIL/USDT",
        "STX/USDT", "IMX/USDT", "AR/USDT", "FTM/USDT", "GRT/USDT",
        "RUNE/USDT", "ALGO/USDT", "XLM/USDT", "HBAR/USDT", "VET/USDT",
        "THETA/USDT", "JUP/USDT", "PENDLE/USDT", "WIF/USDT", "BONK/USDT",
        "FLOKI/USDT", "KAS/USDT", "AKT/USDT", "TON/USDT", "EOS/USDT",
    ]

    exchange = ccxt.binance({"enableRateLimit": True})
    since_timestamp = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)

    print("Pattern:")
    print("  gestern Anstieg → heute Morgen < gestern Abend")
    print("  gestern Abstieg → heute Morgen > gestern Abend")
    print("Fenster: Kauf = Offset, Verkauf = Offset + 12h (auch über Mitternacht)")
    print(f"Tage: {days}  |  Offsets: 0–23\n")

    summary_rows = []

    for i, symbol in enumerate(top_50_symbols, start=1):
        print(f"[{i}/{len(top_50_symbols)}] Lade {symbol} ...")
        try:
            ohlcv = fetch_all_ohlcv(exchange, symbol, "1h", since_timestamp)
        except Exception as e:
            print(f"  übersprungen: {e}\n")
            continue

        if not ohlcv:
            print("  keine Daten\n")
            continue

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df["date"] = df.index.date

        rates = {}
        for offset in offsets:
            stats = pattern_rate_for_offset(df, offset, interval_hours)
            if stats is None:
                rates[offset] = None
                continue
            rates[offset] = stats["Pattern %"]
            summary_rows.append({
                "Symbol": symbol,
                "Offset": offset,
                "Kauf": f"{offset:02d}:00",
                "Verkauf": f"{(offset + interval_hours) % 24:02d}:00",
                "Tage": stats["Tage"],
                "Hits": stats["Hits"],
                "Pattern %": stats["Pattern %"],
            })

        printable = [
            f"{o:02d}:{rates[o]:.1f}%" if rates[o] is not None else f"{o:02d}:—"
            for o in offsets
        ]
        print("  " + "  ".join(printable[:12]))
        print("  " + "  ".join(printable[12:]))
        valid = [r for r in rates.values() if r is not None]
        if valid:
            print(
                f"  Mittel {sum(valid)/len(valid):.1f}% | "
                f"Max {max(valid):.1f}% | Min {min(valid):.1f}%"
            )
        print()
        time.sleep(0.2)

    if not summary_rows:
        print("Keine Ergebnisse.")
        return

    long_df = pd.DataFrame(summary_rows)
    pivot = long_df.pivot(index="Symbol", columns="Offset", values="Pattern %")

    pd.set_option("display.float_format", lambda x: f"{x:.1f}")
    pd.set_option("display.max_columns", 24)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 60)

    print("\n==========================================")
    print(" PATTERN % JE COIN UND OFFSET")
    print("==========================================")
    print(pivot.to_string())

    best = long_df.sort_values("Pattern %", ascending=False).head(20)
    print("\nTop 20 Kombinationen:")
    print(best.to_string(index=False))

    per_coin = (
        long_df.groupby("Symbol")["Pattern %"]
        .agg(Mittel="mean", Max="max", Min="min")
        .sort_values("Max", ascending=False)
    )
    print("\nPro Coin über alle Offsets:")
    print(per_coin.to_string())


if __name__ == "__main__":
    main()