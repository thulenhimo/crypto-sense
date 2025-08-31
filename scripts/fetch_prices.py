#! /usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path
import yfinance as yf

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize a raw pandas DataFrame from yfinance into a consistent OHLCV format with a UTC timestamp,
    validated ordering, and numeric types.
    """
    if df.empty:
        raise RuntimeError(
            "Empty frame received from yfinance - try a shorter interval or different period"
        )

    # Reset index and make sure column names are lowercase
    data = df.copy()
    # The dataframe has multi-level column index where other "level" contains always the price symbol, let us drop it
    data.columns = data.columns.droplevel(1)
    data = data.rename(columns={c: c.lower() for c in data.columns})
    data = data.reset_index()

    # yfinance uses different index names depending on interval
    if "datetime" in data.columns:
        ts_col = "datetime"
    elif "date" in data.columns:
        ts_col = "date"
    else:
        # Fall back to first column if needed
        ts_col = data.columns[0]

    data = data.rename(columns={ts_col: "timestamp"})
    missing_required_cols = set(REQUIRED_COLS) - set(data.columns)
    if missing_required_cols:
        raise RuntimeError(
            f"Missing expected columns in dataframe return from yfinance: {sorted(missing_required_cols)}"
        )

    # sort by timestamp, drop duplicates, enforce dtypes
    data = data[REQUIRED_COLS].sort_values(by="timestamp")
    data = data.drop_duplicates()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # quick validation/sanity check before returning the data
    if not data["timestamp"].is_monotonic_increasing:
        raise AssertionError(
            "Timestamps are not strictly monotonic increasing after sorting"
        )
    if data[REQUIRED_COLS].isnull().any().any():
        # we'll allow a few NaNs in volume; drop rows with NaNs to keep file clean
        df = df.dropna()

    return data


def fetch(symbol: str, interval: str, period: str) -> pd.DataFrame:
    raw = yf.download(symbol, interval=interval, period=period, progress=False)
    return _normalize_frame(raw)


def main():
    parser = argparse.ArgumentParser(description="Fetch BTC prices")
    parser.add_argument(
        "--symbol", default="BTC-USD", help="Ticker symbol (default: BTC-USD)"
    )
    parser.add_argument(
        "--interval", default="1h", help="Candle interval, e.g., 1h, 1d"
    )
    parser.add_argument(
        "--period", default="730d", help="History period (e.g., 30d, 365d, 730d, max)"
    )
    parser.add_argument(
        "--out",
        default="data/raw/btc_1h.parquet",
        help="Output Parquet path (directories created automatically)",
    )
    args = parser.parse_args()

    df = fetch(args.symbol, args.interval, args.period)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path)

    # Echo a tiny summary so one can eyeball success
    start = df["timestamp"].min()
    end = df["timestamp"].max()
    print(
        f"Saved {len(df):,} rows → {out_path}\n"
        f"Range: {start} → {end}\n"
        f"Columns: {list(df.columns)}"
    )


if __name__ == "__main__":
    main()
