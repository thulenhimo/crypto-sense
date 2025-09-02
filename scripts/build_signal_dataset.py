import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    # Ensure exactly hourly frequency; if needed, round to hour
    df["timestamp"] = df["timestamp"].dt.floor("h")
    # If gaps exist, keep them (targets will naturally be NaN at boundaries)
    # Compute log returns (close-to-close)
    df["ret_1h"] = np.log(df["close"]).diff()
    # Future returns: 1h and 4h ahead
    df["futret_1h"] = df["ret_1h"].shift(-1)
    df["ret_4h"] = np.log(df["close"]).diff(4)
    df["futret_4h"] = df["ret_4h"].shift(-4)  # equivalent to log(close_t+4/close_t)
    return df[["timestamp", "close", "ret_1h", "futret_1h", "futret_4h"]]


def load_sentiment(path: str) -> pd.DataFrame:
    s = pd.read_parquet(path)
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True)
    s = s.sort_values("timestamp").drop_duplicates("timestamp")
    # Expected columns: sent_mean, sent_std, sent_n, pos_share, neg_share
    keep = ["timestamp", "sent_mean", "sent_std", "sent_n", "pos_share", "neg_share"]
    missing = [c for c in keep if c not in s.columns]
    if missing:
        raise SystemExit(f"Missing sentiment columns: {missing}")
    return s[keep]


def make_features(sent: pd.DataFrame) -> pd.DataFrame:
    s = sent.copy()
    # Replace NaNs (hours with no news) by neutral stats
    s[["sent_mean", "sent_std", "pos_share", "neg_share"]] = s[
        ["sent_mean", "sent_std", "pos_share", "neg_share"]
    ].fillna(0.0)
    s["sent_n"] = s["sent_n"].fillna(0.0)
    # let us ceate a bonud feature that is a tiny bit of memory with 3h rolling means (no look‑ahead)
    s = s.set_index("timestamp").sort_index()
    s["sent_mean_3h"] = s["sent_mean"].rolling(3, min_periods=1).mean()
    s["pos_share_3h"] = s["pos_share"].rolling(3, min_periods=1).mean()
    s = s.reset_index()
    return s


def main():
    ap = argparse.ArgumentParser(
        description="Join prices with hourly sentiment and build targets"
    )
    ap.add_argument("--prices", default="data/raw/btc_1h.parquet")
    ap.add_argument("--sent", required=True, help="Hourly sentiment parquet from Day 3")
    ap.add_argument("--out", default="data/processed/signal_dataset.parquet")
    args = ap.parse_args()

    prices = load_prices(args.prices)
    sent = load_sentiment(args.sent)
    sent = make_features(sent)

    # Left join on hourly timestamp; features are at time t, targets are beyond t
    df = prices.merge(sent, on="timestamp", how="left")

    # Drop rows where targets are not available (end of series)
    df = df.dropna(subset=["futret_1h", "futret_4h"]).reset_index(drop=True)

    # Final column order
    cols_front = ["timestamp", "close", "futret_1h", "futret_4h"]
    # Rest of the columns are features except for the ret_*h used to compute the future return
    feature_cols = [
        c for c in df.columns if c not in cols_front and c not in ("ret_1h", "ret_4h")
    ]
    df = df[cols_front + feature_cols]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    print(df.tail())
    print(f"Saved dataset with {len(df)} rows, {len(feature_cols)} features → {out}")
    print("Features:", feature_cols)


if __name__ == "__main__":
    main()
