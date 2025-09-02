import argparse
from pathlib import Path
import numpy as pd
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Aggreagate news sentiment to hourly features"
    )
    ap.add_argument("--in", dest="inp", required=True, help="Scored news Parquet")
    ap.add_argument("--out", default="data/processed/news_sent_hourly.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.inp))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["sent_score"])

    pos = (df["sent_label"].str.lower() == "positive").astype(int)
    neg = (df["sent_label"].str.lower() == "negative").astype(int)

    g = (
        df.set_index("timestamp")
        .resample("1H")
        .agg(
            {
                "sent_score": ["mean", "std", "count"],
            }
        )
    )
    g.columns = ["sent_mean", "sent_std", "sent_n"]

    shares = (
        pd.DataFrame({"pos": pos, "neg": neg, "timestamp": df["timestamp"]})
        .set_index("timestamp")
        .resample("1H")
        .sum()
    )

    # Convert sums to shares (by count in each hour)
    counts = g["sent_n"].replace(0, pd.NA)
    shares["pos_share"] = shares["pos"] / counts
    shares["neg_share"] = shares["neg"] / counts

    out = pd.concat([g, shares[["pos_share", "neg_share"]]], axis=1).reset_index()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"Saved hourly sentiment features ({len(out)} rows) → {args.out}")
    print(out.tail())


if __name__ == "__main__":
    main()
