import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    ap = argparse.ArgumentParser(
        description="Train on full dataset and export scaler+clf"
    )
    ap.add_argument("--data", required=True, help="signal_dataset.parquet")
    ap.add_argument("--horizon", choices=["1h", "4h"], default="1h")
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    df = pd.read_parquet(args.data).sort_values("timestamp")
    y = (df[f"futret_{args.horizon}"].values > 0).astype(int)
    drop = {"timestamp", "close", "futret_1h", "futret_4h"}
    X = df.drop(columns=[c for c in drop if c in df.columns]).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xs, y)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, outdir / f"scaler_{args.horizon}.pkl")
    joblib.dump(clf, outdir / f"clf_{args.horizon}.pkl")

    print("Saved:", outdir / f"scaler_{args.horizon}.pkl")
    print("Saved:", outdir / f"clf_{args.horizon}.pkl")


if __name__ == "__main__":
    main()
