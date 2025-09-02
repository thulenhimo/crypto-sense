import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    mean_squared_error,
)
from scipy.stats import spearmanr


def time_split_idx(n: int, train_frac=0.7, val_frac=0.15):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    test_idx = slice(n_train + n_val, n)
    return train_idx, val_idx, test_idx


def spearman_ic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    res = spearmanr(y_true, y_score, nan_policy="omit")
    return float(res.correlation if hasattr(res, "correlation") else res[0])


def main():
    ap = argparse.ArgumentParser(
        description="Train baseline signal models on time split"
    )
    ap.add_argument(
        "--data", required=True, help="Parquet from build_signal_dataset.py"
    )
    ap.add_argument("--horizon", choices=["1h", "4h"], default="1h")
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.data))
    df = df.sort_values("timestamp").reset_index(drop=True)

    target_col = f"futret_{args.horizon}"

    # Feature set contains all numeric columns except timestamp, close, targets
    drop_cols = {"timestamp", "close", "futret_1h", "futret_4h"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).values
    y = df[target_col].values

    # Train/val/test split by time
    tr, va, te = time_split_idx(len(df))
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    # Scale features (fit on train only)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)

    # Classification: up/down
    ytr_cls = (ytr > 0).astype(int)
    yva_cls = (yva > 0).astype(int)
    yte_cls = (yte > 0).astype(int)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr_cls)
    score_tr = clf.decision_function(Xtr)
    score_va = clf.decision_function(Xva)
    score_te = clf.decision_function(Xte)

    # Classification metrics on test
    auc = roc_auc_score(yte_cls, score_te)
    prc = average_precision_score(yte_cls, score_te)
    acc = accuracy_score(yte_cls, (score_te > 0).astype(int))
    f1 = f1_score(yte_cls, (score_te > 0).astype(int))
    ic_rank_cls = spearman_ic(yte, score_te)

    # Regression on return magnitude
    reg = Ridge(alpha=1.0)
    reg.fit(Xtr, ytr)
    pred_te = reg.predict(Xte)
    mse = mean_squared_error(yte, pred_te)
    ic_rank_reg = spearman_ic(yte, pred_te)

    # Save predictions
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    preds = pd.DataFrame(
        {
            "timestamp": df.loc[te, "timestamp"].values,
            "true_ret": yte,
            "clf_score": score_te,
            "clf_pred": (score_te > 0).astype(int),
            "reg_pred": pred_te,
        }
    )
    preds.to_parquet(outdir / f"signal_test_preds_{args.horizon}.parquet")

    metrics = {
        "horizon": args.horizon,
        "n_train": (
            len(range(*tr.indices(len(df)))) if isinstance(tr, slice) else len(tr)
        ),
        "n_test": (
            len(range(*te.indices(len(df)))) if isinstance(te, slice) else len(te)
        ),
        "clf_auc": float(auc),
        "clf_pr_auc": float(prc),
        "clf_acc": float(acc),
        "clf_f1": float(f1),
        "rank_ic_cls": float(ic_rank_cls),
        "reg_mse": float(mse),
        "rank_ic_reg": float(ic_rank_reg),
    }

    print("=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
