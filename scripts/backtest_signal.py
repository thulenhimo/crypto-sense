import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def annualization_factor(horizon: str) -> float:
    # Hours per year ~ 24*365 = 8760
    if horizon == "1h":
        return float(8760)
    if horizon == "4h":
        return float(8760 / 4)  # 4h bars per year
    raise ValueError("horizon must be '1h' or '4h'")


def make_positions(
    scores: np.ndarray, mode: str, thr_high: float, thr_low: float
) -> np.ndarray:
    if mode == "longonly":
        pos = np.where(scores > thr_high, 1.0, 0.0)
    elif mode == "longshort":
        pos = np.where(scores > thr_high, 1.0, np.where(scores < thr_low, -1.0, 0.0))
    else:
        raise ValueError("mode must be 'longonly' or 'longshort'")
    return pos.astype(float)


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    dd = equity - peaks
    return float(dd.min())  # negative number (max drop in cumulative PnL)


def backtest(
    df: pd.DataFrame,
    mode: str,
    thr_high: float,
    thr_low: float,
    cost_bps: float,
    horizon: str,
) -> tuple[pd.DataFrame, dict]:
    # Preconditions
    for col in ("timestamp", "true_ret", "clf_score"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in predictions parquet")

    df = df.sort_values("timestamp").reset_index(drop=True)

    scores = df["clf_score"].to_numpy(float)
    rets = df["true_ret"].to_numpy(
        float
    )  # already futret_{H}; i.e., next‑period log return

    pos = make_positions(scores, mode=mode, thr_high=thr_high, thr_low=thr_low)

    # Transaction costs per change in position (one‑way cost in decimal)
    cost_rate = float(cost_bps) / 10_000.0
    delta_pos = np.abs(pos - np.r_[0.0, pos[:-1]])
    costs = cost_rate * delta_pos

    gross = pos * rets
    net = gross - costs
    equity = np.cumsum(net)

    # Metrics
    ann = annualization_factor(horizon)
    mu = float(np.nanmean(net))
    sd = float(np.nanstd(net, ddof=1)) if len(net) > 1 else float("nan")
    sharpe = float(np.sqrt(ann) * mu / sd) if sd > 0 else float("nan")

    # Hit rate on *gross* PnL when in a position
    in_pos = np.abs(pos) > 0
    hit = float((gross[in_pos] > 0).mean()) if in_pos.any() else float("nan")

    # Turnover per bar: average absolute change in position
    turnover = float(delta_pos.mean())

    mdd = max_drawdown_from_equity(equity)  # negative
    total_gross = float(gross.sum())
    total_cost = float(costs.sum())
    total_net = float(net.sum())

    out = df[["timestamp"]].copy()
    out["score"] = scores
    out["position"] = pos
    out["true_ret"] = rets
    out["gross"] = gross
    out["cost"] = costs
    out["net"] = net
    out["equity"] = equity

    metrics = {
        "mode": mode,
        "horizon": horizon,
        "thr_high": float(thr_high),
        "thr_low": float(thr_low),
        "cost_bps": float(cost_bps),
        "n_bars": int(len(out)),
        "in_pos_frac": float(in_pos.mean()),
        "turnover": turnover,
        "hit_rate": hit,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "gross_pnl": total_gross,
        "cost_paid": total_cost,
        "net_pnl": total_net,
    }
    return out, metrics


def main():
    ap = argparse.ArgumentParser(
        description="Tiny backtest on classifier scores with costs"
    )
    ap.add_argument(
        "--preds",
        required=True,
        help="Parquet with columns: timestamp,true_ret,clf_score",
    )
    ap.add_argument("--horizon", choices=["1h", "4h"], default="1h")
    ap.add_argument("--mode", choices=["longonly", "longshort"], default="longshort")
    ap.add_argument(
        "--thr_high", type=float, default=0.6, help="Go long above this score"
    )
    ap.add_argument(
        "--thr_low",
        type=float,
        default=0.4,
        help="Go short below this score (longshort mode)",
    )
    ap.add_argument(
        "--cost_bps", type=float, default=4.0, help="Per trade side cost in bps"
    )
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.preds))

    out, metrics = backtest(
        df=df,
        mode=args.mode,
        thr_high=args.thr_high,
        thr_low=args.thr_low,
        cost_bps=args.cost_bps,
        horizon=args.horizon,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.mode}_{args.horizon}_cost{int(args.cost_bps)}"
    out_path = outdir / f"bt_{tag}.parquet"
    out.to_parquet(out_path)

    metrics_path = outdir / f"bt_{tag}_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved backtest series → {out_path}")
    print(f"Saved metrics → {metrics_path}\n")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
