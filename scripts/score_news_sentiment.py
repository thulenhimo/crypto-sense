import argparse
from pathlib import Path
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

# TODO: move these functions into one suitable module / subpackage
from eval_fpb_baselines import ensure_vader, vader_predict, finbert_predict


def vader_score(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    res = vader_predict(texts)
    return res["label"], res["score"]


def finbert_score(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    res = finbert_predict(texts)
    return res["label"], res["score"]


def main():
    ap = argparse.ArgumentParser(
        description="Score crypto news Parquet with a sentiment model"
    )
    ap.add_argument("--in", dest="inp", default="data/raw/crypto_news.parquet")
    ap.add_argument("--model", choices=["vader", "finbert"], required=True)
    ap.add_argument("--out", default="data/raw/news_scored.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.inp))
    texts = (
        (df["title"].fillna("") + ". " + df["summary"].fillna("")).astype(str).tolist()
    )

    if args.model == "vader":
        lab, score = vader_score(texts)
    else:
        lab, score = finbert_score(texts)

    out = df.copy()
    out["sent_label"] = lab
    out["sent_score"] = score
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)

    print(f"Scored {len(out):,} articles → {args.out}")
    print(out[["timestamp", "sent_label", "sent_score"]].head())


if __name__ == "__main__":
    main()
