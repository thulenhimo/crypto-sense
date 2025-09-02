import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
)
import torch
import matplotlib.pyplot as plt

# VADER
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# FinBERT
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    AlbertTokenizerFast,
)


LABELS = ["negative", "neutral", "positive"]


def ensure_vader() -> SentimentIntensityAnalyzer:
    """Makes sure that the vader is downloaded and returns the sentiment analyzer"""
    try:
        _ = nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

    return SentimentIntensityAnalyzer()


def vader_predict(texts: list[str]) -> dict[str, np.ndarray]:
    analyzer = ensure_vader()
    compound_scores = np.array(
        [analyzer.polarity_scores(t)["compound"] for t in texts], dtype=float
    )
    labels = np.where(
        compound_scores >= 0.05,
        "positive",
        np.where(compound_scores <= -0.05, "negative", "neutral"),
    )
    return {"label": labels, "score": compound_scores}


def finbert_predict(texts: list[str], device: str = None) -> dict[str, np.ndarray]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = "ProsusAI/finbert"  # https://huggingface.co/ProsusAI/finbert
    tok: AlbertTokenizer | AlbertTokenizerFast = AutoTokenizer.from_pretrained(name)
    model: AlbertForSequenceClassification = (
        AutoModelForSequenceClassification.from_pretrained(name)
    )
    model.to(device)

    model.eval()
    all_labels = []
    all_scores = []
    batch_size = 32
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tok(
                batch,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            pred_ids = probs.argmax(axis=1)

            # FinBERT label order: ['positive','negative','neutral'] or similar depending on config
            # We map using model.config.id2label to be safe
            id2label = model.config.id2label
            labels = [id2label[j] for j in pred_ids]

            # Build p(pos)-p(neg)
            # Find indices by label name (case-insensitive)
            lab2id = {v.lower(): k for k, v in id2label.items()}
            pos_idx = lab2id.get("positive")
            neg_idx = lab2id.get("negative")
            score = probs[:, pos_idx] - probs[:, neg_idx]
            all_labels.extend(labels)
            all_scores.extend(score.tolist())

    return {
        "label": np.array(all_labels, dtype=object),
        "score": np.array(all_scores, dtype=float),
    }


def load_split_files(
    in_dir: Path, splits=("fpb_val.parquet", "fpb_test.parquet")
) -> dict[str, pd.DataFrame]:
    out = {}
    for s in splits:
        p = in_dir / s
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {s}")
        out[s.split(".")[0].split("_")[-1]] = pd.read_parquet(p)
    return out


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate baselines on Financial PhraseBank"
    )
    ap.add_argument(
        "--in_dir", default="data/processed", help="Directory with fpb_*.parquet"
    )
    ap.add_argument("--model", choices=["vader", "finbert"], required=True)
    ap.add_argument(
        "--out_dir", default="data/processed", help="Where to save scored parquet"
    )
    ap.add_argument("-p", "--plot", action="store_true", help="Plot confusion matrices")
    args = ap.parse_args()

    splits = load_split_files(Path(args.in_dir))

    for split_name, df in splits.items():
        texts = df["text"].fillna("").astype(str).tolist()
        if args.model == "vader":
            pred = vader_predict(texts)
        else:
            pred = finbert_predict(texts)

        df_out = df.copy()
        df_out["pred_label"] = pred["label"]
        df_out["pred_score"] = pred["score"]

        m = metrics(df_out["label"].values, df_out["pred_label"].values)
        cm = confusion_matrix(
            df_out["label"].values, df_out["pred_label"].values, labels=LABELS
        )

        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(out_path / f"fpb_{split_name}_{args.model}.parquet")

        print(f"=== {args.model.upper()} on FPB {split_name} ===")
        print({k: round(v, 4) for k, v in m.items()})
        print("labels:", LABELS)
        print("confusion_matrix:", cm)

        if args.plot:
            ConfusionMatrixDisplay(confusion_matrix=cm).plot()
            plt.title(f"{args.model.upper()} on FPB {split_name}")
            plt.show()


if __name__ == "__main__":
    main()
