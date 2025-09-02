import argparse
from datasets import load_dataset
import pandas as pd
from pathlib import Path


CONFIG = "sentences_allagree"  # alternatives: sentences_50agree, _66agree, _75agree

LABELS = ["negative", "neutral", "positive"]  # dataset uses string labels per HF card
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

TRAIN, VAL, TEST = 0.8, 0.1, 0.1


def main():
    ap = argparse.ArgumentParser(
        description="Download Financial PhraseBank and make splits"
    )
    ap.add_argument(
        "--outdir", default="data/processed", help="Output directory for Parquet files"
    )
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    ds = load_dataset("takala/financial_phrasebank", CONFIG)
    # There's no train/validation/test split.
    # However the dataset is available in four possible configurations depending on
    # the percentage of agreement of annotators:
    # sentences_50agree; Number of instances with >=50% annotator agreement: 4846
    # sentences_66agree: Number of instances with >=66% annotator agreement: 4217
    # sentences_75agree: Number of instances with >=75% annotator agreement: 3453
    # sentences_allagree: Number of instances with 100% annotator agreement: 2264
    # HF returns a single split named 'train' for this dataset
    df = pd.DataFrame(ds["train"])
    df = df.rename(columns={"sentence": "text", "label": "label"})
    # Map the numerical labels to corresponding string represantion
    df["label"] = df["label"].map(LABEL_MAP)
    # keep only allowed labels
    df = df[df["label"].isin(LABELS)]

    # Shuffle and split the data 80/10/10 % parts (train/validation/test)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # shuffling
    n = len(df)
    n_train = int(TRAIN * n)
    n_val = int(VAL * n)
    train = df.iloc[:n_train, :]
    val = df.iloc[n_train : n_train + n_val, :]
    test = df.iloc[n_train + n_val :, :]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save the data as parquet
    train.to_parquet(outdir / "fpb_train.parquet")
    val.to_parquet(outdir / "fpb_val.parquet")
    test.to_parquet(outdir / "fpb_test.parquet")

    print(
        f"Saved FPB splits → {outdir} "
        f"train={len(train)}, val={len(val)}, test={len(test)} "
        f"labels: {train['label'].unique().tolist()}"
    )


if __name__ == "__main__":
    main()
