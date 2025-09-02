#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Optional

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel

# Sentiment models
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Config
HORIZON = os.getenv("CS_HORIZON", "1h")
SENT_PARQUET = os.getenv(
    "CS_SENT_PARQUET", "data/processed/news_sent_hourly_finbert.parquet"
)
SCALER_PATH = os.getenv("CS_SCALER", f"data/processed/scaler_{HORIZON}.pkl")
CLF_PATH = os.getenv("CS_CLF", f"data/processed/clf_{HORIZON}.pkl")

# Load classifier artifacts
scaler = joblib.load(SCALER_PATH)
clf = joblib.load(CLF_PATH)

# FinBERT (lazy init)
_FINBERT = None
_FINBERT_TOK = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_finbert():
    global _FINBERT, _FINBERT_TOK
    if _FINBERT is None:
        name = "ProsusAI/finbert"
        _FINBERT_TOK = AutoTokenizer.from_pretrained(name)
        _FINBERT = AutoModelForSequenceClassification.from_pretrained(name).to(_DEVICE)
        _FINBERT.eval()
    return _FINBERT_TOK, _FINBERT


# VADER
def get_vader():
    try:
        _ = nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


# The app
app = FastAPI(title="CryptoSense API")


class SentimentReq(BaseModel):
    text: str
    model: str = "finbert"  # or "vader"


@app.get("/health")
def health():
    return {"ok": True, "horizon": HORIZON}


@app.post("/sentiment")
def sentiment(req: SentimentReq):
    text = req.text or ""
    if req.model.lower() == "vader":
        sia = get_vader()
        comp = float(sia.polarity_scores(text)["compound"])
        label = (
            "positive" if comp >= 0.05 else ("negative" if comp <= -0.05 else "neutral")
        )
        return {"model": "vader", "label": label, "score": comp}
    else:
        tok, mdl = get_finbert()
        enc = tok(
            [text], truncation=True, padding=True, max_length=256, return_tensors="pt"
        ).to(_DEVICE)
        with torch.inference_mode():
            probs = torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()[0]
        id2label = mdl.config.id2label
        pred = int(np.argmax(probs))
        label = id2label[pred]
        lab2id = {v.lower(): int(k) for k, v in id2label.items()}
        score = float(probs[lab2id["positive"]] - probs[lab2id["negative"]])
        return {"model": "finbert", "label": label, "score": score}


@app.get("/signal")
def signal(
    timestamp: Optional[str] = Query(
        None, description="ISO8601; defaults to latest row"
    ),
    thr_high: float = 0.6,
    thr_low: float = 0.4,
):
    # Load hourly sentiment and prepare feature vector to mirror training
    s = pd.read_parquet(SENT_PARQUET).sort_values("timestamp")
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True)
    row = None
    if timestamp:
        ts = (
            pd.to_datetime(timestamp).tz_convert("UTC")
            if pd.to_datetime(timestamp).tzinfo
            else pd.to_datetime(timestamp).tz_localize("UTC")
        )
        row = s[s["timestamp"] == ts]
        if row.empty:
            # fallback to nearest previous hour
            row = s[s["timestamp"] <= ts].tail(1)
    else:
        row = s.tail(1)
    if row.empty:
        return {"error": "No sentiment row found for given timestamp"}

    # Features used in Day-4 dataset builder (keep names in sync)
    feats = [
        "sent_mean",
        "sent_std",
        "sent_n",
        "pos_share",
        "neg_share",
        "sent_mean_3h",
        "pos_share_3h",
    ]
    for f in feats:
        if f not in row.columns:
            row[f] = 0.0
    x = row[feats].values.astype(float)
    xs = scaler.transform(x)
    score = float(clf.decision_function(xs)[0])

    action = "flat"
    if score > thr_high:
        action = "long"
    elif score < thr_low:
        action = "short"

    return {
        "timestamp": row["timestamp"].iloc[0].isoformat(),
        "score": score,
        "action": action,
        "thr_high": thr_high,
        "thr_low": thr_low,
    }
