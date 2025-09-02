# CryptoSense

# How to Run (Essentials Only)

A tiny, end-to-end project that tests whether crypto text sentiment helps predict short-horizon BTC returns. Everything runs as standalone **scripts**—no extra scaffolding needed.

---

### Use uv (recommended)

```bash
# 1) Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"

# 2) Create a virtualenv and install deps from pyproject/lock
uv venv            # creates .venv
uv sync            # installs all dependencies

# 3) Run any script/command inside the env (no manual activation needed)
uv run python scripts/fetch_prices.py --help
uv run python scripts/build_signal_dataset.py --help

# Examples
uv run python scripts/train_signal_baselines.py --data data/processed/signal_dataset_finbert.parquet --horizon 1h --outdir data/processed
uv run uvicorn scripts.api_fastapi:app --host 0.0.0.0 --port 8000
uv run streamlit run scripts/streamlit_dashboard.py
```

### Or if you prefer venv; Create and activate a virtual environment


```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
```

---

### Install only the required packages
```bash
pip install pandas numpy pyarrow yfinance feedparser beautifulsoup4 langdetect datasets \
            nltk scikit-learn transformers torch joblib fastapi "uvicorn[standard]" \
            streamlit matplotlib
```

---

## Run Order (start → finish)

> Replace paths as you like; each script writes its outputs to `data/` automatically.

### 1) Prices
Fetch BTC hourly candles (≈2 years) → Parquet.
```bash
python scripts/fetch_prices.py \
  --symbol BTC-USD \
  --interval 1h \
  --period 730d \
  --out data/raw/btc_1h.parquet
```
**You can tweak:** `--interval` (`1h`, `4h`, `1d`), `--period` (`30d`, `365d`, `730d`, `max`).

---

### 2) Crypto news + labeled data

**2a. News via RSS** → Parquet  
```bash
python scripts/fetch_crypto_news_rss.py \
  --feeds https://www.coindesk.com/arc/outboundfeeds/rss/ https://cointelegraph.com/rss \
  --out data/raw/crypto_news.parquet
```
**You can tweak:** add/remove RSS feeds; change `--out`.

**2b. Financial PhraseBank splits** → Parquets  
```bash
python scripts/make_financial_phrasebank.py \
  --outdir data/processed \
  --seed 17
```
**You can tweak:** inside the script, switch config (`sentences_allagree` ↔ `sentences_75agree`, etc.).

---

### 3) Baseline sentiment + aggregation

**3a. Evaluate baselines on FPB** (metrics printed; predictions saved)  
```bash
python scripts/eval_fpb_baselines.py --model vader   --in_dir data/processed --out_dir data/processed
python scripts/eval_fpb_baselines.py --model finbert --in_dir data/processed --out_dir data/processed
```

**3b. Score your news** (choose one or both)  
```bash
python scripts/score_news_sentiment.py --model vader   --in data/raw/crypto_news.parquet --out data/processed/news_scored_vader.parquet
python scripts/score_news_sentiment.py --model finbert --in data/raw/crypto_news.parquet --out data/processed/news_scored_finbert.parquet
```

**3c. Aggregate to hourly features**  
```bash
python scripts/aggregate_news_sentiment.py --in data/processed/news_scored_vader.parquet   --out data/processed/news_sent_hourly_vader.parquet
python scripts/aggregate_news_sentiment.py --in data/processed/news_scored_finbert.parquet --out data/processed/news_sent_hourly_finbert.parquet
```
**You can tweak:** use VADER or FinBERT (or both) and compare later.

---

### 4) Join, target & train baselines

**4a. Build the signal dataset** (prices × hourly sentiment)  
```bash
python scripts/build_signal_dataset.py \
  --prices data/raw/btc_1h.parquet \
  --sent   data/processed/news_sent_hourly_finbert.parquet \
  --out    data/processed/signal_dataset_finbert.parquet
```
**You can tweak:** swap `--sent` to the VADER file for a second dataset.

**4b. Train minimal baselines** (time split; test predictions saved)  
```bash
python scripts/train_signal_baselines.py --data data/processed/signal_dataset_finbert.parquet --horizon 1h --outdir data/processed
python scripts/train_signal_baselines.py --data data/processed/signal_dataset_finbert.parquet --horizon 4h --outdir data/processed
```
**Outputs:** `signal_test_preds_{1h,4h}.parquet` (has `timestamp, true_ret, clf_score, …`).  
**You can tweak:** try VADER dataset; adjust LogisticRegression/Ridge params inside the script if desired.

---

### 5) Backtest

Turn classifier scores into trades with costs; save equity and metrics.
```bash
# 1h, long/short with neutral zone
python scripts/backtest_signal.py \
  --preds data/processed/signal_test_preds_1h.parquet \
  --horizon 1h \
  --mode longshort \
  --thr_high 0.6 --thr_low 0.4 \
  --cost_bps 4 \
  --outdir data/processed

# 4h, long-only
python scripts/backtest_signal.py \
  --preds data/processed/signal_test_preds_4h.parquet \
  --horizon 4h \
  --mode longonly \
  --thr_high 0.55 \
  --cost_bps 4 \
  --outdir data/processed
```
**Outputs:** `bt_<mode>_<horizon>_cost<bps>.parquet` and `_metrics.txt`.  
**You can tweak:** `--mode`, thresholds (`--thr_high`, `--thr_low`), `--cost_bps`.

---

### 6) Export model + API + Dashboard)

**6a. Export the deployed classifier** (train on the full dataset)
```bash
python scripts/export_signal_model.py \
  --data data/processed/signal_dataset_finbert.parquet \
  --horizon 1h \
  --outdir data/processed
```
**Outputs:** `scaler_1h.pkl`, `clf_1h.pkl`.

**6b. Run FastAPI**  
```bash
CS_SENT_PARQUET=data/processed/news_sent_hourly_finbert.parquet \
CS_SCALER=data/processed/scaler_1h.pkl \
CS_CLF=data/processed/clf_1h.pkl \
uvicorn scripts.api_fastapi:app --host 0.0.0.0 --port 8000
```
**Test endpoints**
- Health: `GET http://localhost:8000/health`
- Sentiment: `POST /sentiment` with `{"text":"Bitcoin jumps on ETF news","model":"finbert"}` (or `"vader"`)
- Signal: `GET /signal` (latest) or `GET /signal?timestamp=2025-08-30T12:00:00Z`

**6c. Streamlit dashboard**  
```bash
streamlit run scripts/streamlit_dashboard.py
```
Inside the app, choose which backtest Parquet and hourly sentiment Parquet to view.

---

## Common Tweaks

- **Sentiment source:** run the pipeline twice—once with VADER, once with FinBERT—and compare backtests.
- **Horizon:** train/backtest at `1h` and `4h` using the same features.
- **Costs & thresholds:** adjust `--cost_bps`, `--thr_high`, `--thr_low` to test robustness.
- **RSS freshness:** rerun the RSS fetch → re-score → re-aggregate to update the pipeline.

---

## Troubleshooting

- **Empty/parquet errors:** check that prior steps produced the expected files/columns.
- **FinBERT slow:** use VADER for faster iteration; keep FinBERT for final runs.
- **Timezones:** scripts normalize timestamps to UTC; ensure custom timestamps passed to the API include `Z` or a timezone offset.
