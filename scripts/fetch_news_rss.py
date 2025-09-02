#!/usr/bin/env python3
"""
Main function in this module fetches and cleans recent crypto news from a few reliable RSS feeds, then writes a single Parquet.

Default feeds (can be overridden by CLI):

CoinDesk: https://www.coindesk.com/arc/outboundfeeds/rss/ (official feed)

Cointelegraph (all): https://cointelegraph.com/rss (official feeds page lists RSS options)
"""
from typing import Any
import argparse
import pandas as pd
from pathlib import Path
import feedparser
from feedparser.util import FeedParserDict
from datetime import datetime, timezone
from langdetect import detect, DetectorFactory
from bs4 import BeautifulSoup

DetectorFactory.seed = 42

DEFAULT_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]
REQUIRED_COLS = ["timestamp", "source", "title", "summary", "url"]


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ")
    return text


def parse_feed(url: str) -> list[dict[str, Any]]:
    data: FeedParserDict = feedparser.parse(url)
    out = []
    for e in data.entries:
        # extract timestamp
        # prefer published, fallback to updated
        ts = None
        for key in ("published_parsed", "updated_parsed"):
            if getattr(e, key, None):
                ts = datetime(*getattr(e, key)[:6], tzinfo=timezone.utc)
                break
        if ts is None:
            # skip entries without a parseable timestamp
            continue
        title = getattr(e, "title", "").strip()
        summary_html = getattr(e, "summary", "")
        summary = html_to_text(summary_html)
        link = getattr(e, "link", None)
        source = data.feed.get("title", url)
        text_blob = f"{title}\n\n{summary}".strip()

        # Language filter (EN only); ignore if detection fails on very short text
        keep = True
        try:
            if len(text_blob) >= 20:
                keep = detect(text_blob) == "en"
        except Exception:
            keep = True
        if not keep:
            continue
        out.append(
            {
                "timestamp": ts,
                "source": source,
                "title": title,
                "summary": summary,
                "url": link,
            }
        )

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Fetch crypto news via RSS and save Parquet"
    )
    ap.add_argument("--feeds", nargs="*", default=DEFAULT_FEEDS, help="RSS feed URLs")
    ap.add_argument(
        "--out", default="data/raw/crypto_news.parquet", help="Output Parquet path"
    )
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for url in args.feeds:
        rows.extend(parse_feed(url))

    if not rows:
        raise SystemExit("No items fetched. Try different feeds or check your network.")

    df = pd.DataFrame(rows)

    # Deduplicate by URL if present, else by title+timestamp minute
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    else:
        df["ts_minute"] = df["timestamp"].dt.floor("min")
        df = df.drop_duplicates(subset=["title", "ts_minute"], keep="first")
        df = df.drop(columns=["ts_minute"], errors="ignore")

    # # Sort and enforce schema
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[REQUIRED_COLS]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    print(
        f"Fetched {len(df):,} articles from {len(args.feeds)} feeds → {out}"
        f"Range: {df['timestamp'].min()} → {df['timestamp'].max()}"
    )


if __name__ == "__main__":
    main()
