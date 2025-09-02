import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def main():
    st.set_page_config(page_title="CryptoSense Dashboard", layout="wide")
    st.title("CryptoSense — Backtest & Sentiment")

    # Sidebar controls
    preds = st.sidebar.text_input(
        "Backtest Parquet", "data/processed/bt_longshort_1h_cost4.parquet"
    )
    sent = st.sidebar.text_input(
        "Hourly Sentiment Parquet", "data/processed/news_sent_hourly_finbert.parquet"
    )

    # Load data
    try:
        bt = pd.read_parquet(preds)
        bt["timestamp"] = pd.to_datetime(bt["timestamp"])
    except Exception as e:
        st.error(f"Failed to load backtest: {e}")
        return
    try:
        s = pd.read_parquet(sent)
        s["timestamp"] = pd.to_datetime(s["timestamp"])
    except Exception as e:
        st.warning(f"Sentiment overlay not loaded: {e}")
        s = None

    # Equity curve
    st.subheader("Equity Curve (Net)")
    fig1 = plt.figure()
    plt.plot(bt["timestamp"], bt["equity"])
    plt.xlabel("Time")
    plt.ylabel("Cumulative Net PnL (log-returns sum)")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Metrics quickview (computed previously; recompute a few here)
    st.subheader("Backtest Snapshot")
    net = bt["net"].dropna()
    hit = (
        (bt.loc[bt["position"].abs() > 0, "gross"] > 0).mean()
        if (bt["position"].abs() > 0).any()
        else float("nan")
    )
    turnover = bt["position"].diff().abs().fillna(0).mean()
    st.write(
        {
            "n_bars": len(bt),
            "in_pos_frac": float((bt["position"].abs() > 0).mean()),
            "hit_rate": float(hit) if pd.notnull(hit) else None,
            "turnover": float(turnover),
            "total_net": float(net.sum()),
        }
    )

    # Sentiment overlay (mean score, rescaled) if available
    if s is not None and "sent_mean" in s.columns:
        st.subheader("Hourly Sentiment (mean)")
        fig2 = plt.figure()
        plt.plot(s["timestamp"], s["sent_mean"])
        plt.xlabel("Time")
        plt.ylabel("sent_mean")
        st.pyplot(fig2)


if __name__ == "__main__":
    main()
