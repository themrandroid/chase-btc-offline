# backtest_tab.py
import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from backtest.backtest import backtest_from_probabilities

# Paths
FEATURES_FILE = Path("data/features/features_labeled.parquet")
PROB_FILE = Path("results/df_probabilities.parquet")

def run_local_backtest(start_date, end_date, threshold, sl, tp, initial_capital, position_size):
    if not FEATURES_FILE.exists() or not PROB_FILE.exists():
        st.warning("Required feature or probability files not found. Run the pipeline first.")
        return None

    # Load features and probabilities
    df_features = pd.read_parquet(FEATURES_FILE)
    df_probs = pd.read_parquet(PROB_FILE)

    df_features.reset_index(inplace=True)  # Ensure timestamp is a column

    # Apply date filter
    df_filtered = df_features[(df_features["timestamp"] >= start_date) & (df_features["timestamp"] <= end_date)]
    df_probs_filtered = df_probs[df_probs["timestamp"].isin(df_filtered["timestamp"])]

    if df_filtered.empty or df_probs_filtered.empty:
        st.warning("No data available for the selected date range.")
        return None

    prices = df_filtered["close"].values
    dates = df_filtered["timestamp"].astype(str).tolist()

    # Run backtest
    bt_results = backtest_from_probabilities(
        prices=prices,
        y_prob=df_probs_filtered,
        dates=dates,
        threshold=threshold,
        stop_loss=sl,
        take_profit=tp,
        initial_capital=initial_capital,
        position_size=position_size
    )
    return bt_results

def show_backtest_tab():
    # ----------------------
    # Scenario Settings
    # ----------------------
    with st.expander("⚙️ Scenario Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            initial_equity = st.number_input("Initial Equity ($)", 100.0, 1_000_000.0, 1000.0, step=100.0)
            threshold_scenario = st.slider("Decision Threshold", 0.0, 1.0, 0.27, 0.01)
            position_size = st.number_input("Position Size %", 0.0, 1.0, 1.0, 0.01)
        with col2:
            start_date = st.date_input("Start Date", datetime(2020, 1, 1))
            end_date = st.date_input("End Date", datetime.today())
            sl_scenario = st.number_input("Stop Loss %", 0.0, 1.0, 0.05, 0.01)
            tp_scenario = st.number_input("Take Profit %", 0.0, 2.0, 0.3, 0.01)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 Run Scenario"):
        bt_results = run_local_backtest(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            threshold=threshold_scenario,
            sl=sl_scenario,
            tp=tp_scenario,
            initial_capital=initial_equity,
            position_size=position_size
        )

        if bt_results:
            metrics = bt_results["metrics"]

            # KPI Card
            st.markdown("""
            <div style="background-color:#1E1E1E; padding:20px; border-radius:15px; margin-bottom:20px;">
                <h3 style="color:#00FF00; text-align:center;">Backtest Metrics</h3>
            """, unsafe_allow_html=True)

            cols = st.columns(4)
            cols[0].markdown(f"""
                <div style="text-align:center; color:white;">
                    <h4>Final Equity</h4>
                    <h2 style="color:#00FF00;">${metrics['final_equity']:.2f}</h2>
                    <p style="font-size:12px; color:#aaa;">💡 Balance at the end of the test.</p>
                </div>
            """, unsafe_allow_html=True)

            cols[1].markdown(f"""
                <div style="text-align:center; color:white;">
                    <h4>Cumulative Return</h4>
                    <h2 style="color:#00FF00;">{metrics['cumulative_return']*100:.2f}%</h2>
                    <p style="font-size:12px; color:#aaa;">💡 Growth/loss over the test period.</p>
                </div>
            """, unsafe_allow_html=True)

            cols[2].markdown(f"""
                <div style="text-align:center; color:white;">
                    <h4>Sharpe Ratio</h4>
                    <h2 style="color:#00FF00;">{metrics['sharpe_ratio']:.2f}</h2>
                    <p style="font-size:12px; color:#aaa;">💡 Risk-adjusted return (higher is better).</p>
                </div>
            """, unsafe_allow_html=True)

            cols[3].markdown(f"""
                <div style="text-align:center; color:white;">
                    <h4>Max Drawdown</h4>
                    <h2 style="color:#00FF00;">{metrics['max_drawdown_pct']*100:.2f}%</h2>
                    <p style="font-size:12px; color:#aaa;">💡 Largest drop from peak balance.</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Equity Curve
            equity_df = pd.DataFrame(bt_results["equity_curve"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df["date"], y=equity_df["strategy"],
                mode="lines", name="Strategy",
                line=dict(color="cyan", width=3)
            ))

            # Trade markers
            trades = bt_results.get("trades", [])
            for trade in trades:
                action = trade["action"]
                color = "green" if action == "BUY" else "red" if action == "STOP_LOSS" else "yellow"
                fig.add_trace(go.Scatter(
                    x=[equity_df["date"][trade["date_idx"]]],
                    y=[trade["size_usd"]],
                    mode="markers",
                    marker=dict(size=12, color=color, symbol="triangle-up"),
                    name=action
                ))

            fig.update_layout(
                title="📈 Equity Curve",
                plot_bgcolor="#121212",
                paper_bgcolor="#121212",
                font=dict(color="#ffffff"),
                xaxis_title="Date",
                yaxis_title="Equity ($)"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("💡 This chart shows how your account value changes over time. Spikes = wins, dips = losses.")
            st.success("Backtest updated!")