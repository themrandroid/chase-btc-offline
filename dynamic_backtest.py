import pandas as pd
from pathlib import Path
from backtest.backtest import backtest_from_probabilities

# Paths
FEATURES_FILE = Path("data/features/features_labeled.parquet")
PROBS_FILE = Path("results/df_probabilities.parquet")

def dynamic_backtest(
    start_date: str = "2015-01-01",
    end_date: str = None,
    threshold: float = 0.27,
    stop_loss: float = 0.05,
    take_profit: float = 0.30,
    initial_capital: float = 1000.0,
    position_size: float = 1.0
):
    """
    Run a dynamic backtest using cached probabilities and full feature data.
    """
    if not FEATURES_FILE.exists() or not PROBS_FILE.exists():
        raise FileNotFoundError("Required files not found. Ensure run_pipeline + probabilities generation is done.")

    # Load full features
    df_features = pd.read_parquet(FEATURES_FILE)
    df_features.reset_index(inplace=True)  # ensure 'timestamp' is a column

    # Apply date filter
    if start_date:
        df_features = df_features[df_features["timestamp"] >= start_date]
    if end_date:
        df_features = df_features[df_features["timestamp"] <= end_date]

    if df_features.empty:
        raise ValueError("No data available for the given date range.")

    # Load cached probabilities
    probs_df = pd.read_parquet(PROBS_FILE)

    # Align lengths if necessary
    min_len = min(len(df_features), len(probs_df))
    df_features = df_features.tail(min_len)
    probs_df = probs_df.tail(min_len)

    prices = df_features["close"].values
    dates = df_features["timestamp"].astype(str).tolist()

    # Run backtest
    bt_results = backtest_from_probabilities(
        prices=prices,
        y_prob=probs_df,
        dates=dates,
        threshold=threshold,
        stop_loss=stop_loss,
        take_profit=take_profit,
        initial_capital=initial_capital,
        position_size=position_size
    )

    # Return structured backtest data
    return {
        "metrics": {
            "sharpe": bt_results["metrics"]["sharpe_ratio"],
            "max_drawdown": bt_results["metrics"]["max_drawdown_pct"],
            "cumulative_return": bt_results["metrics"]["cumulative_return"],
            "final_equity": bt_results["metrics"]["final_equity"],
            "total_trades": bt_results["metrics"]["total_trades"],
            "win_rate_pct": bt_results["metrics"]["win_rate_pct"],
            "avg_profit_per_closed_trade": bt_results["metrics"]["avg_profit_per_closed_trade"],
        },
        "equity_curve": bt_results["equity_curve"],
        "trades": bt_results.get("trades", []),
        "raws": bt_results.get("raw", {})
    }

# # =======================
# # Example usage
# # =======================
# if __name__ == "__main__":
#     # User-defined parameters
#     params = {
#         "start_date": "2022-01-01",
#         "end_date": "2023-12-31",
#         "threshold": 0.27,
#         "stop_loss": 0.05,
#         "take_profit": 0.3,
#         "initial_capital": 5000.0,
#         "position_size": 0.5
#     }

#     results = dynamic_backtest(**params)
#     print("[INFO] Backtest metrics:", results["metrics"])
