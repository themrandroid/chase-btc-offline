import os
import json
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional


def generate_signals(y_prob: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    """Turn probabilities into binary signals (1 = buy, 0 = flat/hold)."""
    return (y_prob > threshold).astype(int)

def simulate_trades(
    prices: np.ndarray,
    signals: np.ndarray,
    fee: float = 0.001,
    slippage: float = 0.0005,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    initial_capital: float = 1000.0,
    position_size: float = 1.0
) -> Dict[str, Any]:
    """
    Simulate trading given daily prices and signals.

    Notes:
      - signals[i] indicates action at day i (use price[i] to enter/exit).
      - position is held until a 0 signal, stop_loss, or take_profit triggers.
      - position_size is fraction of capital allocated (1.0 = all in).
      - returns equity curve (per-day), trades list, buy&hold curve.
    """
    n = len(signals)
    if len(prices) != n:
        raise ValueError("prices and signals must have same length")

    capital = float(initial_capital)
    position = 0.0  # amount of  held (BTC)
    entry_price = None
    equity_curve = []
    trades = []

    # Buy-and-hold for benchmark (buy at first price with all capital)
    if n >= 1 and not math.isnan(prices[0]) and prices[0] > 0:
        bh_amount = (initial_capital / (prices[0] * (1 + fee + slippage)))
        buy_and_hold = bh_amount * prices  # value each day (ignoring fees later)
    else:
        bh_amount = 0.0
        buy_and_hold = np.zeros(n)

    for i in range(n - 1):
        price_today = float(prices[i])
        price_next = float(prices[i + 1])

        # Enter trade (signal==1 and no open position)
        if signals[i] == 1 and position == 0:
            alloc = capital * position_size
            # BTC amount bought
            btc_bought = alloc / (price_today * (1 + fee + slippage))
            position = btc_bought
            capital -= alloc  # reduce capital by allocated funds
            entry_price = price_today
            trades.append({
                "date_idx": i,
                "action": "BUY",
                "price": price_today,
                "size_asset": float(position),
                "size_usd": float(alloc)
            })

        # Exit trade by signal==0
        elif signals[i] == 0 and position > 0:
            proceeds = position * price_today * (1 - fee - slippage)
            capital += proceeds
            trades.append({
                "date_idx": i,
                "action": "SELL",
                "price": price_today,
                "size_asset": float(position),
                "size_usd": float(proceeds)
            })
            position = 0.0
            entry_price = None

        # Risk controls (stop loss / take profit)
        if position > 0 and entry_price is not None:
            drawdown = (price_today - entry_price) / entry_price
            if stop_loss is not None and drawdown <= -abs(stop_loss):
                proceeds = position * price_today * (1 - fee - slippage)
                capital += proceeds
                trades.append({
                    "date_idx": i,
                    "action": "STOP_LOSS",
                    "price": price_today,
                    "size_asset": float(position),
                    "size_usd": float(proceeds),
                    "drawdown": float(drawdown)
                })
                position = 0.0
                entry_price = None
            elif take_profit is not None and drawdown >= abs(take_profit):
                proceeds = position * price_today * (1 - fee - slippage)
                capital += proceeds
                trades.append({
                    "date_idx": i,
                    "action": "TAKE_PROFIT",
                    "price": price_today,
                    "size_asset": float(position),
                    "size_usd": float(proceeds),
                    "drawdown": float(drawdown)
                })
                position = 0.0
                entry_price = None

        # Compute equity at close of day i
        equity = capital + position * price_today
        equity_curve.append(float(equity))

    # Append final day equity (using last price)
    if n >= 1:
        final_price = float(prices[-1])
        equity_curve.append(float(capital + position * final_price))
    else:
        equity_curve = [float(initial_capital)]

    equity_curve = np.array(equity_curve, dtype=float)
    buy_and_hold_curve = buy_and_hold if len(buy_and_hold) == len(equity_curve) else buy_and_hold[:len(equity_curve)]

    return {
        "equity_curve": equity_curve,
        "buy_and_hold_curve": buy_and_hold_curve,
        "trades": trades
    }

def calculate_metrics(equity_curve: np.ndarray, initial_capital: float = 1000.0) -> Dict[str, Any]:
    """
    Compute standard backtest metrics:
     - cumulative return, annualized sharpe (assume daily returns), max drawdown (abs and pct)
    """
    if len(equity_curve) < 2:
        return {
            "final_equity": float(equity_curve[-1]) if len(equity_curve) else initial_capital,
            "cumulative_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "daily_returns_mean": 0.0,
            "daily_returns_std": 0.0
        }

    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
    cum_return = float(equity_curve[-1] / initial_capital - 1.0)
    mean_r = float(np.nanmean(returns))
    std_r = float(np.nanstd(returns, ddof=0)) if len(returns) > 1 else 0.0
    sharpe = (mean_r / (std_r + 1e-9)) * math.sqrt(252) if std_r > 0 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    max_dd = float(np.max(drawdowns))
    max_dd_pct = float(max_dd / (np.max(running_max) + 1e-9)) if np.max(running_max) > 0 else 0.0

    return {
        "final_equity": float(equity_curve[-1]),
        "cumulative_return": cum_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "daily_returns_mean": mean_r,
        "daily_returns_std": std_r
    }

def prepare_chart_data(dates: List[str], equity_curve: np.ndarray, buy_and_hold_curve: np.ndarray) -> List[Dict[str, Any]]:
    """
    Create list of dicts: [{date, strategy, buy_and_hold}, ...] suitable for JSON output and plotting.
    """
    n = len(equity_curve)
    out = []
    for i in range(n):
        date = dates[i] if i < len(dates) else i
        out.append({
            "date": str(date),
            "strategy": float(equity_curve[i]),
            "buy_and_hold": float(buy_and_hold_curve[i]) if i < len(buy_and_hold_curve) else None
        })
    return out

def save_report_json(report: Dict[str, Any], output_dir: str, filename: str = "backtest_report.json") -> str:
    """
    Save the report dict as JSON into output_dir and return the path.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    return path

def backtest_from_probabilities(
    prices: np.ndarray,
    y_prob: np.ndarray,
    dates: Optional[List[Any]] = None,
    threshold: float = 0.27,
    stop_loss: Optional[float] = 0.05,
    take_profit: Optional[float] = 0.10,
    initial_capital: float = 10000.0,
    fee: float = 0.001,
    slippage: float = 0.0005,
    position_size: float = 1.0,
    output_dir: str = "backtest_results",
    return_json: bool = True
) -> Dict[str, Any]:
    """
    High-level function that:
      - converts probabilities -> signals
      - simulates trades
      - calculates metrics
      - prepares chart-ready output
      - optionally saves a JSON report and returns the report dict
    """
    prices = np.asarray(prices, dtype=float)
    y_prob = np.asarray(y_prob["probability"], dtype=float)

    if len(prices) != len(y_prob):
        # Allow y_prob length to be len(prices) or len(prices)-?; simplest is to align from the end
        min_len = min(len(prices), len(y_prob))
        prices = prices[-min_len:]
        y_prob = y_prob[-min_len:]
        if dates is not None:
            dates = dates[-min_len:]

    signals = generate_signals(y_prob, threshold=threshold)

    sim = simulate_trades(
        prices=prices,
        signals=signals,
        fee=fee,
        slippage=slippage,
        stop_loss=stop_loss,
        take_profit=take_profit,
        initial_capital=initial_capital,
        position_size=position_size
    )

    equity_curve = sim["equity_curve"]
    bh_curve = sim["buy_and_hold_curve"]
    trades = sim["trades"]

    metrics = calculate_metrics(equity_curve, initial_capital=initial_capital)

    # extra trade stats: total trades, win rate (approx using SELL vs BUY P&L)
    total_trades = sum(1 for t in trades if t["action"] in ("SELL", "TAKE_PROFIT", "STOP_LOSS"))
    # approximate win-rate: compute P&L for each closed pair (BUY/SELL)
    wins = 0
    profits = []
    stack = []
    for t in trades:
        if t["action"] == "BUY":
            stack.append(t)
        else:
            if stack:
                buy = stack.pop(0)
                sell = t
                pnl = sell["size_usd"] - buy["size_usd"]
                profits.append(float(pnl))
                if pnl > 0:
                    wins += 1
    win_rate = (wins / len(profits) * 100.0) if len(profits) else None
    avg_profit = float(np.mean(profits)) if len(profits) else None

    # Prepare chart data (dates optional)
    if dates is None:
        # generate simple date indices if none provided
        dates_out = list(range(len(equity_curve)))
    else:
        dates_out = [str(d) for d in dates[-len(equity_curve):]]

    equity_chart = prepare_chart_data(dates_out, equity_curve, bh_curve)

    report = {
        "config": {
            "threshold": threshold,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "initial_capital": initial_capital,
            "fee": fee,
            "slippage": slippage,
            "position_size": position_size
        },
        "metrics": {
            "final_equity": metrics["final_equity"],
            "cumulative_return": metrics["cumulative_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "avg_profit_per_closed_trade": avg_profit
        },
        "equity_curve": equity_chart,
        "trades": trades,
        "raw": {
            "prices_length": int(len(prices)),
            "probs_min": float(np.min(y_prob)),
            "probs_max": float(np.max(y_prob)),
            "probs_mean": float(np.mean(y_prob))
        }
    }

    if return_json:
        saved = save_report_json(report, output_dir=output_dir)
        report["report_path"] = saved

    return report