import logging
from datetime import datetime, time
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, CommandHandler, ContextTypes, ConversationHandler,
    MessageHandler, CallbackQueryHandler, filters, JobQueue
)
import pandas as pd
import json
import pytz
from backtest.backtest import backtest_from_probabilities
from pathlib import Path

TELEGRAM_TOKEN = "8431683550:AAEaG2o2qd3fZggVBru3Botr5cD1QgKlhYk"

if TELEGRAM_TOKEN is None:
    raise RuntimeError("Missing TELEGRAM_TOKEN")

# --- Config & State ---
subscribed_users = set()
user_configs = {}

DEFAULT_CONFIG = {
    "threshold": 0.27,
    "sl": 0.05,
    "tp": 0.3,
    "position_size": 1.0
}

# Paths to cached files
LIVE_SIGNAL_FILE = Path("results/live_signal.json")
FEATURES_FILE = Path("data/features/features_labeled.parquet")
PROBS_FILE = Path("results/df_probabilities.parquet")

# Logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

THRESHOLD, STOP_LOSS, TAKE_PROFIT, POSITION_SIZE = range(4)

# --- Helpers ---
def load_live_signal():
    with open(LIVE_SIGNAL_FILE, "r") as f:
        return json.load(f)

def load_features():
    df = pd.read_parquet(FEATURES_FILE)
    df.reset_index(inplace=True)
    return df

def run_backtest(config):
    df = load_features()
    prices = df["close"].values
    dates = df["timestamp"].astype(str).tolist()

    # Use cached probabilities
    probs = pd.read_parquet(PROBS_FILE)

    bt_results = backtest_from_probabilities(
        prices=prices,
        y_prob=probs,
        dates=dates,
        threshold=config["threshold"],
        stop_loss=config["sl"],
        take_profit=config["tp"],
        initial_capital=1000,
        position_size=config["position_size"]
    )
    return bt_results

# --- Bot commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    subscribed_users.add(user_id)
    await update.message.reply_text(
        "👋 Welcome to ChaseBTC Bot!\n"
        "You are now subscribed to daily signals ✅\n"
        "Commands:\n"
        "/signal – Today's prediction\n"
        "/backtest – Run backtest\n"
        "/learn – Trading guide\n"
        "/config – Configure preferences"
    )

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    config = user_configs.get(user_id, DEFAULT_CONFIG)

    try:
        prediction = load_live_signal()
        signal = prediction["signal"]
        prob = prediction["confidence"]
        confidence = prob if signal == "🟢BUY" else 100 - prob
        text = (
            f"📊 *ChaseBTC Daily Signal*\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"Action: *{signal}*\n"
            f"Confidence: {confidence:.1f}%\n"
        )
        await update.message.reply_markdown(text)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error reading live signal: {e}")

async def daily_signal_job(context: ContextTypes.DEFAULT_TYPE):
    for user_id in subscribed_users:
        try:
            config = user_configs.get(user_id, DEFAULT_CONFIG)
            prediction = load_live_signal()
            signal_val = prediction["signal"]
            prob = prediction["confidence"]
            confidence = prob if signal_val == "🟢BUY" else 100 - prob

            bt = run_backtest(config)
            metrics = bt["metrics"]

            text = (
                f"🌅 *Daily Signal*\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
                f"Action: *{signal_val}*\n"
                f"Confidence: {confidence:.1f}%\n"
                f"SL: {config['sl']*100:.1f}%\n"
                f"TP: {config['tp']*100:.1f}%\n\n"
                f"📊 Backtest Metrics:\n"
                f"• Cumulative Return: {metrics['cumulative_return']*100:.1f}%\n"
                f"• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"• Max Drawdown: {metrics['max_drawdown_pct']*100:.1f}%"
            )
            await context.bot.send_message(chat_id=user_id, text=text, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Failed sending daily signal to {user_id}: {e}")

## ---- /backtest ----
async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    config = user_configs.get(user_id, DEFAULT_CONFIG)

    try:
        bt = run_backtest(config)
        metrics = bt["metrics"]
        text = (
            f"📈 *Backtest Results:*\n" 
            f"Initial Capital: $1000.00\n"
            f"Final Equity: ${metrics['final_equity']:.2f}\n"
            f"Cumulative Return: {metrics['cumulative_return']*100:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown_pct']*100:.2f}%\n"
            f"Start Date: 2015-01-01\n"
            f"End Date: Today"
        )
        await update.message.reply_markdown(text)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error running backtest: {e}")


# ---- /learn ----
async def learn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    messages = [
        "📚 *Welcome to ChaseBTC Learn Corner!*\n\n"
        "Let’s go step by step so you get a clear picture of Bitcoin and trading.",

        "💡 *What is Bitcoin (BTC)?*\n"
        "Bitcoin is the first and most popular cryptocurrency — a digital currency not controlled by banks or governments. "
        "People trade it like stocks, hoping to profit from price changes.",

        "💹 *What is Crypto Trading?*\n"
        "Just like stocks, you buy BTC when you think the price will rise, and sell (or hold) when you think it may fall. "
        "Because BTC is volatile, tools like ChaseBTC help guide decisions.",

        "📈 *Sharpe Ratio*: Think of this as 'profit quality'.\n"
        "It compares your gains to the risk you took. High Sharpe = steady profits with less stress. "
        "Low Sharpe = wild swings even if profitable.",

        "📉 *Max Drawdown*: The worst drop from a peak to a bottom.\n"
        "It tells you how much pain you’d feel before recovery. Lower drawdown = safer ride.",

        "📊 *Cumulative Return*: The total % gain/loss over time.\n"
        "Example: Start with $1000 → end with $1500 = +50% cumulative return.",

        "🛑 *Stop Loss (SL)*: Auto-sell to limit losses.\n"
        "Example: Buy at $100, SL=5%. If BTC falls to $95, it sells to protect you.",

        "✅ *Take Profit (TP)*: Auto-sell to secure profits.\n"
        "Example: Buy at $100, TP=20%. If BTC rises to $120, it sells and locks gains.",

        "🎯 *Threshold*: The model’s cutoff for making a BUY vs HOLD decision.\n"
        "Example: Threshold=0.60 → the model buys only if >60% confident.",

        "🚀 *Remember*: Trading always carries risk. ChaseBTC is here to *teach and assist*, not give financial advice. "
        "The goal is to help you learn and practice smarter!"
    ]

    # Send them one by one
    for msg in messages:
        await update.message.reply_markdown(msg)


# ---- /config (interactive wizard) ----
async def config_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚙️ Let's set up your trading preferences.\n\n"
        "Step 1️⃣: Choose your decision threshold (0.0–1.0).",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("0.3", callback_data="0.3"),
                                            InlineKeyboardButton("0.5", callback_data="0.5"),
                                            InlineKeyboardButton("0.7", callback_data="0.7")]])
    )
    return THRESHOLD

async def set_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    threshold = float(query.data)
    user_configs[user_id] = {"threshold": threshold}
    await query.edit_message_text(f"✅ Threshold set to {threshold}\n\nStep 2️⃣: Enter Stop Loss % (e.g., 0.05)")
    return STOP_LOSS

async def set_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        sl = float(update.message.text)
        user_configs[user_id]["sl"] = sl
        await update.message.reply_text("✅ Stop Loss set.\n\nStep 3️⃣: Enter Take Profit % (e.g., 0.3)")
        return TAKE_PROFIT
    except:
        await update.message.reply_text("⚠️ Enter a valid number (e.g., 0.05).")
        return STOP_LOSS

async def set_take_profit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        tp = float(update.message.text)
        user_configs[user_id]["tp"] = tp
        await update.message.reply_text("✅ Take Profit set.\n\nStep 4️⃣: Enter Position Size % (e.g., 1.0)")
        return POSITION_SIZE
    except:
        await update.message.reply_text("⚠️ Enter a valid number (e.g., 0.3).")
        return TAKE_PROFIT

async def set_position_size(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        ps = float(update.message.text)
        user_configs[user_id]["position_size"] = ps
        config = user_configs[user_id]
        text = (
            f"🎉 Your config has been saved:\n"
            f"Threshold: {config['threshold']}\n"
            f"SL: {config['sl']*100:.1f}%\n"
            f"TP: {config['tp']*100:.1f}%\n"
            f"Position Size: {config['position_size']*100:.1f}%"
        )
        await update.message.reply_text(text)
        return ConversationHandler.END
    except:
        await update.message.reply_text("⚠️ Enter a valid number (e.g., 1.0).")
        return POSITION_SIZE

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Configuration cancelled.")
    return ConversationHandler.END

def telegram_bot():
    app = Application.builder().token(TELEGRAM_TOKEN).job_queue(JobQueue()).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("backtest", backtest))
    app.add_handler(CommandHandler("learn", learn))

    # Config conversation
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("config", config_start)],
        states={
            THRESHOLD: [CallbackQueryHandler(set_threshold)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_stop_loss)],
            TAKE_PROFIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_take_profit)],
            POSITION_SIZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_position_size)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(conv_handler)

    nigeria_tz = pytz.timezone("Africa/Lagos")
    app.job_queue.run_daily(daily_signal_job, time=time(hour=8, minute=00, tzinfo=nigeria_tz))
    app.run_polling()

telegram_bot()

if __name__ == "__main__":
    telegram_bot()