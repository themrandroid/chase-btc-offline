# app.py
import sys, os
sys.path.append(os.path.dirname(__file__))
import streamlit as st
from new_live import display_prediction_card, fetch_prediction
from new_backtest import show_backtest_tab

# ----------------------
# General Config
# ----------------------
st.set_page_config(page_title="Chase BTC Terminal", 
                   layout="wide", 
                   page_icon=":money_with_wings:")

st.markdown("""
<div style="text-align:center; font-size:48px; color:#00FF00;">
        💰<b>ChaseBTC</b>🏃‍♂️ 
</div>
""", unsafe_allow_html=True)

# --- Live Signal ---
prediction = fetch_prediction()
display_prediction_card(prediction)

# -- Backtest ----
show_backtest_tab()

# -- Redirect to Telegram Bot --
st.markdown("""
<div style="background-color:#1E1E1E;
            border:2px solid #00FF00;
            border-radius:12px;
            padding:20px;
            margin:20px 0;
            text-align:center;">
    <h3 style="color:#00FF00; margin-bottom:10px;">📢 Get Real-time Alerts</h3>
    <p style="color:white;">Join our Telegram Bot for instant trading updates!</p>
    <a href="https://t.me/chase_btc_bot" target="_blank" 
       style="display:inline-block; padding:10px 20px; background-color:#0088cc; 
              color:white; font-weight:bold; border-radius:8px; text-decoration:none;">
       👉 Open Telegram
    </a>
</div>
""", unsafe_allow_html=True)

st.write("")


# Disclaimer Banner
st.markdown("""
<div style='background-color:#b22222;padding:10px;border-radius:2px;color:white;text-align:center;'>
⚠️ This is a trading simulator. Not financial advice. Use responsibly.
</div>
""", unsafe_allow_html=True)