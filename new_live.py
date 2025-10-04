# live_signal.py
import json
import streamlit as st
from pathlib import Path

LIVE_SIGNAL_FILE = Path("results/live_signal.json")

@st.cache_data(ttl=60*60*24)
def fetch_prediction():
    """
    Fetch the latest live prediction from local JSON file.
    Returns a dict with 'signal', 'probability', 'confidence', and 'timestamp'.
    """
    if not LIVE_SIGNAL_FILE.exists():
        st.warning("Live signal not found. Run the data pipeline first.")
        return None

    try:
        with open(LIVE_SIGNAL_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading live signal: {e}")
        return None

def display_prediction_card(prediction: dict):
    """
    Display a single card with the BUY/SELL signal and the confidence score.
    """
    if not prediction:
        st.warning("No prediction available.")
        return

    signal = prediction["signal"]
    score = prediction["confidence"]
    confidence = score if signal == "🟢BUY" else 100 - score  # convert to %
    color = "green" if signal == "🟢BUY" else "red"

    st.markdown("#### Today's Signal")

    st.markdown(
        f"""
        <div style="
            border:2px solid {color};
            border-radius:20px;
            padding:12px;
            text-align:center;
            width:320px;
            margin:auto;
            background-color:#1E1E1E;
        ">
            <h1 style='color:{color}; font-size:75px; margin:0;'>{signal}</h1>
            <p style='color:#FFFFFF; font-size:28px; margin:0px;'>Confidence: {confidence:.1f}%</p>
            <p style='color:#AAAAAA; font-size:16px; margin:0px;'>Timestamp: {prediction.get("timestamp", "N/A")}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("")