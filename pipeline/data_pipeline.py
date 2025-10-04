import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import pickle
from pathlib import Path

# ============================
# CONFIG
# ============================
FEATURES_DIR = "data/features"
RAW_DATA_DIR = "data/raw"
FEATURES_FILE = os.path.join(FEATURES_DIR, "features.parquet")
LABELED_FILE = os.path.join(FEATURES_DIR, "features_labeled.parquet")
SCALER_FILE = os.path.join(FEATURES_DIR, "scaler.pkl")
MANIFEST_FILE = os.path.join(FEATURES_DIR, "manifest.json")

SEQ_LEN = 20          # sequence length for models
LOOK_AHEAD = 3        # days to look ahead for target
THRESHOLD = 0.01      # 1% threshold for labeling

# Ensure directories exist
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# ============================
# 1. FETCH RAW DATA
# ============================
def fetch_raw_data(days_back: int = 730, start_date: datetime | None = None) -> pd.DataFrame:
    """
    Fetch BTC-USD historical data from Yahoo Finance.
    """
    end_date = datetime.today()

    today_str = datetime.today().strftime("%Y-%m-%d")
    raw_path = Path(RAW_DATA_DIR) / f"btc_raw_{today_str}.csv"

    if raw_path.exists():
        print(f"[INFO] Loading BTC-USD data from local file: {raw_path}")
        df = pd.read_csv(raw_path)
    
    else:
        if start_date:
            print(f"[INFO] Fetching BTC-USD data from {start_date.date()} to {end_date.date()}")
        elif days_back:
            start_date = end_date - timedelta(days=days_back)
            print(f"[INFO] Fetching BTC-USD data from {start_date.date()} to {end_date.date()}")

        df = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d")
        df.reset_index(inplace=True)

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Rename columns
        df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        # Save raw copy
        raw_path = os.path.join(RAW_DATA_DIR, f"btc_raw_{datetime.today().date()}.csv")
        df.to_csv(raw_path, index=False)

    return df

# ============================
# 2. CLEAN DATA
# ============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw BTC data.
    """
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    print("Columns in df_raw:", df.columns.tolist())
    df.sort_values("timestamp", inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    df.set_index("timestamp", inplace=True)
    return df

# ============================
# 3. FEATURE ENGINEERING
# ============================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate ML features.
    """
    df_feat = df.copy()

    # Daily returns
    df_feat['log_return'] = np.log(df_feat['close'] / df_feat['close'].shift(1))

    # Rolling volatility
    df_feat['volatility_10d'] = df_feat['log_return'].rolling(window=10).std()
    df_feat['volatility_21d'] = df_feat['log_return'].rolling(window=21).std()

    # Rolling returns
    df_feat['return_3d'] = df_feat['close'].pct_change(3)
    df_feat['return_14d'] = df_feat['close'].pct_change(14)

    # Bollinger Bands (lower)
    rolling_mean = df_feat['close'].rolling(window=20).mean()
    rolling_std = df_feat['close'].rolling(window=20).std()
    df_feat['bollinger_down'] = rolling_mean - (2 * rolling_std)

    df_feat.dropna(inplace=True)
    return df_feat

# ============================
# 4. SCALE FEATURES
# ============================
def scale_features(df: pd.DataFrame, scaler: StandardScaler = None):
    feature_cols = ['volatility_10d', 'volatility_21d', 'return_3d', 'return_14d', 'bollinger_down']
    X = df[feature_cols].values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    df_scaled = df.copy()
    df_scaled[feature_cols] = X_scaled
    return df_scaled, scaler

# ============================
# 5. GENERATE LABELS
# ============================
def generate_labels(df: pd.DataFrame, look_ahead: int = LOOK_AHEAD, threshold: float = THRESHOLD):
    future_close = df['close'].shift(-look_ahead)
    df['target'] = (future_close > df['close'] * (1 + threshold)).astype(int)
    df.dropna(inplace=True)
    return df

# ============================
# 6. SAVE MANIFEST
# ============================
def update_manifest(latest_timestamp: str, version: str = "v1.0"):
    manifest = {
        "last_updated": latest_timestamp,
        "feature_version": version,
        "updated_at": datetime.utcnow().isoformat()
    }
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)

# ============================
# 7. RUN FULL PIPELINE
# ============================
def run_pipeline(start_date: datetime = datetime(2015, 1, 1), force: bool = False):
    print("[PIPELINE] Starting data pipeline...")

    # Step 1: Fetch
    df_raw = fetch_raw_data(start_date=start_date)

    # Step 2: Clean
    df_clean = clean_data(df_raw)

    # Step 3: Features
    df_features = build_features(df_clean)

    # Step 4: Scale
    df_scaled, scaler = scale_features(df_features)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)

    # Save features without labels
    df_scaled.to_parquet(FEATURES_FILE)

    # Step 5: Labels
    df_labeled = generate_labels(df_scaled)
    df_labeled.to_parquet(LABELED_FILE)

    # Step 6: Manifest
    latest_ts = str(df_labeled.index[-1])
    update_manifest(latest_ts)

    print(f"[PIPELINE] Completed successfully. Latest date: {latest_ts}")

run_pipeline()