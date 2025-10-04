from pipeline.data_pipeline import run_pipeline, fetch_raw_data, build_features
from prediction.prediction import PredictionEngine
import pandas as pd
import json
from pathlib import Path

# Paths
FEATURES_FILE = Path("data/features/features_labeled.parquet")
LIVE_SIGNAL_FILE = Path("results/live_signal.json")
PROBS_FILE = Path("results/df_probabilities.parquet")

# Step 1: Refresh full pipeline
run_pipeline()

# Step 2: Generate live signal (latest 60 days is enough for sequence)
df_recent = build_features(fetch_raw_data(days_back=60))

engine = PredictionEngine()
engine.load_models()
seq = engine.prepare_sequence(df_recent)
prob = engine.predict_single_sequence(seq)
adjusted_confidence = min((prob / 0.7) * 100, 100.0)
signal = engine.generate_signal(prob)

# Save live signal
LIVE_SIGNAL_FILE.parent.mkdir(exist_ok=True, parents=True)
with open(LIVE_SIGNAL_FILE, "w") as f:
    json.dump({
        "timestamp": str(df_recent.index[-1]),
        "probability": float(prob),
        "confidence": round(adjusted_confidence, 2),
        "signal": signal
    }, f)

print(f"[INFO] Live signal saved to {LIVE_SIGNAL_FILE}")

# Step 3: Load full labeled features for historical probabilities
if not FEATURES_FILE.exists():
    raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

df_full = pd.read_parquet(FEATURES_FILE)
df_full.reset_index(inplace=True)  # Ensure 'timestamp' is a column

# Step 4: Generate historical probabilities
probs_df = engine.predict_dataframe(df_full)

# Save probabilities for dynamic backtest
probs_df.to_parquet(PROBS_FILE, index=False)
print(f"[INFO] Historical probabilities saved to {PROBS_FILE}")