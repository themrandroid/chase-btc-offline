import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model #type: ignore

# ==============================
# CONFIG CONSTANTS
# ==============================
MODEL_PATH = "models/final"
SEQ_LEN = 20

TOP_FEATURES = [
    "volatility_21d",
    "volatility_10d",
    "return_14d",
    "return_3d",
    "bollinger_down",
]

THRESHOLD = 0.27  # Buy signal threshold


# ==============================
# PREDICTION ENGINE
# ==============================
class PredictionEngine:
    """
    Handles loading models, caching them, preparing sequences,
    and generating live/historical predictions using ensemble logic.
    """

    def __init__(self, model_path=MODEL_PATH, seq_len=SEQ_LEN):
        self.model_path = model_path
        self.seq_len = seq_len
        self.models_cache = {}  # { "lstm": [fold1_model, fold2_model, ...], ... }
        self.base_models = ["lstm", "gru", "conv1d"]

    # --------------------------
    # Load and cache models
    # --------------------------
    def load_models(self):
        """Load and cache all fold models for each architecture."""
        if self.models_cache:
            return  # Models already loaded

        for model_name in self.base_models:
            model_dir = os.path.join(self.model_path, model_name)
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Missing model folder: {model_dir}")

            fold_models = []
            for fname in sorted(os.listdir(model_dir)):
                if fname.endswith(".h5") and "fold" in fname.lower():
                    model_path = os.path.join(model_dir, fname)
                    fold_models.append(load_model(model_path))

            if not fold_models:
                raise ValueError(f"No fold models found in {model_dir}")

            self.models_cache[model_name] = fold_models

        print(f"[PredictionEngine] Loaded models for: {list(self.models_cache.keys())}")

    # --------------------------
    # Prepare sequence for live prediction
    # --------------------------
    def prepare_sequence(self, df, feature_cols=TOP_FEATURES):
        """
        Prepare a single sequence from the latest rows of a feature DataFrame.
        Output shape: (1, seq_len, n_features)
        """
        if len(df) < self.seq_len:
            raise ValueError("Not enough data to build sequence")

        seq = df[feature_cols].tail(self.seq_len).values
        return np.expand_dims(seq, axis=0)

    # --------------------------
    # Predict a single sequence
    # --------------------------
    def predict_single_sequence(self, seq):
        """
        Run ensemble prediction for a single prepared sequence.
        Returns a float probability between 0 and 1.
        """
        if not self.models_cache:
            self.load_models()

        all_model_probs = []

        for model_name, fold_models in self.models_cache.items():
            fold_preds = [model.predict(seq, verbose=0).flatten() for model in fold_models]
            model_avg = np.mean(fold_preds, axis=0)  # Average folds of this architecture
            all_model_probs.append(model_avg)

        # Final ensemble average across base models
        final_prob = float(np.mean(all_model_probs))
        return final_prob

    # --------------------------
    # Predict historical dataframe
    # --------------------------
    def predict_dataframe(self, df, feature_cols=TOP_FEATURES, batch_size=64):
        """
        Generate rolling predictions for an entire feature DataFrame using bulk inference.
        Returns a DataFrame with columns: [timestamp, close, probability].
        """
        if len(df) < self.seq_len:
            raise ValueError("Data too short for sequence generation")

        if not self.models_cache:
            self.load_models()

        # ----------------------
        # 1. Build all sequences
        # ----------------------
        values = df[feature_cols].values
        num_samples = len(values) - self.seq_len

        X = np.zeros((num_samples, self.seq_len, len(feature_cols)))
        for i in range(num_samples):
            X[i] = values[i:i + self.seq_len]

        timestamps = df.iloc[self.seq_len:].timestamp.values
        closes = df.iloc[self.seq_len:].close.values

        # ----------------------
        # 2. Bulk predict
        # ----------------------
        all_model_probs = []

        for model_name, fold_models in self.models_cache.items():
            fold_preds = []
            for model in fold_models:
                preds = model.predict(X, batch_size=batch_size, verbose=0).flatten()
                fold_preds.append(preds)

            model_avg = np.mean(fold_preds, axis=0)  # Average across folds
            all_model_probs.append(model_avg)

        # Final ensemble average across architectures
        final_probs = np.mean(all_model_probs, axis=0)

        # ----------------------
        # 3. Return DataFrame
        # ----------------------
        return pd.DataFrame({
            "timestamp": timestamps,
            "close": closes,
            "probability": final_probs
        })

        # --------------------------
        # Generate trading signal
        # --------------------------
    def generate_signal(self, prob, threshold=THRESHOLD):
        """
        Convert probability into trading signal.
        """
        return "🟢BUY" if prob > threshold else "🔴HOLD"