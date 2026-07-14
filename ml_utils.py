
import numpy as np
from typing import Union

def preprocess_features(features: list[float]) -> np.ndarray:
    """Validate and preprocess input features."""
    if not features:
        raise ValueError("Features list cannot be empty")

    arr = np.array(features, dtype=float)

    if not np.all(np.isfinite(arr)):
        raise ValueError("All features must be finite (no NaN or Inf)")

    # normalise to 0-1 range
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def compute_confidence(probability: float) -> str:
    """Convert probability to confidence label."""
    if not 0 <= probability <= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")
    if probability >= 0.8:
        return "high"
    elif probability >= 0.5:
        return "medium"
    else:
        return "low"

def batch_predict(model, samples: list[list[float]]) -> list[int]:
    """Run batch prediction."""
    if not samples:
        raise ValueError("Samples list cannot be empty")
    X = np.array(samples)
    return model.predict(X).tolist()
