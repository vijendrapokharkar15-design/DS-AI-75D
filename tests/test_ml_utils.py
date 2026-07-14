
import pytest
import numpy as np
from ml_utils import preprocess_features, compute_confidence

# ===== preprocess_features tests =====

def test_preprocess_basic():
    result = preprocess_features([0.0, 5.0, 10.0])
    expected = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_preprocess_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        preprocess_features([])

def test_preprocess_nan_raises():
    with pytest.raises(ValueError, match="finite"):
        preprocess_features([1.0, float("nan"), 3.0])

def test_preprocess_inf_raises():
    with pytest.raises(ValueError, match="finite"):
        preprocess_features([1.0, float("inf"), 3.0])

def test_preprocess_single_value():
    # all same values -> normalise to zeros
    result = preprocess_features([5.0, 5.0, 5.0])
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

def test_preprocess_negative_values():
    result = preprocess_features([-2.0, 0.0, 2.0])
    expected = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_almost_equal(result, expected)

# ===== compute_confidence tests =====

def test_confidence_high():
    assert compute_confidence(0.9) == "high"
    assert compute_confidence(0.8) == "high"
    assert compute_confidence(1.0) == "high"

def test_confidence_medium():
    assert compute_confidence(0.7) == "medium"
    assert compute_confidence(0.5) == "medium"

def test_confidence_low():
    assert compute_confidence(0.3) == "low"
    assert compute_confidence(0.0) == "low"

def test_confidence_invalid_raises():
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_confidence(1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_confidence(-0.1)

# ===== boundary tests =====

def test_confidence_boundary_08():
    assert compute_confidence(0.8) == "high"

def test_confidence_boundary_05():
    assert compute_confidence(0.5) == "medium"

def test_preprocess_two_values():
    result = preprocess_features([3.0, 7.0])
    np.testing.assert_array_almost_equal(result, [0.0, 1.0])
