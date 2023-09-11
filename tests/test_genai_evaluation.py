"""Tests for `genai_evaluation` package."""

import numpy as np
import pandas as pd
import pytest
from genai_evaluation import ks_statistic

# Define test cases
def test_ks_statistic():
    # Basic test cases
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert ks_statistic(a, b) == 0.0

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert ks_statistic(a, b) == 3.0

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 2.5])
    assert ks_statistic(a, b) == 0.5  # Approximate value

    assert ks_statistic(np.array([1.0]), np.array([1.0])) == 0.0  
    # Single-element arrays

    # Large arrays
    a = np.arange(1, 1001)
    b = np.arange(1001, 2001)
    assert ks_statistic(a, b) == 1000

    # Mismatched length arrays (Expecting a ValueError)
    with pytest.raises(ValueError):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        ks_statistic(a, b)
    

    # Edge cases. Empty Arrays
    with pytest.raises(ValueError):    
        a = np.array([]) 
        b = np.array([])
        ks_statistic(a, b)