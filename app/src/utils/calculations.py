import numpy as np
from numpy.typing import NDArray

def clamp(x: float, min_val: float = 0.1, max_val: float = 0.9) -> float:
    return max(min(x, max_val), min_val)

def normalize(data: NDArray):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data