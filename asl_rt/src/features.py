# src/features.py
import numpy as np

def flatten_landmarks(landmarks, img_w, img_h):
    pts = np.array([[lm.x * img_w, lm.y * img_h] for lm in landmarks], dtype=np.float32)
    return pts.flatten()  # 42-dim: [0x,0y, ..., 20x,20y]

def row_box_normalize(X):
    """
    X: (n_samples, 42) -> normalize per row to the hand's bounding box.
    Returns same shape.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 42:
        raise ValueError(f"Expected (n,42) features, got {X.shape}")
    XY = X.reshape(-1, 21, 2)
    mins = XY.min(axis=1, keepdims=True)
    maxs = XY.max(axis=1, keepdims=True)
    size = np.linalg.norm(maxs - mins, axis=2, keepdims=True) + 1e-6
    XYn = (XY - mins) / size
    return XYn.reshape(-1, 42)
