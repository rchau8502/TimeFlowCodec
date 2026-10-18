import math
import struct
from typing import List
import numpy as np
from timeflowstream.core.types import MLPSegment

class TinyMLPFitter:
    """
    Tiny Fixed-Basis Multilayer Perceptron (Coordinate Network / ELM) with INT8 weights.
    Hidden layer computes non-linear ReLU basis activations h_i(t) = max(0, t - c_i).
    Output weights are solved via closed-form least-squares and quantized to INT8.
    """
    def __init__(self, hidden_dim: int = 8, max_epochs: int = 40, lr: float = 0.05):
        self.hidden_dim = hidden_dim
        # Fixed center thresholds spread evenly across [0, 1]
        self.centers = [float(i) / max(hidden_dim - 1, 1) for i in range(hidden_dim)]

    def _compute_basis(self, t_norm: List[float]) -> np.ndarray:
        # X: [N, 2 + H] -> [1 (bias), t (linear), ReLU(t - c_0), ..., ReLU(t - c_{H-1})]
        n = len(t_norm)
        H = self.hidden_dim
        t_arr = np.array(t_norm, dtype=np.float64)
        c_arr = np.array(self.centers, dtype=np.float64)
        
        basis = np.maximum(0.0, t_arr[:, None] - c_arr[None, :])
        X = np.hstack([np.ones((n, 1), dtype=np.float64), t_arr[:, None], basis])
        return X

    def fit(self, channel_id: int, t_start: float, t_vals: List[float], y_vals: List[float]) -> MLPSegment:
        n = len(t_vals)
        if n == 0:
            raise ValueError("Cannot fit empty sequence")

        t_min = t_vals[0]
        t_max = t_vals[-1] if t_vals[-1] > t_min else t_min + 1.0
        t_norm = [(t - t_min) / (t_max - t_min) for t in t_vals]

        X = self._compute_basis(t_norm)
        y_arr = np.array(y_vals, dtype=np.float64)

        # Solve output weights w in closed form: (X^T X + lambda I) w = X^T y
        w, _, _, _ = np.linalg.lstsq(X, y_arr, rcond=None)

        # Quantize w to INT8
        max_w = float(np.max(np.abs(w)))
        if max_w < 1e-7:
            max_w = 1e-7
        scale = 127.0 / max_w
        w_int8 = np.clip(np.round(w * scale), -128, 127).astype(np.int8)

        # Header: [max_w (float32)] + int8 weights
        header = struct.pack("<f", max_w)
        payload = header + w_int8.tobytes()

        return MLPSegment(
            channel_id=channel_id,
            t_start=t_start,
            length=n,
            latent_weights=payload,
            mse=0.0
        )

    def reconstruct(self, segment: MLPSegment, t_vals: List[float]) -> List[float]:
        header_size = struct.calcsize("<f")
        max_w = struct.unpack("<f", segment.latent_weights[:header_size])[0]
        raw_int8 = segment.latent_weights[header_size:]

        w_int8 = np.frombuffer(raw_int8, dtype=np.int8)
        w = w_int8.astype(np.float64) * (max_w / 127.0)

        t_min = t_vals[0]
        t_max = t_vals[-1] if t_vals[-1] > t_min else t_min + 1.0
        t_norm = [(t - t_min) / (t_max - t_min) for t in t_vals]

        X = self._compute_basis(t_norm)
        preds = X @ w
        return preds.tolist()
