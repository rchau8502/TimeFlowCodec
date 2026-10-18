import math
import pytest
from timeflowstream.core.cascade_engine import CascadeStreamEngine

def test_cascade_engine_encodes_and_decodes_mixed_stream():
    engine = CascadeStreamEngine(tolerance=0.08, min_window=4, max_window=64)

    # 100 samples: 50 steady cruising (Tier 1) + 50 non-linear braking pulse (Tier 2/Fallback)
    t_vals = [float(i) for i in range(100)]
    y_vals = []
    for t in t_vals:
        if t < 50:
            y_vals.append(30.0 + 0.1 * t)  # Linear cruising
        else:
            y_vals.append(35.0 + 10.0 * math.sin((t - 50) * 0.2))  # Dynamic braking

    compressed_bytes = engine.compress_stream(channel_id=1, t_vals=t_vals, y_vals=y_vals)
    raw_size = len(y_vals) * 8  # 800 bytes raw float64
    assert len(compressed_bytes) < raw_size / 4  # At least 4x-10x compression on mixed stream

    recon_t, recon_y = engine.decompress_stream(compressed_bytes)
    assert len(recon_y) == len(y_vals)
    # Validate reconstruction accuracy
    avg_err = sum(abs(y - r) for y, r in zip(y_vals, recon_y)) / len(y_vals)
    assert avg_err < 0.5
