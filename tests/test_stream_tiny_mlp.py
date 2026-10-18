import math
import pytest
from timeflowstream.core.tiny_mlp import TinyMLPFitter

def test_tiny_mlp_fits_nonlinear_transient():
    # Sinusoidal non-linear pulse (cannot be represented by a single linear line)
    fitter = TinyMLPFitter(hidden_dim=8, max_epochs=40, lr=0.08)
    t_vals = [i / 20.0 for i in range(20)]
    y_vals = [math.sin(t * math.pi) for t in t_vals]

    seg = fitter.fit(channel_id=1, t_start=0.0, t_vals=t_vals, y_vals=y_vals)
    assert seg is not None
    assert len(seg.latent_weights) <= 256  # Compact footprint: under 256 bytes

    recon = fitter.reconstruct(seg, t_vals)
    assert len(recon) == len(y_vals)
    max_err = max(abs(y - r) for y, r in zip(y_vals, recon))
    assert max_err < 0.25  # Substantial non-linear shape capture
