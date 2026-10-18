import pytest
from timeflowstream.core.types import LinearSegment
from timeflowstream.core.linear_regression import StreamingLinearRegression

def test_constant_signal_compression():
    # Constant signal y = 42.0 for 50 samples
    alr = StreamingLinearRegression(tolerance=0.01, min_window=4, max_window=100)
    segments = []
    for t in range(50):
        seg = alr.push(float(t), 42.0)
        if seg is not None:
            segments.append(seg)
    final_seg = alr.flush()
    if final_seg:
        segments.append(final_seg)

    assert len(segments) == 1
    assert segments[0].length == 50
    assert pytest.approx(segments[0].slope, abs=1e-3) == 0.0
    assert pytest.approx(segments[0].intercept, abs=1e-3) == 42.0

def test_linear_ramp_compression():
    # Linear ramp y = 2.5 * t + 10.0
    alr = StreamingLinearRegression(tolerance=0.05, min_window=4, max_window=100)
    segments = []
    for t in range(40):
        val = 2.5 * t + 10.0
        seg = alr.push(float(t), val)
        if seg is not None:
            segments.append(seg)
    final_seg = alr.flush()
    if final_seg:
        segments.append(final_seg)

    assert len(segments) == 1
    assert segments[0].length == 40
    assert pytest.approx(segments[0].slope, abs=1e-2) == 2.5
    assert pytest.approx(segments[0].intercept, abs=1e-2) == 10.0

def test_step_change_triggers_new_segment():
    # Signal with abrupt step at t = 20
    alr = StreamingLinearRegression(tolerance=0.1, min_window=4, max_window=100)
    segments = []
    for t in range(40):
        val = 10.0 if t < 20 else 50.0
        seg = alr.push(float(t), val)
        if seg is not None:
            segments.append(seg)
    final_seg = alr.flush()
    if final_seg:
        segments.append(final_seg)

    assert len(segments) >= 2
    assert segments[0].length == 20
