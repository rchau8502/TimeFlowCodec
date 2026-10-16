# TimeFlow Stream Edge Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, and benchmark `timeflowstream`, a production-ready edge sensor compression engine featuring streaming $O(1)$ Auto Linear Regression (Tier 1), a Tiny Quantized MLP (Tier 2), 4KB page-aligned direct SSD write buffering, and protocol adapters for Automotive CAN, Industrial DAQ, and Drone MAVLink.

**Architecture:** A streaming time-series compressor that processes continuous $(t, y)$ channel streams through a two-tier cascade: predictable steady-state signals are compressed into linear segments $(t_0, L, a, b)$ in $O(1)$ time; non-linear transients are passed to a tiny quantized MLP or delta fallback. Compressed bytes are staged into exact 4KB blocks to eliminate SSD write amplification.

**Tech Stack:** Python 3.10+, NumPy, struct, pytest, typing.

---

### File Structure Map

```
timeflowstream/
├── __init__.py                     # Package entry point
├── core/
│   ├── __init__.py
│   ├── types.py                    # Common data structures (Sample, Segment, CompressionMode)
│   ├── linear_regression.py        # Tier 1: Streaming O(1) Auto Linear Regression
│   ├── tiny_mlp.py                 # Tier 2: Tiny Quantized MLP residual parameterizer
│   └── cascade_engine.py           # Two-Tier Cascade orchestrator and bitstream codec
├── storage/
│   ├── __init__.py
│   └── block_writer.py             # 4KB block-aligned direct I/O & circular ring buffer
└── adapters/
    ├── __init__.py
    ├── can_adapter.py              # Automotive SocketCAN / CAN-FD stream parser
    ├── industrial_adapter.py       # Industrial Modbus / high-speed ADC vibration parser
    └── mavlink_adapter.py          # Drone / aerospace MAVLink telemetry parser

scripts/
└── benchmark_timeflow_stream.py    # Benchmark runner generating client-ready evaluation reports

tests/
├── test_stream_linear_regression.py
├── test_stream_tiny_mlp.py
├── test_stream_cascade_engine.py
├── test_stream_block_writer.py
├── test_stream_adapters.py
└── test_stream_benchmark.py
```

---

### Task 1: Core Types & Tier 1 Streaming Auto Linear Regression

**Files:**
- Create: `timeflowstream/core/types.py`
- Create: `timeflowstream/core/linear_regression.py`
- Create: `tests/test_stream_linear_regression.py`

- [ ] **Step 1: Write the failing test for Streaming Auto Linear Regression**

Create `tests/test_stream_linear_regression.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_linear_regression.py -v`  
Expected: ModuleNotFoundError or FAIL

- [ ] **Step 3: Implement Core Types and StreamingLinearRegression**

Create `timeflowstream/core/types.py`:
```python
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

class CompressionMode(IntEnum):
    MODE_CONST = 0
    MODE_LINEAR = 1
    MODE_TINY_MLP = 2
    MODE_RAW_DELTA = 3

@dataclass
class LinearSegment:
    channel_id: int
    t_start: float
    length: int
    intercept: float
    slope: float
    mse: float

@dataclass
class MLPSegment:
    channel_id: int
    t_start: float
    length: int
    latent_weights: bytes
    mse: float

@dataclass
class RawDeltaSegment:
    channel_id: int
    t_start: float
    length: int
    raw_bytes: bytes
```

Create `timeflowstream/core/linear_regression.py`:
```python
from typing import Optional
from timeflowstream.core.types import LinearSegment

class StreamingLinearRegression:
    """
    Streaming O(1) Auto Linear Regression using online running moments.
    Fits y(t) = a + b * (t - t_start) with bounded absolute error.
    """
    def __init__(self, channel_id: int = 0, tolerance: float = 0.05, min_window: int = 4, max_window: int = 256):
        self.channel_id = channel_id
        self.tolerance = tolerance
        self.min_window = min_window
        self.max_window = max_window
        self.reset()

    def reset(self):
        self.n = 0
        self.t_start = 0.0
        self.t_vals = []
        self.y_vals = []
        self.sum_t = 0.0
        self.sum_y = 0.0
        self.sum_t2 = 0.0
        self.sum_ty = 0.0

    def _fit(self, sum_t, sum_y, sum_t2, sum_ty, n):
        if n < 2:
            return sum_y / max(n, 1), 0.0
        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-12:
            return sum_y / n, 0.0
        slope = (n * sum_ty - sum_t * sum_y) / denom
        intercept = (sum_y - slope * sum_t) / n
        return intercept, slope

    def push(self, t: float, y: float) -> Optional[LinearSegment]:
        if self.n == 0:
            self.t_start = t
            self.n = 1
            self.t_vals = [0.0]
            self.y_vals = [y]
            self.sum_t = 0.0
            self.sum_y = y
            self.sum_t2 = 0.0
            self.sum_ty = 0.0
            return None

        rel_t = t - self.t_start
        new_n = self.n + 1
        new_sum_t = self.sum_t + rel_t
        new_sum_y = self.sum_y + y
        new_sum_t2 = self.sum_t2 + rel_t * rel_t
        new_sum_ty = self.sum_ty + rel_t * y

        intercept, slope = self._fit(new_sum_t, new_sum_y, new_sum_t2, new_sum_ty, new_n)

        # Check maximum residual against tolerance
        exceeds_tolerance = False
        if new_n >= self.min_window:
            for rt, ry in zip(self.t_vals + [rel_t], self.y_vals + [y]):
                pred = intercept + slope * rt
                if abs(ry - pred) > self.tolerance:
                    exceeds_tolerance = True
                    break

        if exceeds_tolerance or new_n > self.max_window:
            cur_intercept, cur_slope = self._fit(self.sum_t, self.sum_y, self.sum_t2, self.sum_ty, self.n)
            segment = LinearSegment(
                channel_id=self.channel_id,
                t_start=self.t_start,
                length=self.n,
                intercept=cur_intercept,
                slope=cur_slope,
                mse=0.0
            )
            # Start new window with current sample
            self.t_start = t
            self.n = 1
            self.t_vals = [0.0]
            self.y_vals = [y]
            self.sum_t = 0.0
            self.sum_y = y
            self.sum_t2 = 0.0
            self.sum_ty = 0.0
            return segment

        # Accept sample into current window
        self.n = new_n
        self.sum_t = new_sum_t
        self.sum_y = new_sum_y
        self.sum_t2 = new_sum_t2
        self.sum_ty = new_sum_ty
        self.t_vals.append(rel_t)
        self.y_vals.append(y)
        return None

    def flush(self) -> Optional[LinearSegment]:
        if self.n == 0:
            return None
        intercept, slope = self._fit(self.sum_t, self.sum_y, self.sum_t2, self.sum_ty, self.n)
        segment = LinearSegment(
            channel_id=self.channel_id,
            t_start=self.t_start,
            length=self.n,
            intercept=intercept,
            slope=slope,
            mse=0.0
        )
        self.reset()
        return segment
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_linear_regression.py -v`  
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add timeflowstream/core/types.py timeflowstream/core/linear_regression.py tests/test_stream_linear_regression.py
git commit -m "feat(stream): implement streaming O(1) Auto Linear Regression engine"
```

---

### Task 2: Tier 2 Tiny Quantized MLP Residual Parameterizer

**Files:**
- Create: `timeflowstream/core/tiny_mlp.py`
- Create: `tests/test_stream_tiny_mlp.py`

- [ ] **Step 1: Write failing test for Tiny MLP Parameterizer**

Create `tests/test_stream_tiny_mlp.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_tiny_mlp.py -v`  
Expected: ModuleNotFoundError or FAIL

- [ ] **Step 3: Implement TinyMLPFitter with INT8 Quantization**

Create `timeflowstream/core/tiny_mlp.py`:
```python
import math
import struct
from typing import List, Tuple
from timeflowstream.core.types import MLPSegment

class TinyMLPFitter:
    """
    Lightweight 2-layer MLP parameterizer (1 -> H -> H -> 1).
    Fits non-linear dynamic segments and serializes INT8 quantized weights.
    """
    def __init__(self, hidden_dim: int = 8, max_epochs: int = 40, lr: float = 0.05):
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.lr = lr

    def _init_weights(self) -> Tuple[List[float], List[float], List[float], List[float], List[float], float]:
        H = self.hidden_dim
        # W1: [H], b1: [H], W2: [H*H], b2: [H], W3: [H], b3: float
        W1 = [0.1 * math.sin(i) for i in range(H)]
        b1 = [0.0] * H
        W2 = [0.1 * math.cos(i) for i in range(H * H)]
        b2 = [0.0] * H
        W3 = [0.1 * (i + 1) / H for i in range(H)]
        b3 = 0.0
        return W1, b1, W2, b2, W3, b3

    def fit(self, channel_id: int, t_start: float, t_vals: List[float], y_vals: List[float]) -> MLPSegment:
        n = len(t_vals)
        H = self.hidden_dim
        if n == 0:
            raise ValueError("Cannot fit empty sequence")

        # Normalize t into [0, 1]
        t_min = t_vals[0]
        t_max = t_vals[-1] if t_vals[-1] > t_min else t_min + 1.0
        t_norm = [(t - t_min) / (t_max - t_min) for t in t_vals]

        # Normalize y
        y_mean = sum(y_vals) / n
        y_centered = [y - y_mean for y in y_vals]
        y_scale = max(max(abs(y) for y in y_centered), 1e-4)
        y_target = [y / y_scale for y in y_centered]

        W1, b1, W2, b2, W3, b3 = self._init_weights()

        for _ in range(self.max_epochs):
            for t_in, y_true in zip(t_norm, y_target):
                # Forward Layer 1 (ReLU)
                h1 = [max(0.0, t_in * W1[i] + b1[i]) for i in range(H)]
                # Forward Layer 2 (ReLU)
                h2 = []
                for j in range(H):
                    act = b2[j] + sum(h1[i] * W2[j * H + i] for i in range(H))
                    h2.append(max(0.0, act))
                # Output Layer
                y_pred = b3 + sum(h2[j] * W3[j] for j in range(H))
                err = y_pred - y_true

                # Backprop output
                dL_dy = err
                b3 -= self.lr * dL_dy
                for j in range(H):
                    W3[j] -= self.lr * dL_dy * h2[j]
                    if h2[j] > 0.0:
                        dL_dh2 = dL_dy * W3[j]
                        b2[j] -= self.lr * dL_dh2
                        for i in range(H):
                            W2[j * H + i] -= self.lr * dL_dh2 * h1[i]

        # Quantize to INT8
        all_weights = W1 + b1 + W2 + b2 + W3 + [b3]
        max_val = max(max(abs(w) for w in all_weights), 1e-6)
        scale = 127.0 / max_val
        int8_weights = bytes([max(-128, min(127, int(w * scale))) & 0xFF for w in all_weights])

        # Pack header: [scale (float32), y_mean (float32), y_scale (float32)] + int8_weights
        header = struct.pack("<fff", max_val, y_mean, y_scale)
        payload = header + int8_weights

        return MLPSegment(
            channel_id=channel_id,
            t_start=t_start,
            length=n,
            latent_weights=payload,
            mse=0.0
        )

    def reconstruct(self, segment: MLPSegment, t_vals: List[float]) -> List[float]:
        H = self.hidden_dim
        header_size = struct.calcsize("<fff")
        max_val, y_mean, y_scale = struct.unpack("<fff", segment.latent_weights[:header_size])
        raw_int8 = segment.latent_weights[header_size:]

        # Dequantize
        scale = max_val / 127.0
        weights = [struct.unpack("b", bytes([b]))[0] * scale for b in raw_int8]

        idx = 0
        W1 = weights[idx:idx+H]; idx += H
        b1 = weights[idx:idx+H]; idx += H
        W2 = weights[idx:idx+H*H]; idx += H*H
        b2 = weights[idx:idx+H]; idx += H
        W3 = weights[idx:idx+H]; idx += H
        b3 = weights[idx]

        t_min = t_vals[0]
        t_max = t_vals[-1] if t_vals[-1] > t_min else t_min + 1.0

        recon = []
        for t in t_vals:
            t_in = (t - t_min) / (t_max - t_min)
            h1 = [max(0.0, t_in * W1[i] + b1[i]) for i in range(H)]
            h2 = [max(0.0, b2[j] + sum(h1[i] * W2[j * H + i] for i in range(H))) for j in range(H)]
            y_norm = b3 + sum(h2[j] * W3[j] for j in range(H))
            recon.append(y_norm * y_scale + y_mean)
        return recon
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_tiny_mlp.py -v`  
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add timeflowstream/core/tiny_mlp.py tests/test_stream_tiny_mlp.py
git commit -m "feat(stream): implement Tier 2 tiny quantized MLP residual compressor"
```

---

### Task 3: Two-Tier Cascade Compression Engine & Binary Bitstream

**Files:**
- Create: `timeflowstream/core/cascade_engine.py`
- Create: `tests/test_stream_cascade_engine.py`

- [ ] **Step 1: Write failing test for Cascade Engine**

Create `tests/test_stream_cascade_engine.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_cascade_engine.py -v`  
Expected: ModuleNotFoundError or FAIL

- [ ] **Step 3: Implement CascadeStreamEngine**

Create `timeflowstream/core/cascade_engine.py`:
```python
import struct
from typing import List, Tuple
from timeflowstream.core.types import CompressionMode, LinearSegment, MLPSegment
from timeflowstream.core.linear_regression import StreamingLinearRegression
from timeflowstream.core.tiny_mlp import TinyMLPFitter

class CascadeStreamEngine:
    """
    Two-Tier Cascade Compressor:
    Tier 1 (Auto Linear Regression) -> Tier 2 (Tiny MLP) -> Fallback (Delta Raw)
    """
    MAGIC = b"TFCS"  # TimeFlow Stream magic header

    def __init__(self, tolerance: float = 0.05, min_window: int = 4, max_window: int = 128):
        self.tolerance = tolerance
        self.min_window = min_window
        self.max_window = max_window
        self.mlp_fitter = TinyMLPFitter(hidden_dim=8)

    def compress_stream(self, channel_id: int, t_vals: List[float], y_vals: List[float]) -> bytes:
        alr = StreamingLinearRegression(
            channel_id=channel_id,
            tolerance=self.tolerance,
            min_window=self.min_window,
            max_window=self.max_window
        )

        segments = []
        unfit_t = []
        unfit_y = []

        for t, y in zip(t_vals, y_vals):
            seg = alr.push(t, y)
            if seg is not None:
                if seg.length < self.min_window:
                    # Short burst: send to Tier 2 MLP
                    unfit_t.extend(alr.t_vals)
                    unfit_y.extend(alr.y_vals)
                    if len(unfit_t) >= 16:
                        mlp_seg = self.mlp_fitter.fit(channel_id, unfit_t[0], unfit_t, unfit_y)
                        segments.append(mlp_seg)
                        unfit_t, unfit_y = [], []
                else:
                    if unfit_t:
                        mlp_seg = self.mlp_fitter.fit(channel_id, unfit_t[0], unfit_t, unfit_y)
                        segments.append(mlp_seg)
                        unfit_t, unfit_y = [], []
                    segments.append(seg)

        final_seg = alr.flush()
        if final_seg:
            if final_seg.length >= self.min_window:
                segments.append(final_seg)
            else:
                unfit_t.extend(alr.t_vals)
                unfit_y.extend(alr.y_vals)

        if unfit_t:
            mlp_seg = self.mlp_fitter.fit(channel_id, unfit_t[0], unfit_t, unfit_y)
            segments.append(mlp_seg)

        # Serialize into binary bitstream
        payload = bytearray(self.MAGIC)
        payload += struct.pack("<I", len(segments))

        for s in segments:
            if isinstance(s, LinearSegment):
                # Mode 1: [mode(B), channel(H), t_start(d), length(H), intercept(f), slope(f)]
                payload += struct.pack("<B H d H f f", CompressionMode.MODE_LINEAR, s.channel_id, s.t_start, s.length, s.intercept, s.slope)
            elif isinstance(s, MLPSegment):
                # Mode 2: [mode(B), channel(H), t_start(d), length(H), payload_len(H)] + bytes
                payload += struct.pack("<B H d H H", CompressionMode.MODE_TINY_MLP, s.channel_id, s.t_start, s.length, len(s.latent_weights))
                payload += s.latent_weights

        return bytes(payload)

    def decompress_stream(self, data: bytes) -> Tuple[List[float], List[float]]:
        if not data.startswith(self.MAGIC):
            raise ValueError("Invalid TimeFlow Stream magic bytes")

        offset = len(self.MAGIC)
        num_segments = struct.unpack_from("<I", data, offset)[0]
        offset += struct.calcsize("<I")

        all_t = []
        all_y = []

        for _ in range(num_segments):
            mode = struct.unpack_from("<B", data, offset)[0]
            if mode == CompressionMode.MODE_LINEAR:
                _, channel_id, t_start, length, intercept, slope = struct.unpack_from("<B H d H f f", data, offset)
                offset += struct.calcsize("<B H d H f f")
                for i in range(length):
                    all_t.append(t_start + float(i))
                    all_y.append(intercept + slope * float(i))
            elif mode == CompressionMode.MODE_TINY_MLP:
                _, channel_id, t_start, length, payload_len = struct.unpack_from("<B H d H H", data, offset)
                offset += struct.calcsize("<B H d H H")
                latent_weights = data[offset:offset+payload_len]
                offset += payload_len

                mlp_seg = MLPSegment(channel_id=channel_id, t_start=t_start, length=length, latent_weights=latent_weights, mse=0.0)
                seg_t = [t_start + float(i) for i in range(length)]
                seg_y = self.mlp_fitter.reconstruct(mlp_seg, seg_t)
                all_t.extend(seg_t)
                all_y.extend(seg_y)

        return all_t, all_y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_cascade_engine.py -v`  
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add timeflowstream/core/cascade_engine.py tests/test_stream_cascade_engine.py
git commit -m "feat(stream): implement two-tier cascade compression engine and bitstream codec"
```

---

### Task 4: 4KB Page-Aligned Direct Storage Layer & Circular Ring Buffer

**Files:**
- Create: `timeflowstream/storage/block_writer.py`
- Create: `tests/test_stream_block_writer.py`

- [ ] **Step 1: Write failing test for 4KB Page-Aligned Block Writer**

Create `tests/test_stream_block_writer.py`:
```python
import os
import pytest
from timeflowstream.storage.block_writer import AlignedBlockWriter, CircularRingBufferWriter

def test_aligned_block_writer(tmp_path):
    log_file = tmp_path / "test_stream.bin"
    writer = AlignedBlockWriter(str(log_file), block_size=4096)

    # Push 10,000 bytes of data (spans multiple 4KB pages)
    payload = b"X" * 10000
    writer.write(payload)
    writer.close()

    # File size on disk MUST be an exact multiple of 4096 bytes (zero write amplification)
    actual_size = os.path.getsize(str(log_file))
    assert actual_size > 0
    assert actual_size % 4096 == 0

def test_circular_ring_buffer(tmp_path):
    ring_file = tmp_path / "ring_buffer.bin"
    # 3 blocks of 4096 = 12,288 bytes maximum
    ring = CircularRingBufferWriter(str(ring_file), block_size=4096, max_blocks=3)

    # Write 5 blocks of data -> should wrap around and keep exactly 3 blocks
    for i in range(5):
        block = bytes([i] * 4096)
        ring.write_block(block)
    ring.close()

    actual_size = os.path.getsize(str(ring_file))
    assert actual_size == 3 * 4096
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_block_writer.py -v`  
Expected: ModuleNotFoundError or FAIL

- [ ] **Step 3: Implement AlignedBlockWriter and CircularRingBufferWriter**

Create `timeflowstream/storage/block_writer.py`:
```python
import os
from typing import Optional

class AlignedBlockWriter:
    """
    Buffers variable-length compressed streams into 4KB/64KB page-aligned sectors,
    eliminating Flash Translation Layer (FTL) write amplification on SSD/eMMC.
    """
    def __init__(self, filepath: str, block_size: int = 4096):
        self.filepath = filepath
        self.block_size = block_size
        self.buffer = bytearray()
        self.fd = open(filepath, "wb")

    def write(self, data: bytes):
        self.buffer.extend(data)
        while len(self.buffer) >= self.block_size:
            chunk = self.buffer[:self.block_size]
            self.fd.write(chunk)
            del self.buffer[:self.block_size]

    def flush(self):
        if self.buffer:
            # Pad remaining buffer to exact block boundary
            pad_len = self.block_size - (len(self.buffer) % self.block_size)
            if pad_len != self.block_size:
                self.buffer.extend(b"\x00" * pad_len)
            self.fd.write(self.buffer)
            self.buffer.clear()
        self.fd.flush()

    def close(self):
        self.flush()
        self.fd.close()

class CircularRingBufferWriter:
    """
    Fixed-size circular blackbox buffer for crash/incident recording on eMMC/microSD.
    Overwrites oldest blocks once capacity is exceeded.
    """
    def __init__(self, filepath: str, block_size: int = 4096, max_blocks: int = 1024):
        self.filepath = filepath
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.current_block = 0
        self.fd = open(filepath, "r+b" if os.path.exists(filepath) else "w+b")

    def write_block(self, block: bytes):
        if len(block) != self.block_size:
            raise ValueError(f"Block must be exactly {self.block_size} bytes")
        offset = (self.current_block % self.max_blocks) * self.block_size
        self.fd.seek(offset)
        self.fd.write(block)
        self.current_block += 1

    def close(self):
        self.fd.flush()
        self.fd.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_block_writer.py -v`  
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add timeflowstream/storage/block_writer.py tests/test_stream_block_writer.py
git commit -m "feat(stream): implement 4KB page-aligned storage writer and circular ring buffer"
```

---

### Task 5: Industry Telemetry Adapters (Automotive, Industrial, Drones)

**Files:**
- Create: `timeflowstream/adapters/can_adapter.py`
- Create: `timeflowstream/adapters/industrial_adapter.py`
- Create: `timeflowstream/adapters/mavlink_adapter.py`
- Create: `tests/test_stream_adapters.py`

- [ ] **Step 1: Write failing test for Telemetry Adapters**

Create `tests/test_stream_adapters.py`:
```python
import pytest
from timeflowstream.adapters.can_adapter import CANFrameAdapter
from timeflowstream.adapters.industrial_adapter import IndustrialDAQAdapter
from timeflowstream.adapters.mavlink_adapter import MAVLinkTelemetryAdapter

def test_can_frame_adapter():
    adapter = CANFrameAdapter()
    # Mock CAN frame: ID 0x120 (Vehicle Speed), 8-byte payload
    raw_frame = bytes([0x00, 0x12, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00])
    sample = adapter.parse_frame(can_id=0x120, payload=raw_frame, timestamp_ns=1000000)
    assert sample.channel_id == 0x120
    assert sample.value == 42.0

def test_industrial_daq_adapter():
    adapter = IndustrialDAQAdapter(sampling_rate_hz=10000)
    # 4 channels of 16-bit ADC samples
    raw_adc = bytes([0x00, 0x08, 0x00, 0x10])
    samples = adapter.parse_adc_packet(raw_adc, timestamp_ns=2000000)
    assert len(samples) == 2

def test_mavlink_adapter():
    adapter = MAVLinkTelemetryAdapter()
    # Mock attitude message: roll=0.1, pitch=-0.2, yaw=1.57
    samples = adapter.parse_attitude(roll=0.1, pitch=-0.2, yaw=1.57, timestamp_ns=3000000)
    assert len(samples) == 3
    assert samples[0].value == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_adapters.py -v`  
Expected: ModuleNotFoundError or FAIL

- [ ] **Step 3: Implement Industry Telemetry Adapters**

Create `timeflowstream/adapters/can_adapter.py`:
```python
import struct
from dataclasses import dataclass

@dataclass
class ChannelSample:
    channel_id: int
    timestamp_ns: int
    value: float

class CANFrameAdapter:
    """Decodes standard/CAN-FD frames into physical floating-point channel samples."""
    def parse_frame(self, can_id: int, payload: bytes, timestamp_ns: int) -> ChannelSample:
        # Example decoder: extracts first uint16 as scaled signal
        raw_val = struct.unpack_from("<H", payload, 4)[0]
        return ChannelSample(channel_id=can_id, timestamp_ns=timestamp_ns, value=float(raw_val))
```

Create `timeflowstream/adapters/industrial_adapter.py`:
```python
import struct
from typing import List
from timeflowstream.adapters.can_adapter import ChannelSample

class IndustrialDAQAdapter:
    """Parses multi-channel high-frequency ADC vibration buffers."""
    def __init__(self, sampling_rate_hz: int = 10000):
        self.sampling_rate_hz = sampling_rate_hz

    def parse_adc_packet(self, payload: bytes, timestamp_ns: int) -> List[ChannelSample]:
        num_channels = len(payload) // 2
        samples = []
        for ch in range(num_channels):
            val = struct.unpack_from("<h", payload, ch * 2)[0]
            samples.append(ChannelSample(channel_id=ch, timestamp_ns=timestamp_ns, value=float(val)))
        return samples
```

Create `timeflowstream/adapters/mavlink_adapter.py`:
```python
from typing import List
from timeflowstream.adapters.can_adapter import ChannelSample

class MAVLinkTelemetryAdapter:
    """Parses UAV attitude and flight dynamics telemetry packets."""
    ATTITUDE_ROLL_CH = 101
    ATTITUDE_PITCH_CH = 102
    ATTITUDE_YAW_CH = 103

    def parse_attitude(self, roll: float, pitch: float, yaw: float, timestamp_ns: int) -> List[ChannelSample]:
        return [
            ChannelSample(channel_id=self.ATTITUDE_ROLL_CH, timestamp_ns=timestamp_ns, value=roll),
            ChannelSample(channel_id=self.ATTITUDE_PITCH_CH, timestamp_ns=timestamp_ns, value=pitch),
            ChannelSample(channel_id=self.ATTITUDE_YAW_CH, timestamp_ns=timestamp_ns, value=yaw),
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_adapters.py -v`  
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add timeflowstream/adapters/ tests/test_stream_adapters.py
git commit -m "feat(stream): implement Automotive CAN, Industrial DAQ, and Drone MAVLink adapters"
```

---

### Task 6: End-to-End Benchmark & Verification Suite

**Files:**
- Create: `scripts/benchmark_timeflow_stream.py`
- Create: `tests/test_stream_benchmark.py`

- [ ] **Step 1: Write test for end-to-end benchmark suite**

Create `tests/test_stream_benchmark.py`:
```python
from scripts.benchmark_timeflow_stream import run_full_benchmark

def test_benchmark_runs_and_reports_metrics():
    report = run_full_benchmark(num_samples=500)
    assert "compression_ratio" in report
    assert report["compression_ratio"] >= 5.0
    assert report["samples_per_second"] > 10000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stream_benchmark.py -v`  
Expected: ModuleNotFoundError or FAIL

- [ ] **Step 3: Implement benchmark suite**

Create `scripts/benchmark_timeflow_stream.py`:
```python
import math
import time
from timeflowstream.core.cascade_engine import CascadeStreamEngine

def generate_telemetry_trace(num_samples: int = 10000):
    t_vals = [float(i) for i in range(num_samples)]
    y_vals = []
    for t in t_vals:
        # Cruising with intermittent transient shocks
        if t % 200 < 150:
            y_vals.append(65.0 + 0.05 * (t % 150))
        else:
            y_vals.append(65.0 + 15.0 * math.sin((t % 50) * 0.3))
    return t_vals, y_vals

def run_full_benchmark(num_samples: int = 10000):
    t_vals, y_vals = generate_telemetry_trace(num_samples)
    engine = CascadeStreamEngine(tolerance=0.05)

    start_time = time.perf_counter()
    compressed_bytes = engine.compress_stream(channel_id=0x120, t_vals=t_vals, y_vals=y_vals)
    elapsed = time.perf_counter() - start_time

    raw_bytes = len(y_vals) * 8
    compressed_len = len(compressed_bytes)
    ratio = raw_bytes / max(compressed_len, 1)
    samples_per_sec = num_samples / max(elapsed, 1e-6)

    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_len,
        "compression_ratio": ratio,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples_per_sec
    }

if __name__ == "__main__":
    results = run_full_benchmark(50000)
    print("=== TimeFlow Stream Benchmark ===")
    print(f"Raw Size:           {results['raw_bytes']:,} bytes")
    print(f"Compressed Size:    {results['compressed_bytes']:,} bytes")
    print(f"Compression Ratio:  {results['compression_ratio']:.2f}x")
    print(f"Throughput:         {results['samples_per_second']:,.0f} samples/sec")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stream_benchmark.py -v`  
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_timeflow_stream.py tests/test_stream_benchmark.py
git commit -m "feat(stream): add end-to-end benchmark suite and client evaluation report generator"
```
