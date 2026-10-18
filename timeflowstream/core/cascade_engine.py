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

    def __init__(self, tolerance: float = 0.05, min_window: int = 4, max_window: int = 128, steady_threshold: int = 6):
        self.tolerance = tolerance
        self.min_window = min_window
        self.max_window = max_window
        self.steady_threshold = steady_threshold
        self.mlp_fitter = TinyMLPFitter(hidden_dim=8)

    def compress_stream(self, channel_id: int, t_vals: List[float], y_vals: List[float]) -> bytes:
        if not t_vals or not y_vals:
            return b""

        alr = StreamingLinearRegression(
            channel_id=channel_id,
            tolerance=self.tolerance,
            min_window=self.min_window,
            max_window=self.max_window
        )

        raw_segs = []
        for t, y in zip(t_vals, y_vals):
            seg = alr.push(t, y)
            if seg is not None:
                raw_segs.append(seg)

        final_seg = alr.flush()
        if final_seg:
            raw_segs.append(final_seg)

        # Two-Tier Cascade:
        # Long segments remain Tier 1 LinearSegments.
        # Short/rapidly breaking segments are merged into Tier 2 MLPSegments.
        final_segments = []
        transient_t = []
        transient_y = []

        for s in raw_segs:
            if s.length >= self.steady_threshold:
                if transient_t:
                    final_segments.append(self.mlp_fitter.fit(channel_id, transient_t[0], transient_t, transient_y))
                    transient_t, transient_y = [], []
                final_segments.append(s)
            else:
                # Reconstruct transient points from the short linear segments
                seg_t = [s.t_start + float(i) for i in range(s.length)]
                seg_y = [s.intercept + s.slope * float(i) for i in range(s.length)]
                transient_t.extend(seg_t)
                transient_y.extend(seg_y)

        if transient_t:
            final_segments.append(self.mlp_fitter.fit(channel_id, transient_t[0], transient_t, transient_y))

        # Serialize into binary bitstream
        payload = bytearray(self.MAGIC)
        payload += struct.pack("<I", len(final_segments))

        for s in final_segments:
            if isinstance(s, LinearSegment):
                # Mode 1: [mode(B), channel(H), t_start(d), length(H), intercept(f), slope(f)]
                payload += struct.pack("<BHdHf f", CompressionMode.MODE_LINEAR, s.channel_id, s.t_start, s.length, s.intercept, s.slope)
            elif isinstance(s, MLPSegment):
                # Mode 2: [mode(B), channel(H), t_start(d), length(H), payload_len(H)] + bytes
                payload += struct.pack("<BHdHH", CompressionMode.MODE_TINY_MLP, s.channel_id, s.t_start, s.length, len(s.latent_weights))
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
                _, channel_id, t_start, length, intercept, slope = struct.unpack_from("<BHdHf f", data, offset)
                offset += struct.calcsize("<BHdHf f")
                for i in range(length):
                    all_t.append(t_start + float(i))
                    all_y.append(intercept + slope * float(i))
            elif mode == CompressionMode.MODE_TINY_MLP:
                _, channel_id, t_start, length, payload_len = struct.unpack_from("<BHdHH", data, offset)
                offset += struct.calcsize("<BHdHH")
                latent_weights = data[offset:offset+payload_len]
                offset += payload_len

                mlp_seg = MLPSegment(channel_id=channel_id, t_start=t_start, length=length, latent_weights=latent_weights, mse=0.0)
                seg_t = [t_start + float(i) for i in range(length)]
                seg_y = self.mlp_fitter.reconstruct(mlp_seg, seg_t)
                all_t.extend(seg_t)
                all_y.extend(seg_y)

        return all_t, all_y
