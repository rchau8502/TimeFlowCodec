"""Rate control engine for TimeFlowCodec.

Provides standard video codec rate control mechanisms:
1. CRF (Constant Rate Factor) to parameter mapping conforming to AV1/x264 conventions.
2. VBR (Variable Bitrate) and CBR (Constant Bitrate) virtual leaky bucket models
   that dynamically modulate error tolerance tau and quantization step size Q
   across chunk windows to hit precise target bitrates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

DEFAULT_CRF = 23
CRF_MIN = 0
CRF_MAX = 63

# Base parameters at default CRF 23
BASE_TAU_SCREEN = 0.001
BASE_TAU_ANIME = 0.003
BASE_TAU_NATURAL = 0.008

BASE_SLOPE_THRESH = 0.001


def crf_to_parameters(crf: float, tune: str = "screen") -> Dict[str, Any]:
    """Map a user-facing CRF (0-63) to TimeFlowCodec encoder parameters.

    Conforms to standard AV1 / x264 intuition:
    - Lower CRF = higher quality, higher bitrate.
    - Every ~6 CRF steps doubles/halves tolerance (log-linear scale).
    - CRF 0: Near-lossless (tau=1e-6).
    - CRF 23: Default balanced quality.
    - CRF 50+: Low-bitrate streaming.
    """
    crf_clamped = max(float(CRF_MIN), min(float(CRF_MAX), float(crf)))

    if tune == "anime":
        base_tau = BASE_TAU_ANIME
    elif tune == "natural":
        base_tau = BASE_TAU_NATURAL
    else:  # default "screen"
        base_tau = BASE_TAU_SCREEN

    # Log-linear scaling: tau(crf) = base_tau * 2^((crf - 23) / 6.0)
    exponent = (crf_clamped - DEFAULT_CRF) / 6.0
    tau = base_tau * math.pow(2.0, exponent)
    tau = max(1e-6, min(0.05, tau))

    # Slope threshold scale: higher CRF permits treating slight slopes as constant
    slope_thresh = BASE_SLOPE_THRESH * math.pow(1.5, (crf_clamped - DEFAULT_CRF) / 10.0)
    slope_thresh = max(1e-4, min(0.02, slope_thresh))

    # Quantization step size Q for residual coding:
    # At CRF <= 12: Q = 1 (finest)
    # At CRF 23: Q = 2
    # At CRF 40+: Q = 8+
    q_step = max(1, int(round(math.pow(2.0, (crf_clamped - 15.0) / 8.0))))

    # Lagrangian lambda for RDO: lambda ~= 0.85 * 2^((crf - 12) / 4.0)
    lambda_rdo = 0.85 * math.pow(2.0, (crf_clamped - 12.0) / 4.0)

    return {
        "crf": crf_clamped,
        "tau": float(tau),
        "slope_threshold": float(slope_thresh),
        "q_step": int(q_step),
        "lambda_rdo": float(lambda_rdo),
        "tune": tune,
    }


def parse_bitrate_string(bitrate_str: str | int | float) -> float:
    """Parse bitrate string (e.g. '1500k', '2M', '800000') into kbps."""
    if isinstance(bitrate_str, (int, float)):
        val = float(bitrate_str)
        # If greater than 10000, assume bps -> convert to kbps
        return val / 1000.0 if val > 10000.0 else val

    s = str(bitrate_str).strip().lower()
    if s.endswith("k") or s.endswith("kbps"):
        return float(s.rstrip("kbps"))
    elif s.endswith("m") or s.endswith("mbps"):
        return float(s.rstrip("mbps")) * 1000.0
    else:
        val = float(s)
        return val / 1000.0 if val > 10000.0 else val


@dataclass
class RateControlStats:
    chunks_encoded: int
    frames_encoded: int
    total_bits: int
    target_bits: int
    average_bitrate_kbps: float
    target_bitrate_kbps: float
    buffer_fullness_bits: float
    current_tau: float


class RateController:
    """Dynamic rate control engine implementing leaky-bucket buffer regulation."""

    def __init__(
        self,
        target_bitrate_kbps: float,
        fps: float = 30.0,
        mode: str = "vbr",
        buffer_size_seconds: float = 1.5,
        tune: str = "screen",
    ) -> None:
        self.target_bitrate_kbps = max(10.0, float(target_bitrate_kbps))
        self.fps = max(1.0, float(fps))
        self.mode = mode.lower()
        self.tune = tune

        # Buffer capacity in bits
        self.target_bits_per_frame = (self.target_bitrate_kbps * 1000.0) / self.fps
        self.buffer_size_bits = self.target_bitrate_kbps * 1000.0 * max(0.5, float(buffer_size_seconds))
        # Initial buffer fullness set to 50%
        self.buffer_fullness_bits = self.buffer_size_bits * 0.5

        # Initial baseline tau estimate derived from target bitrate
        # Screen content: ~500 kbps aligns with tau ~ 0.001
        self.current_tau = self._estimate_initial_tau()
        self.current_q_step = 2

        self.total_bits = 0
        self.total_frames = 0
        self.chunks_encoded = 0

    def _estimate_initial_tau(self) -> float:
        """Estimate initial tau from target bitrate."""
        # For screen content at 1080p30:
        # 200 kbps -> tau ~ 0.004
        # 1000 kbps -> tau ~ 0.001
        # 4000 kbps -> tau ~ 0.0002
        if self.target_bitrate_kbps >= 2000.0:
            return 0.0003
        elif self.target_bitrate_kbps >= 1000.0:
            return 0.0008
        elif self.target_bitrate_kbps >= 500.0:
            return 0.0015
        elif self.target_bitrate_kbps >= 250.0:
            return 0.003
        else:
            return 0.006

    def get_chunk_parameters(self, chunk_frames: int) -> Dict[str, Any]:
        """Compute encoder parameters for the upcoming chunk."""
        # Virtual buffer error: ratio of current buffer fullness to target (50%)
        target_buffer = self.buffer_size_bits * 0.5
        buffer_ratio = self.buffer_fullness_bits / max(1.0, target_buffer)

        # In CBR mode, adjust aggressively (gamma = 1.2)
        # In VBR mode, allow smoother breathing (gamma = 0.6)
        gamma = 1.2 if self.mode == "cbr" else 0.6

        # Modulate tau based on buffer fullness
        # If buffer is too full (>1.0), increase tau (reduce quality) to save bits
        # If buffer is running empty (<1.0), decrease tau (increase quality) to fill budget
        factor = math.pow(max(0.2, min(5.0, buffer_ratio)), gamma)
        adjusted_tau = self.current_tau * factor
        adjusted_tau = max(1e-6, min(0.05, adjusted_tau))

        # Adjust quantization step accordingly
        if adjusted_tau < 0.0005:
            q = 1
        elif adjusted_tau < 0.002:
            q = 2
        elif adjusted_tau < 0.008:
            q = 4
        else:
            q = 8

        lambda_rdo = adjusted_tau * 1000.0

        return {
            "tau": float(adjusted_tau),
            "slope_threshold": BASE_SLOPE_THRESH,
            "q_step": int(q),
            "lambda_rdo": float(lambda_rdo),
            "tune": self.tune,
        }

    def update(self, chunk_frames: int, actual_bytes: int, actual_psnr: float | None = None) -> None:
        """Update buffer state and rate controller history after encoding a chunk."""
        actual_bits = actual_bytes * 8
        target_chunk_bits = self.target_bits_per_frame * chunk_frames

        self.total_bits += actual_bits
        self.total_frames += chunk_frames
        self.chunks_encoded += 1

        # Update leaky bucket: buffer fills with actual bits, drains at target rate
        self.buffer_fullness_bits += actual_bits - target_chunk_bits
        # Clamp buffer fullness within [0, buffer_size_bits]
        self.buffer_fullness_bits = max(0.0, min(self.buffer_size_bits, self.buffer_fullness_bits))

        # Update base tau with exponential moving average (EMA)
        chunk_bitrate_kbps = (actual_bits / 1000.0) / (chunk_frames / self.fps) if chunk_frames > 0 else 0.0
        if chunk_bitrate_kbps > 0:
            rate_error = chunk_bitrate_kbps / self.target_bitrate_kbps
            # Adjust current_tau towards target
            step_factor = math.pow(rate_error, 0.4)
            self.current_tau = max(1e-6, min(0.05, self.current_tau * step_factor))

    def get_stats(self) -> RateControlStats:
        """Return current rate control statistics."""
        duration_s = self.total_frames / self.fps if self.fps > 0 else 0.0
        avg_bitrate = (self.total_bits / 1000.0) / duration_s if duration_s > 0 else 0.0
        target_bits_total = int(self.target_bits_per_frame * self.total_frames)

        return RateControlStats(
            chunks_encoded=self.chunks_encoded,
            frames_encoded=self.total_frames,
            total_bits=self.total_bits,
            target_bits=target_bits_total,
            average_bitrate_kbps=float(avg_bitrate),
            target_bitrate_kbps=self.target_bitrate_kbps,
            buffer_fullness_bits=float(self.buffer_fullness_bits),
            current_tau=float(self.current_tau),
        )
