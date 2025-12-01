"""Constants for TimeFlowCodec pixel-per-pixel temporal modeling."""

MODE_TFC_LINEAR = 0  # per-pixel linear model
MODE_TFC_CONST = 1   # per-pixel constant model
MODE_FB_RAW = 2      # fallback: raw per-frame pixel values (uint8)
MODE_RESERVED = 3

DEFAULT_TAU = 0.1
DEFAULT_SLOPE_THRESHOLD = 1e-3
