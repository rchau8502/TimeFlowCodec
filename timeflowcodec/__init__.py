"""RGB TimeFlowCodec per-pixel temporal modeling."""
from .constants import (
    PLANE_R,
    PLANE_G,
    PLANE_B,
    MODE_TFC_LINEAR,
    MODE_TFC_CONST,
    MODE_FB_RAW,
    MODE_RESERVED,
    DEFAULT_TAU,
    DEFAULT_SLOPE_THRESHOLD,
    COLOR_FORMAT_RGB,
    BITS_PER_MODE,
)
from .encoder import encode_video_to_tfc
from .decoder import decode_tfc_to_video
from .version import __version__, get_version_string, get_build_meta

__all__ = [
    "PLANE_R",
    "PLANE_G",
    "PLANE_B",
    "MODE_TFC_LINEAR",
    "MODE_TFC_CONST",
    "MODE_FB_RAW",
    "MODE_RESERVED",
    "DEFAULT_TAU",
    "DEFAULT_SLOPE_THRESHOLD",
    "COLOR_FORMAT_RGB",
    "BITS_PER_MODE",
    "encode_video_to_tfc",
    "decode_tfc_to_video",
    "get_version_string",
    "get_build_meta",
    "__version__",
]
