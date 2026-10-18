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
