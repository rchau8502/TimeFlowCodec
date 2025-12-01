"""Utility functions for loading/saving videos and basic metrics."""
from __future__ import annotations

import numpy as np
import imageio.v2 as imageio


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3].astype(np.float32)
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return gray.astype(np.uint8)
    raise ValueError("Unsupported frame shape for grayscale conversion: %s" % (arr.shape,))


def load_video_to_array(path: str, max_frames: int | None = None) -> np.ndarray:
    """
    Load a video file and return frames as (T, H, W) uint8, grayscale.
    """
    reader = imageio.get_reader(path)
    frames = []
    for idx, frame in enumerate(reader):
        if max_frames is not None and idx >= max_frames:
            break
        frames.append(_to_grayscale(frame))
    reader.close()
    if not frames:
        raise ValueError("No frames read from video: %s" % path)
    arr = np.stack(frames, axis=0)
    return arr.astype(np.uint8)


def save_array_to_video(frames: np.ndarray, path: str, fps: int = 30) -> None:
    """
    Save (T, H, W) uint8 frames to a video file at the given fps.
    """
    if frames.ndim != 3:
        raise ValueError("frames must have shape (T, H, W)")
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute mean squared error between two arrays of the same shape.
    """
    if a.shape != b.shape:
        raise ValueError("Shape mismatch: %s vs %s" % (a.shape, b.shape))
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))
