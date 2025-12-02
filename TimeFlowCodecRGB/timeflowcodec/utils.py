"""Utility helpers for RGB TimeFlowCodec."""
from __future__ import annotations

import numpy as np
import imageio.v2 as imageio


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        # Grayscale -> replicate to RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.shape[2] == 3:
            return arr.astype(np.uint8)
    raise ValueError(f"Unsupported frame shape for RGB conversion: {arr.shape}")


def load_video_rgb(path: str, max_frames: int | None = None) -> np.ndarray:
    """
    Load a video file as uint8 RGB frames with shape (T, H, W, 3).
    """
    reader = imageio.get_reader(path)
    frames = []
    for idx, frame in enumerate(reader):
        if max_frames is not None and idx >= max_frames:
            break
        frames.append(_ensure_rgb(frame))
    reader.close()
    if not frames:
        raise ValueError(f"No frames read from video: {path}")
    arr = np.stack(frames, axis=0)
    return arr.astype(np.uint8)


def save_video_from_rgb(frames: np.ndarray, path: str, fps: int = 30) -> None:
    """
    Save uint8 RGB frames with shape (T, H, W, 3) to a video file.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (T, H, W, 3)")
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean squared error between two arrays."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))
