from __future__ import annotations

import numpy as np

from timeflowcodec import decode_tfc_to_video, encode_video_to_tfc
from timeflowcodec.utils import mse


def test_encode_decode_roundtrip(monkeypatch, tmp_path):
    T, H, W = 8, 8, 8
    rng = np.random.default_rng(0)
    frames = np.zeros((T, H, W), dtype=np.uint8)

    for t in range(T):
        frames[t, 0:2, 0:2] = 50  # constant region
        frames[t, 2:4, 2:4] = np.clip(20 + 10 * t, 0, 255)  # linear region
        frames[t, 4:, 4:] = rng.integers(0, 256, size=(H - 4, W - 4), dtype=np.uint8)  # noisy region

    captured: dict[str, np.ndarray] = {}

    def fake_load_video(path: str, max_frames=None):  # noqa: ARG001
        return frames

    def fake_save_video(arr: np.ndarray, path: str, fps: int = 30):  # noqa: ARG001
        captured["frames"] = arr.copy()

    monkeypatch.setattr("timeflowcodec.utils.load_video_to_array", fake_load_video)
    monkeypatch.setattr("timeflowcodec.encoder.load_video_to_array", fake_load_video)
    monkeypatch.setattr("timeflowcodec.utils.save_array_to_video", fake_save_video)
    monkeypatch.setattr("timeflowcodec.decoder.save_array_to_video", fake_save_video)

    input_path = tmp_path / "dummy.mp4"
    output_tfc = tmp_path / "video.tfc"
    recon_video = tmp_path / "recon.mp4"

    encode_video_to_tfc(str(input_path), str(output_tfc))
    decode_tfc_to_video(str(output_tfc), str(recon_video))

    assert "frames" in captured
    recon = captured["frames"]

    overall_mse = mse(frames, recon)
    assert overall_mse < 1e-3

    const_region_orig = frames[:, 0, 0].astype(np.float32)
    const_region_recon = recon[:, 0, 0].astype(np.float32)
    assert mse(const_region_orig, const_region_recon) < 1e-6

    linear_region_orig = frames[:, 2, 2].astype(np.float32)
    linear_region_recon = recon[:, 2, 2].astype(np.float32)
    assert mse(linear_region_orig, linear_region_recon) < 1e-3
