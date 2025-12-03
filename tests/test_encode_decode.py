from __future__ import annotations

import numpy as np

from timeflowcodec import decode_tfc_to_video, encode_video_to_tfc
from timeflowcodec.utils import mse


def test_encode_decode_roundtrip(monkeypatch, tmp_path):
    T, H, W = 6, 6, 6
    rng = np.random.default_rng(42)
    frames = np.zeros((T, H, W, 3), dtype=np.uint8)

    for t in range(T):
        # Red channel constant block
        frames[t, 0:2, 0:2, 0] = 120
        # Green channel linear block
        frames[t, 2:4, 2:4, 1] = np.clip(30 + 15 * t, 0, 255)
        # Blue channel noisy region
        frames[t, 4:, 4:, 2] = rng.integers(0, 256, size=(H - 4, W - 4), dtype=np.uint8)
        # Mixed pixel varying in all channels
        frames[t, 1, 4, :] = np.clip([10 * t, 5 * t, 20 * t], 0, 255)

    captured: dict[str, np.ndarray] = {}

    def fake_save_video(arr: np.ndarray, path: str, fps: int = 30):  # noqa: ARG001
        captured["frames"] = arr.copy()

    class FakeReader:
        def __iter__(self):
            for f in frames:
                yield f

    def fake_get_reader(path):  # noqa: ARG001
        return FakeReader()

    monkeypatch.setattr("timeflowcodec.encoder.imageio.get_reader", fake_get_reader)
    monkeypatch.setattr("timeflowcodec.encoder._ensure_rgb", lambda f: f)
    monkeypatch.setattr("timeflowcodec.utils.save_video_from_rgb", fake_save_video)
    monkeypatch.setattr("timeflowcodec.decoder.save_video_from_rgb", fake_save_video)

    input_path = tmp_path / "dummy.mp4"
    output_tfc = tmp_path / "video.tfc"
    recon_video = tmp_path / "recon.mp4"

    encode_video_to_tfc(str(input_path), str(output_tfc))
    decode_tfc_to_video(str(output_tfc), str(recon_video))

    assert "frames" in captured
    recon = captured["frames"]

    overall_mse = mse(frames, recon)
    assert overall_mse < 1e-2

    # Check constant red block
    assert mse(frames[:, 0, 0, 0].astype(np.float32), recon[:, 0, 0, 0].astype(np.float32)) < 1e-6
    # Check linear green block
    assert mse(frames[:, 2, 2, 1].astype(np.float32), recon[:, 2, 2, 1].astype(np.float32)) < 1e-3
