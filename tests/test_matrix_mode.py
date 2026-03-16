from __future__ import annotations

import numpy as np

from timeflowcodec import decode_tfc_to_video, encode_video_to_tfc
from timeflowcodec.encoder import _fit_rank1_matrix
from timeflowcodec.utils import mse


def test_fit_rank1_matrix_exact_reconstruction():
    t = np.array([0.0, 1.0, -2.0, 0.5], dtype=np.float32)
    s = np.array([2.0, -1.0, 0.25], dtype=np.float32)
    x = 17.0 + np.outer(t, s)

    mean, temporal, spatial, recon = _fit_rank1_matrix(x)

    assert mse(x, recon) < 0.01
    assert abs(mean - float(np.mean(x))) < 1e-6
    assert temporal.shape == (x.shape[0],)
    assert spatial.shape == (x.shape[1],)


def test_matrix_mode_roundtrip_smoke(monkeypatch, tmp_path):
    T, H, W = 8, 4, 4
    base_t = np.sin(np.linspace(0, np.pi, T)).astype(np.float32)
    scale = np.array(
        [
            [1.0, 1.4, 0.7, 0.9],
            [1.2, 0.6, 1.1, 1.3],
            [0.8, 1.5, 1.0, 0.5],
            [0.9, 1.1, 0.4, 1.6],
        ],
        dtype=np.float32,
    )

    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    for i in range(T):
        tile = np.clip(120.0 + 70.0 * base_t[i] * scale, 0, 255).astype(np.uint8)
        frames[i, :, :, 0] = tile
        frames[i, :, :, 1] = tile
        frames[i, :, :, 2] = tile

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
    output_tfc = tmp_path / "matrix.tfc"
    recon_video = tmp_path / "recon.mp4"

    encode_video_to_tfc(
        str(input_path),
        str(output_tfc),
        tau=0.01,
        container_version=2,
        tiling=2,
        matrix_mode=True,
        matrix_tau=0.2,
    )
    decode_tfc_to_video(str(output_tfc), str(recon_video), stream_output=False)

    assert "frames" in captured
    recon = captured["frames"]
    assert recon.shape == frames.shape
    # Quantized matrix factors are lossy; keep threshold modest.
    assert mse(frames.astype(np.float32), recon.astype(np.float32)) < 25.0
