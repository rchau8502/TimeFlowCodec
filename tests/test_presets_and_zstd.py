from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from timeflowcodec import COMP_ZSTD, decode_tfc_to_video, encode_video_to_tfc
from timeflowcodec.encoder import _apply_preset_defaults


def test_preset_anime_defaults_upgrade_compression_and_tools():
    out = _apply_preset_defaults(
        preset="anime",
        tau=0.1,
        slope_threshold=1e-3,
        payload_comp_type=1,
        tiling=None,
        max_ram_mb=None,
        scene_cut="off",
        scene_threshold=0.35,
        matrix_mode=False,
        matrix_tau=0.12,
        matrix_rate_ratio=0.95,
    )
    (
        _tau,
        _slope,
        payload_comp_type,
        tiling,
        max_ram_mb,
        scene_cut,
        _scene_threshold,
        matrix_mode,
        _matrix_tau,
        _matrix_rate_ratio,
    ) = out
    assert payload_comp_type == COMP_ZSTD
    assert tiling == 16
    assert max_ram_mb == 2500
    assert scene_cut == "auto"
    assert matrix_mode is True


@pytest.mark.skipif(
    importlib.util.find_spec("zstandard") is None,
    reason="zstandard not installed",
)
def test_encode_decode_zstd_roundtrip(monkeypatch, tmp_path):
    t, h, w = 4, 4, 4
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        frames[i, :, :, 0] = 40 + i * 5
        frames[i, :, :, 1] = 80
        frames[i, :, :, 2] = 120 - i * 5

    captured: dict[str, np.ndarray] = {}

    def fake_save_video(arr: np.ndarray, path: str, fps: int = 30):  # noqa: ARG001
        captured["frames"] = arr.copy()

    class FakeReader:
        def __iter__(self):
            for f in frames:
                yield f

    monkeypatch.setattr("timeflowcodec.encoder.imageio.get_reader", lambda _p: FakeReader())
    monkeypatch.setattr("timeflowcodec.encoder._ensure_rgb", lambda f: f)
    monkeypatch.setattr("timeflowcodec.utils.save_video_from_rgb", fake_save_video)
    monkeypatch.setattr("timeflowcodec.decoder.save_video_from_rgb", fake_save_video)

    out_tfc = tmp_path / "zstd.tfc"
    out_mp4 = tmp_path / "recon.mp4"
    encode_video_to_tfc(
        "in.mp4",
        str(out_tfc),
        payload_comp_type=COMP_ZSTD,
        container_version=2,
        preset="custom",
    )
    decode_tfc_to_video(str(out_tfc), str(out_mp4), stream_output=False)
    assert "frames" in captured
    np.testing.assert_array_equal(captured["frames"], frames)

