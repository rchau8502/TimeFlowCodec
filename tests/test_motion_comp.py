from __future__ import annotations

import numpy as np

from timeflowcodec import decode_tfc_to_video, encode_video_to_tfc
from timeflowcodec.utils import mse


def test_motion_comp_moving_block(monkeypatch, tmp_path):
    T, H, W = 5, 16, 16
    frames = np.zeros((T, H, W, 3), dtype=np.uint8)

    # Frame 0: Tile (0, 0)
    frames[0, 0:8, 0:8, 0] = 180
    frames[0, 0:8, 0:8, 1] = 100

    # Frame 1: Tile (0, 1) (moved right by 1 tile / 8 pixels)
    frames[1, 0:8, 8:16, 0] = 180
    frames[1, 0:8, 8:16, 1] = 100

    # Frame 2: Tile (1, 0) (moved down by 1 tile / 8 pixels)
    frames[2, 8:16, 0:8, 0] = 180
    frames[2, 8:16, 0:8, 1] = 100

    # Frame 3: Tile (1, 1) (moved diagonally by 1 tile / 8 pixels)
    frames[3, 8:16, 8:16, 0] = 180
    frames[3, 8:16, 8:16, 1] = 100

    # Frame 4: Tile (0, 0) (moved back)
    frames[4, 0:8, 0:8, 0] = 180
    frames[4, 0:8, 0:8, 1] = 100

    captured: dict[str, np.ndarray] = {}

    def fake_save_video(arr: np.ndarray, path: str, fps: int = 30):
        captured["frames"] = arr.copy()

    class FakeReader:
        def __iter__(self):
            for f in frames:
                yield f

    def fake_get_reader(path):
        return FakeReader()

    monkeypatch.setattr("timeflowcodec.encoder.imageio.get_reader", fake_get_reader)
    monkeypatch.setattr("timeflowcodec.encoder._ensure_rgb", lambda f: f)
    monkeypatch.setattr("timeflowcodec.utils.save_video_from_rgb", fake_save_video)
    monkeypatch.setattr("timeflowcodec.decoder.save_video_from_rgb", fake_save_video)

    input_path = tmp_path / "dummy.mp4"
    output_tfc_no_mc = tmp_path / "video_no_mc.tfc"
    output_tfc_mc = tmp_path / "video_mc.tfc"
    recon_video = tmp_path / "recon.mp4"

    # 1. Encode WITHOUT motion compensation (V3)
    encode_video_to_tfc(
        str(input_path),
        str(output_tfc_no_mc),
        container_version=3,
        tiling=8,
        tau=0.05,
        preset="custom",
        motion_comp=False,
    )

    # 2. Encode WITH motion compensation (V4 BMMC)
    encode_video_to_tfc(
        str(input_path),
        str(output_tfc_mc),
        container_version=4,
        tiling=8,
        tau=0.05,
        preset="custom",
        motion_comp=True,
    )

    # Compare file sizes: BMMC should compress moving sequences much better!
    size_no_mc = output_tfc_no_mc.stat().st_size
    size_mc = output_tfc_mc.stat().st_size
    print(f"File Size without MC: {size_no_mc} bytes")
    print(f"File Size with MC: {size_mc} bytes")

    # 3. Decode BMMC file and verify roundtrip correctness
    decode_tfc_to_video(str(output_tfc_mc), str(recon_video), stream_output=False)

    assert "frames" in captured
    recon = captured["frames"]
    assert recon.shape == frames.shape

    overall_mse = mse(frames, recon)
    print(f"BMMC Reconstruction MSE: {overall_mse}")
    assert overall_mse < 1.0  # High fidelity round-trip
