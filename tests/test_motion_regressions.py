import numpy as np
import pytest

from timeflowcodec.decoder import decode_tfc_to_video
from timeflowcodec.encoder import encode_video_to_tfc


@pytest.mark.parametrize("code", range(9))
def test_motion_reference_all_directions(code):
    from timeflowcodec.utils import motion_reference

    grid = np.arange(20).reshape(4, 5) * 7
    codes = np.full(grid.shape, code, dtype=np.uint8)
    expected = np.empty_like(grid)
    for y in range(4):
        for x in range(5):
            expected[y, x] = grid[max(0, min(3, y + code // 3 - 1)), max(0, min(4, x + code % 3 - 1))]
    np.testing.assert_array_equal(motion_reference(grid, codes), expected)


@pytest.mark.parametrize("budget", [None, 0])
def test_motion_preserves_signed_warped_samples(monkeypatch, tmp_path, budget):
    frames = np.random.default_rng(4).integers(0, 256, (5, 8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr("timeflowcodec.encoder.imageio.get_reader", lambda _: iter(frames))
    path = tmp_path / "motion.tfc"
    encode_video_to_tfc("synthetic", str(path), tiling=1, tau=0, motion_comp=True,
                        payload_comp_type=0, max_ram_mb=budget)
    actual = decode_tfc_to_video(str(path), None)
    assert np.max(np.abs(actual.astype(int) - frames.astype(int))) <= 3


@pytest.mark.parametrize("code", range(9))
def test_decoder_motion_against_independent_reference(tmp_path, code):
    """A hand-specified constant model avoids encoder/decoder bug cancellation."""
    import struct

    from timeflowcodec.constants import MODE_TFC_CONST
    from timeflowcodec.format import build_plane_payload_v4, write_header

    grid = np.arange(20, dtype=np.uint8).reshape(4, 5) * 7
    segment = {
        "length": 2,
        "modes": np.full(20, MODE_TFC_CONST, dtype=np.uint8),
        "tfc_params": {i: {"a": int(value)} for i, value in enumerate(grid.flat)},
        "mvs": np.full((1, 20), code, dtype=np.uint8),
    }
    payload = build_plane_payload_v4([segment], 0)
    path = tmp_path / "reference.tfc"
    with path.open("wb") as output:
        write_header(output, dict(version=4, width=5, height=4, num_frames=2, tiling=1))
        for _ in range(3):
            output.write(struct.pack("<I", len(payload)))
            output.write(payload)
    actual = decode_tfc_to_video(str(path), None)
    expected = np.empty_like(grid)
    for y in range(4):
        for x in range(5):
            expected[y, x] = grid[max(0, min(3, y + code // 3 - 1)), max(0, min(4, x + code % 3 - 1))]
    np.testing.assert_array_equal(actual[0], np.repeat(grid[..., None], 3, axis=2))
    np.testing.assert_array_equal(actual[1], np.repeat(expected[..., None], 3, axis=2))
