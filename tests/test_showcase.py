import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def showcase():
    source = Path(__file__).parents[1] / "scripts/create_showcase.py"
    spec = importlib.util.spec_from_file_location("create_showcase", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_scene_is_deterministic_and_linear_inside_native_windows(showcase):
    frames = np.array([showcase.original_frame("linear_color", t, 160, 160) for t in range(32)])
    assert frames.dtype == np.uint8
    assert frames.shape == (32, 160, 160, 3)
    np.testing.assert_array_equal(frames[24], showcase.original_frame("linear_color", 24, 160, 160))
    for window in (frames[:16], frames[16:]):
        changes = np.diff(window.astype(np.int16), axis=0)
        np.testing.assert_array_equal(changes, np.broadcast_to(changes[0], changes.shape))


def test_original_hero_survives_actual_native_encode_exactly(showcase, tmp_path):
    from timeflowcodec.decoder import decode_tfc_to_video
    from timeflowcodec.encoder import encode_video_to_tfc

    frames = np.array([showcase.original_frame("linear_color", t, 160, 160) for t in range(16)])
    destination = tmp_path / "hero.tfc"
    encode_video_to_tfc(frames, str(destination), preset="screen", tau=0)
    result = showcase.measure_rgb(frames, decode_tfc_to_video(str(destination), None))
    assert result == dict(exact_rgb=True, max_rgb_error=0, rgb_psnr_db=None)


def test_measurement_rejects_truncated_and_empty_sequences(showcase):
    reference = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        showcase.measure_rgb(reference, reference[:2])
    with pytest.raises(ValueError):
        showcase.measure_rgb(reference[:0], reference[:0])


def test_single_pixel_error_cannot_be_reported_as_exact(showcase):
    reference = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    decoded = reference.copy()
    decoded[1, 3, 4, 2] = 1
    result = showcase.measure_rgb(reference, decoded)
    assert result["exact_rgb"] is False
    assert result["max_rgb_error"] == 1
    assert np.isfinite(result["rgb_psnr_db"])
