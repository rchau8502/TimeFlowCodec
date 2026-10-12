"""Size controls must affect encoded data, without silently averaging pixels."""

import numpy as np
import pytest

from timeflowcodec.decoder import decode_tfc_to_video
from timeflowcodec.encoder import encode_video_to_tfc
from timeflowcodec.format import (
    build_plane_payload_v3,
    build_plane_payload_v4,
    parse_plane_payload_v3,
    parse_plane_payload_v4,
)


@pytest.mark.parametrize("builder,parser", [
    (build_plane_payload_v3, parse_plane_payload_v3),
    (build_plane_payload_v4, parse_plane_payload_v4),
])
def test_integer_residuals_are_not_expanded_to_fractional_16bit_scale(builder, parser):
    residuals = np.array([[0, 2, -3, 6]], dtype=np.int16)
    segment = dict(length=4, modes=np.array([2], dtype=np.uint8), tfc_params={},
                   raw_predictors={0: dict(a=100, b=0)}, raw_residuals=residuals)
    decoded = parser(builder([segment], 0), 0)[0]
    np.testing.assert_array_equal(decoded["raw_residuals"], residuals)


@pytest.mark.parametrize("version", [3, 4])
def test_residual_quantization_reduces_size_with_bounded_error(tmp_path, version):
    frames = np.random.default_rng(3).integers(40, 200, (16, 24, 32, 3), dtype=np.uint8)
    sizes = []
    for step in (1, 8):
        target = tmp_path / f"q{step}.tfc"
        encode_video_to_tfc(frames, str(target), container_version=version,
                            tau=0, tiling=1, payload_comp_type=3, q_step=step)
        decoded = decode_tfc_to_video(str(target), None)
        assert np.max(np.abs(frames.astype(int) - decoded.astype(int))) <= step / 2 + 1
        sizes.append(target.stat().st_size)
    assert sizes[1] < sizes[0] * .8


@pytest.mark.parametrize("preset", ["anime", "lownoise", "general"])
def test_presets_do_not_silently_average_fine_spatial_detail(tmp_path, preset):
    frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    frames[:, :, ::2] = 255
    target = tmp_path / "detail.tfc"
    encode_video_to_tfc(frames, str(target), preset=preset)
    actual = decode_tfc_to_video(str(target), None)
    assert np.max(np.abs(frames.astype(int) - actual.astype(int))) <= 2


def test_unreachable_size_target_preserves_existing_destination(tmp_path):
    from test_media import make_video

    source, target = tmp_path / "source.mp4", tmp_path / "target.tfc"
    make_video(source)
    target.write_bytes(b"existing result")
    with pytest.raises(ValueError, match="size target"):
        encode_video_to_tfc(str(source), str(target), container_version=5,
                            max_output_ratio=.001, preset="screen")
    assert target.read_bytes() == b"existing result"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source.mp4", "target.tfc"]


def test_unreachable_bitrate_preserves_destination(tmp_path):
    from test_media import make_video

    source, target = tmp_path / "source.mp4", tmp_path / "target.tfc"
    make_video(source)
    target.write_bytes(b"existing result")
    with pytest.raises(ValueError, match="bitrate target"):
        encode_video_to_tfc(str(source), str(target), container_version=5,
                            target_kbps=.01, preset="screen")
    assert target.read_bytes() == b"existing result"


def test_size_target_can_succeed_without_changing_frames(tmp_path):
    import imageio.v2 as imageio

    frames = np.full((24, 32, 32, 3), 80, dtype=np.uint8)
    source, target = tmp_path / "static.mp4", tmp_path / "static.tfc"
    with imageio.get_writer(source, fps=24, codec="libx264rgb", pixelformat="rgb24",
                            ffmpeg_params=["-crf", "0"]) as writer:
        for frame in frames:
            writer.append_data(frame)
    encode_video_to_tfc(str(source), str(target), container_version=5,
                        preset="screen", max_output_ratio=.4)
    assert target.stat().st_size <= source.stat().st_size * .4
    np.testing.assert_array_equal(decode_tfc_to_video(str(target), None), frames)


@pytest.mark.parametrize("options", [
    {"q_step": 0}, {"q_step": True}, {"target_kbps": float("nan")},
    {"target_kbps": -1}, {"max_output_ratio": float("inf")}, {"max_output_ratio": 0},
])
def test_invalid_size_controls_rejected_before_source_access(tmp_path, options):
    target = tmp_path / "existing.tfc"
    target.write_bytes(b"existing")
    with pytest.raises(ValueError):
        encode_video_to_tfc("missing.mp4", str(target), container_version=5, **options)
    assert target.read_bytes() == b"existing"


def test_cli_passes_quantization_and_actual_bitrate_target(monkeypatch, tmp_path):
    from timeflowcodec.cli import main
    from timeflowcodec.rate_control import crf_to_parameters

    calls = []
    monkeypatch.setattr("timeflowcodec.cli.encode_video_to_tfc", lambda *a, **kw: calls.append(kw))
    main(["encode", "source.mp4", str(tmp_path / "crf.tfc"), "--crf", "35"])
    assert calls[-1]["q_step"] == crf_to_parameters(35)["q_step"]
    main(["encode", "source.mp4", str(tmp_path / "rate.tfc"), "--bitrate", "500k"])
    assert calls[-1]["target_kbps"] == 500
    main(["encode", "source.mp4", str(tmp_path / "size.tfc"), "--max-output-ratio", ".4"])
    assert calls[-1]["max_output_ratio"] == .4


def test_gui_defaults_to_size_check_and_reports_measured_expansion(tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    import gui

    app = QApplication.instance() or QApplication([])
    window = gui.MainWindow()
    try:
        assert window.size_target_combo.currentData() == 1.0
        assert window.preset_combo.currentText() == "screen"
        source, target = tmp_path / "source.mp4", tmp_path / "target.tfc"
        source.write_bytes(b"s" * 100)
        target.write_bytes(b"t" * 200)
        message = gui._compression_result_message(str(source), str(target))
        assert "larger" in message and "100.0%" in message
        target.write_bytes(b"t" * 40)
        assert "60.0% smaller" in gui._compression_result_message(str(source), str(target))
    finally:
        window.close()
        app.processEvents()
