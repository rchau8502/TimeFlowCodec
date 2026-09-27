import numpy as np
import pytest

from timeflowcodec.decoder import decode_tfc_to_video
from timeflowcodec.encoder import encode_video_to_tfc


@pytest.mark.parametrize("macbook", [False, True])
def test_screen_preserves_colored_single_pixel_lines(monkeypatch, tmp_path, macbook):
    frames = np.zeros((4, 17, 19, 3), dtype=np.uint8)
    frames[:, :, ::2, 0] = 255
    frames[:, ::2, :, 1] = 173
    frames[:, 7, :, 2] = 211
    monkeypatch.setattr("timeflowcodec.encoder.imageio.get_reader", lambda _: iter(frames))
    path = tmp_path / "screen.tfc"
    encode_video_to_tfc("unused", str(path), preset="screen", macbook_profile=macbook, payload_comp_type=0)
    decoded = decode_tfc_to_video(str(path), None)
    np.testing.assert_array_equal(decoded, frames)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_decode_arrays_without_video_export(monkeypatch, tmp_path, version):
    frames = np.full((3, 8, 8, 3), 80, dtype=np.uint8)
    monkeypatch.setattr("timeflowcodec.encoder.imageio.get_reader", lambda _: iter(frames))
    path = tmp_path / "sample.tfc"
    encode_video_to_tfc("unused", str(path), container_version=version, payload_comp_type=0)
    np.testing.assert_array_equal(decode_tfc_to_video(str(path), None), frames)


def test_ssim_identical_and_brightness_shift():
    from timeflowcodec.bench import compute_metrics_numpy

    frames = np.full((2, 12, 12, 3), 80, dtype=np.uint8)
    psnr, ssim = compute_metrics_numpy(frames, frames)
    assert psnr == float("inf")
    assert ssim == pytest.approx(1.0)
    _, shifted = compute_metrics_numpy(frames, frames + 20)
    expected = (2 * 80 * 100 + (0.01 * 255) ** 2) / (80**2 + 100**2 + (0.01 * 255) ** 2)
    assert shifted == pytest.approx(expected)


def test_metrics_reject_misaligned_frames():
    from timeflowcodec.bench import compute_metrics_numpy

    frames = np.zeros((2, 12, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        compute_metrics_numpy(frames, frames[:1])


def test_array_output_rejects_streaming():
    with pytest.raises(ValueError, match="stream_output"):
        decode_tfc_to_video("unused", None, stream_output=True)


def test_benchmark_measures_pixels_without_export(tmp_path, monkeypatch):
    import imageio.v2 as imageio

    from timeflowcodec.bench import bench_tfc

    clip = tmp_path / "source.mp4"
    frames = np.full((3, 16, 16, 3), 80, dtype=np.uint8)
    with imageio.get_writer(
        str(clip), fps=12, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    def no_export(*args, **kwargs):
        pytest.fail("Benchmark must not export decoded video")

    monkeypatch.setattr("timeflowcodec.decoder.save_video_from_rgb", no_export)
    _, result = bench_tfc(clip, tmp_path, None, 0.001, 0, "screen")
    assert result.psnr == float("inf")
    assert result.ssim == pytest.approx(1.0)


def test_screen_cli_and_gui(monkeypatch, tmp_path):
    from gui import _preset_ui_defaults
    from timeflowcodec.cli import compress_main, main

    calls = []
    monkeypatch.setattr("timeflowcodec.cli.encode_video_to_tfc", lambda *a, **kw: calls.append(kw))
    main(["encode", "in.mp4", str(tmp_path / "out.tfc"), "--preset", "screen"])
    compress_main(["in.mp4", str(tmp_path / "out.tfc"), "--preset", "screen"])
    assert all(call["preset"] == "screen" for call in calls)
    defaults = _preset_ui_defaults("screen")
    assert defaults["colorspace_index"] == 0
    assert defaults["tau"] == 0.0001


def test_report_uses_valid_json_for_perfect_reconstruction(tmp_path):
    import json

    from timeflowcodec.bench import BenchResult, write_results

    result = BenchResult("tfc", "screen", "-", "test.mp4", 1, 1, 3, 3, 120, float("inf"), 1.0)
    write_results(tmp_path, [result], {}, [])

    def invalid_constant(value):
        pytest.fail(f"Invalid JSON numeric constant: {value}")

    data = json.loads((tmp_path / "results.json").read_text(), parse_constant=invalid_constant)
    assert data["results"][0]["lossless_reconstruction"] is True


def test_benchmark_respects_codec_selection(monkeypatch, tmp_path):
    from timeflowcodec import bench

    (tmp_path / "input.mp4").touch()

    def unwanted_tfc(*args, **kwargs):
        pytest.fail("TFC was not selected")

    monkeypatch.setattr(bench, "bench_tfc", unwanted_tfc)
    monkeypatch.setattr(bench, "bench_ffmpeg", lambda *args: None)
    monkeypatch.setattr(bench, "collect_env", lambda: {})
    monkeypatch.setattr(bench, "write_results", lambda *args: None)
    bench.main(["--clips", str(tmp_path), "--out", str(tmp_path / "out"), "--codecs", "x264"])
