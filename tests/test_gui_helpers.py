from __future__ import annotations

from gui import _is_apple_silicon, _preset_ui_defaults, _suggest_output_path


def test_is_apple_silicon_detection():
    assert _is_apple_silicon("Darwin", "arm64") is True
    assert _is_apple_silicon("Darwin", "x86_64") is False
    assert _is_apple_silicon("Linux", "aarch64") is False


def test_suggest_output_path_swaps_suffix():
    assert _suggest_output_path("/tmp/input.mov", ".tfc") == "/tmp/input.tfc"
    assert _suggest_output_path("/tmp/input.tfc", ".mp4") == "/tmp/input.mp4"


def test_preset_ui_defaults_for_anime():
    defaults = _preset_ui_defaults("anime")
    assert defaults["tau"] == 0.035
    assert defaults["scene_cut"] == "auto"
    assert defaults["matrix_mode"] is False
    assert defaults["compression_index"] == 3
    assert defaults["colorspace_index"] == 2
