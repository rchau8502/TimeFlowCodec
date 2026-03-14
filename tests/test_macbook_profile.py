from __future__ import annotations

from timeflowcodec.encoder import _apply_macbook_profile_defaults


def test_macbook_profile_applies_safe_defaults():
    payload_comp_type, tiling, max_ram_mb, dtype, scene_cut = (
        _apply_macbook_profile_defaults(
            payload_comp_type=2,
            tiling=None,
            max_ram_mb=None,
            dtype="float16",
            scene_cut="off",
        )
    )

    assert payload_comp_type == 1
    assert tiling == 16
    assert max_ram_mb == 1536
    assert dtype == "uint8"
    assert scene_cut == "auto"


def test_macbook_profile_preserves_explicit_values_when_safe():
    payload_comp_type, tiling, max_ram_mb, dtype, scene_cut = (
        _apply_macbook_profile_defaults(
            payload_comp_type=1,
            tiling=32,
            max_ram_mb=1024,
            dtype="uint16",
            scene_cut="auto",
        )
    )

    assert payload_comp_type == 1
    assert tiling == 32
    assert max_ram_mb == 1024
    assert dtype == "uint16"
    assert scene_cut == "auto"
