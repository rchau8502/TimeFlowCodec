from __future__ import annotations

import numpy as np

from timeflowcodec import COMP_ZSTD, MODE_FB_RAW, MODE_TFC_CONST, MODE_TFC_LINEAR
from timeflowcodec.format import build_plane_payload_v3, parse_plane_payload_v3


def test_v3_payload_roundtrip_with_residual_raw():
    modes = np.array([MODE_FB_RAW, MODE_TFC_CONST, MODE_TFC_LINEAR], dtype=np.uint8)
    seg = {
        "length": 4,
        "modes": modes,
        "tfc_params": {
            1: {"mode": MODE_TFC_CONST, "a": 90.0},
            2: {"mode": MODE_TFC_LINEAR, "a": 30.0, "b": 2.0},
        },
        "raw_predictors": {
            0: {"a": 120.0, "b": -1.5},
        },
        "raw_residuals": np.array([[2, -1, 3, 0]], dtype=np.int16),
    }

    payload = build_plane_payload_v3([seg], COMP_ZSTD)
    parsed = parse_plane_payload_v3(payload, COMP_ZSTD)
    assert len(parsed) == 1
    out = parsed[0]
    assert out["length"] == 4
    np.testing.assert_array_equal(out["modes"], modes)
    assert out["tfc_params"][1]["mode"] == MODE_TFC_CONST
    assert out["tfc_params"][2]["mode"] == MODE_TFC_LINEAR
    assert out["raw_predictors"][0]["a"] == 120.0
    assert abs(out["raw_predictors"][0]["b"] + 1.5) < 0.05
    np.testing.assert_allclose(out["raw_residuals"], np.array([[2, -1, 3, 0]], dtype=np.float32), atol=0.6)
