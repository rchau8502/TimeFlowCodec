"""Tests for live screen capture pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from timeflowcodec.capture import (
    capture_screen_to_tfc,
    get_capture_command,
)
from timeflowcodec.decoder import decode_tfc_to_video


def test_get_capture_command_structure():
    """Verify get_capture_command builds valid FFmpeg arguments."""
    cmd, dev = get_capture_command(fps=30, scale_width=1280)
    assert len(cmd) > 5
    assert "-pix_fmt" in cmd
    assert "rgb24" in cmd
    assert "pipe:1" in cmd


def test_mocked_capture_screen_to_tfc(tmp_path):
    """Verify live screen capture creates valid decodable TFC5 container."""
    output_tfc = tmp_path / "live_test.tfc"
    width, height = 320, 240
    fps = 30
    chunk_frames = 10
    total_to_generate = 20

    # Generate 20 frames of synthetic RGB data
    raw_bytes = bytearray()
    for t in range(total_to_generate):
        frame = np.full((height, width, 3), (t * 10) % 256, dtype=np.uint8)
        raw_bytes.extend(frame.tobytes())

    class MockStdout:
        def __init__(self, data: bytes, frame_sz: int):
            self.data = data
            self.frame_sz = frame_sz
            self.offset = 0

        def read(self, n: int) -> bytes:
            if self.offset >= len(self.data):
                return b""
            chunk = self.data[self.offset : self.offset + n]
            self.offset += len(chunk)
            return chunk

    mock_proc = MagicMock()
    mock_proc.stdout = MockStdout(bytes(raw_bytes), width * height * 3)
    mock_proc.terminate = MagicMock()
    mock_proc.wait = MagicMock()

    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("timeflowcodec.capture.probe_screen_dimensions", return_value=(width, height)):
        frames_captured = capture_screen_to_tfc(
            output_path=output_tfc,
            fps=fps,
            scale_width=width,
            duration=None,
            chunk_frames=chunk_frames,
        )

    assert frames_captured == total_to_generate
    assert output_tfc.exists()

    # Verify that the captured output decodes cleanly
    decoded = decode_tfc_to_video(str(output_tfc), None)
    assert decoded is not None
    assert decoded.shape == (total_to_generate, height, width, 3)
