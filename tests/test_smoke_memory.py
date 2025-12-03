from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from timeflowcodec.encoder import encode_video_to_tfc
from timeflowcodec.decoder import decode_tfc_to_video


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_smoke_memory(tmp_path):
    clip = tmp_path / "clip.mp4"
    out_tfc = tmp_path / "out.tfc"
    recon = tmp_path / "recon.mp4"

    subprocess.run([
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=10",
        "-t",
        "1",
        str(clip),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    encode_video_to_tfc(str(clip), str(out_tfc), max_frames=20, payload_comp_type=0, tiling=16)
    decode_tfc_to_video(str(out_tfc), str(recon), stream_output=False)

    assert out_tfc.exists()
    assert recon.exists()
