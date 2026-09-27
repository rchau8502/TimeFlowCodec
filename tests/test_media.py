import subprocess
from fractions import Fraction

import imageio_ffmpeg
import pytest

from timeflowcodec import media


def make_video(path, rate="24", audio=False, extra=()):
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=32x24:rate={rate}:duration=0.5",
    ]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", "-c:a", "aac"]
    subprocess.run(command + ["-c:v", "libx264", *extra, str(path)], check=True)


@pytest.mark.parametrize("rate", ["24", "25", "30000/1001", "60"])
def test_exact_rate(tmp_path, rate):
    source = tmp_path / "source.mp4"
    make_video(source, rate)
    info = media.probe_media(source)
    assert Fraction(info["fps_num"], info["fps_den"]) == Fraction(rate)
    assert (info["width"], info["height"]) == (32, 24)
    assert not info["has_audio"]


def test_audio_roundtrip(tmp_path):
    source, audio, video, output = [tmp_path / name for name in ["source.mp4", "audio.mka", "silent.mp4", "output.mp4"]]
    make_video(source, audio=True)
    make_video(video)
    assert media.probe_media(source)["has_audio"]
    assert media.extract_audio(source, audio)
    media.mux_audio(video, audio, output)
    assert media.probe_media(output)["has_audio"]
    assert not media.extract_audio(video, tmp_path / "absent.mka")


def test_rejects_vfr(tmp_path):
    source = tmp_path / "vfr.mp4"
    make_video(source, extra=["-vf", "setpts='if(lt(N,5),N,2*N)/(24*TB)'", "-fps_mode", "vfr"])
    with pytest.raises(ValueError, match="[Vv]ariable|[Ii]rregular"):
        media.probe_media(source)


def test_cancellation(tmp_path):
    class Cancelled(Exception):
        pass

    def cancel():
        raise Cancelled()

    with pytest.raises(Cancelled):
        media.probe_media(tmp_path / "unused", cancel_check=cancel)
