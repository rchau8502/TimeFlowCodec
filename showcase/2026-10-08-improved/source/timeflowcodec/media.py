"""Source timing and audio through the FFmpeg runtime shipped with imageio.

Probe every video timestamp to explicitly reject variable frame rate input.
The diagnostic pipe is drained incrementally, never retained for the full clip.
"""

import re
import subprocess
from collections import deque
from fractions import Fraction
from queue import Empty, Queue
from threading import Thread

import imageio_ffmpeg

from .atomic import atomic_destination
from .jobs import check_cancel


def _run(arguments, cancel_check=None, on_line=None):
    check_cancel(cancel_check)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-nostats", *arguments]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    lines = Queue(maxsize=128)

    # A bounded queue keeps verbose per-frame diagnostics from growing memory.
    # On cancellation the consumer drains it until the reader exits.
    def read_lines():
        try:
            for line in iter(process.stderr.readline, b""):
                lines.put(line.decode("utf-8", errors="replace"))
        finally:
            lines.put(None)

    reader = Thread(target=read_lines, daemon=True)
    reader.start()
    tail = deque(maxlen=20)
    try:
        while True:
            check_cancel(cancel_check)
            try:
                line = lines.get(timeout=0.05)
            except Empty:
                continue
            if line is None:
                break
            tail.append(line)
            if on_line is not None:
                on_line(line)
        if process.wait() != 0:
            raise ValueError("FFmpeg failed: " + "".join(tail)[-4000:])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        while reader.is_alive():
            try:
                lines.get(timeout=0.05)
            except Empty:
                pass
        reader.join()
        process.stderr.close()


def probe_media(source, cancel_check=None):
    info = {"has_audio": False, "num_frames": 0}
    time_base = rate = previous = first = step = None
    dimensions = None

    def consume(line):
        nonlocal time_base, rate, previous, first, step, dimensions
        if re.search(r"Stream #0:\d+.*Audio:", line):
            info["has_audio"] = True
        config = re.search(r"config in time_base: (\d+/\d+), frame_rate: (\d+/\d+)", line)
        if config:
            try:
                new_base, new_rate = map(Fraction, config.groups())
            except ZeroDivisionError as exc:
                raise ValueError("Source has no usable frame timing") from exc
            if time_base is not None and (new_base != time_base or new_rate != rate):
                raise ValueError("Variable source timing is not supported")
            time_base, rate = new_base, new_rate
        frame = re.search(r" n:\s*\d+ pts:\s*(-?\d+).*? s:(\d+)x(\d+)", line)
        if frame:
            pts, width, height = map(int, frame.groups())
            if dimensions is not None and dimensions != (width, height):
                raise ValueError("Variable video dimensions are not supported")
            dimensions = (width, height)
            if previous is not None:
                delta = pts - previous
                if delta <= 0 or (step is not None and delta != step):
                    raise ValueError(
                        "Variable or irregular frame rate is not supported; convert to constant frame rate first"
                    )
                step = delta
            else:
                first = pts
            previous = pts
            info["num_frames"] += 1

    _run(
        ["-i", str(source), "-map", "0:v:0", "-an", "-vf", "showinfo", "-fps_mode", "passthrough", "-f", "null", "-"],
        cancel_check,
        consume,
    )
    if not dimensions or time_base is None or rate is None or rate <= 0:
        raise ValueError("No usable video stream or frame timing")
    fps = 1 / (step * time_base) if step is not None else rate
    info.update(
        width=dimensions[0],
        height=dimensions[1],
        fps_num=fps.numerator,
        fps_den=fps.denominator,
        start_num=(first * time_base).numerator,
        start_den=(first * time_base).denominator,
    )
    return info


def extract_audio(source, destination, cancel_check=None):
    if not probe_media(source, cancel_check)["has_audio"]:
        return False
    with atomic_destination(destination) as temporary:
        _run(["-y", "-i", str(source), "-map", "0:a:0", "-vn", "-c:a", "copy", temporary], cancel_check)
    return True


def mux_audio(video, audio, destination, cancel_check=None):
    with atomic_destination(destination) as temporary:
        _run(
            [
                "-y",
                "-i",
                str(video),
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                temporary,
            ],
            cancel_check,
        )
