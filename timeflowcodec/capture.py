"""Live screen capture streaming pipeline for TimeFlowCodec.

Captures desktop frames via native OS capture pipelines (macOS AVFoundation / ScreenCaptureKit,
Linux x11grab, Windows gdigrab) and streams them into chunked v5 TFC containers in real-time.
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from .chunked import RECORD, _encode_rgb_chunk, _write_record
from .constants import COMP_ZSTD


def get_capture_command(
    fps: int = 30,
    scale_width: int | None = 1280,
    device: str | None = None,
) -> tuple[list[str], str]:
    """Return the platform-specific FFmpeg command and detected screen input."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    system = platform.system()

    cmd = [ffmpeg_exe, "-hide_banner", "-loglevel", "warning"]

    if system == "Darwin":
        # macOS AVFoundation (ScreenCaptureKit display capture)
        screen_dev = device if device is not None else "2:none"
        cmd.extend([
            "-f", "avfoundation",
            "-capture_cursor", "1",
            "-framerate", str(fps),
            "-i", screen_dev,
        ])
    elif system == "Linux":
        screen_dev = device if device is not None else os.environ.get("DISPLAY", ":0.0")
        cmd.extend([
            "-f", "x11grab",
            "-framerate", str(fps),
            "-i", screen_dev,
        ])
    elif system == "Windows":
        screen_dev = device if device is not None else "desktop"
        cmd.extend([
            "-f", "gdigrab",
            "-framerate", str(fps),
            "-i", screen_dev,
        ])
    else:
        raise RuntimeError(f"Unsupported operating system for native screen capture: {system}")

    # Video filters: scaling if requested (ensuring even dimensions for video codecs)
    vf_filters = []
    if scale_width is not None and scale_width > 0:
        vf_filters.append(f"scale={scale_width}:-2")

    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    # Output raw RGB24 to stdout pipe
    cmd.extend([
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ])

    return cmd, screen_dev


def probe_screen_dimensions(
    fps: int = 30,
    scale_width: int | None = 1280,
    device: str | None = None,
) -> tuple[int, int]:
    """Probe the captured frame dimensions by running FFmpeg for 1 frame."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    system = platform.system()

    probe_cmd = [ffmpeg_exe, "-hide_banner"]
    if system == "Darwin":
        screen_dev = device if device is not None else "2:none"
        probe_cmd.extend(["-f", "avfoundation", "-i", screen_dev])
    elif system == "Linux":
        screen_dev = device if device is not None else os.environ.get("DISPLAY", ":0.0")
        probe_cmd.extend(["-f", "x11grab", "-i", screen_dev])
    elif system == "Windows":
        screen_dev = device if device is not None else "desktop"
        probe_cmd.extend(["-f", "gdigrab", "-i", screen_dev])
    else:
        raise RuntimeError(f"Unsupported system: {system}")

    probe_cmd.extend(["-vframes", "1", "-f", "null", "-"])

    res = subprocess.run(probe_cmd, capture_output=True, text=True)
    # Search for resolution pattern in stderr like "3420x2224" or "1920x1080"
    import re
    m = re.search(r",\s*(\d{3,5})x(\d{3,5})", res.stderr)
    if not m:
        # Fallback default if detection fails
        if scale_width:
            return scale_width, int(scale_width * 9 / 16)
        return 1920, 1080

    orig_w, orig_h = int(m.group(1)), int(m.group(2))
    if scale_width and scale_width < orig_w:
        calc_h = int((orig_h * scale_width) / orig_w)
        if calc_h % 2 != 0:
            calc_h += 1
        return scale_width, calc_h

    return orig_w, orig_h


def capture_screen_to_tfc(
    output_path: str | Path,
    fps: int = 30,
    scale_width: int | None = 1280,
    duration: float | None = None,
    chunk_frames: int = 30,
    preset: str = "screen",
    device: str | None = None,
) -> int:
    """Capture live screen frames and encode directly into a v5 TFC streaming container."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = probe_screen_dimensions(fps=fps, scale_width=scale_width, device=device)
    frame_bytes = width * height * 3

    cmd, dev_name = get_capture_command(fps=fps, scale_width=scale_width, device=device)

    print("TimeFlowCodec Live Screen Capture")
    print(f"Device: {dev_name} | Resolution: {width}x{height} | Target FPS: {fps}")
    print(f"Output: {output_path} | Preset: {preset} | Chunk: {chunk_frames} frames")
    if duration:
        print(f"Duration: {duration}s (Ctrl+C to stop early)")
    else:
        print("Streaming live... Press Ctrl+C to stop and finalize .tfc container.")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_bytes * 4,
    )

    stop_requested = False

    def sig_handler(sig, frame):
        nonlocal stop_requested
        stop_requested = True

    prev_sigint = signal.signal(signal.SIGINT, sig_handler)

    total_frames = 0
    t_start = time.perf_counter()
    chunk_buffer: list[np.ndarray] = []

    # Container options
    opts = {
        "preset": preset,
        "tau": 0.1 if preset != "screen" else 0.05,
        "slope_threshold": 1e-3,
        "payload_comp_type": COMP_ZSTD,
        "container_version": 3,
    }

    import json

    metadata = {
        "version": 5,
        "width": width,
        "height": height,
        "num_frames": 0,  # Updated on completion
        "fps_num": fps,
        "fps_den": 1,
        "chunk_frames": chunk_frames,
        "has_audio": False,
        "audio_codec": None,
    }

    with tempfile.TemporaryDirectory() as work_dir:
        chunk_idx = 0
        chunks_file_path = Path(work_dir) / "live_chunks.bin"

        with open(chunks_file_path, "wb") as chunks_out:
            try:
                while not stop_requested:
                    if duration and (time.perf_counter() - t_start) >= duration:
                        break

                    raw = proc.stdout.read(frame_bytes)
                    if len(raw) < frame_bytes:
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                    chunk_buffer.append(frame)
                    total_frames += 1

                    if len(chunk_buffer) >= chunk_frames:
                        chunk_path = Path(work_dir) / f"live_{chunk_idx}.tfc"
                        _encode_rgb_chunk(np.asarray(chunk_buffer), chunk_path, opts)
                        _write_record(chunks_out, b"VIDS", chunk_path)
                        chunk_buffer.clear()
                        chunk_idx += 1
                        elapsed = time.perf_counter() - t_start
                        sys.stdout.write(f"\rCaptured {total_frames} frames ({total_frames / elapsed:.1f} FPS) ...")
                        sys.stdout.flush()

                # Flush remaining frames
                if chunk_buffer:
                    chunk_path = Path(work_dir) / f"live_{chunk_idx}.tfc"
                    _encode_rgb_chunk(np.asarray(chunk_buffer), chunk_path, opts)
                    _write_record(chunks_out, b"VIDS", chunk_path)
                    chunk_buffer.clear()

            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                signal.signal(signal.SIGINT, prev_sigint)

        # Finalize output file with exact metadata and concatenated records
        metadata["num_frames"] = total_frames
        meta_json = json.dumps(metadata).encode("utf-8")

        with open(output_path, "wb") as out_f:
            out_f.write(b"TFC5")
            out_f.write(len(meta_json).to_bytes(4, "little"))
            out_f.write(meta_json)

            # Stream chunks
            if chunks_file_path.exists():
                with open(chunks_file_path, "rb") as cf:
                    while block := cf.read(1024 * 1024):
                        out_f.write(block)

            # Terminal DONE record
            out_f.write(RECORD.pack(b"DONE", 0, 0))

    elapsed_total = time.perf_counter() - t_start
    avg_fps = (total_frames / elapsed_total) if elapsed_total > 0 else 0
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print()
    print("Capture complete!")
    print(f"Total Frames: {total_frames} | Time: {elapsed_total:.2f}s | Avg FPS: {avg_fps:.1f}")
    print(f"Output: {output_path} ({size_mb:.2f} MB)")
    return total_frames


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TimeFlowCodec Live Screen Capture")
    parser.add_argument("output", help="Output .tfc file path")
    parser.add_argument("--fps", type=int, default=30, help="Target capture frame rate (default: 30)")
    parser.add_argument("--scale", type=int, default=1280, help="Horizontal resolution scale (default: 1280, 0=native)")
    parser.add_argument("--duration", type=float, default=None, help="Capture duration in seconds (default: unlimited)")
    parser.add_argument("--chunk-frames", type=int, default=30, help="Frames per streaming chunk (default: 30)")
    parser.add_argument("--preset", type=str, default="screen", choices=["screen", "anime", "general"])
    parser.add_argument("--device", type=str, default=None, help="Custom FFmpeg capture device identifier")

    args = parser.parse_args(argv)
    scale_w = None if args.scale == 0 else args.scale
    capture_screen_to_tfc(
        output_path=args.output,
        fps=args.fps,
        scale_width=scale_w,
        duration=args.duration,
        chunk_frames=args.chunk_frames,
        preset=args.preset,
        device=args.device,
    )


if __name__ == "__main__":
    main()
