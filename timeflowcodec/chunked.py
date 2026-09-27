"""Version 5: bounded legacy chunks with exact timing and an optional audio track.

Layout: TFC5, uint32 JSON length, JSON, then (tag, uint64 length, uint32
CRC32) records. VIDS records each hold a complete v3/v4 container. A single
optional AUDI record holds a Matroska audio stream. DONE is empty and terminal.
No record may be interpreted before its length and checksum are validated.
"""

import json
import shutil
import struct
import tempfile
import zlib
from contextlib import closing, nullcontext
from fractions import Fraction
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .atomic import atomic_destination
from .jobs import check_cancel, report_progress
from .media import extract_audio, mux_audio, probe_media

MAGIC = b"TFC5"
MAX_METADATA = 16 * 1024
MAX_CHUNK_BYTES = 128 * 1024 * 1024
MAX_AUDIO_BYTES = 1024 * 1024 * 1024
CHUNK_RGB_BYTES = 16 * 1024 * 1024
RECORD = struct.Struct("<4sQI")


def read_metadata(stream):
    raw = stream.read(4)
    if len(raw) != 4:
        raise ValueError("Incomplete v5 metadata length")
    size = struct.unpack("<I", raw)[0]
    if not 0 < size <= MAX_METADATA:
        raise ValueError("Metadata size limit exceeded")
    try:
        metadata = json.loads(stream.read(size))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid v5 metadata") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Invalid v5 metadata object")
    for key in ("width", "height", "num_frames", "fps_num", "fps_den", "chunk_frames"):
        value = metadata.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Invalid v5 {key}")
    w, h, count = (metadata[k] for k in ("width", "height", "num_frames"))
    if w > 16384 or h > 16384 or w * h > 16_777_216 or count > 10_000_000:
        raise ValueError("Video dimensions or duration exceed supported limits")
    if metadata["chunk_frames"] * w * h * 3 > max(CHUNK_RGB_BYTES, w * h * 3):
        raise ValueError("Chunk RGB size limit exceeded")
    if not Fraction(1, 1000) <= Fraction(metadata["fps_num"], metadata["fps_den"]) <= 1000:
        raise ValueError("Unsupported frame rate")
    if type(metadata.get("has_audio")) is not bool:
        raise ValueError("Invalid audio metadata")
    metadata["version"] = 5
    return metadata


def _write_record(output, tag, source, cancel_check=None):
    size = Path(source).stat().st_size
    limit = MAX_AUDIO_BYTES if tag == b"AUDI" else MAX_CHUNK_BYTES
    if size > limit:
        raise ValueError("Record size limit exceeded")
    crc = 0
    with open(source, "rb") as data:
        while block := data.read(1024 * 1024):
            check_cancel(cancel_check)
            crc = zlib.crc32(block, crc)
        output.write(RECORD.pack(tag, size, crc))
        data.seek(0)
        while block := data.read(1024 * 1024):
            check_cancel(cancel_check)
            output.write(block)


def records(source, cancel_check=None, audio_path=None):
    """Yield validated record bytes, with at most one bounded record in memory."""
    with open(source, "rb") as stream:
        if stream.read(4) != MAGIC:
            raise ValueError("Invalid v5 magic")
        metadata = read_metadata(stream)
        seen_audio = False
        chunks = 0
        expected_chunks = (metadata["num_frames"] + metadata["chunk_frames"] - 1) // metadata["chunk_frames"]
        while True:
            check_cancel(cancel_check)
            raw = stream.read(RECORD.size)
            if len(raw) != RECORD.size:
                raise ValueError("Missing or incomplete record header")
            tag, size, crc = RECORD.unpack(raw)
            if tag == b"DONE":
                if size or crc or stream.read(1) or chunks != expected_chunks or seen_audio != metadata["has_audio"]:
                    raise ValueError("Invalid terminal record or record count")
                return
            if tag not in (b"VIDS", b"AUDI"):
                raise ValueError("Unknown v5 record")
            if tag == b"VIDS":
                chunks += 1
                if seen_audio or chunks > expected_chunks:
                    raise ValueError("Invalid video record order")
            else:
                if seen_audio or chunks != expected_chunks or not metadata["has_audio"]:
                    raise ValueError("Invalid audio record order")
                seen_audio = True
            limit = MAX_AUDIO_BYTES if tag == b"AUDI" else MAX_CHUNK_BYTES
            if not 0 < size <= limit:
                raise ValueError("Record size limit exceeded")
            if tag == b"AUDI":
                checksum = 0
                remaining = size
                with open(audio_path, "wb") if audio_path is not None else nullcontext() as audio:
                    while remaining:
                        check_cancel(cancel_check)
                        block = stream.read(min(remaining, 1024 * 1024))
                        if not block:
                            raise ValueError("Truncated audio record")
                        remaining -= len(block)
                        checksum = zlib.crc32(block, checksum)
                        if audio is not None:
                            audio.write(block)
                if checksum != crc:
                    raise ValueError("Audio checksum mismatch")
                yield tag, b""
            else:
                data = stream.read(size)
                if len(data) != size or zlib.crc32(data) != crc:
                    raise ValueError("Truncated record or checksum mismatch")
                yield tag, data


def encode(source, destination, options, cancel_check=None, progress_callback=None):
    from .encoder import encode_video_to_tfc

    check_cancel(cancel_check)
    window = options.pop("window", None)
    if window is None:
        window = 16
    if type(window) is not int or window <= 0:
        raise ValueError("window must be a positive integer")
    report_progress(progress_callback, "Checking source timing", 0, 0)
    metadata = probe_media(source, cancel_check)
    requested = options.pop("max_frames", None)
    if requested is not None:
        if requested <= 0:
            raise ValueError("max_frames must be positive")
        metadata["num_frames"] = min(metadata["num_frames"], requested)
    frame_bytes = metadata["width"] * metadata["height"] * 3
    metadata["chunk_frames"] = min(window, max(1, CHUNK_RGB_BYTES // frame_bytes))
    # Truncated encodes currently omit audio to avoid silently retaining a full
    # source soundtrack with a shorter video. Full encodes preserve it.
    if requested is not None and metadata["has_audio"]:
        raise ValueError("Audio with max_frames is not supported; trim the source first")
    metadata["version"] = 5
    encoded_metadata = json.dumps(metadata, separators=(",", ":")).encode()
    options.pop("container_version", None)
    options["container_version"] = 4 if options.get("motion_comp") else 3
    with tempfile.TemporaryDirectory(prefix="timeflow-encode-") as work, atomic_destination(destination) as temporary:
        chunk_path, audio_path = Path(work) / "chunk.tfc", Path(work) / "audio.mka"
        if metadata["has_audio"]:
            extract_audio(source, audio_path, cancel_check)
        with open(temporary, "wb") as output, closing(imageio.get_reader(source)) as reader:
            output.write(MAGIC + struct.pack("<I", len(encoded_metadata)) + encoded_metadata)
            frames = iter(reader)
            completed = 0
            while completed < metadata["num_frames"]:
                check_cancel(cancel_check)
                count = min(metadata["chunk_frames"], metadata["num_frames"] - completed)
                chunk = []
                for _ in range(count):
                    check_cancel(cancel_check)
                    try:
                        frame = next(frames)
                    except StopIteration as exc:
                        raise ValueError("Source ended before its probed frame count") from exc
                    if frame.shape != (metadata["height"], metadata["width"], 3):
                        raise ValueError("Source frame dimensions changed")
                    chunk.append(frame)
                encode_video_to_tfc(np.asarray(chunk), str(chunk_path), **options, cancel_check=cancel_check)
                _write_record(output, b"VIDS", chunk_path, cancel_check)
                completed += count
                report_progress(progress_callback, "Encoding frames", completed, metadata["num_frames"])
            if metadata["has_audio"]:
                _write_record(output, b"AUDI", audio_path, cancel_check)
            output.write(RECORD.pack(b"DONE", 0, 0))
            check_cancel(cancel_check)


def iter_frames(source, cancel_check=None, progress_callback=None, audio_path=None, work_dir=None):
    from .decoder import decode_tfc_to_video
    from .format import read_header

    with open(source, "rb") as stream:
        metadata = read_header(stream)
    with tempfile.TemporaryDirectory(prefix="timeflow-decode-", dir=work_dir) as work:
        chunk_path = Path(work) / "chunk.tfc"
        completed = 0
        for tag, data in records(source, cancel_check, audio_path):
            if tag == b"AUDI":
                continue
            chunk_path.write_bytes(data)
            with chunk_path.open("rb") as stream:
                header = read_header(stream)
            count = min(metadata["chunk_frames"], metadata["num_frames"] - completed)
            if (
                header["version"] not in (3, 4)
                or header["width"] != metadata["width"]
                or header["height"] != metadata["height"]
                or header["num_frames"] != count
            ):
                raise ValueError("Chunk header does not match container")
            frames = decode_tfc_to_video(str(chunk_path), None, cancel_check=cancel_check)
            for frame in frames:
                check_cancel(cancel_check)
                yield frame
                completed += 1
            report_progress(progress_callback, "Decoding frames", completed, metadata["num_frames"])
        if completed != metadata["num_frames"]:
            raise ValueError("Decoded frame total does not match metadata")


def decode(source, destination, fps=None, stream_output=False, cancel_check=None, progress_callback=None):
    from .format import read_header

    with open(source, "rb") as stream:
        metadata = read_header(stream)
    rate = Fraction(metadata["fps_num"], metadata["fps_den"]) if fps is None else Fraction(str(fps))
    if rate <= 0:
        raise ValueError("FPS must be positive")
    if destination is None:
        if stream_output:
            raise ValueError("Array output cannot be combined with stream_output")
        if metadata["width"] * metadata["height"] * metadata["num_frames"] * 3 > 256 * 1024 * 1024:
            raise ValueError("Buffered RGB output size limit exceeded; use streaming output")
        return np.stack(list(iter_frames(source, cancel_check, progress_callback)))
    with tempfile.TemporaryDirectory(prefix="timeflow-export-") as work, atomic_destination(destination) as temporary:
        silent, audio = Path(work) / ("silent" + Path(destination).suffix), Path(work) / "audio.mka"
        writer = imageio.get_writer(str(silent), fps=float(rate), macro_block_size=1, output_params=["-r", str(rate)])
        try:
            for frame in iter_frames(source, cancel_check, progress_callback, audio):
                writer.append_data(frame)
        finally:
            writer.close()
        check_cancel(cancel_check)
        if metadata["has_audio"] and fps is None:
            mux_audio(silent, audio, temporary, cancel_check)
        elif metadata["has_audio"]:
            raise ValueError("FPS override with audio is unsupported; use source timing")
        else:
            shutil.copyfile(silent, temporary)
