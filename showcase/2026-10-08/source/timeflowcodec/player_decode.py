"""Short-clip RGB cache decoder, run in a disposable player subprocess."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from .decoder import decode_tfc_to_video
from .format import read_header

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RGB_BYTES = 256 * 1024 * 1024


def decode_cache(source: str | Path, destination: str | Path) -> None:
    """Decode without a lossy intermediate; limits are not a payload sandbox."""
    source = Path(source)
    with source.open("rb") as stream:
        header = read_header(stream)
    if header["version"] == 5:
        return _decode_chunked_cache(source, Path(destination), header)
    if source.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("This preview player has a 64 MiB input limit.")
    with source.open("rb") as stream:
        header = read_header(stream)
    width, height, count = (int(header[key]) for key in ("width", "height", "num_frames"))
    if min(width, height, count) <= 0:
        raise ValueError("Video dimensions and frame count must be positive.")
    if width * height * count * 3 > MAX_RGB_BYTES:
        raise ValueError("This preview player has a 256 MiB decoded RGB limit. Use a shorter clip.")
    frames = decode_tfc_to_video(str(source), None)
    if frames is None or frames.shape != (count, height, width, 3) or frames.dtype != np.uint8:
        raise ValueError("Decoded frames do not match the file header.")
    np.save(destination, frames, allow_pickle=False)


def _decode_chunked_cache(source, destination, header):
    from .chunked import iter_frames

    shape = (header["num_frames"], header["height"], header["width"], 3)
    required = int(np.prod(shape))
    # Keep space for the audio track and codec scratch chunks as well as RGB.
    if required + source.stat().st_size + 512 * 1024 * 1024 > shutil.disk_usage(destination.parent).free:
        raise ValueError("Not enough temporary disk space for lossless playback cache")
    audio = destination.with_suffix(".mka")
    with destination.open("wb") as output:
        np.lib.format.write_array_header_2_0(output, {"descr": "|u1", "fortran_order": False, "shape": shape})
        for frame in iter_frames(source, audio_path=audio, work_dir=destination.parent):
            output.write(frame.tobytes())
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "fps_num": header["fps_num"],
                "fps_den": header["fps_den"],
                "has_audio": header["has_audio"],
                "version": 5,
            }
        )
    )


class DiskFrames:
    """Read one RGB frame at a time without retaining mapped pages for a clip."""

    def __init__(self, path):
        self._mmap = open(path, "rb")
        try:
            if np.lib.format.read_magic(self._mmap) != (2, 0):
                raise ValueError("Unsupported cache version")
            self.shape, order, self.dtype = np.lib.format.read_array_header_2_0(self._mmap)
            if order or self.dtype != np.uint8 or len(self.shape) != 4 or self.shape[-1] != 3:
                raise ValueError("Invalid RGB frame cache")
            self.ndim = 4
            self.offset = self._mmap.tell()
            self.frame_bytes = int(np.prod(self.shape[1:]))
            if Path(path).stat().st_size != self.offset + self.frame_bytes * len(self):
                raise ValueError("Incomplete frame cache")
        except Exception:
            self._mmap.close()
            raise

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._mmap.seek(self.offset + index * self.frame_bytes)
        raw = self._mmap.read(self.frame_bytes)
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.shape[1:])


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("Expected an input .tfc and an output .npy cache.", file=sys.stderr)
        return 2
    try:
        decode_cache(*args)
    except Exception as exc:  # Child errors are displayed by the player.
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
