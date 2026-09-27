"""Generate a tiny deterministic detail stress clip; not a representative video corpus."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(args.output), fps=12, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]
    ) as writer:
        for t in range(12):
            frame = np.full((48, 64, 3), 235, dtype=np.uint8)
            frame[4:22:2, 4:35] = [30, 50, 90]
            frame[26:44, 4:35:3] = [180, 40, 30]
            frame[4:44, 40] = [0, 180, 90]
            frame[30:34, 44+t:46+t] = [20, 20, 20]
            writer.append_data(frame)


if __name__ == "__main__":
    main()
