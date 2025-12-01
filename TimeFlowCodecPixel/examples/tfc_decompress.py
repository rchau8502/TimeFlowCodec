"""CLI wrapper for TimeFlowCodec decoder."""
from __future__ import annotations

import argparse

from timeflowcodec import decode_tfc_to_video


def main():
    parser = argparse.ArgumentParser(description="Per-pixel TimeFlowCodec decoder")
    parser.add_argument("input", help="Input .tfc path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    args = parser.parse_args()

    decode_tfc_to_video(args.input, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
