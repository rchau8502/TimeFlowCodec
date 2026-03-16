"""CLI wrapper for RGB TimeFlowCodec decoder."""

from __future__ import annotations

import argparse

from timeflowcodec import decode_tfc_to_video


def main():
    parser = argparse.ArgumentParser(
        description="RGB TimeFlowCodec decoder (per-pixel)"
    )
    parser.add_argument("input", help="Input .tfc path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    parser.add_argument(
        "--stream-output",
        action="store_true",
        default=True,
        help="Write progressively to reduce memory",
    )
    parser.add_argument(
        "--no-stream-output",
        action="store_false",
        dest="stream_output",
        help="Buffer full decoded video in memory",
    )
    args = parser.parse_args()

    decode_tfc_to_video(
        args.input, args.output, fps=args.fps, stream_output=args.stream_output
    )


if __name__ == "__main__":
    main()
