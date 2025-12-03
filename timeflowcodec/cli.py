"""Command-line entrypoints for TimeFlowCodec."""
from __future__ import annotations

import argparse

from . import DEFAULT_SLOPE_THRESHOLD, DEFAULT_TAU
from .decoder import decode_tfc_to_video
from .encoder import encode_video_to_tfc


def compress_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TimeFlowCodec RGB encoder (per-pixel)")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output .tfc path")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Error ratio threshold")
    parser.add_argument(
        "--slope-threshold", type=float, default=DEFAULT_SLOPE_THRESHOLD, help="Slope threshold for const mode"
    )
    parser.add_argument(
        "--payload-comp-type",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Payload compression: 0=none, 1=zlib, 2=LZMA",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames processed")
    parser.add_argument("--window", type=int, default=None, help="Sliding window (frames) for processing")
    parser.add_argument("--tiling", type=int, default=None, help="Tile size (e.g., 16 or 32) to share params")
    parser.add_argument("--max-ram-mb", type=int, default=None, help="Soft RAM cap; warns when exceeded")
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        choices=["uint8", "uint16", "float16"],
        help="Internal dtype for accumulation (uint8 recommended)",
    )
    args = parser.parse_args(argv)

    encode_video_to_tfc(
        args.input,
        args.output,
        tau=args.tau,
        slope_threshold=args.slope_threshold,
        payload_comp_type=args.payload_comp_type,
        max_frames=args.max_frames,
        window=args.window,
        tiling=args.tiling,
        max_ram_mb=args.max_ram_mb,
        dtype=args.dtype,
    )


def decompress_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TimeFlowCodec RGB decoder (per-pixel)")
    parser.add_argument("input", help="Input .tfc path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    args = parser.parse_args(argv)

    decode_tfc_to_video(args.input, args.output, fps=args.fps)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TimeFlowCodec (per-pixel RGB)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("compress", help="Compress video to .tfc")
    p_enc.add_argument("input")
    p_enc.add_argument("output")
    p_enc.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p_enc.add_argument("--slope-threshold", type=float, default=DEFAULT_SLOPE_THRESHOLD)
    p_enc.add_argument("--payload-comp-type", type=int, default=2, choices=[0, 1, 2])
    p_enc.add_argument("--max-frames", type=int, default=None)

    p_dec = sub.add_parser("decompress", help="Decompress .tfc to video")
    p_dec.add_argument("input")
    p_dec.add_argument("output")
    p_dec.add_argument("--fps", type=int, default=30)

    args = parser.parse_args(argv)

    if args.cmd == "compress":
        encode_video_to_tfc(
            args.input,
            args.output,
            tau=args.tau,
            slope_threshold=args.slope_threshold,
            payload_comp_type=args.payload_comp_type,
            max_frames=args.max_frames,
        )
    elif args.cmd == "decompress":
        decode_tfc_to_video(args.input, args.output, fps=args.fps)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
