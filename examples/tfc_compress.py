"""CLI wrapper for RGB TimeFlowCodec encoder."""
from __future__ import annotations

import argparse

from timeflowcodec import encode_video_to_tfc


def main():
    parser = argparse.ArgumentParser(description="RGB TimeFlowCodec encoder (per-pixel)")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output .tfc path")
    parser.add_argument("--tau", type=float, default=0.1, help="Error ratio threshold")
    parser.add_argument("--slope-threshold", type=float, default=1e-3, help="Slope threshold for const mode")
    parser.add_argument(
        "--payload-comp-type",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Payload compression: 0=none, 1=zlib, 2=LZMA",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames processed")
    parser.add_argument("--window", type=int, default=None, help="Sliding window (frames) for processing")
    parser.add_argument("--tiling", type=int, default=None, help="Tile size (share params per tile)")
    parser.add_argument("--max-ram-mb", type=int, default=None, help="Soft RAM cap; warns when exceeded")
    parser.add_argument("--dtype", type=str, default="uint8", choices=["uint8", "uint16", "float16"], help="Internal dtype")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
