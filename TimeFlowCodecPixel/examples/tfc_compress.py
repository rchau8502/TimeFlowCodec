"""CLI wrapper for TimeFlowCodec encoder."""
from __future__ import annotations

import argparse

from timeflowcodec import encode_video_to_tfc


def main():
    parser = argparse.ArgumentParser(description="Per-pixel TimeFlowCodec encoder")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output .tfc path")
    parser.add_argument("--tau", type=float, default=0.1, help="Error ratio threshold")
    parser.add_argument("--slope-threshold", type=float, default=1e-3, help="Slope threshold for const mode")
    parser.add_argument(
        "--payload-comp-type",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Payload compression: 0=none, 1=zlib, 2=LZMA",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames processed")
    args = parser.parse_args()

    encode_video_to_tfc(
        args.input,
        args.output,
        tau=args.tau,
        slope_threshold=args.slope_threshold,
        payload_comp_type=args.payload_comp_type,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
