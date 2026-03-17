"""Command-line entrypoints for TimeFlowCodec."""

from __future__ import annotations

import argparse

from . import (
    COMP_LZMA,
    COMP_NONE,
    COMP_ZLIB,
    COMP_ZSTD,
    DEFAULT_SLOPE_THRESHOLD,
    DEFAULT_TAU,
)
from .decoder import decode_tfc_to_video
from .encoder import encode_video_to_tfc
from .version import get_version_string


def compress_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TimeFlowCodec RGB encoder (per-pixel)"
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output .tfc path")
    parser.add_argument(
        "--tau", type=float, default=DEFAULT_TAU, help="Error ratio threshold"
    )
    parser.add_argument(
        "--slope-threshold",
        type=float,
        default=DEFAULT_SLOPE_THRESHOLD,
        help="Slope threshold for const mode",
    )
    parser.add_argument(
        "--payload-comp-type",
        type=int,
        default=COMP_ZSTD,
        choices=[COMP_NONE, COMP_ZLIB, COMP_LZMA, COMP_ZSTD],
        help="Payload compression: 0=none, 1=zlib, 2=LZMA, 3=zstd",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="anime",
        choices=["custom", "anime", "lownoise"],
        help="Compression preset (anime/lownoise are optimized for coherent content)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Limit number of frames processed"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Sliding window (frames) for processing",
    )
    parser.add_argument(
        "--tiling",
        type=int,
        default=None,
        help="Tile size (e.g., 16 or 32) to share params",
    )
    parser.add_argument(
        "--max-ram-mb", type=int, default=None, help="Soft RAM cap; warns when exceeded"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        choices=["uint8", "uint16", "float16"],
        help="Internal dtype for accumulation (uint8 recommended)",
    )
    parser.add_argument(
        "--macbook-profile",
        action="store_true",
        help="Apply MacBook-friendly defaults (tiling/max-ram/zlib/scene-cut)",
    )
    parser.add_argument(
        "--scene-cut",
        type=str,
        default="off",
        choices=["off", "auto"],
        help="Scene cut mode for segmented encoding",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.35,
        help="Scene cut threshold (used when --scene-cut auto)",
    )
    parser.add_argument(
        "--matrix-mode",
        action="store_true",
        help="Enable low-rank matrix mode for RAW-heavy tiles (v2 + tiling)",
    )
    parser.add_argument(
        "--matrix-tau",
        type=float,
        default=0.12,
        help="Error ratio threshold for matrix mode acceptance",
    )
    parser.add_argument(
        "--matrix-rate-ratio",
        type=float,
        default=0.95,
        help="Require estimated matrix payload to be < raw * ratio",
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
        macbook_profile=args.macbook_profile,
        scene_cut=args.scene_cut,
        scene_threshold=args.scene_threshold,
            matrix_mode=args.matrix_mode,
            matrix_tau=args.matrix_tau,
            matrix_rate_ratio=args.matrix_rate_ratio,
            preset=args.preset,
    )


def decompress_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TimeFlowCodec RGB decoder (per-pixel)"
    )
    parser.add_argument("input", help="Input .tfc path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")
    parser.add_argument(
        "--stream-output",
        action="store_true",
        default=True,
        help="Write decoded frames progressively to keep memory bounded",
    )
    parser.add_argument(
        "--no-stream-output",
        action="store_false",
        dest="stream_output",
        help="Disable streaming decode and buffer full output in memory",
    )
    args = parser.parse_args(argv)

    decode_tfc_to_video(
        args.input, args.output, fps=args.fps, stream_output=args.stream_output
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TimeFlowCodec (per-pixel RGB)")
    parser.add_argument(
        "--version", action="store_true", help="Show version/build info and exit"
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_enc = sub.add_parser(
        "encode", aliases=["compress"], help="Compress video to .tfc"
    )
    p_enc.add_argument("input")
    p_enc.add_argument("output")
    p_enc.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p_enc.add_argument("--slope-threshold", type=float, default=DEFAULT_SLOPE_THRESHOLD)
    p_enc.add_argument(
        "--payload-comp-type",
        type=int,
        default=COMP_ZSTD,
        choices=[COMP_NONE, COMP_ZLIB, COMP_LZMA, COMP_ZSTD],
        help="Payload compression: 0=none, 1=zlib, 2=LZMA, 3=zstd",
    )
    p_enc.add_argument(
        "--preset",
        type=str,
        default="anime",
        choices=["custom", "anime", "lownoise"],
        help="Compression preset",
    )
    p_enc.add_argument(
        "--container-version",
        type=int,
        default=2,
        choices=[1, 2],
        help="Container version (2 preferred)",
    )
    p_enc.add_argument("--max-frames", type=int, default=None)
    p_enc.add_argument(
        "--window", type=int, default=None, help="Sliding window (frames)"
    )
    p_enc.add_argument(
        "--tiling", type=int, default=None, help="Tile size to share params"
    )
    p_enc.add_argument(
        "--max-ram-mb", type=int, default=None, help="Soft RAM cap; warns when exceeded"
    )
    p_enc.add_argument(
        "--dtype", type=str, default="uint8", choices=["uint8", "uint16", "float16"]
    )
    p_enc.add_argument(
        "--macbook-profile",
        action="store_true",
        help="Apply MacBook-friendly defaults (tiling/max-ram/zlib/scene-cut)",
    )
    p_enc.add_argument(
        "--scene-cut",
        type=str,
        default="off",
        choices=["off", "auto"],
        help="Scene cut mode for segmented encoding",
    )
    p_enc.add_argument(
        "--scene-threshold",
        type=float,
        default=0.35,
        help="Scene cut threshold (used when --scene-cut auto)",
    )
    p_enc.add_argument(
        "--matrix-mode",
        action="store_true",
        help="Enable low-rank matrix mode for RAW-heavy tiles (v2 + tiling)",
    )
    p_enc.add_argument(
        "--matrix-tau",
        type=float,
        default=0.12,
        help="Error ratio threshold for matrix mode acceptance",
    )
    p_enc.add_argument(
        "--matrix-rate-ratio",
        type=float,
        default=0.95,
        help="Require estimated matrix payload to be < raw * ratio",
    )

    p_dec = sub.add_parser(
        "decode", aliases=["decompress"], help="Decompress .tfc to video"
    )
    p_dec.add_argument("input")
    p_dec.add_argument("output")
    p_dec.add_argument("--fps", type=int, default=30)
    p_dec.add_argument(
        "--stream-output",
        action="store_true",
        default=True,
        help="Write decoded frames progressively to keep memory bounded",
    )
    p_dec.add_argument(
        "--no-stream-output",
        action="store_false",
        dest="stream_output",
        help="Disable streaming decode and buffer full output in memory",
    )

    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        print(get_version_string())
        return

    if args.cmd in {"encode", "compress"}:
        encode_video_to_tfc(
            args.input,
            args.output,
            tau=args.tau,
            slope_threshold=args.slope_threshold,
            payload_comp_type=args.payload_comp_type,
            container_version=args.container_version,
            max_frames=args.max_frames,
            window=args.window,
            tiling=args.tiling,
            max_ram_mb=args.max_ram_mb,
            dtype=args.dtype,
            macbook_profile=args.macbook_profile,
            scene_cut=args.scene_cut,
            scene_threshold=args.scene_threshold,
            matrix_mode=args.matrix_mode,
            matrix_tau=args.matrix_tau,
            matrix_rate_ratio=args.matrix_rate_ratio,
            preset=args.preset,
        )
    elif args.cmd in {"decode", "decompress"}:
        decode_tfc_to_video(
            args.input, args.output, fps=args.fps, stream_output=args.stream_output
        )
    else:
        parser.error("Specify 'encode'/'decode' (or 'compress'/'decompress').")


if __name__ == "__main__":
    main()
