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
    parser = argparse.ArgumentParser(description="TimeFlowCodec encoder")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output .tfc path")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Error ratio threshold")
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
        choices=["custom", "anime", "lownoise", "general", "screen"],
        help="Compression preset (general enables motion compensation)",
    )
    parser.add_argument(
        "--colorspace",
        type=str,
        default="rgb",
        choices=["rgb", "yuv444", "yuv420"],
        help="Internal coding colorspace (anime usually benefits from yuv420)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames processed")
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
    parser.add_argument("--max-ram-mb", type=int, default=None, help="Soft RAM cap; warns when exceeded")
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
    parser.add_argument(
        "--motion-comp",
        "--mc",
        action="store_true",
        help="Enable block-matching motion compensation",
    )
    parser.add_argument("--crf", type=float, default=None, help="Constant Rate Factor (0-63)")
    parser.add_argument("--bitrate", type=str, default=None, help="Target bitrate (e.g. '2M', '800k')")
    parser.add_argument(
        "--tune",
        type=str,
        default=None,
        choices=["screen", "anime", "natural"],
        help="Content tuning: screen (zero halo), anime (yuv420), natural (motion comp)",
    )
    parser.add_argument("--rdo", action="store_true", help="Enable Lagrangian Rate-Distortion Optimization")
    parser.add_argument("--lambda-rdo", type=float, default=None, help="Lagrangian multiplier for RDO")
    parser.add_argument("--threads", type=int, default=1, help="Number of worker threads (0=auto)")
    parser.add_argument("--container-version", type=int, default=5, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args(argv)

    opts = _resolve_encode_options(args)
    encode_video_to_tfc(args.input, args.output, **opts)


def _resolve_encode_options(args: argparse.Namespace) -> dict:
    tau = args.tau
    slope_threshold = args.slope_threshold
    preset = args.preset
    colorspace = args.colorspace
    motion_comp = args.motion_comp
    rdo = getattr(args, "rdo", False)
    lambda_rdo = getattr(args, "lambda_rdo", None)
    threads = getattr(args, "threads", 1)

    tune = getattr(args, "tune", None)
    if tune:
        if tune == "screen":
            preset = "screen"
            colorspace = "rgb"
        elif tune == "anime":
            preset = "anime"
            colorspace = "yuv420"
        elif tune == "natural":
            preset = "general"
            motion_comp = True

    crf = getattr(args, "crf", None)
    if crf is not None:
        from .rate_control import crf_to_parameters

        crf_params = crf_to_parameters(crf, tune=tune or "screen")
        tau = crf_params["tau"]
        slope_threshold = crf_params["slope_threshold"]
        if rdo and lambda_rdo is None:
            lambda_rdo = crf_params["lambda_rdo"]
    elif getattr(args, "bitrate", None) is not None:
        from .rate_control import parse_bitrate_string

        target_kbps = parse_bitrate_string(args.bitrate)
        est_tau = max(1e-5, min(0.05, 5.0 / max(target_kbps, 100.0)))
        tau = est_tau

    return {
        "tau": tau,
        "slope_threshold": slope_threshold,
        "payload_comp_type": args.payload_comp_type,
        "container_version": args.container_version,
        "max_frames": args.max_frames,
        "window": args.window,
        "tiling": args.tiling,
        "max_ram_mb": args.max_ram_mb,
        "dtype": args.dtype,
        "colorspace": colorspace,
        "macbook_profile": args.macbook_profile,
        "scene_cut": args.scene_cut,
        "scene_threshold": args.scene_threshold,
        "matrix_mode": args.matrix_mode,
        "matrix_tau": args.matrix_tau,
        "matrix_rate_ratio": args.matrix_rate_ratio,
        "preset": preset,
        "motion_comp": motion_comp,
        "rdo": rdo,
        "lambda_rdo": lambda_rdo,
        "threads": threads,
    }


def decompress_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TimeFlowCodec RGB decoder (per-pixel)")
    parser.add_argument("input", help="Input .tfc path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--fps", type=float, default=None, help="Override source FPS (legacy default: 30)")
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
    parser.add_argument("--threads", type=int, default=1, help="Number of worker threads (0=auto)")
    args = parser.parse_args(argv)

    decode_tfc_to_video(
        args.input,
        args.output,
        fps=args.fps,
        stream_output=args.stream_output,
        threads=args.threads,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TimeFlowCodec")
    parser.add_argument("--version", action="store_true", help="Show version/build info and exit")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_enc = sub.add_parser("encode", aliases=["compress"], help="Compress video to .tfc")
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
        choices=["custom", "anime", "lownoise", "general", "screen"],
        help="Compression preset",
    )
    p_enc.add_argument(
        "--container-version",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Container version (5 stores timing, audio, and bounded chunks)",
    )
    p_enc.add_argument(
        "--colorspace",
        type=str,
        default="rgb",
        choices=["rgb", "yuv444", "yuv420"],
        help="Internal coding colorspace",
    )
    p_enc.add_argument("--max-frames", type=int, default=None)
    p_enc.add_argument("--window", type=int, default=None, help="Sliding window (frames)")
    p_enc.add_argument("--tiling", type=int, default=None, help="Tile size to share params")
    p_enc.add_argument("--max-ram-mb", type=int, default=None, help="Soft RAM cap; warns when exceeded")
    p_enc.add_argument("--dtype", type=str, default="uint8", choices=["uint8", "uint16", "float16"])
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
    p_enc.add_argument(
        "--motion-comp",
        "--mc",
        action="store_true",
        help="Enable block-matching motion compensation",
    )
    p_enc.add_argument("--crf", type=float, default=None, help="Constant Rate Factor (0-63)")
    p_enc.add_argument("--bitrate", type=str, default=None, help="Target bitrate (e.g. '2M', '800k')")
    p_enc.add_argument(
        "--tune",
        type=str,
        default=None,
        choices=["screen", "anime", "natural"],
        help="Content tuning: screen (zero halo), anime (yuv420), natural (motion comp)",
    )
    p_enc.add_argument("--rdo", action="store_true", help="Enable Lagrangian Rate-Distortion Optimization")
    p_enc.add_argument("--lambda-rdo", type=float, default=None, help="Lagrangian multiplier for RDO")
    p_enc.add_argument("--threads", type=int, default=1, help="Number of worker threads (0=auto)")

    p_dec = sub.add_parser("decode", aliases=["decompress"], help="Decompress .tfc to video")
    p_dec.add_argument("input")
    p_dec.add_argument("output")
    p_dec.add_argument("--fps", type=float, default=None)
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
    p_dec.add_argument("--threads", type=int, default=1, help="Number of worker threads (0=auto)")

    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        print(get_version_string())
        return

    if args.cmd in {"encode", "compress"}:
        opts = _resolve_encode_options(args)
        encode_video_to_tfc(args.input, args.output, **opts)
    elif args.cmd in {"decode", "decompress"}:
        decode_tfc_to_video(
            args.input,
            args.output,
            fps=args.fps,
            stream_output=args.stream_output,
            threads=args.threads,
        )
    else:
        parser.error("Specify 'encode'/'decode' (or 'compress'/'decompress').")


if __name__ == "__main__":
    main()
