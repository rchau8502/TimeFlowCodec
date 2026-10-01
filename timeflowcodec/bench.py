from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import imageio_ffmpeg
import numpy as np

from .constants import COMP_ZSTD
from .decoder import decode_tfc_to_video
from .encoder import encode_video_to_tfc
from .utils import load_video_rgb


@dataclass
class BenchResult:
    codec: str
    preset: str
    crf: str
    clip: str
    encode_time: float
    decode_time: float
    encode_fps: float
    decode_fps: float
    size_bytes: int
    psnr: float | None
    ssim: float | None


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    if cmd[0] == "ffmpeg":
        cmd = [shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe(), *cmd[1:]]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def has_ffmpeg_codec(name: str) -> bool:
    proc = run_cmd(["ffmpeg", "-hide_banner", "-encoders"])
    return name in proc.stdout


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_metrics_numpy(ref_frames: np.ndarray, test_frames: np.ndarray) -> tuple[float, float]:
    """RGB PSNR and mean per-frame, per-channel windowed SSIM (uint8 range)."""
    from skimage.metrics import structural_similarity

    if (
        ref_frames.shape != test_frames.shape
        or ref_frames.ndim != 4
        or ref_frames.shape[-1] != 3
        or not ref_frames.shape[0]
    ):
        raise ValueError("Expected matching nonempty (T, H, W, 3) frame shapes")
    side = min(ref_frames.shape[1:3])
    if side < 3:
        raise ValueError("SSIM requires frames at least 3 pixels wide and high")
    window = min(7, side if side % 2 else side - 1)
    mse = np.mean((ref_frames.astype(np.float64) - test_frames.astype(np.float64)) ** 2)
    psnr = float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)
    ssim = np.mean(
        [
            structural_similarity(ref, test, data_range=255, channel_axis=-1, win_size=window)
            for ref, test in zip(ref_frames, test_frames)
        ]
    )
    return psnr, float(ssim)


def load_frames(path: Path) -> np.ndarray:
    arr = []
    reader = load_video_rgb(str(path))
    arr.append(reader)
    return arr[0]


def bench_tfc(
    clip: Path,
    tmp: Path,
    tau: float | None,
    slope: float,
    payload_comp_type: int,
    preset: str,
) -> tuple[Path, BenchResult]:
    out_tfc = tmp / (clip.stem + ".tfc")

    start = time.perf_counter()
    encode_video_to_tfc(
        str(clip),
        str(out_tfc),
        **({"tau": tau} if tau is not None else {}),
        slope_threshold=slope,
        payload_comp_type=payload_comp_type,
        container_version=3,
        preset=preset,
    )
    enc_time = time.perf_counter() - start

    start = time.perf_counter()
    recon = decode_tfc_to_video(str(out_tfc), None)
    dec_time = time.perf_counter() - start

    frames = load_video_rgb(str(clip))
    t, h, w, _ = frames.shape
    enc_fps = t / enc_time if enc_time > 0 else 0.0
    dec_fps = t / dec_time if dec_time > 0 else 0.0
    size_bytes = out_tfc.stat().st_size

    psnr, ssim = compute_metrics_numpy(frames, recon)

    res = BenchResult(
        codec="tfc",
        preset=preset,
        crf="-",
        clip=clip.name,
        encode_time=enc_time,
        decode_time=dec_time,
        encode_fps=enc_fps,
        decode_fps=dec_fps,
        size_bytes=size_bytes,
        psnr=psnr,
        ssim=ssim,
    )
    return out_tfc, res


def bench_ffmpeg(clip: Path, tmp: Path, codec: str, preset: str, crf: str) -> Optional[tuple[Path, BenchResult]]:
    codec_map = {"x264": "libx264", "x265": "libx265", "av1": "libaom-av1"}
    if codec not in codec_map:
        return None
    lib = codec_map[codec]
    if not has_ffmpeg_codec(lib):
        return None

    out_vid = tmp / f"{clip.stem}_{codec}_{preset}_crf{crf}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(clip),
        "-c:v",
        lib,
        "-crf",
        crf,
        "-an",
        str(out_vid),
    ]
    if codec == "av1":
        speed = ["-cpu-used", {"fast": "6", "medium": "4", "slow": "2"}.get(preset, "4")]
    else:
        speed = ["-preset", preset]
    cmd[-1:-1] = speed
    start = time.perf_counter()
    proc = run_cmd(cmd)
    enc_time = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{codec} encode failed: {proc.stderr[-2000:]}")

    frames = load_video_rgb(str(clip))
    t, h, w, _ = frames.shape
    enc_fps = t / enc_time if enc_time > 0 else 0.0

    start = time.perf_counter()
    # decoding just via imageio read
    recon = load_video_rgb(str(out_vid))
    dec_time = time.perf_counter() - start
    dec_fps = t / dec_time if dec_time > 0 else 0.0

    psnr, ssim = compute_metrics_numpy(frames, recon)

    res = BenchResult(
        codec=codec,
        preset=preset,
        crf=crf,
        clip=clip.name,
        encode_time=enc_time,
        decode_time=dec_time,
        encode_fps=enc_fps,
        decode_fps=dec_fps,
        size_bytes=out_vid.stat().st_size,
        psnr=psnr,
        ssim=ssim,
    )
    return out_vid, res


def collect_env() -> dict:
    data = {
        "python": sys.version,
        "platform": sys.platform,
    }
    git = run_cmd(["git", "rev-parse", "HEAD"])
    if git.returncode == 0:
        data["git_commit"] = git.stdout.strip()
    status = run_cmd(["git", "status", "--porcelain"])
    data["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    data["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    ff = run_cmd(["ffmpeg", "-version"])
    if ff.returncode == 0:
        data["ffmpeg"] = ff.stdout.splitlines()[0]
    return data


def write_results(out_dir: Path, results: list[BenchResult], env: dict, missing: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    records = []
    for result in results:
        record = asdict(result)
        record["lossless_reconstruction"] = result.psnr == float("inf")
        # JSON has no Infinity literal; the flag distinguishes perfect reconstruction.
        if record["lossless_reconstruction"]:
            record["psnr"] = None
        records.append(record)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"env": env, "results": records, "missing": missing}, f, indent=2, allow_nan=False)

    # report.md
    lines = []
    lines.append("# TimeFlowCodec Benchmark Report\n")
    lines.append(f"Env: {env}\n")
    lines.append(
        "Metrics use decoded RGB frames directly: RGB PSNR and mean per-frame windowed RGB SSIM. "
        "TFC decode timing excludes video export. All metrics buffer short clips in RAM. "
        "Compare file sizes only at similar measured quality; equal CRF is not equal quality.\n"
    )
    if missing:
        lines.append(f"Missing codecs: {', '.join(missing)}\n")

    # Compute Bjontegaard Delta metrics where 4+ points exist
    bd_records = {}
    from .bjontegaard import bd_psnr, bd_rate
    clips = sorted(set(r.clip for r in results))
    for clip in clips:
        clip_res = [r for r in results if r.clip == clip]
        tfc_res = [r for r in clip_res if r.codec == "tfc" and r.psnr is not None and r.psnr != float("inf")]
        if len(tfc_res) >= 4:
            tfc_rates = [r.size_bytes for r in tfc_res]
            tfc_psnrs = [r.psnr for r in tfc_res]
            for codec in sorted(set(r.codec for r in clip_res if r.codec != "tfc")):
                anchor_res = [r for r in clip_res if r.codec == codec and r.psnr is not None and r.psnr != float("inf")]
                if len(anchor_res) >= 4:
                    anchor_rates = [r.size_bytes for r in anchor_res]
                    anchor_psnrs = [r.psnr for r in anchor_res]
                    try:
                        rate_pct = bd_rate(anchor_rates, anchor_psnrs, tfc_rates, tfc_psnrs)
                        psnr_gain = bd_psnr(anchor_rates, anchor_psnrs, tfc_rates, tfc_psnrs)
                        bd_records[f"{clip} vs {codec}"] = {
                            "clip": clip,
                            "anchor": codec,
                            "bd_rate_pct": rate_pct,
                            "bd_psnr_db": psnr_gain,
                        }
                    except Exception:
                        pass

    if bd_records:
        lines.append("## Bjontegaard Delta Metrics (TFC vs Baselines)\n")
        lines.append("| Clip | Anchor Codec | BD-Rate (%) | BD-PSNR (dB) |")
        lines.append("|---|---|---|---|")
        for key, bd in bd_records.items():
            lines.append(f"| {bd['clip']} | {bd['anchor']} | {bd['bd_rate_pct']:+.2f}% | {bd['bd_psnr_db']:+.2f} dB |")
        lines.append("\n*Negative BD-Rate indicates bitrate reduction for equal quality (superior efficiency).*\n")

    lines.append("## Scoreboard\n")
    lines.append("| codec | preset | crf | clip | size (bytes) | enc time (s) | dec time (s) | psnr | ssim |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.codec} | {r.preset} | {r.crf} | {r.clip} | {r.size_bytes} | "
            f"{r.encode_time:.2f} | {r.decode_time:.2f} | {r.psnr if r.psnr is not None else 'NA'} | "
            f"{r.ssim if r.ssim is not None else 'NA'} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # plots
    ensure_dir(out_dir / "plots")
    if results:
        # bitrate vs psnr
        for metric in ("psnr", "ssim"):
            plt.figure()
            for codec in set(r.codec for r in results):
                xs = []
                ys = []
                for r in results:
                    if r.codec != codec:
                        continue
                    xs.append(r.size_bytes)
                    val = getattr(r, metric)
                    ys.append(val if val is not None else np.nan)
                plt.scatter(xs, ys, label=codec)
            plt.xlabel("Size (bytes)")
            plt.ylabel(metric.upper())
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "plots" / f"bitrate_{metric}.png")
            plt.close()
        # speed bars
        plt.figure()
        labels = [f"{r.codec}-{r.clip}" for r in results]
        enc = [r.encode_fps for r in results]
        dec = [r.decode_fps for r in results]
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width / 2, enc, width, label="encode fps")
        plt.bar(x + width / 2, dec, width, label="decode fps")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "speed.png")
        plt.close()


def find_clips(path: Path) -> list[Path]:
    return sorted([p for p in path.glob("*.mp4") if p.is_file()])


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark TimeFlowCodec against ffmpeg codecs")
    parser.add_argument("--clips", type=Path, required=True, help="Directory containing .mp4 clips")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--codecs", type=str, default="tfc,x264,x265,av1", help="Comma separated codecs")
    parser.add_argument("--presets", type=str, default="medium", help="Comma separated presets for ffmpeg codecs")
    parser.add_argument("--crf", type=str, default="23", help="Comma separated CRF values")
    parser.add_argument("--tau", type=float, default=None, help="Override preset tau")
    parser.add_argument("--slope-threshold", type=float, default=1e-3, help="Slope threshold for TFC")
    parser.add_argument(
        "--payload-comp-type",
        type=int,
        default=COMP_ZSTD,
        choices=[0, 1, 2, 3],
        help="Payload compression for TFC (0=none,1=zlib,2=lzma,3=zstd)",
    )
    parser.add_argument(
        "--tfc-preset",
        type=str,
        default="anime",
        choices=["custom", "anime", "lownoise", "general", "screen"],
        help="Preset for TFC runs",
    )
    args = parser.parse_args(argv)

    clips = find_clips(args.clips)
    if not clips:
        raise SystemExit(f"No .mp4 clips found in {args.clips}")

    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    crfs = [c.strip() for c in args.crf.split(",") if c.strip()]

    with tempfile.TemporaryDirectory(prefix="tfc_bench_") as tmp_name:
        tmp = Path(tmp_name)
        results: list[BenchResult] = []
        missing: list[str] = []

        for clip in clips:
            # TFC
            if "tfc" in codecs:
                _rec_path, res = bench_tfc(
                    clip,
                    tmp,
                    args.tau,
                    args.slope_threshold,
                    args.payload_comp_type,
                    args.tfc_preset,
                )
                results.append(res)
            # ffmpeg baselines
            for codec in codecs:
                if codec == "tfc":
                    continue
                for preset in presets:
                    for crf in crfs:
                        out = bench_ffmpeg(clip, tmp, codec, preset, crf)
                        if out is None:
                            missing.append(codec)
                            continue
                        results.append(out[1])

        env = collect_env()
        env["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        write_results(args.out, results, env, sorted(set(missing)))
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
