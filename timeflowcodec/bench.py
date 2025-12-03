from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .decoder import decode_tfc_to_video
from .encoder import encode_video_to_tfc
from .utils import _ensure_rgb, load_video_rgb


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
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def has_ffmpeg_codec(name: str) -> bool:
    proc = run_cmd(["ffmpeg", "-hide_banner", "-codecs"])
    return name in proc.stdout


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_metrics_ffmpeg(ref: Path, test: Path) -> tuple[Optional[float], Optional[float]]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(test),
        "-i",
        str(ref),
        "-lavfi",
        "[0:v][1:v]psnr;[0:v][1:v]ssim",
        "-f",
        "null",
        "-",
    ]
    proc = run_cmd(cmd)
    psnr = None
    ssim = None
    txt = proc.stderr
    for line in txt.splitlines():
        if "psnr_avg" in line:
            try:
                psnr = float(line.split("psnr_avg:")[-1].split()[0])
            except ValueError:
                pass
        if "All:" in line and "ssim" in line:
            try:
                ssim = float(line.split("All:")[-1].split()[0])
            except ValueError:
                pass
    return psnr, ssim


def compute_metrics_numpy(ref_frames: np.ndarray, test_frames: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    if ref_frames.shape != test_frames.shape:
        return None, None
    mse = np.mean((ref_frames.astype(np.float64) - test_frames.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * math.log10((255.0 * 255.0) / mse)
    # Simple SSIM proxy: luminance-only, no windowing (rough but deterministic)
    ref_y = ref_frames.mean(axis=3)
    test_y = test_frames.mean(axis=3)
    mu_x = ref_y.mean()
    mu_y = test_y.mean()
    sigma_x = ref_y.var()
    sigma_y = test_y.var()
    sigma_xy = ((ref_y - mu_x) * (test_y - mu_y)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    denom = (sigma_x + sigma_y + c1) * (sigma_x + sigma_y + c2)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / denom if denom != 0 else None
    return psnr, float(ssim) if ssim is not None else None


def load_frames(path: Path) -> np.ndarray:
    arr = []
    reader = load_video_rgb(str(path))
    arr.append(reader)
    return arr[0]


def bench_tfc(clip: Path, tmp: Path, tau: float, slope: float, payload_comp_type: int) -> tuple[Path, BenchResult]:
    out_tfc = tmp / (clip.stem + ".tfc")
    out_rec = tmp / (clip.stem + "_tfc.mp4")

    start = time.perf_counter()
    encode_video_to_tfc(str(clip), str(out_tfc), tau=tau, slope_threshold=slope, payload_comp_type=payload_comp_type)
    enc_time = time.perf_counter() - start

    start = time.perf_counter()
    decode_tfc_to_video(str(out_tfc), str(out_rec))
    dec_time = time.perf_counter() - start

    frames = load_video_rgb(str(clip))
    t, h, w, _ = frames.shape
    enc_fps = t / enc_time if enc_time > 0 else 0.0
    dec_fps = t / dec_time if dec_time > 0 else 0.0
    size_bytes = out_tfc.stat().st_size

    psnr, ssim = compute_metrics_ffmpeg(clip, out_rec)
    if psnr is None or ssim is None:
        recon = load_video_rgb(str(out_rec))
        psnr, ssim = compute_metrics_numpy(frames, recon)

    res = BenchResult(
        codec="tfc",
        preset="-",
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
    return out_rec, res


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
        "-preset",
        preset,
        "-crf",
        crf,
        "-an",
        str(out_vid),
    ]
    start = time.perf_counter()
    proc = run_cmd(cmd)
    enc_time = time.perf_counter() - start
    if proc.returncode != 0:
        return None

    frames = load_video_rgb(str(clip))
    t, h, w, _ = frames.shape
    enc_fps = t / enc_time if enc_time > 0 else 0.0

    start = time.perf_counter()
    # decoding just via imageio read
    recon = load_video_rgb(str(out_vid))
    dec_time = time.perf_counter() - start
    dec_fps = t / dec_time if dec_time > 0 else 0.0

    psnr, ssim = compute_metrics_ffmpeg(clip, out_vid)
    if psnr is None or ssim is None:
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
    ff = run_cmd(["ffmpeg", "-version"])
    if ff.returncode == 0:
        data["ffmpeg"] = ff.stdout.splitlines()[0]
    return data


def write_results(out_dir: Path, results: list[BenchResult], env: dict, missing: list[str]) -> None:
    ensure_dir(out_dir)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"env": env, "results": [asdict(r) for r in results], "missing": missing}, f, indent=2)

    # report.md
    lines = []
    lines.append("# TimeFlowCodec Benchmark Report\n")
    lines.append(f"Env: {env}\n")
    if missing:
        lines.append(f"Missing codecs: {', '.join(missing)}\n")
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
                    t = r.encode_time
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
    parser.add_argument("--tau", type=float, default=0.1, help="Tau for TFC")
    parser.add_argument("--slope-threshold", type=float, default=1e-3, help="Slope threshold for TFC")
    parser.add_argument("--payload-comp-type", type=int, default=1, choices=[0, 1, 2], help="Payload compression for TFC")
    args = parser.parse_args(argv)

    clips = find_clips(args.clips)
    if not clips:
        raise SystemExit(f"No .mp4 clips found in {args.clips}")

    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    crfs = [c.strip() for c in args.crf.split(",") if c.strip()]

    tmp = Path(tempfile.mkdtemp(prefix="tfc_bench_"))
    results: list[BenchResult] = []
    missing: list[str] = []

    for clip in clips:
        # TFC
        rec_path, res = bench_tfc(clip, tmp, args.tau, args.slope_threshold, args.payload_comp_type)
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
    write_results(args.out, results, env, sorted(set(missing)))
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
