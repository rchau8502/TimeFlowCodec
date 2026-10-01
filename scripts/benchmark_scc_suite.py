#!/usr/bin/env python3
"""Automated JVET-SCC Benchmark Suite Runner for TimeFlowCodec.

Executes standardized Rate-Distortion sweeps across JVET Screen Content Coding
benchmark sequences, computes ITU-T VCEG-M33 Bjontegaard Delta metrics (BD-Rate & BD-PSNR),
and formats outputs as Markdown, JSON, and publication-ready LaTeX tables.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from timeflowcodec.bench import compute_metrics_numpy  # noqa: E402
from timeflowcodec.bjontegaard import bd_psnr, bd_rate  # noqa: E402
from timeflowcodec.decoder import decode_tfc_to_video  # noqa: E402
from timeflowcodec.encoder import encode_video_to_tfc  # noqa: E402
from timeflowcodec.utils import load_video_rgb  # noqa: E402


@dataclass
class PointResult:
    codec: str
    clip: str
    param_label: str
    rate_kbps: float
    size_bytes: int
    psnr: float
    ssim: float
    encode_time_s: float
    decode_time_s: float


def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if not exe:
        raise RuntimeError("ffmpeg executable not found on system PATH.")
    return exe


def ensure_scc_clips(clips_dir: Path) -> list[Path]:
    """Verify that canonical SCC clips exist, or generate them via generate_scc_benchmarks.py."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        clips_dir / "scc_slideshow.mp4",
        clips_dir / "scc_console_code.mp4",
        clips_dir / "scc_web_browsing.mp4",
        clips_dir / "scc_animation_ui.mp4",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        print(f"[scc_bench] Missing {len(missing)} benchmark clips. Generating canonical clips...")
        gen_script = REPO_ROOT / "examples" / "generate_scc_benchmarks.py"
        subprocess.check_call([sys.executable, str(gen_script), "--out-dir", str(clips_dir)])

    return [p for p in expected if p.exists()]


def run_ffmpeg_encode(
    ffmpeg_exe: str,
    input_mp4: Path,
    encoder: str,
    crf: int,
    out_mp4: Path,
) -> tuple[float, int]:
    t0 = time.perf_counter()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_mp4),
        "-c:v",
        encoder,
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    enc_time = time.perf_counter() - t0
    size_bytes = out_mp4.stat().st_size
    return enc_time, size_bytes


def run_ffmpeg_decode(
    ffmpeg_exe: str,
    in_mp4: Path,
) -> tuple[float, np.ndarray]:
    t0 = time.perf_counter()
    frames = load_video_rgb(str(in_mp4))
    dec_time = time.perf_counter() - t0
    return dec_time, frames


def evaluate_clip_point(
    gt_frames: np.ndarray,
    recon_frames: np.ndarray,
    size_bytes: int,
    duration_s: float,
) -> tuple[float, float, float]:
    rate_kbps = (size_bytes * 8.0) / (duration_s * 1000.0) if duration_s > 0 else 0.0
    psnr_val, ssim_val = compute_metrics_numpy(gt_frames, recon_frames)
    return rate_kbps, psnr_val, ssim_val


def run_suite(
    clips: list[Path],
    out_dir: Path,
    quick: bool = False,
) -> dict[str, Any]:
    ffmpeg_exe = check_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    crf_list = [18, 23, 28, 33] if not quick else [23, 28]
    tau_list = [0.0001, 0.001, 0.005, 0.02] if not quick else [0.001, 0.005]

    all_points: list[PointResult] = []

    print("\n========================================================")
    print("TimeFlowCodec JVET-SCC Standardized Benchmark Suite")
    print(f"Clips: {len(clips)} | Quick mode: {quick}")
    print("========================================================\n")

    for clip_path in clips:
        clip_name = clip_path.stem
        print(f"--> Processing Sequence: {clip_name}")

        reader = imageio.get_reader(str(clip_path))
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 30.0))
        reader.close()
        gt_frames = load_video_rgb(str(clip_path))
        num_frames = len(gt_frames)
        duration_s = num_frames / fps

        # 1. Anchor: libx264
        for crf in crf_list:
            enc_out = temp_dir / f"{clip_name}_x264_crf{crf}.mp4"
            enc_time, size_bytes = run_ffmpeg_encode(ffmpeg_exe, clip_path, "libx264", crf, enc_out)
            dec_time, recon_frames = run_ffmpeg_decode(ffmpeg_exe, enc_out)
            kbps, psnr_val, ssim_val = evaluate_clip_point(gt_frames, recon_frames, size_bytes, duration_s)
            all_points.append(
                PointResult("x264", clip_name, f"CRF={crf}", kbps, size_bytes, psnr_val, ssim_val, enc_time, dec_time)
            )
            print(f"   [x264] CRF {crf:2d}: {kbps:8.1f} kbps | {psnr_val:5.2f} dB PSNR | {ssim_val:.4f} SSIM")

        # 2. Test Codec: TimeFlowCodec
        for tau in tau_list:
            tfc_out = temp_dir / f"{clip_name}_tfc_tau{tau}.tfc"
            t0 = time.perf_counter()
            encode_video_to_tfc(
                str(clip_path),
                str(tfc_out),
                preset="screen",
                tau=tau,
                window=min(32, num_frames),
            )
            enc_time = time.perf_counter() - t0
            size_bytes = tfc_out.stat().st_size

            t0_dec = time.perf_counter()
            recon_frames = decode_tfc_to_video(str(tfc_out), output_path=None)
            dec_time = time.perf_counter() - t0_dec

            kbps, psnr_val, ssim_val = evaluate_clip_point(gt_frames, recon_frames, size_bytes, duration_s)
            all_points.append(
                PointResult("tfc", clip_name, f"tau={tau}", kbps, size_bytes, psnr_val, ssim_val, enc_time, dec_time)
            )
            print(f"   [TFC ] tau {tau:.4f}: {kbps:8.1f} kbps | {psnr_val:5.2f} dB PSNR | {ssim_val:.4f} SSIM")

    # Clean up temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    # 3. Compute Bjontegaard Delta metrics
    bd_results = {}
    clips_set = sorted(set(p.clip for p in all_points))

    for clip in clips_set:
        tfc_pts = sorted([p for p in all_points if p.clip == clip and p.codec == "tfc"], key=lambda x: x.rate_kbps)
        x264_pts = sorted([p for p in all_points if p.clip == clip and p.codec == "x264"], key=lambda x: x.rate_kbps)

        if len(tfc_pts) >= 4 and len(x264_pts) >= 4:
            r_tfc = [p.rate_kbps for p in tfc_pts]
            d_tfc = [p.psnr for p in tfc_pts]
            r_x264 = [p.rate_kbps for p in x264_pts]
            d_x264 = [p.psnr for p in x264_pts]

            rate_gain = bd_rate(r_x264, d_x264, r_tfc, d_tfc)
            psnr_gain = bd_psnr(r_x264, d_x264, r_tfc, d_tfc)

            bd_results[clip] = {
                "bd_rate_pct": float(rate_gain),
                "bd_psnr_db": float(psnr_gain),
            }

    # Summary averages
    if bd_results:
        avg_bd_rate = float(np.mean([v["bd_rate_pct"] for v in bd_results.values()]))
        avg_bd_psnr = float(np.mean([v["bd_psnr_db"] for v in bd_results.values()]))
    else:
        avg_bd_rate = 0.0
        avg_bd_psnr = 0.0

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points": [asdict(p) for p in all_points],
        "bd_results": bd_results,
        "average_bd_rate_pct": avg_bd_rate,
        "average_bd_psnr_db": avg_bd_psnr,
    }

    # Write JSON output
    json_path = out_dir / "scc_benchmark_results.json"
    json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # Generate Markdown Report
    md_lines = [
        "# JVET Screen Content Coding (SCC) Benchmark Report",
        f"**Date:** {summary_data['timestamp']}",
        "",
        "## 1. Bjontegaard Delta Summary (TFC vs H.264/libx264)",
        "| Sequence Name | BD-Rate (%) | BD-PSNR (dB) |",
        "|---|---:|---:|",
    ]
    for c, v in bd_results.items():
        md_lines.append(f"| `{c}` | **{v['bd_rate_pct']:+.2f}%** | **{v['bd_psnr_db']:+.2f} dB** |")
    if bd_results:
        md_lines.append(f"| **Overall Average** | **{avg_bd_rate:+.2f}%** | **{avg_bd_psnr:+.2f} dB** |")
    md_lines.extend(
        [
            "",
            "> *Note: Negative BD-Rate indicates bitstream savings conforming to ITU-T VCEG-M33.*",
            "",
            "## 2. Full Rate-Distortion Sampling",
            "| Codec | Clip | Parameter | Rate (kbps) | PSNR (dB) | SSIM | Enc Time (s) | Dec Time (s) |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for p in all_points:
        line_str = (
            f"| {p.codec} | `{p.clip}` | {p.param_label} | {p.rate_kbps:.1f} | {p.psnr:.2f} | "
            f"{p.ssim:.4f} | {p.encode_time_s:.2f} | {p.decode_time_s:.2f} |"
        )
        md_lines.append(line_str)

    (out_dir / "scc_benchmark_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Generate LaTeX Table
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Bjontegaard Delta (BD-Rate & BD-PSNR) of TFC vs. H.264 Anchor (ITU-T VCEG-M33)}",
        r"\label{tab:scc_bd_rates}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Sequence} & \textbf{BD-Rate (\%)} & \textbf{BD-PSNR (dB)} \\",
        r"\midrule",
    ]
    for c, v in bd_results.items():
        esc_c = c.replace("_", r"\_")
        rate_str = f"$\\mathbf{{{v['bd_rate_pct']:+.1f}\\%}}$"
        psnr_str = f"$\\mathbf{{{v['bd_psnr_db']:+.2f}}}$ dB"
        tex_lines.append(f"{esc_c} & {rate_str} & {psnr_str} \\\\")
    if bd_results:
        tex_lines.append(r"\midrule")
        tex_lines.append(
            f"\\textbf{{Average}} & $\\mathbf{{{avg_bd_rate:+.1f}\\%}}$ & "
            f"$\\mathbf{{{avg_bd_psnr:+.2f}}}$ dB \\\\"
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (out_dir / "scc_bd_rate_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    print("\n========================================================")
    print("BENCHMARK SUMMARY (TFC vs x264)")
    print("========================================================")
    for c, v in bd_results.items():
        print(f"  {c:<24}: BD-Rate = {v['bd_rate_pct']:+6.2f}% | BD-PSNR = {v['bd_psnr_db']:+5.2f} dB")
    if bd_results:
        print(f"  {'AVERAGE':<24}: BD-Rate = {avg_bd_rate:+6.2f}% | BD-PSNR = {avg_bd_psnr:+5.2f} dB")
    print(f"Outputs written to: {out_dir}\n")

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standardized JVET-SCC benchmarks for TimeFlowCodec.")
    parser.add_argument("--clips-dir", type=Path, default=REPO_ROOT / "clips" / "scc_benchmarks")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "scc_results")
    parser.add_argument("--quick", action="store_true", help="Run fast 2-point test instead of full 4-point sweep")
    args = parser.parse_args()

    clips = ensure_scc_clips(args.clips_dir)
    run_suite(clips, args.out_dir, quick=args.quick)


if __name__ == "__main__":
    main()
