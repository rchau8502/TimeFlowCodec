#!/usr/bin/env python3
"""Generate publication-quality vector figures (PDF) and web images (PNG)
for the TimeFlowCodec research thesis and presentation.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Use clean, publication-ready styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "tfc_screen": "#0064A4",   # UCI Anteater Blue
    "tfc_anime": "#17BECF",    # Cyan
    "x264": "#D9534F",         # Coral red
    "x265": "#F0AD4E",         # Amber/Orange
    "av1": "#2CA02C",          # Forest green
    "const": "#1F77B4",        # Blue
    "linear": "#FF7F0E",       # Orange
    "raw": "#D62728",          # Red
    "matrix": "#9467BD",       # Purple
}


def save_fig(fig, base_name: str) -> None:
    if not base_name.startswith("rd_") and base_name != "tfc_architecture_v5":
        fig.text(.5, .005, "ILLUSTRATION — NOT MEASURED CODEC PERFORMANCE",
                 ha="center", color="darkred", fontsize=10, weight="bold")
    pdf_path = FIG_DIR / f"{base_name}.pdf"
    png_path = FIG_DIR / f"{base_name}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated {pdf_path.name} and {png_path.name}")


REPORT_FILE = Path(__file__).resolve().parent.parent / "benchmarks/scc_results/scc_benchmark_results.json"


def load_rd_points(path: Path, sequence: str) -> list[dict]:
    """Read actual saved samples; never invent a rate/quality curve."""
    report = json.loads(path.read_text())
    points = [point for point in report["points"] if point["clip"] == sequence]
    if not points:
        raise ValueError(f"No measured points for {sequence} in {path}")
    for point in points:
        if not math.isfinite(point["rate_kbps"]) or point["rate_kbps"] <= 0:
            raise ValueError("Measured rates must be positive and finite")
    return points


def plot_rd_points(sequence: str, filename: str, title: str) -> None:
    points = load_rd_points(REPORT_FILE, sequence)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for codec in sorted({point["codec"] for point in points}):
        samples = [point for point in points if point["codec"] == codec]
        for ax, metric in zip(axes, ("psnr", "ssim")):
            finite = [point for point in samples if point.get(metric) is not None
                      and math.isfinite(point[metric])]
            if finite:
                ax.scatter([point["rate_kbps"] for point in finite],
                           [point[metric] for point in finite], label=codec)
    for ax, metric in zip(axes, ("RGB PSNR (dB)", "Mean RGB SSIM")):
        ax.set_xlabel("Measured bitrate (kbps)")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()
    fig.suptitle(title + " — saved samples, not matched-quality conclusions")
    fig.text(.5, .01, "Source: " + REPORT_FILE.name + "; perfect PSNR points omitted",
             ha="center", fontsize=8)
    save_fig(fig, filename)


def plot_rd_ui() -> None:
    plot_rd_points("scc_web_browsing", "rd_ui", "Synthetic scrolling UI")


def plot_rd_slides() -> None:
    plot_rd_points("scc_slideshow", "rd_slides", "Synthetic slides")


def plot_rd_anime() -> None:
    plot_rd_points("scc_animation_ui", "rd_anime", "Synthetic UI animation")


def plot_rd_natural() -> None:
    raise ValueError("No measured natural-video corpus is supplied; no curve can be generated")


def plot_temporal_fitting_concept() -> None:
    """Conceptual diagram illustrating per-pixel 1D temporal regression."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    t = np.arange(0, 30)

    # 1. CONSTANT mode
    np.random.seed(42)
    s_const_true = 180 + np.random.normal(0, 1.2, size=len(t))
    s_const_fit = np.full_like(t, fill_value=np.mean(s_const_true), dtype=float)
    axes[0].plot(t, s_const_true, "k.", label="True $s(t)$", alpha=0.7)
    axes[0].plot(t, s_const_fit, "-", color=COLORS["const"], lw=2.5, label="CONST Model: $\\hat{s}=a$")
    axes[0].fill_between(
        t, s_const_fit - 4, s_const_fit + 4, color=COLORS["const"], alpha=0.15, label="Tolerance $\\tau$"
    )
    axes[0].set_title("(a) Static Region $\\rightarrow$ CONST Mode\n(1 parameter stored: $a$)")
    axes[0].set_xlabel("Time Step $t$ (Frames)")
    axes[0].set_ylabel("Pixel Intensity (0-255)")
    axes[0].set_ylim(140, 240)
    axes[0].grid(True)
    axes[0].legend(loc="upper right", fontsize=9)

    # 2. LINEAR mode
    s_lin_true = 150 + 2.2 * t + np.random.normal(0, 1.5, size=len(t))
    p = np.polyfit(t, s_lin_true, 1)
    s_lin_fit = np.polyval(p, t)
    axes[1].plot(t, s_lin_true, "k.", label="True $s(t)$", alpha=0.7)
    axes[1].plot(t, s_lin_fit, "-", color=COLORS["linear"], lw=2.5, label="LINEAR Model: $\\hat{s}=a+bt$")
    axes[1].fill_between(
        t, s_lin_fit - 4, s_lin_fit + 4, color=COLORS["linear"], alpha=0.15, label="Tolerance $\\tau$"
    )
    axes[1].set_title("(b) Smooth Transition $\\rightarrow$ LINEAR Mode\n(2 parameters stored: $a, b$)")
    axes[1].set_xlabel("Time Step $t$ (Frames)")
    axes[1].grid(True)
    axes[1].legend(loc="upper left", fontsize=9)

    # 3. RAW FALLBACK mode
    s_raw_true = 160 + np.sin(t / 2.5) * 25 + np.random.normal(0, 2.0, size=len(t))
    s_raw_true[12:18] += 40
    p_bad = np.polyfit(t, s_raw_true, 1)
    s_bad_fit = np.polyval(p_bad, t)
    axes[2].plot(t, s_raw_true, "k.", label="True $s(t)$", alpha=0.7)
    axes[2].plot(t, s_bad_fit, "--", color="gray", lw=1.5, label="Linear Fit (Exceeds $\\tau$)")
    axes[2].plot(t, s_raw_true, "-", color=COLORS["raw"], lw=1.8, label="RAW Fallback: uint8[$T$]")
    axes[2].set_title("(c) Complex Dynamic $\\rightarrow$ RAW Fallback\n(Store full samples or Matrix)")
    axes[2].set_xlabel("Time Step $t$ (Frames)")
    axes[2].grid(True)
    axes[2].legend(loc="upper left", fontsize=9)

    save_fig(fig, "temporal_fitting_concept")


def plot_visual_artifacts() -> None:
    """Qualitative comparison of text edge reconstruction:
    Ground Truth vs TimeFlowCodec vs Hybrid Block DCT, showing Gibbs mosquito ringing.
    """
    from scipy.fft import dctn, idctn

    H, W = 32, 32
    gt = np.full((H, W, 3), 245.0)

    # Vertical stems of letter 'H'
    gt[4:28, 8:14, :] = 25.0
    gt[14:18, 8:24, :] = 25.0
    gt[4:28, 18:24, :] = 25.0

    # Colored syntax accent (blue dot)
    gt[20:26, 26:30, :] = [30.0, 100.0, 220.0]

    # TimeFlowCodec: perfect 4:4:4 color preservation & 0 spatial transform
    tfc_recon = gt.copy()

    # Block DCT: 8x8 block transform, frequency quantization, 4:2:0 subsampling
    dct_recon = np.zeros_like(gt)
    mask = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if i + j <= 3:
                mask[i, j] = 1.0

    for c in range(3):
        ch = gt[:, :, c].copy()
        if c > 0:
            small = ch.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
            ch = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        recon_ch = np.zeros((H, W))
        for by in range(0, H, 8):
            for bx in range(0, W, 8):
                blk = ch[by:by+8, bx:bx+8]
                c_blk = dctn(blk, norm="ortho")
                c_quant = c_blk * mask
                recon_ch[by:by+8, bx:bx+8] = idctn(c_quant, norm="ortho")
        dct_recon[:, :, c] = np.clip(recon_ch, 0, 255)

    fig = plt.figure(figsize=(13, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.45])

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(gt.astype(np.uint8), interpolation="nearest")
    ax1.set_title("(a) Ground Truth\n(Pristine 4:4:4 RGB)", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(tfc_recon.astype(np.uint8), interpolation="nearest")
    ax2.set_title("(b) TimeFlowCodec\n(Zero Ringing, 4:4:4)", fontsize=11)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(dct_recon.astype(np.uint8), interpolation="nearest")
    ax3.set_title("(c) Block DCT / 4:2:0\n(Gibbs Ringing & Bleed)", fontsize=11)
    ax3.axis("off")

    row_idx = 10
    x_axis = np.arange(W)
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(x_axis, gt[row_idx, :, 0], "k--", lw=1.8, label="Ground Truth (Step Edge)")
    ax4.plot(x_axis, tfc_recon[row_idx, :, 0], "-", color=COLORS["tfc_screen"], lw=2.2, label="TimeFlowCodec (Exact)")
    ax4.plot(
        x_axis, dct_recon[row_idx, :, 0], "-o", color=COLORS["x264"], lw=1.8, ms=4, label="Block DCT (Gibbs Ripples)"
    )
    ax4.axvspan(5.5, 15.5, color="red", alpha=0.08, label="Ringing Zone")
    ax4.set_title("(d) 1D Horizontal Edge Profile", fontsize=11)
    ax4.set_xlabel("Horizontal Pixel Coordinate $x$")
    ax4.set_ylabel("Luminance (0-255)")
    ax4.set_ylim(-15, 285)
    ax4.grid(True)
    ax4.legend(loc="lower right", fontsize=8.5)

    save_fig(fig, "visual_artifacts_crop")


def plot_mode_distribution() -> None:
    """Mode selection distribution across categories."""
    categories = ["Screen / UI", "Slide Decks", "Animation", "Natural Video"]
    const_pct = [74.2, 85.6, 48.0, 18.5]
    linear_pct = [18.5, 9.8, 32.5, 24.2]
    matrix_pct = [3.8, 2.4, 8.2, 14.5]
    raw_pct = [3.5, 2.2, 11.3, 42.8]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_width = 0.55
    indices = np.arange(len(categories))

    ax.bar(indices, const_pct, bar_width, label="CONST Mode ($a$)", color=COLORS["const"])
    ax.bar(indices, linear_pct, bar_width, bottom=const_pct, label="LINEAR Mode ($a, b$)", color=COLORS["linear"])
    ax.bar(
        indices,
        matrix_pct,
        bar_width,
        bottom=np.array(const_pct) + np.array(linear_pct),
        label="MATRIX Mode (Rank-1)",
        color=COLORS["matrix"],
    )
    bottom_raw = np.array(const_pct) + np.array(linear_pct) + np.array(matrix_pct)
    ax.bar(indices, raw_pct, bar_width, bottom=bottom_raw, label="RAW Fallback (uint8[$T$])", color=COLORS["raw"])

    ax.set_ylabel("Mode Percentage (%)")
    ax.set_title("TimeFlowCodec Coding Mode Selection by Video Category")
    ax.set_xticks(indices)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95)

    for i in range(len(categories)):
        if const_pct[i] > 15:
            ax.text(
                i, const_pct[i] / 2, f"{const_pct[i]:.1f}%",
                ha="center", va="center", color="white", fontweight="bold"
            )
        if raw_pct[i] > 10:
            ax.text(
                i, bottom_raw[i] + raw_pct[i] / 2, f"{raw_pct[i]:.1f}%",
                ha="center", va="center", color="white", fontweight="bold"
            )

    save_fig(fig, "mode_distribution")


def plot_tfc_architecture() -> None:
    """System architecture diagram for TimeFlowCodec v5 streaming pipeline."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    box_style_input = dict(boxstyle="round,pad=0.5", facecolor="#E6F2FF", edgecolor="#0064A4", lw=1.8)
    box_style_core = dict(boxstyle="round,pad=0.5", facecolor="#FFF2E6", edgecolor="#FF7F0E", lw=1.8)
    box_style_entropy = dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5", edgecolor="#2CA02C", lw=1.8)
    box_style_out = dict(boxstyle="round,pad=0.5", facecolor="#F5EEF8", edgecolor="#8E44AD", lw=1.8)

    ax.text(
        0.10, 0.75, "Source Video\n$(T \\times H \\times W \\times 3)$\nRGB / YUV",
        ha="center", va="center", bbox=box_style_input, fontsize=10
    )
    ax.text(
        0.32, 0.75, "Chunking Window\n& Scene Boundary\nDetection",
        ha="center", va="center", bbox=box_style_input, fontsize=10
    )

    ax.text(
        0.58, 0.88, "Per-Pixel Temporal\nRegression ($O(T)$ OLS)\nCONST ($a$) / LINEAR ($a,b$)",
        ha="center", va="center", bbox=box_style_core, fontsize=9.5
    )
    ax.text(
        0.58, 0.50,
        "Mode Decision Gate\n$r = \\frac{\\mathrm{MSE}(\\hat{s}, s)}{\\mathrm{Var}(s) + \\epsilon} \\leq \\tau$\n"
        "Slope Check $|b| \\geq \\theta$",
        ha="center", va="center", bbox=box_style_core, fontsize=9.5
    )
    ax.text(
        0.58, 0.15, "Residual Fallback\nRank-1 Matrix / RAW\n(Disk memmap fallback)",
        ha="center", va="center", bbox=box_style_core, fontsize=9.5
    )

    ax.text(
        0.85, 0.75, "Parameter Quantizer\n& 2-bit RLE Mode Map\n+ Zstandard Entropy",
        ha="center", va="center", bbox=box_style_entropy, fontsize=10
    )
    ax.text(
        0.85, 0.25, ".tfc v5 Container\nChunked Records\nAudio + Exact FPS",
        ha="center", va="center", bbox=box_style_out, fontsize=10
    )

    arrow = dict(arrowstyle="->", lw=1.8, color="#333333")
    ax.annotate("", xy=(0.22, 0.75), xytext=(0.18, 0.75), arrowprops=arrow)
    ax.annotate("", xy=(0.46, 0.88), xytext=(0.42, 0.75), arrowprops=arrow)
    ax.annotate("", xy=(0.46, 0.50), xytext=(0.42, 0.75), arrowprops=arrow)
    ax.annotate("", xy=(0.46, 0.15), xytext=(0.42, 0.75), arrowprops=arrow)

    ax.annotate("", xy=(0.73, 0.75), xytext=(0.70, 0.88), arrowprops=arrow)
    ax.annotate("", xy=(0.73, 0.75), xytext=(0.70, 0.50), arrowprops=arrow)
    ax.annotate("", xy=(0.73, 0.35), xytext=(0.70, 0.15), arrowprops=arrow)

    ax.annotate("", xy=(0.85, 0.40), xytext=(0.85, 0.60), arrowprops=arrow)

    ax.set_title("TimeFlowCodec v5 Architecture: Per-Pixel Temporal Streaming Pipeline", fontsize=13, pad=15)
    save_fig(fig, "tfc_architecture_v5")


def plot_complexity_scaling() -> None:
    """Scaling of memory (RAM) and execution time vs video duration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    duration_sec = np.array([2, 5, 10, 20, 30, 60])
    unbounded_ram = np.array([250, 480, 890, 1680, 2450, 4800])
    v5_streaming_ram = np.array([250, 258, 265, 274, 278, 282])

    ax1.plot(duration_sec, unbounded_ram, "r--o", label="Unbounded In-Memory (Legacy v1-v3)")
    ax1.plot(
        duration_sec, v5_streaming_ram, "o-",
        color=COLORS["tfc_screen"], lw=2.5, label="Bounded v5 Chunked Streaming"
    )
    ax1.axhline(y=300, color="gray", linestyle=":", alpha=0.7, label="Soft Fallback Bound (300 MiB)")
    ax1.set_xlabel("Video Duration (Seconds at 1080p, 30 FPS)")
    ax1.set_ylabel("Peak Resident Memory (MiB)")
    ax1.set_title("Memory Scaling vs. Video Duration")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    codecs = ["TFC (Screen)", "TFC (Anime)", "x264 (med)", "x265 (med)", "AV1 (cpu-4)"]
    enc_fps = [28.5, 45.2, 95.0, 18.4, 4.2]
    dec_fps = [240.0, 310.0, 140.0, 85.0, 52.0]

    x = np.arange(len(codecs))
    w = 0.35
    ax2.bar(x - w/2, enc_fps, w, label="Encode FPS", color="#0064A4")
    ax2.bar(x + w/2, dec_fps, w, label="Decode FPS", color="#2CA02C")
    ax2.set_ylabel("Throughput (Frames Per Second)")
    ax2.set_title("Codec Throughput Comparison (1080p)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(codecs, rotation=20, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right")

    save_fig(fig, "complexity_scaling")


def main() -> None:
    global REPORT_FILE, FIG_DIR
    parser = argparse.ArgumentParser(description="Plot saved codec measurements and labeled illustrations")
    parser.add_argument("--results", type=Path, default=REPORT_FILE)
    parser.add_argument("--out", type=Path, default=FIG_DIR)
    parser.add_argument("--include-illustrations", action="store_true")
    args = parser.parse_args()
    REPORT_FILE, FIG_DIR = args.results, args.out
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_rd_ui()
    plot_rd_slides()
    plot_rd_anime()
    plot_tfc_architecture()
    if args.include_illustrations:
        plot_temporal_fitting_concept()
        plot_visual_artifacts()
        plot_mode_distribution()
        plot_complexity_scaling()
    print("Generated saved measurement plots. Legacy PDFs require editorial revision.")


if __name__ == "__main__":
    main()
