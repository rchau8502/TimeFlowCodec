"""Create original CC0 clips and a reproducible, offline codec comparison.

The hero intentionally favors per-pixel temporal regression. A scrolling control
is always retained. Every exact-quality claim is checked against generated RGB.
No external footage, fonts, music, logos, or artwork are used.
"""

# HTML/document templates intentionally retain long lines.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timeflowcodec.decoder import decode_tfc_to_video  # noqa: E402
from timeflowcodec.encoder import encode_video_to_tfc  # noqa: E402
from timeflowcodec.utils import load_video_rgb  # noqa: E402

# Hand-specified geometric lettering, not a third-party font asset.
GLYPHS = {
    "A": [14, 17, 17, 31, 17, 17, 17],
    "B": [30, 17, 17, 30, 17, 17, 30],
    "C": [14, 17, 16, 16, 16, 17, 14],
    "D": [30, 17, 17, 17, 17, 17, 30],
    "E": [31, 16, 16, 30, 16, 16, 31],
    "F": [31, 16, 16, 30, 16, 16, 16],
    "G": [14, 17, 16, 23, 17, 17, 14],
    "H": [17, 17, 17, 31, 17, 17, 17],
    "I": [31, 4, 4, 4, 4, 4, 31],
    "J": [7, 2, 2, 2, 2, 18, 12],
    "K": [17, 18, 20, 24, 20, 18, 17],
    "L": [16, 16, 16, 16, 16, 16, 31],
    "M": [17, 27, 21, 21, 17, 17, 17],
    "N": [17, 25, 21, 19, 17, 17, 17],
    "O": [14, 17, 17, 17, 17, 17, 14],
    "P": [30, 17, 17, 30, 16, 16, 16],
    "Q": [14, 17, 17, 17, 21, 18, 13],
    "R": [30, 17, 17, 30, 20, 18, 17],
    "S": [15, 16, 16, 14, 1, 1, 30],
    "T": [31, 4, 4, 4, 4, 4, 4],
    "U": [17, 17, 17, 17, 17, 17, 14],
    "V": [17, 17, 17, 17, 17, 10, 4],
    "W": [17, 17, 17, 21, 21, 21, 10],
    "X": [17, 17, 10, 4, 10, 17, 17],
    "Y": [17, 17, 10, 4, 4, 4, 4],
    "Z": [31, 1, 2, 4, 8, 16, 31],
    "0": [14, 17, 19, 21, 25, 17, 14],
    "1": [4, 12, 4, 4, 4, 4, 14],
    "2": [14, 17, 1, 2, 4, 8, 31],
    "3": [30, 1, 1, 14, 1, 1, 30],
    "4": [2, 6, 10, 18, 31, 2, 2],
    "5": [31, 16, 16, 30, 1, 1, 30],
    "6": [14, 16, 16, 30, 17, 17, 14],
    "7": [31, 1, 2, 4, 8, 8, 8],
    "8": [14, 17, 17, 14, 17, 17, 14],
    "9": [14, 17, 17, 15, 1, 1, 14],
    ".": [0, 0, 0, 0, 0, 4, 4],
    ":": [0, 4, 4, 0, 4, 4, 0],
    "-": [0, 0, 0, 31, 0, 0, 0],
    "/": [1, 1, 2, 4, 8, 16, 16],
    "%": [25, 25, 2, 4, 8, 19, 19],
    " ": [0] * 7,
}


def lettering(frame, text, x, y, scale=2, color=(229, 238, 250)):
    for char in text.upper():
        glyph = GLYPHS.get(char, GLYPHS[" "])
        for row, bits in enumerate(glyph):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    frame[y + row * scale : y + (row + 1) * scale, x + col * scale : x + (col + 1) * scale] = color
        x += 6 * scale


def original_frame(scene: str, t: int, width=640, height=360) -> np.ndarray:
    """Deterministic originals; complete RGB frames, no resize or imported assets."""
    if scene not in {"linear_color", "scrolling_cards"}:
        raise ValueError("Unknown original scene")
    frame = np.full((height, width, 3), (12, 20, 32), dtype=np.uint8)
    lettering(frame, "COLOR LAB" if scene == "linear_color" else "SCROLL LAB", 24, 20, 3)
    lettering(frame, "ORIGINAL SYNTHETIC VIDEO", 24, height - 25, 1, (136, 157, 184))
    frame[height - 43 : height - 42, 24 : width - 24] = (36, 53, 74)
    h, w = height - 124, width - 48
    y, x = np.mgrid[:h, :w]
    if scene == "linear_color":
        base = np.stack((50 + (x // 3 % 5) * 20, 60 + (y // 3 % 5) * 20, 70 + ((x + y) // 3 % 5) * 20), axis=-1)
        slope = np.stack((x % 5 - 2, y % 5 - 2, (x + y) % 5 - 2), axis=-1)
        position = t % 32 if t % 32 < 16 else 31 - t % 32
        frame[66 : 66 + h, 24 : 24 + w] = (base + slope * position).astype(np.uint8)
        lettering(frame, "LINEAR COLOR ANIMATION", 270, 28, 1, (116, 230, 190))
    else:
        offset = t * 2
        panel = np.full((h, w, 3), (18, 28, 43), dtype=np.uint8)
        for card in range(12):
            cy = card * 74 - offset
            top, bottom = max(0, cy), min(h, cy + 62)
            if bottom <= top:
                continue
            panel[top:bottom, 0:w] = (28, 43, 60)
            panel[top:bottom, :5] = ((48 + card * 13) % 200, 180, (200 - card * 9) % 220)
            for row in range(3):
                yy = cy + 10 + row * 14
                if 0 <= yy and yy + 3 < h:
                    panel[yy : yy + 3, 22 : 100 + ((card + row) % 4) * 72] = (120 + row * 25, 154 + row * 15, 190)
            if 0 <= cy + 8 and cy + 51 < h:
                panel[cy + 8 : cy + 51, w - 70 : w - 14] = (60 + card * 9, 100 + card * 5, 170)
        frame[66 : 66 + h, 24 : 24 + w] = panel
        lettering(frame, "MOVING SCREEN CONTENT", 270, 28, 1, (240, 180, 112))
    return frame


def measure_rgb(reference, decoded):
    if reference.shape != decoded.shape or reference.ndim != 4 or reference.shape[-1] != 3 or not reference.size:
        raise ValueError("Matching complete RGB frame sequences are required")
    squared_error, max_error = 0.0, 0
    for original, restored in zip(reference, decoded):
        diff = restored.astype(np.int16) - original.astype(np.int16)
        max_error = max(max_error, int(np.max(np.abs(diff))))
        squared_error += float(np.sum(diff.astype(np.float64) ** 2))
    mse = squared_error / reference.size
    return dict(
        exact_rgb=max_error == 0,
        max_rgb_error=max_error,
        rgb_psnr_db=None if mse == 0 else 10 * math.log10(255**2 / mse),
    )


def encode_rgb(frames, destination, fps, options, ffmpeg):
    iterator = iter(frames)
    first = next(iterator)
    command = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{first.shape[1]}x{first.shape[0]}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        *options,
        str(destination),
    ]
    start = time.monotonic()
    # Stream frames so command input does not require another full-clip copy.
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            for frame in itertools.chain((first,), iterator):
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            error = process.stderr.read().decode(errors="replace")
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Encode failed: {error}")
    return command, time.monotonic() - start


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def byte_label(count):
    return f"{count / 1000:,.1f} KB" if count < 1_000_000 else f"{count / 1_000_000:,.2f} MB"


BASELINES = [
    ("h264_rgb_lossless", "H.264 RGB lossless", ".mp4", ["-c:v", "libx264rgb", "-preset", "veryslow", "-crf", "0"]),
    (
        "av1_rgb_lossless",
        "AV1 RGB lossless",
        ".mp4",
        [
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "2",
            "-row-mt",
            "1",
            "-threads",
            "4",
            "-pix_fmt",
            "gbrp",
            "-crf",
            "0",
            "-b:v",
            "0",
            "-aom-params",
            "lossless=1",
        ],
    ),
    (
        "h264_444_crf18",
        "H.264 YUV444 CRF 18",
        ".mp4",
        ["-c:v", "libx264", "-preset", "veryslow", "-pix_fmt", "yuv444p", "-crf", "18"],
    ),
    (
        "h264_420_crf23",
        "H.264 YUV420 CRF 23",
        ".mp4",
        ["-c:v", "libx264", "-preset", "veryslow", "-pix_fmt", "yuv420p", "-crf", "23"],
    ),
    (
        "hevc_444_crf18",
        "HEVC YUV444 CRF 18",
        ".mp4",
        [
            "-c:v",
            "libx265",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv444p",
            "-crf",
            "18",
            "-x265-params",
            "pools=4:log-level=error",
        ],
    ),
]


def run_scene(scene, out, fps, count, ffmpeg):
    directory = out / scene
    directory.mkdir(parents=True, exist_ok=True)
    frames = np.array([original_frame(scene, t) for t in range(count)])
    source = directory / "original_rgb_lossless.mkv"
    command, seconds = encode_rgb(frames, source, fps, ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0"], ffmpeg)
    master_metrics = measure_rgb(frames, load_video_rgb(str(source)))
    if not master_metrics["exact_rgb"]:
        raise RuntimeError("Generated master is not exact RGB")
    entries = [
        dict(
            method="ffv1_master",
            label="FFV1 lossless master",
            file=str(source.relative_to(out)),
            bytes=source.stat().st_size,
            encode_seconds=seconds,
            command=command,
            **master_metrics,
        )
    ]
    target = directory / "native.tfc"
    start = time.monotonic()
    native_options = dict(preset="screen", container_version=5, tau=0, q_step=1, window=16)
    encode_video_to_tfc(str(source), str(target), **native_options)
    elapsed = time.monotonic() - start
    decoded = decode_tfc_to_video(str(target), None)
    entries.append(
        dict(
            method="tfc",
            label="TFC native RGB",
            file=str(target.relative_to(out)),
            bytes=target.stat().st_size,
            encode_seconds=elapsed,
            options=native_options,
            **measure_rgb(frames, decoded),
        )
    )
    if scene == "linear_color" and not entries[-1]["exact_rgb"]:
        raise RuntimeError("The exact-quality showcase must reconstruct every RGB pixel")
    for method, label, suffix, options in BASELINES:
        path = directory / (method + suffix)
        command, seconds = encode_rgb(frames, path, fps, options, ffmpeg)
        comparison = load_video_rgb(str(path))
        metrics = measure_rgb(frames, comparison)
        if "lossless" in method and not metrics["exact_rgb"]:
            raise RuntimeError(f"{label} is not an exact RGB baseline")
        entries.append(
            dict(
                method=method,
                label=label,
                file=str(path.relative_to(out)),
                bytes=path.stat().st_size,
                encode_seconds=seconds,
                command=command,
                **metrics,
            )
        )
        if method == "h264_420_crf23":
            Image.fromarray(comparison[24]).save(directory / "h264_420_frame24.png")
        print(scene, label, byte_label(path.stat().st_size), metrics, flush=True)
        del comparison
    Image.fromarray(frames[24]).save(directory / "original_frame24.png")
    Image.fromarray(decoded[24]).save(directory / "tfc_frame24.png")
    # A browser-friendly proxy is explicitly separate from codec measurements.
    encode_rgb(
        frames,
        directory / "viewing_proxy.mp4",
        fps,
        ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ffmpeg,
    )
    return dict(
        scene=scene,
        width=640,
        height=360,
        fps=fps,
        frames=count,
        duration_seconds=count / fps,
        generated_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
        results=entries,
    )


def write_presentation(out, scenes, ffmpeg):
    hero, control = scenes
    native = next(row for row in hero["results"] if row["method"] == "tfc")
    anchors = [row for row in hero["results"] if row["method"] in {"h264_rgb_lossless", "av1_rgb_lossless"}]

    def presentation_frames():
        for scene in scenes:
            for t in range(scene["frames"]):
                frame = np.full((720, 1280, 3), (10, 17, 28), dtype=np.uint8)
                lettering(frame, "TIMEFLOWCODEC / ORIGINAL TEST", 40, 28, 4)
                lettering(
                    frame, "EXACT RGB PROOF CASE" if scene is hero else "SCROLLING CONTROL", 40, 88, 3, (114, 225, 190)
                )
                clip = original_frame(scene["scene"], t)
                frame[145:505, 40:680] = clip
                lettering(frame, "640 X 360 / 20 FPS / 8 SECONDS", 40, 537, 2, (157, 178, 206))
                rows = [
                    row for row in scene["results"] if row["method"] in {"tfc", "h264_rgb_lossless", "av1_rgb_lossless"}
                ]
                for index, row in enumerate(rows):
                    yy = 158 + index * 100
                    lettering(frame, row["label"], 725, yy, 2)
                    lettering(frame, byte_label(row["bytes"]), 725, yy + 28, 3, (114, 225, 190))
                    lettering(frame, "EXACT RGB" if row["exact_rgb"] else "LOSSY RGB", 725, yy + 62, 1, (157, 178, 206))
                if scene is hero:
                    saving = 100 * (1 - native["bytes"] / anchors[0]["bytes"])
                    lettering(frame, f"{saving:.2f}% SMALLER VS H264 RGB", 40, 599, 3, (114, 225, 190))
                else:
                    lettering(frame, "RESULTS DEPEND ON THE CONTENT", 40, 599, 3, (240, 180, 112))
                lettering(frame, "SYNTHETIC CASE SELECTED FOR TEMPORAL LINEARITY", 40, 659, 2, (157, 178, 206))
                yield frame

    encode_rgb(
        presentation_frames(),
        out / "SHOWCASE.mp4",
        20,
        ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ffmpeg,
    )

    sections = []
    for scene in scenes:
        rows = []
        for row in scene["results"]:
            quality = "Exact RGB — every pixel" if row["exact_rgb"] else f"{row['rgb_psnr_db']:.2f} dB RGB PSNR"
            rows.append(
                f"<tr><td>{html.escape(row['label'])}</td><td>{row['bytes']:,}</td>"
                f'<td>{quality}</td><td><a download href="{row["file"]}">File</a></td></tr>'
            )
        name = scene["scene"]
        title = "01 / Temporal color ramps" if scene is hero else "02 / Scrolling screen control"
        sections.append(f'''<section><h2>{title}</h2><p>640 × 360 · 160 frames · 20 FPS · 8 seconds · no audio</p>
        <video controls loop preload="metadata" src="{name}/viewing_proxy.mp4"></video>
        <p class="caption">Browser viewing proxy only. All measurements use original RGB frames and actual decoded files.</p>
        <table><thead><tr><th>Method</th><th>Total bytes</th><th>Measured quality</th><th>Download</th></tr></thead>
        <tbody>{"".join(rows)}</tbody></table>
        <details><summary>Inspect original, TFC, and H.264 YUV420 frame 24</summary>
        <div class="frames">{"".join(f'<figure><img src="{name}/{file}"><figcaption>{label}</figcaption></figure>' for file, label in [("original_frame24.png", "Original RGB"), ("tfc_frame24.png", "Decoded TFC RGB"), ("h264_420_frame24.png", "Decoded H.264 YUV420 / CRF 23")])}</div>
        <p>These PNGs are actual decoded samples. Lossy settings are context, not matched-quality anchors.</p></details></section>''')
    saving = 100 * (1 - native["bytes"] / anchors[0]["bytes"])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>TFC — Original video proof case</title><style>
    :root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b111b;color:#e5edf8;font:17px/1.55 system-ui,sans-serif}}
    main{{max-width:1080px;margin:auto;padding:48px 24px}}h1{{font-size:clamp(36px,6vw,70px);line-height:1.06;letter-spacing:-.045em;max-width:850px}}
    h2{{font-size:30px;margin-top:0}}.eyebrow{{color:#76e5bd;letter-spacing:.14em;font-size:13px;text-transform:uppercase}}
    .lead{{max-width:790px;font-size:21px}}.metric{{font-size:48px;color:#76e5bd;font-weight:700;margin:28px 0}}
    section{{margin:52px 0;padding-top:30px;border-top:1px solid #2b3a4e}}a{{color:#9bc6ff}}video{{width:100%;max-width:800px;border-radius:10px}}
    table{{border-collapse:collapse;width:100%;font-size:15px;margin:24px 0}}th,td{{text-align:left;border-bottom:1px solid #293548;padding:13px 8px}}
    .caption{{font-size:13px;color:#a5b5cd}}.frames{{display:grid;grid-template-columns:1fr;gap:18px}}figure{{margin:0}}img{{width:100%;image-rendering:pixelated}}
    details{{padding:18px;background:#141f30;border-radius:10px}}summary{{cursor:pointer}}footer{{font-size:14px;color:#a5b5cd}}
    </style><main><div class="eyebrow">TimeFlowCodec · Reproducible proof case · October 8, 2026</div>
    <h1>One kind of video.<br>Every RGB pixel preserved.</h1>
    <p class="lead">An original color-ramp animation demonstrates where native per-pixel temporal modeling can use fewer bytes than conventional lossless video encoding.</p>
    <div class="metric">{saving:.2f}% smaller</div><p>Than <strong>libx264rgb / veryslow / CRF 0</strong> on this exact-RGB synthetic case. Both decodes match every original pixel.</p>
    <p>This animation was intentionally designed with linear per-pixel color changes. It is a favorable proof case, not evidence of universal 60% savings or superiority on camera footage. The scrolling comparison is retained below.</p>
    <p><a download href="SHOWCASE.mp4">Download the 16-second presentation video</a> · <a href="results.json">Raw results and commands</a> · <a href="README.md">Reproduction guide</a> · <a href="ASSET-LICENSE.txt">Media license</a></p>
    {"".join(sections)}<footer><p>Newly generated videos and stills are dedicated under <a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0 1.0</a>.
    No stock footage, music, logos, artwork, or third-party font files were used. Codec software retains its separate MIT license.</p>
    <p>Measurements include complete container bytes. Encoding times in JSON are single-run observations, not speed rankings. MP4 is a container, not a codec. No browser-native TFC decode is claimed.</p></footer></main></html>"""
    (out / "index.html").write_text(document)


def write_share_notes(out, scenes):
    import csv

    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "codec_settings", "container_bytes", "exact_rgb", "max_rgb_error", "rgb_psnr_db"])
        for scene in scenes:
            for result in scene["results"]:
                writer.writerow(
                    [
                        scene["scene"],
                        result["label"],
                        result["bytes"],
                        result["exact_rgb"],
                        result["max_rgb_error"],
                        result["rgb_psnr_db"],
                    ]
                )
    hero = {row["method"]: row for row in scenes[0]["results"]}
    saving_h264 = 100 * (1 - hero["tfc"]["bytes"] / hero["h264_rgb_lossless"]["bytes"])
    saving_av1 = 100 * (1 - hero["tfc"]["bytes"] / hero["av1_rgb_lossless"]["bytes"])
    (out / "PRESENTATION-NOTES.md").write_text(
        "# Suggested showcase wording\n\n"
        f'"On this original 8-second, 640x360 color animation, native TimeFlowCodec preserved every RGB pixel in {hero["tfc"]["bytes"]:,} bytes. '
        f"That is {saving_h264:.2f}% smaller than H.264 RGB with libx264rgb / veryslow / CRF 0, and {saving_av1:.2f}% smaller than AV1 RGB with libaom-av1 / cpu-used 2 / lossless 1. "
        "The animation deliberately uses linear per-pixel color changes, which match TFC's temporal model.\"\n\n"
        "Show the scrolling control next: lossless H.264 uses 39,718 bytes; TFC uses 240,365 bytes. Motion prediction wins there.\n\n"
        "The supported claim is that TFC can outperform these named encoder settings on this specific synthetic content at exact RGB quality. "
        "These results do not support a universal 60% saving, a general codec ranking, or a patent claim. Encoding times are not controlled speed comparisons.\n\n"
        "Share SHOWCASE.mp4 for a quick presentation; share the full folder for original masters, native TFC files, comparison encodes, decoded stills, raw measurements and reproduction source. "
        "Media is original and dedicated under CC0 1.0. Presentation playback uses H.264 for compatibility and is not a live TFC decoder.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "showcase/2026-10-08")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    scenes = [run_scene(scene, out, 20, 160, ffmpeg) for scene in ("linear_color", "scrolling_cards")]
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    result = dict(
        date="2026-10-08",
        ffmpeg=version,
        python=sys.version,
        scenes=scenes,
        methodology="All bytes include containers. Exact RGB requires every pixel to match. No rescaling, truncation, or audio.",
        selection="Original favorable linear animation plus scrolling control; not a representative general-video corpus.",
        code_sha256={
            str(path.relative_to(ROOT)): file_hash(path)
            for path in [Path(__file__), *sorted((ROOT / "timeflowcodec").glob("*.py"))]
        },
    )
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_presentation(out, scenes, ffmpeg)
    write_share_notes(out, scenes)
    (out / "ASSET-LICENSE.txt").write_text(
        "Newly generated videos, images, and presentation assets in this package: CC0 1.0 Universal.\n"
        "To the extent copyright exists, these new media assets are dedicated to the public domain under CC0 1.0.\n"
        "https://creativecommons.org/publicdomain/zero/1.0/\n"
        "No third-party footage, music, logos, artwork, or font files were incorporated.\n"
        "The TimeFlowCodec software and included reproduction code retain their separate MIT license.\n"
    )
    (out / "README.md").write_text(
        "# Original TFC showcase\n\nOpen `index.html` locally or play `SHOWCASE.mp4`. Nothing needs to be uploaded.\n\n"
        "The original masters use FFV1 RGB lossless. Native `.tfc` files use v5, RGB, one pixel per coding cell, window 16, tau 0, residual step 1.\n"
        "H.264 RGB / veryslow / CRF 0 and AV1 RGB / cpu-used 2 / lossless 1 are exact-quality anchors, verified against every generated pixel.\n"
        "Other codec settings are lossy context only. Raw sizes, PSNR, maximum pixel error, commands and hashes are in `results.json`.\n\n"
        "## Reproduce\n\nFrom the included source snapshot:\n\n```sh\npython -m pip install numpy 'imageio[ffmpeg]' zstandard Pillow\n"
        "python scripts/create_showcase.py --out reproduced-showcase\n```\n\n"
        "Clips are 640x360, 160 frames, 20 FPS, 8 seconds each. The presentation is 16 seconds. No audio.\n"
        "The first scene intentionally fits per-pixel linear temporal models. The second scene scrolls cards and must remain in comparisons.\n"
        "These two clips do not establish performance on real camera footage, a standard test corpus, or all MP4 files.\n\n"
        "Viewing proxies and the presentation use ordinary H.264 YUV420 for compatibility; they are not measured TFC files.\n"
        "PNG comparisons are saved directly from decoded RGB. `.tfc` playback requires TimeFlowCodec.\n\n"
        "Generated media: CC0 1.0, see ASSET-LICENSE.txt. Software: MIT, see source/LICENSE.\n"
    )
    source = out / "source"
    (source / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, source / "scripts/create_showcase.py")
    shutil.copytree(
        ROOT / "timeflowcodec",
        source / "timeflowcodec",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "LICENSE", source / "LICENSE")
    files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text("".join(f"{file_hash(path)}  {path.relative_to(out)}\n" for path in files))
    print(f"Created {out}", flush=True)


if __name__ == "__main__":
    main()
