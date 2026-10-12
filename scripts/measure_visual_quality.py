"""Compare native TFC and H.264 RGB at equal dimensions and timing.

PSNR/SSIM screen for quality changes; human review is still needed to establish
visual indistinguishability. Input media is not licensed for redistribution here.
"""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_showcase import encode_rgb, file_hash, measure_rgb  # noqa: E402

from timeflowcodec.decoder import decode_tfc_to_video  # noqa: E402
from timeflowcodec.encoder import encode_video_to_tfc  # noqa: E402
from timeflowcodec.media import probe_media  # noqa: E402


def main():
    import imageio_ffmpeg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    info = probe_media(args.source)
    if info["has_audio"]:
        raise ValueError("Use a video-only source so audio bytes cannot bias codec comparisons")
    fps = info["fps_num"] / info["fps_den"]
    with imageio.get_reader(args.source) as reader:
        frames = np.array(list(reader.iter_data()))
    preview = len(frames) // 2
    Image.fromarray(frames[preview]).save(args.out / "source-preview.png")
    rows = []

    def measure(label, path, decoded, settings):
        metrics = measure_rgb(frames, decoded)
        metrics["mean_rgb_ssim"] = float(
            np.mean([structural_similarity(a, b, channel_axis=-1, data_range=255) for a, b in zip(frames, decoded)])
        )
        row = dict(method=label, file=path.name, bytes=path.stat().st_size, settings=settings, metrics=metrics)
        rows.append(row)
        Image.fromarray(decoded[preview]).save(args.out / (path.stem + "-preview.png"))
        print(json.dumps(row), flush=True)

    for crf in (12, 18, 23):
        path = args.out / f"h264-rgb-crf{crf}.mp4"
        command, _ = encode_rgb(
            frames,
            path,
            fps,
            ["-c:v", "libx264rgb", "-preset", "veryslow", "-crf", str(crf)],
            imageio_ffmpeg.get_ffmpeg_exe(),
        )
        with imageio.get_reader(path) as reader:
            decoded = np.array(list(reader.iter_data()))
        measure(f"H264 RGB CRF {crf}", path, decoded, command)
    for step in (4, 8, 16):
        path = args.out / f"tfc-q{step}.tfc"
        settings = dict(container_version=5, preset="screen", q_step=step, window=16)
        encode_video_to_tfc(str(args.source), str(path), **settings)
        measure(f"TFC q{step}", path, decode_tfc_to_video(str(path), None), settings)

    comparisons = []
    def psnr(row):
        value = row["metrics"]["rgb_psnr_db"]
        return float("inf") if value is None else value

    for anchor in rows[:3]:
        eligible = [
            row
            for row in rows[3:]
            if psnr(row) >= psnr(anchor)
            and row["metrics"]["mean_rgb_ssim"] >= anchor["metrics"]["mean_rgb_ssim"]
        ]
        winner = min(eligible, key=lambda row: row["bytes"]) if eligible else None
        comparisons.append(
            dict(
                h264=anchor["method"],
                h264_bytes=anchor["bytes"],
                tfc=winner["method"] if winner else None,
                tfc_bytes=winner["bytes"] if winner else None,
                size_saving_percent=100 * (1 - winner["bytes"] / anchor["bytes"]) if winner else None,
                target_60_percent_met=bool(winner and winner["bytes"] <= 0.4 * anchor["bytes"]),
            )
        )
    report = dict(
        source=str(args.source),
        source_sha256=file_hash(args.source),
        width=info["width"],
        height=info["height"],
        frames=len(frames),
        fps=fps,
        rows=rows,
        matched_metric_comparisons=comparisons,
        limits="PSNR/SSIM are screening metrics, not proof of visual indistinguishability. "
        "Input media is not dedicated to CC0 by this script. No rescaling or frame dropping.",
    )
    (args.out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
