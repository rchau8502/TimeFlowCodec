# Screen detail validation

The new `screen` preset uses full-resolution RGB, one pixel per coding cell,
and a temporal error-ratio threshold of 0.0001. It avoids the spatial averaging
and chroma subsampling used by the anime defaults. It is still lossy, and it
can cost more RAM, encoding time, and storage.

## Reproduction

From the project root, using an environment with the package installed:

```bash
python -m pip install -e ".[bench]"
python examples/generate_screen_clip.py /tmp/tfc-screen-clips/screen_detail.mp4
python -m timeflowcodec.bench --clips /tmp/tfc-screen-clips --out bench_out/screen --codecs tfc,x264,x265,av1 --presets medium --crf 18,23 --tfc-preset screen
python -m timeflowcodec.bench --clips /tmp/tfc-screen-clips --out bench_out/anime --codecs tfc --tfc-preset anime
```

The generator produces 12 frames at 12 FPS, 64×48 pixels, with thin colored
lines and a moving dark block. Input RGB is encoded losslessly with libx264rgb.
The benchmark measures TFC reconstruction directly in memory; no intermediate
MP4 export can contaminate its quality score. Baselines are decoded to RGB
and evaluated with the same metric implementation.

## Observed results

These measurements predate the 2026-09-15 RAW predictor correction. Source hashes
in the saved reports identify the measured implementation; rerun the reproduction
commands for current results.

| Codec | Preset | CRF | Bytes | RGB PSNR (dB) | Mean RGB SSIM |
|---|---|---|---:|---:|---:|
| TFC | anime | — | 293 | 12.4437 | 0.246078 |
| TFC | screen | — | 1,398 | 47.7767 | 0.992778 |
| x264 | medium | 18 | 2,404 | 42.7483 | 0.998033 |
| x264 | medium | 23 | 2,317 | 36.0602 | 0.995335 |
| x265 | medium | 18 | 4,702 | 48.5643 | 0.999387 |
| x265 | medium | 23 | 4,551 | 44.3773 | 0.997881 |
| AV1 | medium | 18 | 1,468 | 77.6338 | 0.999999 |
| AV1 | medium | 23 | 1,516 | 76.3409 | 0.999998 |

Full results, elapsed times, environment, settings, and source hashes:
[screen and baselines](benchmarks/screen/results.json),
[anime](benchmarks/anime/results.json).

These numbers validate detail preservation, not codec superiority. AV1 delivers
much higher fidelity at a similar size here. File sizes include each format's
container overhead, which dominates tiny clips. Baselines use their default
pixel formats; this is an end-to-end RGB comparison, not a controlled comparison
at a fixed internal chroma format. PSNR and SSIM can rank artifacts differently.
AV1 medium maps to libaom `cpu-used=4`; fast and slow map to 6 and 2.

Representative long screen captures, multiple runs, peak memory measurements,
and matched-quality rate–distortion sweeps are needed before broader conclusions.
The benchmark currently buffers full clips and is intended for short samples.

## Regression coverage

Tests cover colored single-pixel lines at odd dimensions, MacBook profile
interaction, array decoding for container versions 1–4, known SSIM values,
frame shape mismatches, CLI/GUI preset availability, requested codec selection,
strict JSON for perfect reconstruction, and a real video benchmark that rejects
any decoded-video export.

The tests also exposed a legacy v1 encoder ordering bug. Its writer now places
all three mode maps before the payloads, matching the existing v1 reader.

Audio/timestamps, adaptive tiles, bounded container chunks, and GUI
cancellation/previews are not implemented by this change.
