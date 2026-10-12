# TFC file-size investigation — 2026-10-07

The current implementation does not substantiate a general 60% size reduction
against MP4. Larger files are reproducible. MP4 is a container; the meaningful
comparison is against its video codec at comparable decoded quality, resolution,
frame count, duration, and audio handling.

## Current-code reproduction

Source: `dist/production-validation-2026-09-25/2s.mp4`, 320×240,
48 frames at 24 FPS, no audio, 70,053 bytes. Encoded with the repository's
`.venv/bin/python`, `encode_video_to_tfc`, and otherwise default options.
Quality was measured against decoded source RGB using array output from
`decode_tfc_to_video`, without another MP4 export.

| Preset | Container | Output bytes | Output / input | RGB PSNR |
|---|---|---:|---:|---:|
| screen | v3 | 3,041,057 | 43.41× | 55.30 dB |
| screen | v5 (desktop path) | 3,085,926 | 44.05× | 56.59 dB |
| anime | v5 | 37,782 | 0.539× | 15.04 dB |

A 60% reduction would require 28,021 bytes or less for this source. The anime
result reduces size by 46.1%, but loses substantial fidelity; it is not evidence
of equivalent-quality compression efficiency. PSNR here measures extra damage
relative to the decoded input MP4, not quality relative to a pristine original.

Temporary encodes and JSON measurements are in `/tmp/tfc-size-investigation/`.
The screen v5 output also exactly reproduces the size of the existing saved
`2s.tfc` validation artifact.

## Causes established in the implementation

1. **Expensive fallback data.** The screen preset uses full-resolution RGB,
   one coding cell per pixel, and a strict temporal threshold of 0.0001
   (`timeflowcodec/encoder.py:113`). Pixels that fail constant/linear fitting
   retain a temporal predictor plus residual samples. Residuals are serialized
   as signed 16-bit values and compressed with zstd
   (`timeflowcodec/format.py:625`). On the v5 reproduction, 35.63% of coding
   cells use fallback, producing 3,940,064 residual samples. There is no spatial
   transform stage or default motion compensation in this preset to compact
   moving detail as effectively as a conventional video codec.
2. **This is a re-encode of decoded pixels.** TFC does not further compress the
   original MP4 byte stream. It decodes it and attempts to compress the resulting
   frames with a different model. Compression of raw RGB does not imply a
   smaller result than the already compressed input.
3. **Short independent chunks.** The GUI requests container v5
   (`gui.py:959`). Its default window is 16 frames, further capped by 16 MiB
   of uncompressed RGB per chunk (`timeflowcodec/chunked.py:149`). This caps
   chunks at six frames for 720p, two for 1080p, and one for 4K. Each chunk is
   encoded independently, limiting temporal reuse. On this 320×240 sample,
   however, v5 is only 1.5% larger than v3: chunking is not the main cause of
   its 44× expansion.
4. **Container overhead is negligible here.** The screen v5 file contains
   3,085,555 bytes of compressed plane payload and only 371 bytes outside those
   payloads. Changing the extension or shortening metadata cannot resolve it.
5. **Smaller anime outputs trade away spatial detail.** The preset defaults to
   8×8 tile averaging and YUV420 (`timeflowcodec/encoder.py:129`). Its small
   output should not be presented as the same quality as the screen preset.
6. **Advertised rate controls are incompletely connected.** CLI `--bitrate`
   merely maps the requested rate to a heuristic temporal threshold; it does
   not instantiate the feedback `RateController`. CRF computes a residual
   `q_step`, but the CLI does not pass it to the encoder, which has no such
   argument (`timeflowcodec/cli.py:155`). Neither path enforces a file-size
   target. The GUI also passes neither a target bitrate nor RDO.

## The advertised savings are unsupported by the saved evidence

No general 60% MP4 reduction promise was located in the inspected source/docs.
The README does advertise a measured average H.264 BD-rate saving of 26.8%
(`README.md:26`). Its research plots contain hard-coded bitrate/quality arrays
instead of reading benchmark output (`scripts/generate_research_figures.py:53`).
Those plots do not establish measured performance.

The actual saved SCC report
(`benchmarks/scc_results/scc_benchmark_report.md`) has an empty BD-rate summary.
Its point results are mixed: at x264 CRF 23 versus TFC tau 0.005, slides are
76.8 versus 22.4 kbps, scrolling pages 691.1 versus 40.7 kbps, animation
252.0 versus 50.9 kbps, and the static console 29.3 versus 45.3 kbps.
Quality differs between these points, so these ratios cannot establish
matched-quality superiority. The clips are locally generated synthetic samples,
not evidence of testing an independently sourced standard JVET corpus.

## Necessary next work

- Replace unsupported performance claims and illustrative research curves with
  reproducible measured results.
- Connect residual quantization and measured bitrate feedback to the encoding
  path; use an explicit distortion budget rather than achieving size reduction
  through large tile averaging.
- Improve residual coding and spatial/motion prediction, and evaluate reuse
  across chunk boundaries while retaining bounded memory.
- Benchmark real screen recordings, animation, and camera footage at matched
  quality, including the actual desktop v5 path and representative resolutions.

This investigation adds a report only. Codec behavior and existing media files
were not modified. `demo_assets/demo_input.tfc` was already modified before the
investigation.
