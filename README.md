# TimeFlowCodec — Per-Pixel Temporal Video Codec

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23377926.svg)](https://doi.org/10.5281/zenodo.23377926)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/rchau8502/TimeFlowCodec/actions/workflows/ci.yml/badge.svg)](https://github.com/rchau8502/TimeFlowCodec/actions/workflows/ci.yml)

**TimeFlowCodec (TFC)** is an experimental native codec for **screen content, UI capture, presentations, and animation**. It fits constant or linear temporal functions ($\hat{s}(t) = a + bt$) per coordinate, with fallback samples for pixels that do not fit. It supports full-resolution RGB, bounded video chunks, and a native C decoder. Compression performance depends on content and reconstruction quality.

## Key Architectural Advantages

- **Zero Gibbs Ringing on Text & UI Boundaries:**
  Eliminates the oscillatory ripples and mosquito noise intrinsic to block DCT/DST transforms on sharp step edges.
- **Native 4:4:4 Full-Chroma Fidelity:**
  Full-resolution color coding without chroma bleed or blurring on thin colored text (e.g. syntax highlighting in IDEs).
- **Native Decoder:**
  Linear evaluation ($a + bt$) maps to unit-stride memory access with zero inter-block dependencies or inverse transform matrix multiplications.
- **Bounded Chunk Memory:**
  v5 limits decoded RGB data per chunk. Working memory also includes per-pixel model statistics and compressed-payload buffers.
- **Standard Modern Codec Controls:**
  TFC-specific **CRF (0–63)**, measured VBR feedback, Lagrangian **Rate-Distortion Optimization (RDO)**, and SMP multi-threading (`--threads N`).

## Publication & Academic Research

The peer-reviewed / preprint research deliverables are permanently archived with DOI [**10.5281/zenodo.23377926**](https://doi.org/10.5281/zenodo.23377926):
- **Preprint Paper (Zenodo):** [doi.org/10.5281/zenodo.23377926](https://doi.org/10.5281/zenodo.23377926) | Direct PDF: [`TimeFlowCodec_Manuscript.pdf`](TimeFlowCodec_Manuscript.pdf) / [`presentation_pdfs/TimeFlowCodec_IEEE_Paper.pdf`](presentation_pdfs/TimeFlowCodec_IEEE_Paper.pdf).
- **Interactive Web Player & Portal:** [rchau8502.github.io/TimeFlowCodec](https://rchau8502.github.io/TimeFlowCodec/)
- **Presentation Slide Deck:** [`presentation_pdfs/TimeFlowCodec_Presentation_Slides.pdf`](presentation_pdfs/TimeFlowCodec_Presentation_Slides.pdf) (20-slide 16:9 widescreen LaTeX Beamer presentation).
- **Printable Presentation Handout:** [`presentation_pdfs/TimeFlowCodec_Presentation_Handout.pdf`](presentation_pdfs/TimeFlowCodec_Presentation_Handout.pdf) (10-page 2-on-1 handout layout for print & review).
- **Master's Thesis Monograph:** [`presentation_pdfs/TimeFlowCodec_Thesis_Monograph.pdf`](presentation_pdfs/TimeFlowCodec_Thesis_Monograph.pdf) (55 pages, mathematical derivations, container v5 specs, and benchmarks).
- **Interactive Defense Presentation:** [`presentation/index.html`](presentation/index.html) (20 slides with embedded interactive 1D Gibbs overshoot & pixel loupe inspector).
- **Defense Script & Q&A:** [`presentation/SPEAKER_NOTES.md`](presentation/SPEAKER_NOTES.md) (Slide-by-slide 19-minute script + committee Q&A defense guide).

## Compression status and measured evidence

The [improved original CC0 showcase](showcase/2026-10-08-improved/index.html) includes a presentation
video, lossless masters, native TFC files, measured H.264/AV1/HEVC comparisons,
and reproduction source. Its deliberately linear animation demonstrates an
exact-RGB advantage; the scrolling control demonstrates a case where H.264 wins.
See [presentation wording and limits](showcase/2026-10-08-improved/PRESENTATION-NOTES.md).

v5 RGB encodes with residual step 1 now compare temporal chunks against exact
frame-order CONST segments and select the smaller complete native chunk.
This exposes repeated spatial patterns to the compressor while preserving
compatibility with existing v3/v4 chunk decoders. Other colorspaces, quantization
steps, and averaged coding cells retain their existing policy.

TFC is an experimental native temporal codec. **A general 60% reduction against
MP4 has not been demonstrated.** MP4 is a container; useful comparisons require
a named video codec and matched reconstruction quality.

The earlier BD-rate superiority table has been withdrawn: its advertised values
are not supported by the saved benchmark results. The legacy thesis, paper,
presentation, and research figures contain unverified performance claims and
illustrative data; they are drafts, not product validation.

See [the size investigation](docs/evaluation/tfc-size-investigation-2026-10-07.md)
and [the actual saved synthetic SCC measurements](benchmarks/scc_results/scc_benchmark_report.md).
Those results vary by clip and quality. Locally generated clips are not a
standard JVET test corpus. No general codec-superiority claim follows from them.

The desktop encoder now checks its size target before publishing a result.
The default requires output no larger than the source; the optional 60% savings
target requires output at most 40% of the input bytes. Unattainable targets fail
without replacing an existing destination. Choose **Allow larger files** for
archival or experimental use. Successful encodes report measured sizes.

## Core Idea
For RGB video `V(t, y, x, c)` (channels `c ∈ {R,G,B}`):
- Flatten spatial plane to pixel index `p = y * width + x`.
- Fit per-channel temporal models:
  - CONST: `ŝ[t] = a`
  - LINEAR: `ŝ[t] = a + b t`
- Error ratio `r = MSE(s, ŝ_linear) / (MSE(s, 0) + eps)`.
  - If `r <= tau`: choose CONST when `|b| < slope_threshold`, else LINEAR.
  - Else: RAW (store uint8 samples).

### Modes
- `MODE_TFC_CONST` – store `a`
- `MODE_TFC_LINEAR` – store `a, b`
- `MODE_FB_RAW` – store raw uint8 per-frame samples
- `MODE_TFC_MATRIX` – rank-1 low-rank factors for RAW-heavy tiles

### Container Layout (.tfc)
```
[HEADER]
[R PAYLOAD]
[G PAYLOAD]
[B PAYLOAD]
```
- Header (v3 default): width/height/frames, tiling, segment count, color format (`rgb`, `yuv444`, `yuv420`), bits_per_mode=2, payload_comp_type (none/zlib/lzma/zstd).
- Per-plane payload stores segment streams (mode map RLE + quantized CONST/LINEAR + RAW). In v3, streams are scan-order implicit and tile-native.
- Mode maps are bitpacked and RLE-coded. CONST/LINEAR payloads are quantized; no float32 arrays are written on disk in modern containers.

### Spatial detail
With tile size 1, modeling is per pixel per channel. Explicitly choosing larger tiles is lossy. With larger tiles, modern
v3/v4 formats store a value per tile, including fallback samples, and expand it
across the tile on decode. This is lossy spatial averaging. Presets now default to one pixel per coding cell. The `screen` preset
avoids chroma subsampling, but temporal modeling and parameter
quantization still make it a lossy preset. Explicit tiling/colorspace overrides
can trade away its detail preservation.

## Install

See [public-release user requirements](docs/public-release-requirements.md) for
Python/GUI requirements, standalone app packaging, and how recipients play `.tfc` files.
```bash
pip install -r requirements.txt
```
Or install as a package (editable for development):
```bash
pip install -e .
# or build a wheel
pip install build
python -m build
```
From PyPI (once published):
```bash
pip install timeflowcodec
```
Dependencies include `imageio[ffmpeg]` and `zstandard`. Default payload compression is zstd for strong ratio/speed.

## Quickstart (under 2 minutes)
```bash
make install
tfc encode input.mp4 out.tfc --preset anime
tfc decode out.tfc recon.mp4 --fps 30
```
For screen recordings:
```bash
tfc encode input.mp4 out.tfc --preset screen
```

GUI: `python gui.py` (select **screen** for sharp text and UI edges).

### Play a .tfc file

Install the desktop dependencies, then open a clip directly:

```bash
pip install ".[gui]"
timeflowplay clip.tfc --fps 30
# Or open the encoder/decoder app:
timeflowcodec-gui clip.tfc
```

The player supports Open/drag-and-drop, Play/Pause, frame seeking, and changing
FPS. It displays decoded RGB directly, without an intermediate MP4. Set the FPS
to match the source: existing v1–v4 files contain neither audio nor timestamps.
The desktop app opens on a **Play** tab. Use **Open .tfc** or **Play demo**;
after compression, **Play result** opens the completed file. Compress and
Decompress settings scroll to keep all controls accessible on smaller screens.

This first player loads the clip before playback and is limited to 64 MiB input
and 256 MiB decoded RGB. Loading runs in a cancellable child process; those limits
are not a guarantee of total memory use or complete malformed-file protection.
Use trusted short clips until the file-safety and streaming roadmap is complete.

The macOS app bundle declares `.tfc` support, so **Open With → TimeFlowCodec** can
launch playback once the app is registered with macOS. Choosing it as the default
app enables double-click playback. This does not install a decoder in other players.

### MacBook quick profile
Use the low-memory, laptop-friendly profile:
```bash
tfc encode input.mp4 out.tfc --preset anime --macbook-profile
```
This applies a soft memory cap, automatic scene cuts, and uint8 frame storage.
Preset coding cells remain full resolution; explicit `--tiling` is a lossy choice.
The `anime` preset also defaults to the compact v3 container and `yuv420`.

Decode with progressive write (default in CLI):
```bash
timeflowdecompress out.tfc recon.mp4 --stream-output
```
Use `--no-stream-output` only for debugging/small clips.

## CLI Usage & Rate Control
Install (editable or wheel), then use standard modern codec commands:
- **CRF Quality Encoding:** `tfc encode input.mp4 out.tfc --crf 23 --tune screen --threads 4`
- **Checked Target Bitrate:** `tfc encode input.mp4 out.tfc --bitrate 1500k --tune screen`
- **Checked 60% Savings Target:** `tfc encode input.mp4 out.tfc --preset screen --max-output-ratio 0.4`
- **Residual Quantization:** `tfc encode input.mp4 out.tfc --preset screen --q-step 4`

Bitrate feedback runs sequentially across v5 chunks, including residual quantization.
The final total size, including audio/container overhead, must meet the target
or the destination is preserved and an error is returned. CRF is TFC-specific:
equal CRF numbers across codecs are not equal quality. A larger residual step
trades fidelity for size; it does not lower spatial resolution.
- **Rate-Distortion Optimization (RDO):** `tfc encode input.mp4 out.tfc --crf 18 --rdo`
- **Parallel Chunked Decode:** `tfc decode out.tfc recon.mp4 --threads 4`
- **UNIX Streaming Pipeline:** `tfc decode out.tfc - | ffplay -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -`

Supported flags:
- `--crf <0-63>`: TFC-specific quality mapping; omitted by default (residual step 1).
- `--bitrate <rate>`: Target bitrate (e.g. `1500k`, `2M`) using virtual leaky-bucket rate controller.
- `--tune {screen,anime,natural}`: Content-aware tuning (screen = zero halo RGB; anime = yuv420; natural = block motion comp).
- `--rdo`: Evaluates Lagrangian rate-distortion cost $J = D + \lambda R$ per tile.
- `--threads <N>`: Worker thread pool count (0 = auto-detect all available CPU cores).
- `--container-version 5`: Production bounded multi-chunk container with exact timing & audio.

## Native ANSI C Reference Decoder (`libtfc`)

TimeFlowCodec includes a standalone, ANSI C99 reference decoder core with zlib/zstd support:
```bash
# Build libtfc.a and standalone CLI player tfc_dec
make -C libtfc
```

### Decoding & Benchmarking
```bash
# Decode .tfc file directly to raw RGB24 or PPM image
libtfc/build/tfc_dec out.tfc frame0.ppm

# Measure wall-clock decode throughput and FPS
libtfc/build/tfc_dec out.tfc --bench
```
Measure throughput on the actual clip and hardware. The earlier >840 FPS headline
is withdrawn pending reproducible validation; it is not a general playback guarantee.

## GUI
```bash
python gui.py
```
The desktop app includes drag-and-drop inputs, automatic output path suggestions, preset-aware controls, and Apple Silicon auto-detection. On M-series Macs, the GUI enables the safe runtime profile by default.
The latest UI keeps the common path minimal: presets and file selection stay up front, while advanced rate/distortion controls live behind an explicit toggle.

## Reproduce benchmarks
Install the benchmark dependencies first with `pip install -e ".[bench]"`.
```bash
# Prepare clips (see clips/README.md)
python -m timeflowcodec.bench --clips clips --out bench_out --codecs tfc,x264,x265,av1 --presets fast,medium --crf 18,23,28 --tfc-preset anime
```
Outputs: `bench_out/results.json`, `bench_out/report.md`, `bench_out/plots/`.
Metrics compare decoded RGB arrays directly, without exporting another MP4.
SSIM is a per-frame windowed RGB calculation from scikit-image, averaged over
frames; it is not a luminance-only metric. Decoding time excludes video export.
Reports record command settings, source hashes, and whether the checkout is dirty.
Perfect reconstruction uses `psnr: null` with `lossless_reconstruction: true` in
JSON (and infinity in the Markdown report).

The benchmark uses short clips buffered in RAM. The encoder's `--window` option
is currently ignored, and its RAM setting is a soft fallback-buffer limit, not a
process memory guarantee. Streaming decode still loads compressed payloads and
may reconstruct a whole segment. Long-clip bounded-memory processing is future work.

For Python analysis, `decode_tfc_to_video("out.tfc", None)` returns the decoded
`uint8` RGB array shaped `(frames, height, width, 3)`; this buffers the whole clip.

## Run Tests
```bash
pytest
```
Synthetic round-trip and model-fitting tests validate the codec behavior.

## Build Binaries
Windows:
```bash
build_win.bat
```
macOS:
```bash
./build_mac.sh
```
This produces `dist/TimeFlowCodec.app` and `dist/TimeFlowCodec_macbook_installer.dmg`.

## Releases
- Package metadata is defined in `pyproject.toml` (`timeflowcodec` version `2.0.0`).
- To publish artifacts, build with `python -m build` and upload your wheel/sdist to your chosen index (e.g., `twine upload dist/*`).

## Project Layout
- `timeflowcodec/` – core codec engine (models, container v5, encoder/decoder, CLI)
- `libtfc/` – ANSI C99 and WebAssembly high-throughput reference decoder
- `web/` – in-browser HTML5 `<canvas>` WebAssembly player
- `presentation_pdfs/` – compiled presentation slides, handouts, monograph, and paper
- `gui.py` – PySide6 desktop GUI application
- `timeflowcodec_gui.spec` – PyInstaller spec for standalone desktop application
- `main.tex`, `chapters/`, `appendices/`, `figures/` – LaTeX thesis monograph materials
- `paper/` – IEEE conference manuscript materials
- `tests/` – comprehensive pytest test suite

## Citation

If you use TimeFlowCodec in your research or system, please cite the permanent Zenodo preprint:

```bibtex
@article{chau2026timeflowcodec,
  title     = {TimeFlowCodec: An Orthogonal Video Coding Architecture for Screen and Synthetic Content via Per-Pixel Temporal Regression},
  author    = {Chau, Rich},
  year      = {2026},
  month     = {oct},
  doi       = {10.5281/zenodo.23377926},
  url       = {https://doi.org/10.5281/zenodo.23377926},
  publisher = {Zenodo}
}
```

## License & Copyright

**Copyright (c) 2025–2026 Rich Chau. All rights reserved.**

This project uses a modular dual-licensing structure (see [`LICENSE`](LICENSE)):
- **Core Codec & Reference Decoders:** The ANSI C99 library (`libtfc/`), WebAssembly player (`web/`), and core Python codec/CLI tools (`timeflowcodec/`) are licensed under the open-source **MIT License** for broad interoperability.
- **Standalone Desktop Application:** The graphical application (`TimeFlowCodec.app`, `gui.py`, `timeflowcodec/player.py`) and branding are governed by the **Proprietary Desktop Application License** (free for personal and evaluation use; commercial redistribution and packaging reserved).
- **Publications & Research:** Thesis monograph, conference paper, and presentation slide decks are Copyright (c) 2025–2026 Rich Chau. All rights reserved.
