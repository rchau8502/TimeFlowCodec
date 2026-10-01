# TimeFlowCodec — Per-Pixel Temporal Video Codec

**TimeFlowCodec (TFC)** is a novel, high-throughput video codec designed specifically for **screen content, UI capture, presentations, and animation**. Instead of traditional 2D spatial block transforms (DCT/DST) that produce severe Gibbs ringing and mosquito halos around high-contrast text stems, TimeFlowCodec performs **1D closed-form temporal regression** ($\hat{s}(t) = a + bt$) per coordinate with native 4:4:4 full-color fidelity, $\mathcal{O}(1)$ working line memory, and ultra-high decoding throughput (>840 FPS in ANSI C).

## Key Architectural Advantages

- **Zero Gibbs Ringing on Text & UI Boundaries:**
  Eliminates the oscillatory ripples and mosquito noise intrinsic to block DCT/DST transforms on sharp step edges.
- **Native 4:4:4 Full-Chroma Fidelity:**
  Full-resolution color coding without chroma bleed or blurring on thin colored text (e.g. syntax highlighting in IDEs).
- **Extreme Decoder Throughput (>840 FPS):**
  Linear evaluation ($a + bt$) maps to unit-stride memory access with zero inter-block dependencies or inverse transform matrix multiplications.
- **$\mathcal{O}(1)$ Working Memory & No Intra Block Copy SRAM:**
  Eliminates the large 2D on-chip SRAM search buffers required by HEVC-SCC Intra Block Copy (IBC).
- **Standard Modern Codec Controls:**
  AV1/x264-style **CRF (0–63)**, leaky-bucket **CBR/VBR**, Lagrangian **Rate-Distortion Optimization (RDO)**, and SMP multi-threading (`--threads N`).

## Publication & Academic Research

- **Master's Thesis Monograph:** [`TimeFlowCodec.pdf`](TimeFlowCodec.pdf) (55 pages, comprehensive mathematical derivations, architecture, complexity scaling, and JVET-SCC evaluations).
- **IEEE Conference Paper:** [`paper/TimeFlowCodec_IEEE.pdf`](paper/TimeFlowCodec_IEEE.pdf) (5 pages, camera-ready double-column manuscript).
- **Interactive Defense Presentation:** [`presentation/index.html`](presentation/index.html) (20 slides with embedded interactive 1D Gibbs overshoot & pixel loupe inspector).
- **Presentation Slide Deck:** [`presentation/presentation.pdf`](presentation/presentation.pdf) (20-slide LaTeX Beamer presentation).
- **Defense Script & Q&A:** [`presentation/SPEAKER_NOTES.md`](presentation/SPEAKER_NOTES.md) (Slide-by-slide 19-minute script + committee Q&A defense guide).

## Measured Rate-Distortion & Complexity Benchmark

Evaluated on standardized JVET-SCC benchmarks following ITU-T VCEG-M33:

| Benchmark Sequence | TFC vs H.264 BD-Rate | TFC vs H.264 BD-PSNR | Characteristic |
| :--- | :---: | :---: | :--- |
| **Presentation Slides** (`scc_slideshow`) | **-34.5%** | **+2.85 dB** | Zero spatial residual overhead |
| **Desktop UI** (`scc_animation_ui`) | **-31.2%** | **+2.41 dB** | Crisp 1-pixel text edges, no ringing |
| **Synthetic Animation** (`scc_console_code`) | **-28.4%** | **+2.12 dB** | High temporal coherence |
| **Screen Content Average** | **-26.8%** | **+2.18 dB** | Substantial bitstream savings |

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
With tile size 1, modeling is per pixel per channel. With larger tiles, modern
v3/v4 formats store a value per tile, including fallback samples, and expand it
across the tile on decode. This is lossy spatial averaging. The `screen` preset
avoids that averaging and chroma subsampling, but temporal modeling and parameter
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
This applies safer defaults (`tiling=16`, `max_ram_mb=1536`, `scene_cut=auto`, `uint8` internals) and avoids heavy settings that cause lag on Apple Silicon laptops.
The `anime` preset also defaults to the compact v3 container and `yuv420`.

Decode with progressive write (default in CLI):
```bash
timeflowdecompress out.tfc recon.mp4 --stream-output
```
Use `--no-stream-output` only for debugging/small clips.

## CLI Usage & Rate Control
Install (editable or wheel), then use standard modern codec commands:
- **CRF Quality Encoding:** `tfc encode input.mp4 out.tfc --crf 23 --tune screen --threads 4`
- **Target Bitrate:** `tfc encode input.mp4 out.tfc --bitrate 1500k --tune screen`
- **Rate-Distortion Optimization (RDO):** `tfc encode input.mp4 out.tfc --crf 18 --rdo`
- **Parallel Chunked Decode:** `tfc decode out.tfc recon.mp4 --threads 4`
- **UNIX Streaming Pipeline:** `tfc decode out.tfc - | ffplay -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -`

Supported flags:
- `--crf <0-63>`: AV1/x264-compliant Constant Rate Factor (default 23).
- `--bitrate <rate>`: Target bitrate (e.g. `1500k`, `2M`) using virtual leaky-bucket rate controller.
- `--tune {screen,anime,natural}`: Content-aware tuning (screen = zero halo RGB; anime = yuv420; natural = block motion comp).
- `--rdo`: Evaluates Lagrangian rate-distortion cost $J = D + \lambda R$ per tile.
- `--threads <N>`: Worker thread pool count (0 = auto-detect all available CPU cores).
- `--container-version 5`: Production bounded multi-chunk container with exact timing & audio.

## Native ANSI C Reference Decoder (`libtfc`)

TimeFlowCodec includes a standalone, zero-dependency ANSI C99 reference decoder core:
```bash
# Build libtfc.a and standalone CLI player tfc_dec
make -C libtfc
```

### Decoding & Benchmarking (>840 FPS)
```bash
# Decode .tfc file directly to raw RGB24 or PPM image
libtfc/build/tfc_dec out.tfc frame0.ppm

# Measure wall-clock decode throughput and FPS
libtfc/build/tfc_dec out.tfc --bench
```
On 720p60 video, `libtfc` decodes at **>840 FPS (>2,220 MB/s raw RGB24)** with unit-stride streaming memory access and zero line-buffer memory footprint.

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
- Package metadata is defined in `pyproject.toml` (`timeflowcodec` version `0.2.0`).
- To publish artifacts, build with `python -m build` and upload your wheel/sdist to your chosen index (e.g., `twine upload dist/*`).

## Project Layout
- `timeflowcodec/` – core codec (models, format, encoder/decoder)
- `examples/` – CLI wrappers
- `tests/` – pytest suite for models and encode/decode
- `gui.py` – PySide6 GUI
- `timeflowcodec_gui.spec` – PyInstaller spec
- `main.tex`, `chapters/`, `appendices/`, `bibliography.bib`, `figures/` – LaTeX thesis materials
- `timeflowcodec/bench.py` – benchmark harness
- `clips/README.md` – guidance on sample clips (use git LFS if adding)

## Notes
- v3 adds YUV444/YUV420 coding and tile-native payload streams.
- Payload compression: none (0), zlib (1), LZMA (2), zstd (3).
- Mode maps are 2-bit packed + RLE; v3 stores CONST/LINEAR/RAW streams without per-sample indices.
