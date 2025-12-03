# TimeFlowCodec — Per-Pixel Temporal Codec for UI/Animation

Wedge: excels on **UI/animation/temporally coherent content** with per-channel temporal fits (CONST/LINEAR) and raw fallback. Transparent math, easy parallelism. Includes runnable codec (CLI/GUI), benchmark harness, and thesis.

## Scoreboard (regen with `make bench`)
| codec | preset | crf | clip | size (bytes) | psnr | ssim |
|---|---|---|---|---|---|---|
| (run `python -m timeflowcodec.bench --clips clips --out bench_out --codecs tfc,x264 --presets medium --crf 23` to populate) |

## Where it wins / fails
- Wins: UI captures, slides, animation, low-noise static cams (high temporal coherence).
- Fails: heavy noise, high-motion camera, complex textures (many pixels fall back to raw).
- Transparent: per-pixel regression, no motion search, no chroma subsampling.

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

### Container Layout (.tfc)
```
[HEADER]
[R MODE TABLE][R PAYLOAD]
[G MODE TABLE][G PAYLOAD]
[B MODE TABLE][B PAYLOAD]
```
- Header: width/height/frames, color_format=RGB, bits_per_mode=2, payload_comp_type (none/zlib/lzma).
- Mode tables: 2 bits per pixel per channel, row-major.
- Payloads: params (float32) + raw sections (uint8), compressed per channel.

### Best For
Temporal coherence (animation, UI captures, low-noise footage). No spatial blocking; all modeling is per pixel per channel.

## Install
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
Dependencies include `imageio[ffmpeg]`, which bundles a static ffmpeg backend so video I/O works without extra manual installs. Default payload compression is zlib for speed; choose LZMA if you prefer maximum compression over runtime.

## Quickstart (under 2 minutes)
```bash
make install
timeflowcompress input.mp4 out.tfc --tau 0.1 --slope-threshold 1e-3 --payload-comp-type 1
timeflowdecompress out.tfc recon.mp4 --fps 30
```
GUI: `python gui.py`

## CLI Usage
Install (editable or wheel), then use short commands:
- Compress: `timeflowcompress input.mp4 out.tfc --tau 0.1 --slope-threshold 1e-3 --payload-comp-type 1`
- Decompress: `timeflowdecompress out.tfc recon.mp4 --fps 30`
- Subcommands via dispatcher: `timeflowcodec compress ...` or `timeflowcodec decompress ...`

## GUI
```bash
python gui.py
```
Two tabs for compression/decompression; configure tau, slope threshold, and payload compression (None/zlib/LZMA).

## Reproduce benchmarks
```bash
# Prepare clips (see clips/README.md)
python -m timeflowcodec.bench --clips clips --out bench_out --codecs tfc,x264,x265,av1 --presets fast,medium --crf 18,23,28
```
Outputs: `bench_out/results.json`, `bench_out/report.md`, `bench_out/plots/`.

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

## Releases
- Package metadata is defined in `pyproject.toml` (`timeflowcodec` version `0.1.0`).
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
- Strictly RGB; no YUV420 conversion.
- Payload compression: none (0), zlib (1), LZMA (2).
- Mode tables are 2 bits; parameter payload uses float32; fallback is uint8 per frame.
