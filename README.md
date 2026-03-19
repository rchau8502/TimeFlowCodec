# TimeFlowCodec — Per-Pixel Temporal Codec for UI/Animation

Wedge: excels on **UI/animation/temporally coherent content** with per-channel temporal fits (CONST/LINEAR), scene segmentation, tile sharing, and low-rank matrix fallback. Includes runnable codec (CLI/GUI), benchmark harness, and thesis.

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
- `MODE_TFC_MATRIX` – rank-1 low-rank factors for RAW-heavy tiles

### Container Layout (.tfc)
```
[HEADER]
[R PAYLOAD]
[G PAYLOAD]
[B PAYLOAD]
```
- Header (v2 default): width/height/frames, tiling, segment count, color_format=RGB, bits_per_mode=2, payload_comp_type (none/zlib/lzma/zstd).
- Per-plane payload stores segment streams (mode map RLE + quantized CONST/LINEAR + MATRIX + RAW).
- Mode maps are bitpacked and RLE-coded. CONST/LINEAR payloads are quantized (no float32 arrays on disk in v2).

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
Dependencies include `imageio[ffmpeg]` and `zstandard`. Default payload compression is zstd for strong ratio/speed.

## Quickstart (under 2 minutes)
```bash
make install
tfc encode input.mp4 out.tfc --preset anime
tfc decode out.tfc recon.mp4 --fps 30
```
GUI: `python gui.py`

### MacBook quick profile
Use the low-memory, laptop-friendly profile:
```bash
tfc encode input.mp4 out.tfc --preset anime --macbook-profile
```
This applies safer defaults (`tiling=16`, `max_ram_mb=1536`, `scene_cut=auto`, `uint8` internals) and avoids heavy settings that cause lag on Apple Silicon laptops.

Decode with progressive write (default in CLI):
```bash
timeflowdecompress out.tfc recon.mp4 --stream-output
```
Use `--no-stream-output` only for debugging/small clips.

## CLI Usage
Install (editable or wheel), then use short commands:
- Compress: `tfc encode input.mp4 out.tfc --preset anime`
- Decompress: `tfc decode out.tfc recon.mp4 --fps 30`
- Alternative entrypoints: `timeflowcodec encode/decode`, `timeflowcompress`, `timeflowdecompress`
- Presets: `--preset {anime,lownoise,custom}`
- Scene segmentation controls: `--scene-cut {off,auto}` and `--scene-threshold 0.35`
- Matrix low-rank mode (helps RAW-heavy tiles): `--matrix-mode --matrix-tau 0.12 --matrix-rate-ratio 0.95`

Example high-efficiency run:
```bash
tfc encode input.mp4 out.tfc \
  --container-version 2 \
  --preset anime \
  --tiling 16 \
  --scene-cut auto \
  --matrix-mode
```

## GUI
```bash
python gui.py
```
The desktop app includes drag-and-drop inputs, automatic output path suggestions, preset-aware controls, and Apple Silicon auto-detection. On M-series Macs, the GUI enables the safe runtime profile by default.

## Reproduce benchmarks
```bash
# Prepare clips (see clips/README.md)
python -m timeflowcodec.bench --clips clips --out bench_out --codecs tfc,x264,x265,av1 --presets fast,medium --crf 18,23,28 --tfc-preset anime
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
- Strictly RGB; no YUV420 conversion.
- Payload compression: none (0), zlib (1), LZMA (2), zstd (3).
- Mode maps are 2-bit packed + RLE in v2; fallback payload is uint8 and matrix stream is quantized int16 factors.
