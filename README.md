# TimeFlowCodec — Applied Math M.S. Thesis Project

Per-pixel temporal function codec for **RGB video**, developed as an applied mathematics master’s thesis on Pixel Plane Vectorization (PPV) and per-channel temporal regression. Each pixel/channel time series is modeled with constant or linear functions; poorly fit signals fall back to raw storage. The codec demonstrates how simple per-pixel temporal models can compete with block-based approaches on temporally coherent content. This repository is ready for others to clone, install, and run the codec from CLI or GUI.

## Thesis Context
- **Objective:** Study per-pixel temporal regression as a compression primitive and quantify trade-offs between model complexity, error ratios, and payload composition.
- **Method:** For each pixel/channel, fit closed-form least-squares linear models; select CONST vs LINEAR via slope threshold, or RAW when relative error exceeds `tau`.
- **Contribution:** End-to-end reference implementation (encoder, decoder, container format), reproducible tests, CLI/GUI tooling, and PyInstaller build artifacts.

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

## CLI Usage
Compress:
```bash
python examples/tfc_compress.py input.mp4 out.tfc --tau 0.1 --slope-threshold 1e-3 --payload-comp-type 2
```
Decompress:
```bash
python examples/tfc_decompress.py out.tfc recon.mp4 --fps 30
```

## GUI
```bash
python gui.py
```
Two tabs for compression/decompression; configure tau, slope threshold, and payload compression (None/zlib/LZMA).

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

## Notes
- Strictly RGB; no YUV420 conversion.
- Payload compression: none (0), zlib (1), LZMA (2).
- Mode tables are 2 bits; parameter payload uses float32; fallback is uint8 per frame.
