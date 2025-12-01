# TimeFlowCodec (Per-Pixel Temporal Function Codec)

TimeFlowCodec is a per-pixel temporal function video codec based on **Pixel Plane Vectorization (PPV)**. Each pixel is modeled independently over time using simple temporal functions and only stores lightweight parameters when the fit is good enough. Pixels that cannot be modeled well fall back to raw samples.

## Idea: Per-Pixel Temporal Modeling
For a grayscale video `V(t, y, x)` with `T` frames:
- Flatten the plane to pixels `p = y * width + x`.
- For each pixel, fit a temporal model to its intensity time series `s_p[t]`:
  - **CONST**: `ŝ[t] = a`
  - **LINEAR**: `ŝ[t] = a + b t`
- Compute error ratio `r = MSE(s, ŝ) / (MSE(s, 0) + eps)`.
  - If `r <= tau`, use a model (CONST if |b| < slope_threshold, else LINEAR).
  - Otherwise store raw per-frame samples (fallback RAW).

### Modes
- `MODE_TFC_CONST` – store `a`
- `MODE_TFC_LINEAR` – store `a, b`
- `MODE_FB_RAW` – store raw uint8 samples

### Where It Shines
- Content with strong temporal redundancy per pixel: animation, UI captures, low-noise footage.
- Less effective on heavy sensor noise or chaotic textures (more pixels fall back to RAW).

## Install
```bash
pip install -r requirements.txt
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
Launch the PySide6 GUI:
```bash
python gui.py
```
Choose input/output paths, tweak `tau`, slope threshold, and compression (None/zlib/LZMA), then start compress or decompress without blocking the UI.

## Build Binaries (PyInstaller)
On Windows:
```bash
build_win.bat
```
On macOS:
```bash
./build_mac.sh
```

## Project Layout
- `timeflowcodec/` – core codec (models, format, encoder/decoder)
- `examples/` – CLI scripts for compression/decompression
- `tests/` – pytest-based unit tests
- `gui.py` – PySide6 GUI wrapper
- `timeflowcodec_gui.spec` – PyInstaller spec for one-folder builds

## Notes
- Codec is grayscale-only; videos are converted to luminance internally.
- Payload compression supports none/zlib/LZMA (default LZMA).
- Mode table is 2 bits per pixel; parameters and fallbacks stored per pixel.
