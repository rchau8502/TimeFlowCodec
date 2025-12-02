# TimeFlowCodec RGB (Per-Pixel Temporal Function Codec)

TimeFlowCodec RGB is a per-pixel temporal function codec for **RGB video**. Each color channel of each pixel is modeled independently over time using simple temporal functions (constant or linear). Pixels or channels that do not fit well fall back to raw samples.

## Concept
For an RGB video `V(t, y, x, c)` with `T` frames and channels `c ∈ {R, G, B}`:
- Flatten the spatial plane to pixel index `p = y * width + x`.
- For each channel of each pixel, fit two models over time:
  - **CONST**: `ŝ[t] = a`
  - **LINEAR**: `ŝ[t] = a + b t`
- Compute error ratio `r = MSE(s, ŝ_linear) / (MSE(s, 0) + eps)`.
  - If `r <= tau`: choose CONST when `|b| < slope_threshold`, else LINEAR.
  - Otherwise, store RAW per-frame samples for that channel.

### Modes
- `MODE_TFC_CONST` – store `a`
- `MODE_TFC_LINEAR` – store `a, b`
- `MODE_FB_RAW` – store raw uint8 samples per frame

### Container Layout (.tfc)
```
[HEADER]
[R MODE TABLE][R PAYLOAD]
[G MODE TABLE][G PAYLOAD]
[B MODE TABLE][B PAYLOAD]
```
- Header includes width/height/frames, color_format=RGB, bits_per_mode=2, payload_comp_type (none/zlib/lzma).
- Mode tables are 2 bits per pixel per channel, packed row-major.
- Payload per channel has parameter and raw sections (float32 params, uint8 raw samples) and is compressed.

### Best For
Content with strong per-pixel temporal coherence: animation, UI captures, low-noise footage. No spatial blocks are used; everything is per pixel per channel.

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
```bash
python gui.py
```
Use the two tabs to compress/decompress without blocking the UI. Configure tau, slope threshold, and payload compression (None/zlib/LZMA).

## Build Binaries
Windows:
```bash
build_win.bat
```
macOS:
```bash
./build_mac.sh
```

## Project Layout
- `timeflowcodec/` – core RGB codec (models, format, encoder/decoder)
- `examples/` – CLI wrappers
- `tests/` – pytest suite for models and encode/decode
- `gui.py` – PySide6 GUI
- `timeflowcodec_gui.spec` – PyInstaller spec

## Notes
- Strictly RGB; no YUV420 conversion.
- Payload compression options: none (0), zlib (1), LZMA (2).
- Mode tables are 2 bits; parameter payload uses float32; fallback is uint8 per frame.
