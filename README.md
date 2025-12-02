# TimeFlowCodec

This repo hosts the RGB per-pixel TimeFlowCodec reference implementation.

## Projects
- `TimeFlowCodecRGB/` – RGB per-pixel temporal function codec (per-channel CONST/LINEAR fits with raw fallback), CLI tools, GUI, tests, and PyInstaller specs.

## Quick Start (RGB)
```bash
cd TimeFlowCodecRGB
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/tfc_compress.py input.mp4 out.tfc --tau 0.1 --slope-threshold 1e-3 --payload-comp-type 2
python examples/tfc_decompress.py out.tfc recon.mp4 --fps 30
```
GUI:
```bash
cd TimeFlowCodecRGB
python gui.py
```

## License
MIT License. See `LICENSE`.
