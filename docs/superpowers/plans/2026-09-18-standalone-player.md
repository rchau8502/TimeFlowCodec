# Standalone player implementation plan

**Goal:** Open and play .tfc files through an installable desktop launcher.
**Architecture:** Isolated decoder process -> private .npy cache -> mapped RGB
frames rendered by Qt; existing encoder/decoder container layouts stay unchanged.
**Tech stack:** Python, NumPy, PySide6, setuptools, PyInstaller.

- [x] Add tests/test_player.py: real subprocess load with synthetic v3 file; exact
  pixels; seek, pause, FPS and replay; invalid input and decoded size rejection;
  cancellation/reopen/close cleanup; macOS file-open dispatch.
- [x] Run `.venv/bin/python -m pytest tests/test_player.py -q` to capture failures.
- [x] Implement timeflowcodec/player_decode.py with size preflight and .npy output.
- [x] Implement timeflowcodec/player.py: QProcess lifecycle, private cache cleanup,
  QImage rendering, monotonic playback, seek/FPS, drag/drop, file-open events.
- [x] Add GUI menu integration and startup worker/file handling in gui.py. Package
  root gui.py and both GUI launchers in pyproject.toml. Register the macOS document
  type and conditional BUNDLE in timeflowcodec_gui.spec.
- [x] Run player tests, codec regression suite, targeted Ruff and diff checks.
- [x] Build wheel, inspect contents and run installed launchers outside checkout.
  Build/smoke local app if PyInstaller is installed; do not publish or sign.
- [x] Update README, release requirements, and progress with verified capabilities
  and remaining limits. Leave changes local.

## Verification

33 player tests passed across v1-v4; the implemented-feature suite had 90 passed
and one skipped before eight extra lifecycle cases were added. Known unfinished
media/safety/GUI-job test files were excluded from that broader run. Ruff and
diff whitespace checks pass. Wheel launchers and a real decode/render/seek/cleanup
flow were verified from an isolated target install outside the checkout.

Built and smoke-tested the Apple Silicon macOS directory bundle, including exact
RGB comparison from the bundled decoder, native FPS/seek controls, document-type
metadata, and ad-hoc signature integrity. It is not Developer ID signed/notarized.
Artifacts: dist/player-preview-2026-09-18/. No public upload.
