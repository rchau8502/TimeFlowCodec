# Distribution readiness

Source status as of 2026-09-26. These changes are development source, not a
signed installer or a package-index release.

## Python package

Use Python 3.11 as the CI baseline. Local verification uses Python 3.13;
the declared Python 3.9 minimum still needs a full compatibility matrix.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[gui]'
tfc encode input.mp4 output.tfc --preset screen
tfc decode output.tfc restored.mp4
timeflowplay output.tfc
timeflowcodec-gui
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. NumPy, imageio-ffmpeg,
zstandard, and (with the GUI extra) PySide6 are dependencies. FFmpeg normally
comes from imageio-ffmpeg; IMAGEIO_FFMPEG_EXE can select another installation.
No GPU, account, or API key is required.

CLI and GUI encodes now use v5, retaining rational source FPS and the first audio
stream. Variable frame rates are rejected. Legacy v1–v4 files remain readable;
use an explicit FPS override for legacy files when their source rate was not 30.
See [container-v5.md](container-v5.md) for limits and the audio override policy.

The encoder window bounds frames per chunk, not total process RAM. The player
prepares an exact RGB disk cache before playback; disk consumption and startup
latency grow with duration. Legacy array decoding still has a decoded-size cap.
Malformed-container checks, checksums, atomic output replacement, cooperative
codec cancellation, and progress reporting are implemented and covered by tests.
They are not a claim of exhaustive hostile-input security review.

## Standalone app and playback

The latest built local app remains the September 23 desktop preview under
`dist/desktop-preview2-2026-09-23/`. It does **not** contain the subsequent v5,
audio, safety, or codec cancellation changes. A fresh app build and native GUI
validation are required before distributing the current source as an installer.

The current macOS spec declares macOS 13.0, matching the inspected Qt dependency,
and selects arm64 on Apple Silicon. Preview apps are ad-hoc signed. Developer ID
signing, notarization, Intel Mac, Windows, Linux, and clean-machine validation
remain outstanding. Include dependency licenses in any distributed bundle.

The app can open `.tfc` files directly and declares a macOS document type.
It does not install a QuickTime, VLC, browser, or FFmpeg codec. Export MP4 for
recipients without TimeFlowCodec. Source checkouts and app bundles include demo
assets; Python wheels currently omit the optional demos.

## Remaining release gates

- Rebuild and test the current wheel and app on clean supported systems.
- Validate audio synchronization through native playback and seeking, including
  platform audio failures; arbitrary source timeline offsets remain unvalidated.
- Complete representative quality/performance review without generalizing the
  recorded duration-scaling or decoder-profile measurements.
- Establish OS, architecture, Python support, and practical disk/RAM requirements.
- Complete dependency notices, signing/notarization, and installer checks before
  advertising a public standalone release.
