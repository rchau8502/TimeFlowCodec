# Original TFC showcase

Open `index.html` locally or play `SHOWCASE.mp4`. Nothing needs to be uploaded.

The original masters use FFV1 RGB lossless. Native `.tfc` files use v5, RGB, one pixel per coding cell, window 16, tau 0, residual step 1.
H.264 RGB / veryslow / CRF 0 and AV1 RGB / cpu-used 2 / lossless 1 are exact-quality anchors, verified against every generated pixel.
Other codec settings are lossy context only. Raw sizes, PSNR, maximum pixel error, commands and hashes are in `results.json`.

## Reproduce

From the included source snapshot:

```sh
python -m pip install numpy 'imageio[ffmpeg]' zstandard Pillow
python scripts/create_showcase.py --out reproduced-showcase
```

Clips are 640x360, 160 frames, 20 FPS, 8 seconds each. The presentation is 16 seconds. No audio.
The first scene intentionally fits per-pixel linear temporal models. The second scene scrolls cards and must remain in comparisons.
These two clips do not establish performance on real camera footage, a standard test corpus, or all MP4 files.

Viewing proxies and the presentation use ordinary H.264 YUV420 for compatibility; they are not measured TFC files.
PNG comparisons are saved directly from decoded RGB. `.tfc` playback requires TimeFlowCodec.

Generated media: CC0 1.0, see ASSET-LICENSE.txt. Software: MIT, see source/LICENSE.
