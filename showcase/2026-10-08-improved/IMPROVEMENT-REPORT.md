# Native TFC improvement and the 60% target — 2026-10-08

The product target is visually indistinguishable reconstruction at unchanged
resolution and frame rate. Exact RGB preservation remains a stricter benchmark.
An AI discussion is a hypothesis, not measured evidence of savings.
The general 60% saving is still unachieved.

## Implemented improvement

v5 RGB chunks with residual step 1 now compare temporal encoding with exact
frame-order CONST segments. The encoder picks the smaller complete native v3/v4
chunk, including headers. This exposes repeated spatial patterns to Zstandard
without changing the wire format or substituting a standard video codec.
All three planes use consistent segment boundaries. Other colorspaces, averaged
cells and residual quantization policies retain their existing behavior.

The Python decoder keeps single-frame CONST samples in NumPy arrays instead of
creating per-pixel dictionaries. Mode unpacking is vectorized. Validation and
default parser behavior remain intact. Existing C decoders read the new chunk
representation with exact RGB output.

Measured results, with complete native container bytes:

| Input | Previous TFC bytes | Improved TFC bytes | Reduction versus previous TFC |
|---|---:|---:|---:|
| Original 8-second scrolling animation | 240,365 | 124,643 | 48.14% |
| Existing 2-second MP4 validation input | 1,455,828 | 1,181,878 | 18.82% |

Both improved decodes match every source RGB pixel. These reductions compare
against previous TFC files, not H.264. The scrolling animation's H.264 RGB
lossless result remains smaller at 39,718 bytes. The deliberately linear color
animation remains 49,948 TFC bytes. This round does not establish faster encoding;
recorded single-run times were affected by concurrent build and verification work.

Validation: 225 tests passed, one skipped. New v3/v4 chunks decode exactly with
the C reference decoder. The rebuilt desktop app's player decoder reproduces
all 160 scrolling frames exactly. App signature verification passed.

## Visual-quality screening

Source: the existing local 2s.mp4 validation input, 320x240, 48 frames, 24 FPS.
Both encoders receive identical decoded RGB frames. H.264 uses libx264rgb,
veryslow, CRF 12/18/23; TFC uses screen, v5, window 16, q-step 4/8/16.
Sizes include containers. This input is a local diagnostic, outside the original
CC0 media package. PSNR and mean RGB SSIM are screening metrics; human review is
required before describing an encode as visually indistinguishable.

| Encoder settings | Bytes | RGB PSNR (dB) | Mean RGB SSIM |
|---|---:|---:|---:|
| H.264 RGB CRF 12 | 253,060 | 49.05 | 0.996166 |
| H.264 RGB CRF 18 | 162,088 | 44.30 | 0.991683 |
| H.264 RGB CRF 23 | 111,825 | 40.69 | 0.986177 |
| TFC q4 | 957,726 | 52.09 | 0.996993 |
| TFC q8 | 730,339 | 47.95 | 0.993761 |
| TFC q16 | 551,810 | 44.32 | 0.990150 |

Among these tested TFC settings, q8 is the smallest that meets or exceeds both
quality metrics of H.264 RGB CRF 18: 730,339 versus 162,088 bytes, or 4.51 times
the H.264 size. The 60% saving target would require at most 64,835 bytes.
This sparse settings comparison is not a full rate-distortion curve or proof
that no future TFC algorithm can achieve the target.

## Reproduce and review

```sh
python -m pip install scikit-image Pillow
python scripts/measure_visual_quality.py dist/production-validation-2026-09-25/2s.mp4 --out dist/reproduced-visual-quality
```

Raw results: `tfc-visual-quality-2026-10-08.json`. Decoded review stills and all
comparison encodes are in `dist/visual-quality-2026-10-08/`.
The original shareable animations and comparison encodes are in
`showcase/2026-10-08-improved/`, with their media license and source snapshot.

## Why the model still misses the target

Constant/linear per-coordinate models are effective for stable pixels and linear
color changes. Moving detail changes many coordinates nonlinearly and forces
fallback samples. The current motion path searches only a 3x3 neighborhood of
coding cells against the first frame in a segment. That is limited prediction
for moving content. Spatial packing improves entropy compression, but does not
supply the motion prediction and spatial residual models needed to compete on
the tested moving footage. A broad 60% claim requires further algorithm work,
a named baseline, a representative corpus and visual quality validation.
