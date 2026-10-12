# Native compression fixes — 2026-10-07

The wasteful residual path and disconnected size controls have been corrected.
The original claim of general 60% savings against MP4 remains unachieved. These
changes improve TFC and make its results honest; representative matched-quality
codec superiority still requires further algorithm development and measurement.

## Changes

- v3/v4 writers store integer residuals at scale 1 by default rather than
  expanding their range across 16 bits. Optional residual quantization stores
  a genuine reconstruction step in the existing scale field.
- For compressed planes, the writer tries both fitted temporal predictors and
  zero predictors for fallback samples and selects the smaller actual compressed
  payload. Both use native TFC syntax; no MP4 or standard-codec fallback is added.
  Existing Python and C decoders remain compatible.
- CRF now passes its residual quantization step to the encoder. `--q-step` is
  available explicitly. `--bitrate` drives measured chunk feedback, runs
  sequentially, and checks the final total file size against the requested rate.
- v5 `--max-output-ratio` checks output bytes against input bytes before atomic
  replacement. `.4` requires at least 60% savings. It checks the chosen settings;
  it does not automatically lower quality to force a successful result.
- The GUI defaults to screen quality and no expansion, offers a checked 60%
  savings target or explicit permission for larger outputs, exposes residual
  quantization, and reports measured size changes on success. Failed targets
  preserve any existing destination and remove temporary output.
- Presets use full-resolution coding cells by default. Explicit tile sizes still
  enable lossy averaging. Anime retains YUV420 chroma subsampling and its temporal
  threshold, so it can still lose detail; screen remains the default for fidelity.
- The unsupported README BD-rate table and general decoder-speed guarantee were
  withdrawn. Rate–distortion figures now load saved measurement points instead
  of invented arrays. Other illustrative figures require explicit opt-in and
  are labeled as illustrations. Legacy PDFs and presentation artifacts remain
  unverified drafts, as disclosed in the README.

## Same-source comparison

Source: `dist/production-validation-2026-09-25/2s.mp4`, 70,053 bytes,
48 frames, 320×240, 24 FPS, no audio. RGB PSNR compares reconstruction with decoded
source frames, using direct array decoding, without another MP4 export.

| Settings | Earlier TFC bytes | Corrected TFC bytes | Corrected RGB PSNR |
|---|---:|---:|---:|
| screen, v5, residual step 1 | 3,085,926 | 1,455,828 | 56.60 dB |
| screen, v5, residual step 4 | — | 957,726 | 52.09 dB |
| anime, v5, residual step 1 | 37,782 (8×8 averaging) | 585,290 (full-resolution cells) | 26.97 dB |

The default screen output is **52.8% smaller than the earlier TFC**, with
essentially unchanged measured fidelity (earlier 56.59 dB). It is still
20.8 times the MP4 size. The default GUI rejects this output; selecting
"Allow larger files" makes experimental encoding explicit. None of these
measurements establishes 60% savings against MP4 at equivalent quality.

Temporary reproducible outputs and JSON results: `/tmp/tfc-size-fixed/`.
Measurements, input hash, and source-code hashes are also retained in
`docs/evaluation/tfc-size-fix-2026-10-07.json`.

## Usage

```sh
# Require the promised size reduction at the selected native encoding quality.
tfc encode input.mp4 output.tfc --preset screen --max-output-ratio .4

# Test another residual quality/size tradeoff without lowering resolution.
tfc encode input.mp4 output.tfc --preset screen --q-step 4 --max-output-ratio 1

# Measured feedback and a checked total bitrate, including audio/container bytes.
tfc encode input.mp4 output.tfc --preset screen --bitrate 1500k
```

Targets currently require v5. With a bitrate target, measured feedback determines
the residual step per chunk. CRF and bitrate are mutually exclusive. There is
no promise that an arbitrary file can satisfy an arbitrary budget.

## Validation and app

Regression coverage exercises quantization size/error bounds, full spatial
detail, successful 60% savings on static footage with exact reconstruction,
unreachable budgets preserving destinations, CLI parameter propagation,
GUI defaults/size reporting, old-format fixtures, and Python/C interoperability.
Timing, audio, corruption, cancellation, and player lifecycle tests are retained.

The updated local app build is under `dist/compression-fix-2026-10-07/`.
Earlier apps and DMG installers do not include this fix. Build logs and full
test logs are in `/private/tmp/tfc-fixed-build.log` and
`/private/tmp/tfc-final-tests.log`. Nothing has been uploaded or published.

Full suite: **211 passed, 1 skipped**, with the expected forced-memmap warning.
After the final UI wording correction, 8 relevant desktop checks passed again.
Ruff and whitespace checks pass. The app build completed and its deep signature
check passed. The bundled native GUI was exercised on the original 2-second
clip: it visibly reported 1,455,828 encoded bytes against a 70,053-byte budget
and left no output file. The earlier packaged "compression complete" behavior
is replaced with this measured failure when the target cannot be reached.
The final rebuilt app opened normally outside the execution sandbox, and the
native window showed the corrected "Native compression" title and checked-size
badge. Inside the sandbox, LaunchServices reported a misleading missing-executable
error; direct execution and the normal unsandboxed launch both worked.
