# Screen detail and benchmark accuracy

Approved scope: prioritize sharp screen recordings and credible compression measurements.

Design: reuse the v3 RGB per-pixel path with a screen preset (tile size 1,
strict temporal tolerance). Retain existing formats and local motion-compensation
work. Expose decoded RGB arrays through the existing decoder by allowing a null
output path; this explicitly buffers frames and is intended for short benchmarks.
Use identical RGB metrics for TFC and baseline decodes, and time decoding without
video export. Calculate per-frame windowed RGB SSIM via scikit-image.

Implementation and validation:
1. Add regression tests for colored single-pixel lines, odd dimensions, MacBook
   defaults, direct decoding of versions 1–4, and known SSIM values.
2. Add screen preset in encoder, CLI, GUI; retain explicit parameter overrides.
3. Add array output, fix benchmark metrics and codec selection, record settings.
4. Run regression suite and a small reproducible synthetic screen benchmark.
5. Document measured results and limitations. No claim of general codec superiority.

Deferred: adaptive tiles, audio/timestamp storage, bounded container chunking,
and GUI cancellation/previews require separate changes.
