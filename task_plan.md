# Remaining implementation checklist

Goal: execute the approved TimeFlowCodec improvement roadmap, preserving existing local work.

Completed before this run: screen preset, direct RGB benchmarks, standard SSIM,
shared RAW predictor quantization and compatibility tests (45 passed, one skipped).

- [x] 1. Fix signed motion offsets and preserve warped sample range; independent regressions.
- [x] 2. Validate malformed containers and decoded-size limits; preserve destinations atomically.
- [x] 3. Add bounded chunked encoding/decoding, enforce window, and measure duration scaling.
- [x] 4. Preserve rational source timing and audio; define/test VFR and legacy override policies.
- [ ] 5. Add real progress, cancellation, worker lifecycle protection, and quality inspection.
- [ ] 6. Expand reproducible quality/performance cases, profile correctly, and evaluate adaptive compression gate.
- [ ] 7. Run full tests/lint, compatibility/media/GUI checks; update documentation and release limitations.

Current phase: v5 implementation and integration verification. Safety, atomic output,
media, cancellation, and bounded chunks are implemented; final audits and release
validation remain. See progress.md and docs/container-v5.md.
Source push authorized by the user on 2026-09-26; installer publication remains a separate release gate. Existing v1–v4 layout/readability must remain available.
Research changes to compression are conditional on measured benefit; no superiority claim.

## Decisions
- Add new container version for chunked storage and metadata rather than change legacy layouts.
- Short legacy array decode remains available; bounded output uses the new streaming container.
- Reuse the tested segment codec within bounded chunks.

## Errors and verification
- None in this run yet.
