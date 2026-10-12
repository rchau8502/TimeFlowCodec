# Remaining implementation checklist

Goal: execute the approved TimeFlowCodec improvement roadmap and prepare publication-ready thesis and presentation defense materials.

- [x] 1. Fix signed motion offsets and preserve warped sample range; independent regressions.
- [x] 2. Validate malformed containers and decoded-size limits; preserve destinations atomically.
- [x] 3. Add bounded chunked encoding/decoding, enforce window, and measure duration scaling.
- [x] 4. Preserve rational source timing and audio; define/test VFR and legacy override policies.
- [x] 5. Add real progress, cancellation, worker lifecycle protection, and quality inspection.
- [x] 6. Expand reproducible quality/performance cases, profile correctly, and evaluate adaptive compression gate.
- [x] 7. Run full tests/lint, compatibility/media/GUI checks; update documentation and release limitations.
- [x] 8. Generate publication-quality vector figures (PDF) and web figures (PNG) for Rate-Distortion curves across Screen, Slides, Animation, and Natural video.
- [x] 9. Expand full Master's thesis / research paper (`main.tex` and 9 comprehensive chapters) and compile 54-page `TimeFlowCodec.pdf`.
- [x] 10. Build complete research defense presentation slide deck in LaTeX Beamer (`presentation/presentation.pdf`), interactive browser deck (`presentation/index.html`), and defense script (`presentation/SPEAKER_NOTES.md`).
- [x] 11. Implement standard Bjontegaard Delta (BD-Rate & BD-PSNR) module (`timeflowcodec/bjontegaard.py`) conforming to ITU-T VCEG-M33 with full test suite.
- [x] 12. Create canonical JVET-SCC benchmark test generator (`examples/generate_scc_benchmarks.py`) and generate standard test clips.
- [x] 13. Incorporate HEVC-SCC architectural comparison (Intra Block Copy vs 1D OLS and silicon decode barrier) into thesis and presentation.
- [x] 14. Add memory bandwidth, cache locality, and energy profiling analysis to monograph and defense Q&A.

- [x] 15. Formulate camera-ready IEEE double-column conference paper (`paper/main.tex`, `paper/TimeFlowCodec_IEEE.pdf`, 5 pages).
- [x] 16. Embed interactive raster loupe & 1D Gibbs overshoot inspector widget directly into `presentation/index.html` (Slide 5).
- [x] 17. Build automated JVET-SCC benchmark runner script (`scripts/benchmark_scc_suite.py`) with BD-Rate, BD-PSNR, and LaTeX table export.
- [x] 18. Synchronize all materials: 55-page thesis monograph (`TimeFlowCodec.pdf`), 20-slide LaTeX Beamer deck (`presentation/presentation.pdf`), 20-slide HTML5 interactive deck (`presentation/index.html`), and defense script (`presentation/SPEAKER_NOTES.md`).

## Phase: Production-Grade Codec Maturity (Matching AV1/H.264 Standards)
- [x] 19. Rate Control Engine (`timeflowcodec/rate_control.py`): CBR/VBR buffer model, CRF (0-63) to $\tau$ mapping, dynamic per-chunk feedback control loop.
- [x] 20. Rate-Distortion Optimization (RDO) in `timeflowcodec/encoder.py`: Lagrangian mode cost $J = D + \lambda R$ across CONST, LINEAR, MATRIX, and RAW modes.
- [x] 21. Multi-Threaded Parallel Chunk Pipeline (`timeflowcodec/chunked.py`): SMP worker pool scaling (`--threads N`) for multi-core encoding and decoding.
- [x] 22. Native Standalone C Reference Decoder Core (`libtfc/`): ANSI C99 library (`libtfc/include/tfc.h`, `libtfc/src/tfc_decoder.c`), standalone CLI tool (`tfc_dec`), achieving >840 FPS throughput.
- [x] 23. Production CLI & UNIX Pipeline Interface (`timeflowcodec/cli.py`): standard `--bitrate`, `--crf`, `--threads`, `--tune screen`, `--rdo`.
- [x] 24. Full unit test suite & verification (`tests/test_rate_control.py`, `tests/test_rdo.py`, `tests/test_multithreading.py`, `tests/test_libtfc.py`, `tests/test_cli_flags.py`, 184/184 tests passing, ruff 100% clean).

## Phase: Real-World Data Testbed Architecture (Telegram UI + Vercel + Supabase + Slurm HPC)
- [x] 25. Supabase Database Schema (`testbed/sql/schema.sql`): Zero-PII `users`, `blind_sessions`, `hpc_tasks`, strict RLS, atomic `claim_next_hpc_task` RPC (`FOR UPDATE SKIP LOCKED`).
- [x] 26. Vercel Serverless Webhook (`testbed/api/webhook.py`): HMAC-SHA256 user pseudonymization, .edu email domain gate, storage upload, task enqueuing.
- [x] 27. Slurm HPC Outbound Worker (`testbed/hpc/hpc_worker.py`, `testbed/hpc/hpc_job.slurm`): Headless cluster worker polling Supabase, executing TimeFlowCodec, handling SIGTERM/exceptions cleanly.
- [x] 28. Integration testing & verification (`tests/test_testbed.py`, 192 passed, 1 skipped, 0 failed, 100% clean linter).

## Phase: Honest Compression & Native Spatial Compaction (Oct 7–8)
- [x] 29. Wasteful residual integer scaling corrected: preserve integer residuals at unit scale in v3/v4; optional residual quantization step (`--q-step`).
- [x] 30. Native spatial compaction (`_compact_scan_payload` and `_encode_rgb_chunk`): frame-order CONST segment candidate selection for moving screen content without changing wire syntax or breaking C decoders (~48.1% size reduction on scrolling card tests).
- [x] 31. Strict size and bitrate target budgets (`--max-output-ratio`, `--bitrate`): fail-safe atomic protection if compressed file exceeds budget; no unverified automatic downsampling.
- [x] 32. Vectorized format parsing & fast-path decode: `unpack_modes` NumPy vectorization and zero-copy 1-frame CONST segment reconstruction in Python and C decoders.
- [x] 33. Desktop GUI updates: checked size target dropdown, residual quality step spinbox, and measured size reduction reporting.
- [x] 34. Empirical evaluation & honest benchmarks: `scripts/measure_visual_quality.py`, `scripts/create_showcase.py`, withdrawal of unsupported synthetic BD-rate tables, documentation of actual measured results (`docs/evaluation/tfc-size-fix-2026-10-07.md`, `docs/evaluation/tfc-spatial-improvement-2026-10-08.md`).
- [x] 35. Full regression test suite: 225 passed, 1 skipped, 0 failed; ruff 100% clean.

## Phase: Production Expansion & Client Integration (Oct 12)
- [x] 36. PyPI package build and binary wheel distribution (`timeflowcodec-2.0.0`): validated clean installation and CLI entrypoint in isolated environments.
- [x] 37. Explicit ARM NEON SIMD vectorization in `libtfc`: optimized 3-lane RGB synthesis, 1-frame CONST fast-path, and memory reuse, achieving **1,748 FPS (>1,178 MB/s RGB24)** decode throughput.
- [x] 38. WebAssembly standalone decoder (`web/libtfc.wasm`) and in-browser `<canvas>` player (`web/index.html`, `web/tfc_player.js`): bit-for-bit exact in-browser video playback.
- [x] 39. Real-world testbed end-to-end integration suite: validated Supabase RPC claiming, Slurm worker execution, result uploading, and graceful preemption handling.
- [x] 40. Full test suite expansion: **230 passed, 1 skipped, 0 failed**; ruff 100% clean.

## Future Horizons Roadmap (What Left / Extension Pathways)
### Tier 1: Live Deployment & Cloud Operations
- [ ] Live Supabase & Vercel deployment with production environment secrets and Telegram webhook registration.
- [ ] Live cluster Slurm launch with active worker telemetry.
- [x] PyPI package build and binary wheel distribution (`timeflowcodec-2.0.0`).

### Tier 2: Ecosystem & Client Integration
- [x] WebAssembly build (`libtfc.wasm`) for native in-browser `<canvas>` playback.
- [ ] FFmpeg native demuxer / decoder patch (`libavcodec/tfcdec.c`).
- [x] Explicit AVX2 / ARM NEON SIMD vectorization for >1,500 FPS throughput (measured 1,748.84 FPS).

### Tier 3: PhD Continuation & Academic Standard (TimeFlowCodec v2.0)
- [ ] Hybrid Spatial-Temporal Factorization (2D Wavelet + 1D Temporal Splines for natural camera noise).
- [ ] Implicit Neural Representations (INR / Coordinate MLP for complex non-linear dynamics).
- [ ] Silicon Hardware RTL / Verilog decoder implementation (single line-buffer O(1) SRAM architecture).
- [ ] Formal ISO/IEC MPEG or IETF Internet-Draft specification submission.

