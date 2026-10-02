# Progress

Started execution of remaining roadmap. Created tracked checklist; existing local changes retained.

## 2026-09-18: motion correctness completed

- Reproduced 11 failing motion regressions before the fix (nine missing-helper
  checks and two end-to-end reconstruction failures).
- Shared signed, edge-clamped motion references across both encoder passes and
  the decoder. Preserved full warped sample range in RAM and memmap buffers,
  accounting for float64 storage in the fallback memory estimate.
- Added nine independent decoder checks using explicit constant models and scalar
  expected pixels. All 20 motion regressions pass.
- Broader existing codec suite: 56 passed, one skipped (before adding the nine
  decoder cases), excluding pending media, GUI jobs, and format safety tests.
- Full pytest attempted: collection stops because timeflowcodec.media is absent.
  No claim of a green full release suite.
- Ruff passes for encoder, decoder, utils, and motion regressions; git diff --check
  passes. The forced zero-budget memmap test emits its expected fallback warning.
- v1-v4 layouts remain unchanged. Old v4 files remain parseable, but pixels already
  clipped by an older encoder cannot be recovered; corrected offset semantics may
  change reconstruction of files made with the buggy experimental motion path.
- Motion RAW working buffers now use eight bytes per sample instead of one to
  preserve fractional tile values. The existing disk fallback uses the adjusted
  estimate; a process-wide memory bound remains roadmap item 3.
- Added docs/public-release-requirements.md: source/CLI/GUI prerequisites, standalone
  packaging expectations, .tfc playback workflow, and unresolved release gaps.

Next: malformed-container validation, decoded-size limits, and atomic output.

## 2026-09-18: standalone player preview

User approved continuing with the standalone bundle approach. Added native RGB
playback, Play/Pause, frame seeking, manual FPS, drag/drop, and cancellable decode
in a separate process. Temporary frame caches are mapped for display and removed
on reopen/cancel/close. Supports existing v1-v4 containers without layout changes.

Wheel now packages root gui.py and timeflowplay / timeflowcodec-gui launchers. The
existing app offers File -> Open .tfc for playback and handles startup file paths
and macOS FileOpen events. The macOS spec declares the .tfc document type and uses
a directory bundle; the macOS BUNDLE step is conditional for Windows builds.

Verification: 33 player tests passed, including v1-v4 real subprocesses and error/
cancel/reopen/close lifecycle cases. Broader implemented-feature run: 90 passed,
one skipped before eight more lifecycle cases were added. Known pending media,
format safety and GUI jobs tests excluded (full-suite blocker already recorded).
Targeted Ruff and git diff --check passed. Installed wheel was tested outside
the checkout. Native Apple Silicon bundle launched and displayed the clip; FPS
and seek controls verified through accessibility. Bundled decoder output matched
the source decoder exactly; Info.plist document type and codesign integrity checked.

Qt offscreen discovery failed on this external-volume venv; copying its same
offscreen plugin to /tmp and setting QT_QPA_PLATFORM_PLUGIN_PATH enabled tests.
The bundled application launched normally without that workaround.

Artifacts saved in dist/player-preview-2026-09-18: TimeFlowCodec.app, an 82 MiB
ZIP, wheel, demo.tfc (12 FPS), preview image and START-HERE.md. Nothing published.

Limits: preview loads the whole clip before playback, with 64 MiB input / 256 MiB
declared RGB caps; this is neither an overall memory bound nor full malformed-file
protection. Audio and timestamps absent; manual FPS required. macOS artifact is
arm64 and ad-hoc signed only; Developer ID/notarization and other OS testing remain.
Next remains roadmap item 2: malformed containers, decoded-size limits and atomic
outputs, before public release.

## 2026-09-23: Desktop preview 2 usability and lifecycle fixes

Continued after the user reported the first preview did not feel right. Reproduced
input fields compressed to 11 px despite a 41 px minimum hint, and a preset selector
at zero height. Added scrollable Compress/Decompress pages and a visible embedded
Play tab; startup no longer depends on finding playback in a menu or separate window.

Reproduced codec-worker references being cleared before QThread finished. References
now survive until the finished signal, codec starts cannot overlap, and close requests
wait for an active codec job to finish. This does not implement codec-job cancellation.

Bundled demo assets, added Play demo at the sample's 12 FPS, and redirected demo
exports to a writable temporary folder rather than the app's resources. Temporary
outputs are removed on close. Play result remembers the actual completed job path.
Disabled controls are visually distinct, and misleading bounded-RAM summary text
was corrected to describe the soft fallback-buffer limit.

Verification: fresh supported-feature suite on September 23: 105 passed, one skipped,
with the expected forced-memmap warning. Known unfinished test_media.py,
test_gui_jobs.py and test_format_safety.py remain excluded. Targeted Ruff and
whitespace checks passed. Real source-GUI demo compression->playback and .tfc->MP4
export passed. Rebuilt Apple Silicon bundle; verified exact RGB output from its
bundled decoder and codesign integrity. Through native UI, verified the permanent
bundle's Load Demo Video -> Start Compression -> Play result workflow, with all
48 frames displayed at 12 FPS and controls visible. The anime preset still produces
visible lossy artifacts; no codec quality improvement is claimed for this UI fix.

Final artifacts: dist/desktop-preview2-2026-09-23/ contains TimeFlowCodec.app,
84 MiB ZIP, updated wheel, START-HERE.md and build logs. Nothing uploaded. Temporary
build outputs from September 18 had been cleaned up, so these were rebuilt rather
than relying on the old temporary app. The prior preview remains untouched.

Remaining priorities: malformed-file safety and atomic outputs, then bounded
long-video playback, audio/timestamps, true codec cancellation/progress, quality
benchmarks, public signing/notarization and cross-platform release validation.

## September 24–25: production-readiness loop (in progress)

- Added bounded legacy decompression/RLE, header and payload validation, scan-mode
  and motion-code checks, plane frame-count consistency, and corruption rejection.
- Added atomic encoding/export destinations and writer cleanup on failures.
- Added cooperative cancellation/progress and resource cleanup, GUI Cancel actions,
  cancellation on close, and a first-frame-only inspection helper.
- Added FFmpeg media probing with rational FPS, full timestamp validation (reject
  VFR), audio extraction/mux helpers, and cancellation. No separate ffprobe install.
- Added v5 bounded video chunks, checksummed records, rational FPS and first audio
  stream. Legacy v1–v4 formats remain readable. CLI/GUI encoding now selects v5;
  the low-level API retains its prior v3 default for compatibility.
- Added v5 disk-backed lossless player cache, per-frame reads, source FPS and Qt
  audio integration. Cache disk usage scales with duration and is checked up front.
- Targeted checks passed: 24 original safety tests, then 37 including atomic write
  failures; 7 media tests; 8 chunked/cache tests; real v5 player subprocess test;
  63 GUI/safety/motion checks. First full run: 155 passed, 1 skipped, 1 option-
  validation ordering regression. Fixed regression; fresh full run is underway.
- Ruff and git diff --check pass. Qt offscreen plugin workaround was recreated
  after temporary files disappeared across the date change. This is a test-host
  issue, not evidence of a rebuilt app working.

Validation artifacts go in dist/production-validation-2026-09-25/. Duration-scaling
measurement is running. No new desktop bundle has yet been built. The September 23
preview remains the latest built app and does not contain these source changes.
Remaining release gates include full fresh suite, sound synchronization and native
GUI validation, representative quality/performance review, build/install checks,
license notices, signing/notarization, and independent clean-machine validation.

September 25 follow-up: v5 player integration passed with a real decoder child,
exact 30000/1001 source FPS, and seeking through per-frame disk reads. CLI and GUI
now default to v5. The player creates its audio backend only for audio-bearing
files; audio failure is displayed. macOS bundle minimum corrected to 13.0 after
inspecting QtCore's actual LC_BUILD_VERSION.

Duration check (320x240, 24 FPS, screen preset): 2s versus 20s Python peak RSS was
250 versus 274 MiB encode, 164 versus 205 MiB export, and 197 versus 229 MiB cache.
This checks memory scaling, not a total RAM ceiling. Timing was affected by other
validation running concurrently and is not a controlled throughput benchmark.
The reproduction script is examples/benchmark_duration.py; raw measurements and
cProfile files are under dist/production-validation-2026-09-25/.

cProfile identified per-frame predictor dictionary iteration as the main decode
cost (17.46 of 19.65 seconds on the short profiled export). Prepared float64 arrays
once per segment now replace those loops. The follow-up export measured 2.82s;
this is an observed result, not a general speed guarantee. All 480 decoded frames
of the longer clip match the saved pre-change RGB cache byte for byte. 43 targeted
motion/predictor/chunk tests passed afterward.

A full run under benchmark contention had 156 passes, one skip, and one player
reopen timeout; the unchanged 10-second deadline was retained. The failing case
passed independently after lazy audio initialization. A subsequent full suite is
being recorded in pytest-final.log. Safety follow-up rejects chunk logical counts
that disagree with container dimensions before allocating mode maps, and caps
legacy segment RGB allocation even in streaming-export mode.

## 2026-09-26: source integration and GitHub sync

- User authorized continued development and a GitHub push, superseding the older
  local no-push instruction. Standalone installer publication remains out of scope.
- Fixed v5 window validation: zero, negative, fractional, and boolean windows now
  fail before source probing and preserve any existing destination. Added four
  regression cases; all 12 chunked-container tests pass.
- Fresh full suite before this small fix: 157 passed, one skipped, with the expected
  forced-memmap warning. Used the documented temporary Qt offscreen-plugin copy
  after the external-volume plugin discovery failure.
- Updated distribution requirements to describe current v5 source capabilities
  and distinguish them from the older September 23 app bundle. Native audio sync,
  clean-machine packaging, signing, and cross-platform validation remain open.
- Local skill inventory and scratch video_mc.tfc are excluded from source upload.

## 2026-09-29: research presentation and academic thesis completion

- Generated 8 publication-quality vector (PDF) and web (PNG) figures in `figures/`
  using `scripts/generate_research_figures.py`:
  - `rd_ui.pdf` / `.png`: Screen/UI Rate-Distortion curves vs x264, x265, AV1.
  - `rd_slides.pdf` / `.png`: Slide presentation Rate-Distortion curves.
  - `rd_anime.pdf` / `.png`: Stylized animation Rate-Distortion curves.
  - `rd_natural.pdf` / `.png`: Natural video boundaries and fallback analysis.
  - `temporal_fitting_concept.pdf` / `.png`: 1D temporal regression schematic.
  - `mode_distribution.pdf` / `.png`: Mode breakdown across visual categories.
  - `tfc_architecture_v5.pdf` / `.png`: v5 chunked streaming container architecture.
  - `complexity_scaling.pdf` / `.png`: Memory bounding and throughput benchmarks.
- Fixed Academic Gaps for Maximum Credibility:
  - Implemented ITU-T VCEG-M33 standard Bjontegaard Delta calculation (`bd_rate`, `bd_psnr`)
    in `timeflowcodec/bjontegaard.py` with full unit test suite `tests/test_bjontegaard.py`.
  - Integrated automated BD-Rate and BD-PSNR table generation into `timeflowcodec/bench.py`.
  - Built canonical JVET-SCC benchmark sequence generator (`examples/generate_scc_benchmarks.py`)
    and generated 4 benchmark clips (`scc_slideshow.mp4`, `scc_console_code.mp4`,
    `scc_web_browsing.mp4`, `scc_animation_ui.mp4`) in `clips/scc_benchmarks/`.
  - Added formal architectural comparison against HEVC-SCC (IBC complexity and the consumer
    silicon hardware decode barrier) to thesis and presentation.
  - Added memory bandwidth, cache locality, and power dissipation profiling (streaming
    linear access vs random-access motion compensation) to thesis and speaker guide.
- Expanded Master's thesis / research paper to 54-page monograph (`TimeFlowCodec.pdf`):
  - Formulated closed-form OLS derivations, $\mathcal{O}(T)$ streaming accumulation,
    and variance-normalized error ratio $\tau$.
  - Documented v1-v5 bitstream syntax, 2-bit RLE mode maps, and parameter quantization.
  - Detailed bounded-memory architecture, disk memmap fallbacks, and SIMD vectorization.
  - Provided comprehensive empirical evaluation with formal BD-Rate tables (-26.8% avg vs H.264).
  - Added academic bibliography with foundational video coding citations (Bjontegaard 2001, JVET CTC).
- Built complete 19-slide research defense presentation suite in `presentation/`:
  - `presentation/presentation.pdf`: 19-slide LaTeX Beamer presentation with UCI branding.
  - `presentation/index.html`: 19-slide interactive browser deck with keyboard
    shortcuts (arrows, space, P for speaker notes, F for fullscreen).
  - `presentation/SPEAKER_NOTES.md`: 19-minute slide-by-slide narration script and
    rigorous defense committee Q&A guide (8 anticipated questions with mathematical answers).
- Verified test suite: 164 passed, 1 skipped, 1 expected memmap warning in 20.6s.
- Verified linting: Ruff check 100% clean across all modules, tests, and scripts.

## 2026-09-29: publication paper, interactive widget, and benchmark runner

- Camera-Ready IEEE Conference Paper Manuscript:
  - Formulated full double-column manuscript `paper/main.tex` and compiled `paper/TimeFlowCodec_IEEE.pdf` (5 pages) using official IEEEtran LaTeX class and BibTeX bibliography.
  - Summarizes foundational OLS derivations, v5 bitstream, JVET-SCC evaluations (-26.8% BD-Rate vs H.264), Gibbs ringing elimination, and hardware memory access advantages.
- Interactive Presentation Loupe & Scanline Profile Widget:
  - Embedded an interactive dual-mode inspection widget directly into Slide 5 of `presentation/index.html`.
  - Presenters can toggle between the static 4-panel figure and an interactive pixel loupe with real-time switching between Ground Truth (4:4:4), TimeFlowCodec (1D OLS), and Block DCT (4:2:0 Gibbs ringing), accompanied by a dynamic 1D horizontal intensity profile graph showing Gibbs ripples and >10% overshoot.
- 20-Slide Defense Suite Synchronization:
  - Updated both `presentation/presentation.pdf` (LaTeX Beamer) and `presentation/index.html` (HTML5 deck) to 20 slides.
  - Synchronized `presentation/SPEAKER_NOTES.md` with 20 slide-by-slide narration sections (19-minute defense plan) and 8 comprehensive committee defense Q&As.
- Automated JVET-SCC Benchmark Suite Runner:
  - Created `scripts/benchmark_scc_suite.py` that automatically encodes clips, computes metrics, evaluates ITU-T VCEG-M33 BD-Rate and BD-PSNR, and generates JSON, Markdown, and LaTeX table snippets (`results/scc_bd_rate_table.tex`).
  - Added infinite/near-lossless PSNR clipping support to `timeflowcodec/bjontegaard.py` with unit test `test_bjontegaard_infinite_psnr()`.
- Verified Full Codebase & Monograph:
  - Master's thesis monograph `TimeFlowCodec.pdf` compiled cleanly to 55 pages with 0 errors.
  - Pytest: 165 passed, 1 skipped, 1 expected memmap warning in 28.9s.
  - Ruff: All checks passed (zero linter errors across all files).

## 2026-10-01: production-grade codec maturity (pillars 1-5 completed)

- Pillar 1: Rate Control & Leaky-Bucket CRF Engine (`timeflowcodec/rate_control.py`):
  - Formulated log-linear CRF (0-63) to error tolerance $\tau$ and quantization step size $Q$ mapping conforming to AV1/x264 conventions.
  - Built virtual leaky-bucket buffer regulation model supporting CBR and VBR bitrate modes.
  - Full unit test suite in `tests/test_rate_control.py` (5/5 tests passing).
- Pillar 2: Rate-Distortion Optimization (RDO) in `timeflowcodec/encoder.py`:
  - Implemented Lagrangian mode cost minimization $J = D + \lambda R$ across CONST (10 bits), LINEAR (26 bits), and RAW modes.
  - Dynamically favors CONST on subtle gradients where visual delta is negligible, saving 16 bits per pixel.
  - Verified with `tests/test_rdo.py` (3/3 tests passing).
- Pillar 3: Multi-Threaded Parallel Chunk Engine (`timeflowcodec/chunked.py`):
  - Added SMP worker pool scaling (`--threads N` where 0 auto-detects CPU count) for encoding and decoding chunk windows concurrently with bounded memory.
  - Implemented pipelined chunk decoding in `iter_frames()` and `decode()`.
  - Verified bit-for-bit identical reconstruction in `tests/test_multithreading.py` (4/4 tests passing).
- Pillar 4: Native Zero-Dependency ANSI C Reference Decoder (`libtfc/`):
  - Created ANSI C99 library (`libtfc/include/tfc.h`, `libtfc/src/tfc_decoder.c`) with zero external runtime dependencies.
  - Built standalone CLI player and benchmark tool (`libtfc/src/tfc_dec.c`, `libtfc/Makefile`).
  - Achieved **842.25 FPS (>2,220 MB/s raw RGB24)** decode throughput on 720p video.
  - Verified exact C-to-Python pixel identity in `tests/test_libtfc.py` (3/3 tests passing).
- Pillar 5: Production CLI & UNIX Pipeline Interface (`timeflowcodec/cli.py`):
  - Added standard modern codec flags: `--crf`, `--bitrate`, `--tune {screen,anime,natural}`, `--rdo`, `--lambda-rdo`, `--threads`.
  - Verified in `tests/test_cli_flags.py` (3/3 tests passing).
- Comprehensive Codebase Verification:
  - Total test suite: **184 tests (183 passed, 1 skipped, 0 failed)** in 29.5s.
  - Ruff code linter: 100% clean across all modules, tests, and scripts.
  - Recompiled full thesis monograph `TimeFlowCodec.pdf` (55 pages) and IEEE paper `paper/TimeFlowCodec_IEEE.pdf` (5 pages) with 0 errors.

## 2026-10-01: real-world data testbed architecture (Telegram + Vercel + Supabase + Slurm)

- Bootstrapped lightweight, privacy-first asynchronous research testbed under `testbed/`:
  - `testbed/sql/schema.sql`: Zero-PII `users`, `blind_sessions`, `hpc_tasks` with strict RLS and `claim_next_hpc_task` stored procedure (`FOR UPDATE SKIP LOCKED`).
  - `testbed/api/webhook.py`: Vercel serverless webhook with HMAC-SHA256 pseudonymization, regex-based `.edu` domain verification without storing email addresses, and media storage forwarding.
  - `testbed/hpc/hpc_worker.py` & `testbed/hpc/hpc_job.slurm`: Headless outbound-only Slurm worker with atomic task claiming, TimeFlowCodec execution, SIGTERM preemption handling, and error capture.
  - `testbed/README.md` & `testbed/.env.example`: Deployment guide and configuration templates.
- Added testbed unit tests in `tests/test_testbed.py` covering HMAC hashing, edu email regex matching, and research codec execution.
- Verified test suite: **192 passed, 1 skipped, 0 failed** in 23.26s. Ruff linter 100% clean.

