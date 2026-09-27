# TimeFlowCodec: skill-assisted evaluation and improvement plan

Evaluated 2026-09-14 against the current local working tree, including existing
uncommitted screen-preset and motion-compensation changes. This is an evaluation
and proposed roadmap; production code was not changed during this assessment.

## Implementation update — 2026-09-15

The RAW predictor mismatch is fixed for newly encoded v3/v4 files. Shared
quantization now uses the exact float32 slope scale and clipped intercept carried
by the payload before calculating residuals. Regression coverage includes rising
and falling steps, 1,001-frame sequences, mixed LINEAR/RAW scaling, compressed and
uncompressed payloads, and pre-fix payload fixtures. The original sequence now
reconstructs as `[0, 1, 0, 255, 255]`, within the tested one-level rounding tolerance.
Readers and file layouts are unchanged. Previously encoded pixel errors cannot
be recovered by this encoder fix; re-encode from the source.

This completes the next concrete predictor task below, not all of package 1.
Next: fix unsigned motion offsets and signed warped-sample storage with separate
regressions. Other findings remain open. The original assessment and profiling
figures below are retained as historical evidence.

## Recommendation

Keep the product focused on sharp screen recordings and temporally coherent
content. Fix reconstruction correctness first, establish representative quality
and memory measurements, then add bounded streaming, source timing/audio, and
reliable desktop job controls. Defer new compression modes until those foundations
are verified. The existing small synthetic benchmark does not demonstrate a
competitive advantage over AV1.

## Skills selected and applied

Two missing skills were installed globally into `/Users/richc/.codex/skills/` and
read directly for this assessment. They become automatically discoverable by
Codex on the next turn.

| Skill | Status | Concrete use in this evaluation |
|---|---|---|
| find-skills | Existing | Searched the skill directory and checked source fit and adoption. |
| skill-installer | Existing | Installed the two reviewed skill directories, not entire toolkits. |
| python-performance-optimization | Newly installed | Ran cProfile on encode and direct decode; separated profiling observations from performance promises. |
| ffmpeg | Newly installed | Evaluated probing, explicit media-stream selection, export compatibility, and audio/timing test requirements. |
| requesting-code-review | Existing | Independent read-only codec review with file/line evidence and adversarial reproductions. |
| systematic-debugging | Existing | Traced a wrong-pixel result from residual formation through serialized predictor clipping; reproduced it independently. |
| test-driven-development | Existing | Defined regression-first acceptance cases for proposed fixes; no implementation undertaken. |
| verification-before-completion | Existing | Re-ran the suite and lint; separated passing tests from uncovered defects. |
| brainstorm-ideas-existing | Existing | Compared PM, designer, and engineer opportunities and selected five delivery priorities. |

Sources reviewed:
- [Python performance skill](https://github.com/wshobson/agents/tree/main/plugins/python-development/skills/python-performance-optimization), from wshobson/agents.
- [FFmpeg skill](https://github.com/digitalsamba/claude-code-video-toolkit/tree/main/.claude/skills/ffmpeg), from Digital Samba's video toolkit.
- Discovery pages: [Python performance](https://skills.sh/wshobson/agents/python-performance-optimization) and [FFmpeg](https://skills.sh/digitalsamba/claude-code-video-toolkit/ffmpeg).

The directory reported roughly 32.8K/8.1K installs and 39.6K/2.1K repository stars
respectively at search time. These are adoption signals, not correctness or
security guarantees. The downloaded instructions were inspected. Only relevant
profiling/media guidance was applied: Remotion setup, social-platform recipes,
and claims that a lossy CRF setting is an archival master do not define this
codec's requirements. Generic web-design, database, LLM-agent, and deployment
skills do not fit the current Python/Qt architecture and were not added.

## Verified baseline and important gaps

- Existing tests: **31 passed, 1 skipped**; `ruff check .` passed. The skipped smoke
  test requires a system FFmpeg executable; other real-video tests use bundled FFmpeg.
- Screen-preset quality and direct RGB metrics are already implemented. They are
  foundations to retain, not new roadmap deliverables.
- Tests currently miss significant reconstruction cases. Passing tests are not a
  release-readiness conclusion.

| Priority | Finding | Evidence in current source | Consequence |
|---|---|---|---|
| P1 | RAW residuals are calculated against original coefficients, but the decoder receives clipped/quantized coefficients. | `timeflowcodec/encoder.py:704`, `timeflowcodec/format.py:601` | Ordinary brightness changes can reconstruct with a systematic offset. |
| P1 | Motion offsets subtract 1 while still unsigned. | `encoder.py:495`, `encoder.py:676`, `decoder.py:379` | Negative offsets wrap to 255 and sample the wrong border. |
| P1 | Warped motion-compensated samples are clipped to uint8 before residual encoding. | `encoder.py:680` | Values outside 0–255 are irreversibly discarded before inverse motion. |
| P1 | RAM fallback does not bound full prediction/residual allocations or retained payloads. | `encoder.py:616`, `encoder.py:696`, `decoder.py:85` | Memory can grow with duration even with a RAM setting or streaming output. |
| P2 | Timing/audio are not stored; the v1 streaming writer also omits the supplied FPS. | `encoder.py:984`, `decoder.py:177` | Playback duration can change and sound is lost. |
| P2 | Decompression has no decoded-size ceiling; cross-plane totals need stronger validation. | `format.py:244`, decoder parsing/reconstruction | Malformed files can trigger excessive allocations or inconsistent reconstruction. |
| P2 | Worker emits only completion progress; no cooperative cancellation or close lifecycle. | `gui.py:136`, `gui.py:832` | Long jobs cannot be reliably stopped; busy and memory claims exceed the implementation. |

### Independently reproduced predictor defect

With v3, custom preset, tile size 1, no payload compression and `tau=0`, a five-frame
single-pixel RGB sequence `[0, 0, 0, 255, 255]` reconstructs as
`[51, 51, 51, 255, 255]`. This is a predictor mismatch, not acceptable evidence of
an intentional quality tradeoff. See [saved reproduction](predictor-reproduction-2026-09-14.json).
A zero tolerance is not currently a general lossless guarantee.

The independent review also reproduced much larger errors with v4 motion enabled.
Its two motion findings need isolated tests so that fixing one cannot conceal the other.

### Profiling evidence

Two synthetic RGB clips, 12 frames each, were encoded losslessly using libx264rgb
before profiling the screen preset and direct array decode. At 320×192, payload
serialization accounted for about 4.64 of 7.51 profiled encode seconds; roughly
185K scalar NumPy clip calls were made. Segment reconstruction accounted for
about 1.13 of 1.28 profiled decode seconds.

Use these observations to investigate vectorization of quantization/serialization,
precomputed scan indexes, and array-based reconstruction. Do not start with GPU
code or a Rust rewrite. Correct shared quantization must come first.

[Raw cProfile summaries](profile-2026-09-14.txt) and
[measurements](profile-2026-09-14.json) retain the evidence. These are one-off,
profiler-instrumented, small-clip observations; timings are noisy and do not
support scaling or real-time claims. RSS is the macOS process-lifetime high-water
mark, includes profiling overhead, and excludes FFmpeg children. It is not a
per-phase peak-memory benchmark.

The existing `profile.py` also needs improvement: it reports zero RSS if psutil
is absent, samples RSS after operations, and times decoded MP4 export. Replace
missing values with unavailable/null, sample peak RSS in a fresh worker including
children, and distinguish codec decode from export.

## Opportunity evaluation

Assumed users: people compressing screen recordings locally, plus researchers
measuring a temporal codec. No user-interview or market-validation claim is made.

| Perspective | Five candidate improvements |
|---|---|
| Product | Trustworthy output; preserved audio/timing; evidence of a compression niche; predictable long-job limits; reproducible desktop releases. |
| Designer | Before/after inspection; meaningful progress; cancel/retry; explicit quality/size tradeoffs; readable actionable errors. |
| Engineer | Quantization/motion invariants; validated chunked container; vectorized hot paths; adversarial format tests; reproducible quality/RSS corpus. |

Three approaches considered:
1. **Correctness-first incremental evolution — recommended.** Retain working formats,
   fix proven defects, and introduce one reviewed container extension for streaming
   and metadata. Most directly improves trust and produces measurable milestones.
2. **Add adaptive tiles/new temporal models now.** Potentially better rate–distortion,
   but current defects and weak scaling measurements would obscure results.
3. **Replace the implementation with native/GPU code.** May improve throughput, but
   carries the same mathematical bugs into a costly rewrite without a validated niche.

## Five prioritized work packages

Effort estimates are rough engineering days for implementation and tests, not
calendar promises. Format migration and audio/VFR behavior carry the most uncertainty.

### 1. Trustworthy reconstruction and file handling — highest priority, 3–6 days

**Why:** This affects the core promise that decoded output matches the selected
quality. It must precede compression research or performance rewrites.

**Changes:** Define shared quantize/dequantize helpers; calculate RAW residuals
against exactly the predictor the decoder uses. Convert motion codes to signed
integers before offset arithmetic. Keep warped samples signed until final output.
Validate legal modes/vectors, finite scales, dimensions, segment totals, payload
lengths and maximum decoded sizes before allocating. Preserve existing destination
files on failure by writing temporary files and replacing atomically.

**Files:** `encoder.py`, `decoder.py`, `format.py`, `tests/test_format_v3.py`,
`tests/test_motion_comp.py`, new quantization and malformed-container tests.

**Acceptance:** First add tests reproducing the five-frame defect, every motion
direction on an asymmetric grid, negative/>255 warped samples, and long ramps.
For the no-motion RGB RAW path with integer residuals, target maximum sample
error ≤1 after the shared quantizer; document actual bounds for other modes.
Use independent reference arithmetic, not just encoder/decoder agreement.
Malformed inputs fail predictably before exceeding the configured allocation
ceiling. Legacy golden files remain readable. Truncated writes do not replace
valid existing output.

**Assumption to validate:** The fixes fit the existing v3/v4 representation. Old
files already encoded with destructive clipping cannot generally be repaired;
do not silently reinterpret their semantics.

### 2. Measured, bounded streaming and targeted speed work — 7–12 days

**Why:** A per-pixel screen preset increases working state. Long recordings need
predictable memory before claiming laptop friendliness.

**Changes:** Define an independently decodable segment iterator; make `window`
enforce a maximum segment duration even without scene cuts. Design a new chunked
container version with per-chunk lengths, validation limits and an optional seek
index. Emit/release one completed segment at a time; bound residual conversion,
compression, and decode buffers. Count dtype size and simultaneous buffers.
Optimize the measured scalar serialization and reconstruction loops afterward.

**Files:** `encoder.py`, `decoder.py`, `format.py`, `profile.py`, `cli.py`, memory
and format-compatibility tests. Keep v1–v4 readers and explicit version selection.

**Acceptance:** Fresh-process tests compare 1-, 5-, and 20-minute clips at fixed
resolution/window, with noise, no scene cuts, many scene cuts, and motion enabled.
Proposed gate: after warm-up, peak memory at 20 minutes grows by ≤20% over the
1-minute run; publish the measured fixed overhead and include FFmpeg children.
Fail early when a requested budget cannot fit the minimum working set. Streaming
and buffered decoding agree within the same quality bound. Optimizations preserve
results and report repeated timings, not a promised speedup.

**Assumption:** A segment/chunk design can meet the cap at 1080p on the target Mac;
prove it experimentally before adding higher-resolution claims.

### 3. Preserve media timing and audio — 5–10 days, after the format design

**Why:** Usable recordings need correct duration and synchronized sound.

**Changes:** Probe source streams; retain rational frame rate/timebase, dimensions,
color interpretation and source timestamps where supported. Reuse the new
container extension rather than introducing two competing versions. Add explicit
audio-stream handling and export mapping; store/copy compatible audio rather than
unnecessarily re-encoding it. Define VFR and unsupported-stream policies. Keep an
explicit legacy FPS fallback and a user override.

**Acceptance:** 24/25/30000÷1001/60 FPS inputs preserve frame count and duration;
legacy override works for both decode paths. Test audio/no-audio, A/V start offsets,
multiple source audio streams and odd dimensions. Proposed sync target: drift ≤1
video frame over 10 minutes for supported CFR inputs. VFR either preserves timestamps
or is clearly rejected/normalized with an explicit policy; no silent conversion.
Probe exported outputs and fully decode them to verify integrity.

**Assumption:** Audio codec licensing/distribution and muxer compatibility fit the
chosen package; verify supported formats rather than claiming universal playback.

### 4. Reliable desktop jobs and quality inspection — 3–6 days

**Why:** Users need to understand tradeoffs and recover from slow/failed jobs.

**Changes:** Introduce codec progress and cooperative cancellation callbacks. Track
reading/modeling/writing phases; handle window-close while work is active; retain
worker objects until Qt reports completion. Prevent conflicting jobs from writing
the same destination. Add preview crops and original/output size, duration and
quality summaries. Correct the current bounded-RAM wording until package 2 passes.

**Files:** `gui.py`, codec callback interfaces, GUI worker lifecycle tests.

**Acceptance:** Cancel during each major phase, with cleanup and no overwrite of
valid output; retry works. Proposed cancellation responsiveness is ≤2 seconds at
normal segment boundaries, with documented exceptions for uninterruptible external
calls. Closing/reopening does not crash. Preview explicitly shows lossy detail
and is not used as a substitute for full-video validation.

**Assumption:** Small source/output crops adequately help users choose a preset;
validate with a few real workflows before investing in a full editor.

### 5. Representative quality gates, then adaptive compression — 4–8 days for gates;
research effort contingent on results

**Why:** The existing 64×48 stress clip is useful for regression but cannot justify
codec superiority or a new compression algorithm.

**Changes:** Create a consented/licensed corpus: text, scrolling UI, slides, anime,
camera motion, fades and noise; include 720p/1080p, short clips and long memory cases.
Store checksums, timing/color metadata, dependency versions and effective options.
Sweep TFC quality settings and baseline CRFs, compare at similar measured quality,
report RGB PSNR/windowed SSIM plus worst-frame/crop checks, encode/decode FPS,
peak RSS and bytes per second. Use warm-up and at least three measured repetitions.
Make unavailable metrics/codecs explicit.

**Acceptance:** Reproduction commands regenerate reports; frame/timestamp mismatch
is an error. Performance regressions have measured thresholds on a pinned runner.
Keep synthetic edge tests distinct from corpus scores. Do not aggregate unrelated
clips into a single unqualified win claim.

**Research gate:** Only then prototype adaptive tile splitting/per-pixel residuals
and compare them to the fixed screen preset. A proposed success criterion is ≥10%
median bitrate reduction at matched quality on the screen corpus without worsening
worst-case detail or exceeding the memory gate. If the gate fails, retain the
simpler preset and report the negative result.

## Order and release gate

Start package 1 and the measurement foundation from package 5. Review the shared
streaming/metadata format design, complete package 2, then package 3. Desktop job
controls can follow the segment callback contract; advanced compression remains
last. This ordering avoids optimizing corrupted output or designing metadata twice.

Before release: regression and malformed-input tests, long-clip RSS gates, real
media round trips, GUI cancel/close tests, Linux/macOS CI, and packaged-app smoke
checks must pass. Align supported Python versions with the actual tested matrix.
A clean unit-test run alone is insufficient.

Next concrete implementation task: reproduce the RAW predictor defect in a
permanent failing test, introduce shared predictor quantization, recompute residuals
against the serialized predictor, and verify v3/v4 plus legacy golden files.
