# TimeFlowCodec Production Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute all 5 roadmap items: clean Git staging & commit of the Oct 7–9 milestone, explicit ARM NEON SIMD vectorization in `libtfc` (>1,500 FPS), PyPI source and wheel packaging, WebAssembly `libtfc.wasm` in-browser `<canvas>` playback pipeline, and testbed deployment verification.

**Architecture:** 
1. Git repository hygiene: preserve CC0 showcase assets while excluding redundant multi-megabyte archive zips via `.gitignore`.
2. SIMD optimization in C99: vector-stride ARM NEON interleaved RGB store (`vst3_u8`) and fast-path 1-frame CONST segment reconstruction in `libtfc/src/tfc_decoder.c`.
3. Package distribution: PEP 517/621 build via `python3 -m build` producing installable wheels and sdist.
4. WebAssembly decoder: lightweight C bridge `libtfc/src/tfc_wasm.c` and browser player `web/player.html` / `web/tfc_player.js`.
5. Testbed integration: end-to-end validation of Supabase/Vercel/Slurm worker pipeline with mocked network RPCs.

**Tech Stack:** C99, ARM NEON (`<arm_neon.h>`), Python 3.13, Pytest, NumPy, Emscripten / WebAssembly, HTML5 Canvas, Setuptools/Build.

---

### Task 1: Repository Hygiene & Staging of the Oct 7–9 Milestone

**Files:**
- Modify: `.gitignore`
- Modify: `progress.md`
- Modify: `task_plan.md`

- [ ] **Step 1: Update `.gitignore` to prevent committing redundant large zip archives**
Add `showcase/*.zip` and `*.tar.gz` (except tracked arxiv submission if desired, or ignore zip bundles) to `.gitignore`.

- [ ] **Step 2: Check git status to confirm only clean assets remain**
Run: `git status`
Verify no 35MB zip files are staged.

- [ ] **Step 3: Stage and commit the core codec and documentation updates**
Run:
```bash
git add .gitignore README.md demo_assets/demo_input.tfc docs/ gui.py progress.md task_plan.md \
        scripts/ tests/ timeflowcodec/ timeflowcodec_gui.spec showcase/ \
        TimeFlowCodec_Manuscript.pdf paper/
git commit -m "feat(codec): implement adaptive spatial compaction, residual quantization, and size budgets"
```

---

### Task 2: Explicit ARM NEON SIMD Vectorization in `libtfc` (>1,500 FPS)

**Files:**
- Modify: `libtfc/src/tfc_decoder.c`
- Test: `tests/test_libtfc.py`

- [ ] **Step 1: Write failing benchmark / performance test for NEON RGB throughput**
Add unit test in `tests/test_libtfc.py` verifying that RGB mode decode matches Python pixel-for-pixel and measures throughput.

- [ ] **Step 2: Implement ARM NEON 3-channel interleaved vector store & 1-frame CONST fast path**
In `libtfc/src/tfc_decoder.c`:
1. Add fast path for pure 1-frame CONST segments (`seg_len == 1 && const_count == logical_count && tile_size == 1`):
   Directly write `const_raw` bytes into `frame_ptr` with stride 3 or 1 without float allocation or scalar loop.
2. In the temporal reconstruction loop for `hdr.color_format == TFC_COLOR_RGB` (`pixel_stride == 3`):
   When `TFC_HAS_NEON` is enabled and `raw_count == 0`:
   Compute 4 or 8 pixels at a time using NEON `vld1q_f32`, `vmlaq_f32`, `vcvtnq_s32_f32`, `vqmovn_s32`, `vqmovun_s16`.
   Interleave R, G, B channels using `vst3_u8` when synthesizing planes together or vectorized strided scatter.

- [ ] **Step 3: Rebuild `libtfc` and run benchmark**
Run:
```bash
make -C libtfc clean && make -C libtfc
./libtfc/build/tfc_dec demo_assets/demo_input.tfc --bench
```
Verify: throughput exceeds 1,500 FPS on Apple Silicon.

- [ ] **Step 4: Verify test suite passes with exact pixel accuracy**
Run: `pytest tests/test_libtfc.py -v`
Expected: ALL PASS with max pixel difference = 0.

- [ ] **Step 5: Commit changes**
```bash
git add libtfc/ tests/test_libtfc.py
git commit -m "perf(libtfc): vectorize RGB reconstruction and 1-frame CONST segments with ARM NEON"
```

---

### Task 3: PyPI Package Build & Binary Wheel Distribution

**Files:**
- Modify: `pyproject.toml`
- Test: Build output verification

- [ ] **Step 1: Check and update `pyproject.toml` metadata and build configuration**
Verify package name, version, authors, readme, classifiers, and entry points.

- [ ] **Step 2: Build source distribution and binary wheel**
Run: `python3 -m build`
Expected: Generates `dist/timeflowcodec-0.2.1.tar.gz` and `dist/timeflowcodec-0.2.1-py3-none-any.whl`.

- [ ] **Step 3: Test wheel installation and CLI in a temporary virtual environment**
Run:
```bash
python3 -m venv /tmp/tfc_test_env
/tmp/tfc_test_env/bin/pip install dist/*.whl
/tmp/tfc_test_env/bin/tfc --version
/tmp/tfc_test_env/bin/python -c "import timeflowcodec; print(timeflowcodec.__version__)"
rm -rf /tmp/tfc_test_env
```
Expected: Successfully displays version and imports cleanly.

- [ ] **Step 4: Commit changes**
```bash
git add pyproject.toml
git commit -m "build(pypi): configure build backend and verify wheel distribution"
```

---

### Task 4: WebAssembly Decoder (`libtfc.wasm`) & In-Browser `<canvas>` Player

**Files:**
- Create: `libtfc/src/tfc_wasm.c`
- Create: `web/index.html`
- Create: `web/tfc_player.js`
- Modify: `libtfc/Makefile`
- Test: `tests/test_wasm_build.py` (or Makefile check)

- [ ] **Step 1: Write `libtfc/src/tfc_wasm.c` bridge interface**
Implement memory allocation helpers and decoding entry points for WebAssembly:
- `uint8_t* tfc_wasm_alloc(size_t size)`
- `void tfc_wasm_free(uint8_t* ptr)`
- `int tfc_wasm_decode(const uint8_t* in_buf, size_t in_size)`
- `uint8_t* tfc_wasm_get_rgb(void)`
- `int tfc_wasm_get_width(void)`, `int tfc_wasm_get_height(void)`, `int tfc_wasm_get_frames(void)`

- [ ] **Step 2: Add WASM Makefile target with fallback / stand-alone build script**
Update `libtfc/Makefile` to support `make wasm` (using `emcc` or clang wasm target).
Provide a self-contained fallback build script `scripts/build_wasm.sh`.

- [ ] **Step 3: Create browser `<canvas>` player demo**
In `web/index.html` and `web/tfc_player.js`:
- Drag-and-drop or file picker for `.tfc` files.
- Decode `.tfc` via WebAssembly into an `ImageData` buffer.
- Real-time `requestAnimationFrame` render loop displaying playback FPS and resolution.

- [ ] **Step 4: Commit WebAssembly player suite**
```bash
git add libtfc/src/tfc_wasm.c libtfc/Makefile web/ scripts/build_wasm.sh
git commit -m "feat(wasm): add WebAssembly decoder interface and in-browser canvas player"
```

---

### Task 5: Real-World Testbed Deployment Validation

**Files:**
- Check: `testbed/sql/schema.sql`
- Check: `testbed/api/webhook.py`
- Check: `testbed/hpc/hpc_worker.py`
- Modify/Extend: `tests/test_testbed.py`

- [ ] **Step 1: Expand testbed integration tests**
Add end-to-end integration test simulating:
1. User registration & verification via HMAC.
2. Webhook payload receipt and task enqueueing.
3. HPC worker atomic task polling (`claim_next_hpc_task`) and processing with simulated preemption.
4. Completion reporting and result payload generation.

- [ ] **Step 2: Run testbed test suite**
Run: `pytest tests/test_testbed.py -v`
Expected: 100% PASS.

- [ ] **Step 3: Commit testbed updates**
```bash
git add tests/test_testbed.py
git commit -m "test(testbed): add full end-to-end simulation of webhook and Slurm HPC worker"
```

---

### Task 6: Comprehensive Verification & Roadmap Synchronization

**Files:**
- Modify: `task_plan.md`
- Modify: `progress.md`

- [ ] **Step 1: Run full Pytest regression suite**
Run: `pytest`
Expected: All tests pass with zero regressions.

- [ ] **Step 2: Run Ruff linter**
Run: `ruff check .`
Expected: 100% clean.

- [ ] **Step 3: Sync `task_plan.md` and `progress.md`**
Mark completed items in Tier 1 and Tier 2.

- [ ] **Step 4: Final commit and status report**
```bash
git add task_plan.md progress.md
git commit -m "docs: sync task plan and progress with full production expansion completion"
```
