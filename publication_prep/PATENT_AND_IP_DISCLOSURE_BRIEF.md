# Confidential Invention Disclosure & IP Consultation Brief

**Project:** TimeFlowCodec (TFC)  
**Inventor:** Rich Chau (Undergraduate Senior, Applied & Computational Mathematics, University of California, Irvine)  
**Contact:** richc3@uci.edu  
**Document Purpose:** Prepared for confidential consultation with intellectual property counsel / university technology transfer officers prior to public disclosure.  
**Date:** October 2026  
**Confidentiality Status:** CONFIDENTIAL — Not for public distribution.

---

## 1. Executive Summary
TimeFlowCodec (TFC) is a video compression architecture specifically engineered for synthetic screen content (user interfaces, remote desktops, software consoles, vector presentations, cloud gaming, and collaborative whiteboards). 

Unlike conventional hybrid codecs (H.264/AVC, H.265/HEVC, AV1, H.266/VVC) which rely on 2D spatial block transforms (DCT/DST) and 4:2:0 chroma subsampling, TimeFlowCodec **completely eliminates 2D spatial block transforms**. Instead, it models video as an array of independent 1D temporal pixel trajectories fitted via closed-form Ordinary Least Squares (OLS) regression $s(t) = a + b \cdot t$, combined with adaptive per-pixel mode classification and a low-latency streaming container.

This approach yields:
1. **Mathematically Zero Gibbs Ringing:** Text glyphs and UI borders retain perfect step-edge sharpness without mosquito halos.
2. **Native 4:4:4 Full-Color Fidelity:** No chroma subsampling loss on colored typography or syntax highlighting.
3. **Ultra-Low Memory Footprint ($O(1)$ State):** Zero Decoded Picture Buffer (DPB) requirement, 0 line-buffer latency during raster scanline synthesis.
4. **High-Throughput Software Decoding:** Measured at $>970$ FPS on 640×368 video and $>150$ FPS on 720p HD video on Apple Silicon (M4) in single-threaded pure ANSI C99.

---

## 2. Technical Problem with Existing Art

### A. The Gibbs Ringing Penalty on Synthetic Step Edges
Standard video coding architectures divide frames into 2D spatial blocks and apply Discrete Cosine or Sine Transforms (DCT/DST). For natural photographic scenes, spatial frequencies decay smoothly. However, computer graphics, text, and GUI elements represent sharp mathematical step discontinuities ($f(x) = \Theta(x)$). 

Fourier-basis transforms of step discontinuities decay only as $O(1/k)$. When high-frequency transform coefficients are quantized/truncated, inverse transforms produce the **Wilbraham-Gibbs phenomenon**---a mandatory asymptotic $\approx 8.95\%$ overshoot ripple. In video players, this appears as unsightly ringing ripples, halos, and blurry edges around fonts and window borders.

### B. 4:2:0 Chroma Decimation
Standard distribution profiles default to 4:2:0 subsampling, discarding 75% of chrominance data. On natural video, human contrast sensitivity compensates; on screen content, 1-pixel colored text stems (e.g. IDE code syntax highlighting) are smeared across adjacent background pixels, causing severe legibility degradation.

### C. Silicon Hardware Bottlenecks of Screen Content Coding (IBC)
Existing standards bodies introduced Screen Content Coding extensions (HEVC-SCC and AV1 IntraBC). These tools employ Intra Block Copy (IBC), which performs 2D spatial block-matching within the current frame. However, IBC decoders must maintain large on-chip SRAM search windows ($>128$~KB) to store spatial reference pixels across the current picture row, introducing silicon area overhead, routing congestion, and memory bandwidth bottlenecks that have severely hindered mobile hardware adoption.

---

## 3. The Core Invention and Methodological Novelty

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CORE INVENTIVE PIPELINE                         │
│                                                                        │
│ Video Chunk [W × H × T]                                                │
│   │                                                                    │
│   ▼                                                                    │
│ 1D Per-Pixel Temporal Slicing: s_{x,y}(t) for t ∈ [0, T-1]             │
│   │                                                                    │
│   ▼                                                                    │
│ Closed-Form OLS Parameter Solver:                                      │
│   b* = [T·S_{ts} - S_t·S_s] / Δ,    a* = [S_s - b*·S_t] / T            │
│   (Evaluated in a single O(T) pass; basis moments precomputed)         │
│   │                                                                    │
│   ▼                                                                    │
│ Relative Error-Ratio Mode Selection:                                   │
│   r = (RSS / T) / (Signal_Energy / T + ε)                              │
│   ├─ If r < τ and |b*| > θ_slope ──► Mode 1: LINEAR (a, b)             │
│   ├─ If r < τ and |b*| ≤ θ_slope ──► Mode 0: CONST (mean intensity)    │
│   └─ If r ≥ τ                     ──► Mode 2: RAW Fallback + Residuals  │
│   │                                                                    │
│   ▼                                                                    │
│ Quantization & Stream Formatting:                                      │
│   - Intercept a_q ∈ uint8 [0, 255]                                     │
│   - Scaled Slope b_q ∈ int16 [-32768, 32767] (shared scale s_b)        │
│   - RLE-packed 2-bit mode maps                                         │
│   - Little-endian chunked streaming container (TFC5)                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Technical Aspects for Potential Patent Claims:

1. **Purely 1D Temporal Decomposition without Spatial Block Transforms:**
   - A video encoding method comprising receiving a temporal chunk of video frames of size $W \times H \times T$;
   - Decomposing the video volume into individual 1D temporal pixel trajectories $s_{x,y}(t)$ at fixed spatial coordinates $(x, y)$ across time $t \in [0, T-1]$;
   - Generating reconstructed frames solely by evaluating temporal parametric functions for each coordinate without performing 2D spatial frequency transforms.

2. **Closed-Form Moment-Based OLS Parameter Generation:**
   - Utilizing closed-form algebraic expressions where temporal moments $S_t = \sum t$ and $S_{t^2} = \sum t^2$ and the determinant $\Delta = T \cdot S_{t^2} - (S_t)^2 = \frac{T^2(T^2-1)}{12}$ are precomputed based strictly on chunk length $T$;
   - Performing parameter estimation in a single $O(T)$ scan over the temporal samples without matrix inversion or gradient descent iterations.

3. **Relative Error-Ratio Adaptive Mode Decision:**
   - Computing the residual sum of squares (RSS) and normalizing by the local signal variance/energy $D_{\text{sig}}$ to form a dimensionless error ratio $r = D_{\text{tfc}} / D_{\text{sig}}$;
   - Thresholding $r$ against $\tau$ and the slope magnitude $|b^*|$ against $\theta_{\text{slope}}$ to partition pixels into static (CONST), linear trend (LINEAR), and non-linear fallback (RAW) modes;
   - Dynamically scaling thresholds using spatial edge-strength classifiers to allocate higher fidelity to high-contrast graphical contours.

4. **$O(1)$ Decoder Working Memory State & Zero-Line Buffering:**
   - A decoding system that synthesizes output video frames at frame time $t$ by evaluating $\hat{s}(t) = a_q + b_q \cdot t$ directly from stored parametric coefficients;
   - The reconstruction of any pixel at time $t$ is mathematically independent of neighboring spatial pixels and prior temporal frames, eliminating the need for a multi-frame Decoded Picture Buffer (DPB) or 2D block reconstruction line buffers.

5. **Streaming Chunk Container Format (TFC5):**
   - A bitstream structure comprising an initial JSON metadata header followed by autonomous self-contained temporal chunks, each chunk including a fixed-length chunk descriptor (frame count, compressed byte size, and CRC-32 integrity checksum) enabling immediate seeking and network packetization.

---

## 4. Prior Art Distinctions

| Feature / System | Conventional Codecs (H.264/HEVC/AV1) | HEVC-SCC / AV1 IntraBC | Trajectory Research (Esche / Krutz) | **TimeFlowCodec (Invention)** |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Transform** | 2D Spatial DCT/DST | 2D Spatial DCT/DST | 2D Spatial DCT/DST | **None (Zero Spatial Transforms)** |
| **Temporal Modeling** | 2D Block Motion Vectors | 2D Block Motion Vectors | Motion-path 1D filtering | **1D Per-Pixel OLS Regression** |
| **Color Handling** | 4:2:0 Subsampled default | 4:2:0 / 4:4:4 | 4:2:0 | **Native 4:4:4 RGB** |
| **Spatial Gibbs Ringing** | Severe ($>8.95\%$ overshoot) | Present on transform blocks | Present | **Identically Zero (No spatial transform)** |
| **Decoder Memory (DPB)** | Large ($16+$ frames, $>400$~MB for 4K) | Large ($>128$~KB SRAM for IBC) | Large (Multi-frame DPB) | **$O(1)$ Memory (0 line buffers)** |
| **Computational Mode** | Complex motion search | High-complexity search | Pre/post-processing filter | **Deterministic Closed-Form Arithmetic** |

---

## 5. Commercial Applications & Market Segments

1. **Virtual Desktop Infrastructure (VDI) & Remote Workspaces:** (Citrix, VMware Horizon, Microsoft Remote Desktop / Windows 365, AWS WorkSpaces).
2. **Cloud Browser & Remote Document Collaboration:** (Figma, Google Docs, Canva, Miro, cloud developer environments).
3. **Screen Recording & Streaming Software:** (OBS Studio, Zoom, Microsoft Teams, Loom, Discord screen share).
4. **Low-Power Thin Clients & IoT Displays:** Embedded digital signage, medical telemetry monitors, aircraft/automotive dashboard displays.

---

## 6. Current Intellectual Property & Disclosure Status

- **Source Code Repository:** Hosted privately at `https://github.com/rchau8502/TimeFlowCodec`. The repository is **Private** and has never been made public.
- **Preprint Status:** A 6-page academic preprint manuscript has been prepared locally but **has NOT been uploaded to Zenodo, arXiv, or any public preprint server**.
- **PyPI / Distribution:** A local wheel (`timeflowcodec-2.0.0-py3-none-any.whl`) was built and attached to a private GitHub release; it has **NOT** been published to the public Python Package Index (PyPI).
- **Dual Licensing Scaffolding:**
  - Core reference decoders (`libtfc/` ANSI C99) are designated under the MIT License to encourage ecosystem interoperability.
  - The standalone desktop application and proprietary GUI are licensed under the TimeFlowCodec Desktop Application License (see `LICENSE`).
- **Timing Consideration (Absolute Novelty):**
  - In European (EPO) and international patent jurisdictions, any public disclosure destroys patent novelty immediately.
  - Filing a **U.S. Provisional Patent Application** prior to uploading to Zenodo or making the GitHub repository public preserves international patent rights (under the Paris Convention / PCT 12-month window).

---

## 7. Key Questions for Legal Counsel / Technology Transfer Office

1. **Provisional Patent Filing:** Does the novelty of replacing 2D block DCTs with closed-form 1D temporal regression warrant a U.S. Provisional Patent Application filing before public preprint release?
2. **University IP Ownership Review:** Does UC Irvine's Patent Policy apply to undergraduate independent research developed on personal hardware?
3. **Claim Breadth:** Can claims be structured to cover both general per-pixel temporal regression codecs and the specific streaming container / quantization mechanics?
4. **Publication Timeline:** Once a provisional application is filed with the USPTO, can the Zenodo preprint and GitHub repository be published immediately on the same day?
