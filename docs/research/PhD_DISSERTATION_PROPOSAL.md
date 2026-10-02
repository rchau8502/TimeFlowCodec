# PhD Dissertation Research Proposal

**Title:** Continuous 1D Temporal Spline Manifolds and Domain-Adapted Entropy Coding for Zero-Ringing High-Throughput Video Representation  
**Author:** Candidate in Computer Science & Applied Mathematics  
**Topic:** Video Compression, Information Theory, Applied Numerical Analysis  
**Project Lineage:** TimeFlowCodec (TFC) Architecture v1.0 → v2.0  

---

## Executive Summary

Modern video compression standards (H.264/AVC, H.265/HEVC-SCC, AV1, and H.266/VVC) rely predominantly on two-dimensional spatial discrete cosine transforms (2D DCT) and directional intra-prediction blocks. While highly effective for natural camera photography where spatial textures exhibit continuous Gaussian-Markov statistics, this paradigm degrades severely on **Screen Content, User Interfaces, and Synthetic Graphics**. 

Step discontinuities at text stems and high-contrast window boundaries trigger the classical **Gibbs Phenomenon**, generating approximately $8.95\%$ truncation overshoot that persists as visible ringing halos and mosquito noise regardless of the bit budget allocated. Furthermore, spatial Intra Block Copy (IBC) introduces non-local memory lookups that require substantial on-chip static RAM buffers ($\mathcal{O}(\text{Frame})$ or $\mathcal{O}(\text{Tile})$), severely bottlenecking sequential decoding throughput on energy-constrained mobile devices.

This PhD dissertation proposal introduces **TimeFlowCodec v2.0**, an alternative video coding paradigm grounded in **continuous 1D temporal regression over coordinate manifolds**. By decomposing video $V(t, x, y, c)$ into independent 1D temporal trajectories $s_{x,y,c}(t)$ fitted via higher-order polynomial and Hermite splines, TFC achieves:
1. **Mathematical elimination of Gibbs ringing** on high-contrast synthetic edges.
2. **Native 4:4:4 full-color fidelity** without chroma subsampling artifacts.
3. **$\mathcal{O}(1)$ working line memory**, eliminating on-chip IBC search buffers.
4. **Order-of-magnitude decode acceleration (>3,000 FPS)** via SIMD vector FMA pipelines.

---

## 1. Problem Statement & Theoretical Deficiencies in Current Standards

### 1.1 The Gibbs Phenomenon on 2D Discrete Block Transforms
Let $f(x)$ be a step discontinuity representing a 1-pixel font stem (e.g., black text on white background, $f(x) = H(x) \cdot 255$). The $N$-point truncated discrete Fourier or cosine series representation exhibits a maximum peak deviation of:
$$\lim_{N \to \infty} \left( \max_{x} S_N f(x) - f(0^+) \right) = \frac{1}{\pi} \int_{0}^{\pi} \frac{\sin \xi}{\xi} \, d\xi - \frac{1}{2} \approx 0.08949$$
Because transform quantization aggressively zeroes high-frequency DCT coefficients, the reconstructed pixel values ripple across adjacent smooth regions. In user interfaces, this manifests as chromatic distortion around text stems and blurry drop shadows.

### 1.2 Memory Wall & Intra Block Copy (IBC) Bottlenecks
HEVC-SCC and VVC attempt to mitigate DCT ringing by copying repeated character glyphs using Intra Block Copy. However:
- IBC introduces unstructured 2D pointer dereferences across multi-line search areas.
- Hardware decoders must maintain large on-chip SRAM cache lines (typically $\ge 64\text{ KB}$ per CTU), consuming significant die area and dynamic power.
- IBC fails to compress moving, scrolling, or anti-aliased content efficiently without falling back to DCT residual blocks.

---

## 2. Core Research Hypotheses & Theoretical Framework

### Hypothesis 1: Orthogonal Decomposition into 1D Temporal Trajectories
Synthetic screen content displays high spatial frequency (sharp edges) but predominantly low temporal frequency (stationary windows, uniform mouse trajectories, linear scrolling). By rotating the primary transform dimension from the spatial plane $(x, y)$ to the temporal axis $t$, the signal manifold becomes piecewise smooth:
$$s_{x,y}(t) = \sum_{k=0}^P a_k(x, y) \, t^k + \epsilon_{x,y}(t)$$
where $P \in \{0, 1, 2\}$ captures $>95\%$ of UI pixel energy with zero spatial cross-talk.

### Hypothesis 2: Closed-Form Gram Matrix Inversion in $\mathcal{O}(1)$
For any temporal window $T$, the normal equations for polynomial fitting are governed by the temporal Gram matrix:
$$G_{j,k} = \sum_{t=0}^{T-1} t^{j+k}, \quad j, k \in \{0, \dots, P\}$$
Because $G$ depends exclusively on the temporal grid $\{0, \dots, T-1\}$ and is independent of pixel intensities, $G^{-1}$ can be precomputed. Fitting $N$ image coordinates requires only a single pass accumulator vector:
$$\mathbf{a}_i = G^{-1} \begin{bmatrix} \sum s_i(t) \\ \sum t \, s_i(t) \\ \sum t^2 \, s_i(t) \end{bmatrix}$$
with exact computational complexity $\mathcal{O}(N \cdot T)$ and zero matrix inversion latency during encoding.

---

## 3. Proposed Research Thrusts

### Thrust 1: Continuous Temporal Splines & $C^1$ Hermite Boundary Blending
* **Objective:** Extend 1st-order linear regression ($a + bt$) to 2nd-order quadratic models ($a + bt + ct^2$) and cubic Hermite splines across temporal segment boundaries.
* **Key Challenge:** Enforcing boundary derivative continuity ($s_k'(T) = s_{k+1}'(0)$) without transmitting second-derivative parameters across bitstreams.
* **Theoretical Deliverable:** Formulation of constrained piecewise polynomial OLS minimizing Lagrangian distortion $J = D + \lambda R$ subject to $C^1$ continuity conditions.

### Thrust 2: Global Knot Optimization via Bellman Dynamic Programming
* **Objective:** Determine the globally optimal temporal segment partition $\{t_0 = 0, t_1, \dots, t_M = T_{\text{total}}\}$ for arbitrary non-stationary video.
* **Methodology:** Cast variable group-of-pictures (GOP) partitioning as a shortest-path problem on a Directed Acyclic Graph (DAG):
  $$\mathcal{V}(t) = \min_{0 \le \tau < t} \left\{ \mathcal{V}(\tau) + \text{Cost}(\tau, t) + \lambda \cdot \text{Bits}(\tau, t) \right\}$$
* **Deliverable:** Linear-time approximate dynamic programming algorithm operating within bounded lookahead windows (16–64 frames).

### Thrust 3: Domain-Adapted Table Asymmetric Numeral Systems (tANS)
* **Objective:** Replace generic LZ-variant compressors (Deflate, Zstandard) with entropy coders matched to the mathematical distributions of temporal regression parameters:
  * Slopes ($b$) follow zero-mean Generalized Gaussian / Laplacian distributions: $P(b) \propto \exp(-|b| / \sigma)$.
  * Intercepts ($a$) exhibit spatial delta correlations: $\Delta a = a(x, y) - a(x-1, y)$.
* **Deliverable:** Fast finite-state machine (FSM) tANS entropy codec with precomputed probability state tables achieving within $0.5\%$ of the Shannon entropy limit with single-cycle decode throughput.

### Thrust 4: Zero-IDR Streaming Architecture for Sub-15ms Remote Display
* **Objective:** Design an ultra-low-latency transport framing protocol that completely eliminates Keyframe (IDR) bitrate spikes in cloud gaming, virtual desktop infrastructure (VDI), and remote workstations.
* **Methodology:** Periodic intra-refresh via rolling temporal coordinate stripes rather than full-frame I-frames, guaranteeing uniform bit budgets per network packet.

---

## 4. Expected Contributions & Dissertation Deliverables

1. **Information-Theoretic Foundation:** Formal derivation of the Rate-Distortion function $R(D)$ for coordinate-independent 1D temporal regression versus 2D DCT spatial blocks under screen content image priors.
2. **Reference Implementations:**
   * High-level Python research framework (`TimeFlowCodec v2.0`).
   * High-throughput ANSI C / SIMD reference library (`libtfc v2.0`) with AVX-512 and ARM NEON kernels.
3. **Open-Source Standardization Submission:** Formal specification draft submitted to the **Alliance for Open Media (AOMedia)** Screen Content Working Group or IETF.

---

## 5. Target Publication & Milestone Schedule

| Year / Milestone | Focus Area | Target Venue |
| :--- | :--- | :--- |
| **Year 1, Q1–Q2** | Closed-Form Quadratic & Spline Formulations; Numerical Stability Analysis | **Data Compression Conference (DCC)** |
| **Year 1, Q3–Q4** | Bellman DP Knot Optimization; BD-Rate Evaluations on JVET-SCC | **ACM Multimedia (ACM MM)** |
| **Year 2, Q1–Q2** | Domain-Adapted tANS Entropy Engine; Laplacian Prior Modeling | **IEEE Transactions on Image Processing (TIP)** |
| **Year 2, Q3–Q4** | SIMD Hardware Vectorization & $\mathcal{O}(1)$ Line Buffer Validation | **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)** |
| **Year 3, Q1–Q2** | Zero-IDR WebRTC Framing & Low-Latency Remote Display Architecture | **IEEE INFOCOM / ACM SIGCOMM Workshop** |
| **Year 3, Q3–Q4** | AOMedia / IETF Specification Draft Finalization & Dissertation Defense | **Ph.D. Defense & Open Specification Release** |

---

## 6. Required Resources & Feasibility

* **Hardware:** Apple Silicon (M-series NEON profiling), x86-64 workstations (AVX2/AVX-512 benchmarking), and embedded Raspberry Pi / ARM boards for mobile complexity profiling.
* **Datasets:** Standardized ITU-T/ISO/IEC JVET-SCC test sequences (RGB 4:4:4 1080p and 4K), synthetic IDE typing benchmarks, dynamic web browsing traces.
* **Software Baseline:** TimeFlowCodec v1.0.0 (production-tested, 187 green tests, fully established rate control and RDO framework).
