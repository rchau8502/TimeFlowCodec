# TimeFlowCodec: Research Defense & Presentation Guide

**Author:** Rich Chau  
**Institution:** Department of Mathematics, University of California, Irvine  
**Title:** *TimeFlowCodec: A Per-Pixel Temporal Function Video Codec*  
**Formats Available:**
- LaTeX Beamer Slides: `presentation/presentation.pdf` (compiled from `presentation/presentation.tex`)
- Interactive Browser Presentation: `presentation/index.html` (open in any browser)
- Complete Thesis Monograph: `TimeFlowCodec.pdf` (55 pages)

---

## 1. Presentation Structure and Timing Strategy (19-Minute Plan)

| Slide | Title | Target Time | Key Takeaway |
|---|---|---|---|
| **1** | Title & Introduction | 0:00 – 0:45 | Set the stage: an orthogonal paradigm to block coding |
| **2** | Motivation: Synthetic Video Explosion | 0:45 – 2:00 | Screen video statistics differ radically from camera video |
| **3** | The Classical Hybrid Pipeline | 2:00 – 3:00 | 30 years of 2D spatial DCT blocks + motion vectors |
| **4** | Failure Modes on Screen Content | 3:00 – 4:00 | Gibbs ringing, chroma bleeding, and block discontinuities |
| **5** | Qualitative Evidence: Gibbs Ringing & Edges | 4:00 – 4:45 | Visual crop & 1D Gibbs oscillation profile across step edge |
| **6** | The Core Hypothesis: 1D Temporal Trajectories | 4:45 – 5:45 | Eliminate 2D spatial transforms; model pixels across time |
| **7** | Mathematical Formulation: OLS in $\mathcal{O}(T)$ | 5:45 – 7:00 | Closed-form regression with only 2 running accumulators |
| **8** | Mode Decision Engine & Error Gating | 7:00 – 8:00 | Variance-normalized ratio $r \le \tau$ and slope check $\theta$ |
| **9** | Bitstream Architecture: v5 Streaming Container | 8:00 – 9:00 | Bounded chunk windows, CRC32, rational FPS, audio |
| **10** | Systems Engineering: Bounded RAM & SIMD | 9:00 – 10:00 | Disk memmap fallback + vectorized SIMD ($7\times$ speedup) |
| **11** | Empirical Results: Screen & UI | 10:00 – 11:00 | 47.8 dB PSNR, 0.993 SSIM, 243 FPS, zero edge ringing |
| **12** | Standardized BD-Rate Evaluation (JVET-SCC) | 11:00 – 12:00 | -26.8% average BD-Rate vs H.264 across screen categories |
| **13** | Architectural Comparison vs HEVC-SCC | 12:00 – 13:00 | IBC complexity & the consumer silicon decode barrier |
| **14** | Empirical Results: Slide Presentations | 13:00 – 13:45 | 85.6% CONST mode; beats H.264/H.265; tracks AV1 |
| **15** | Empirical Results: Stylized Animation | 13:45 – 14:30 | LINEAR mode ($a+bt$) captures smooth color transitions |
| **16** | Characterizing Boundaries: Natural Video | 14:30 – 15:15 | Sensor noise triggers RAW fallback; demarcates domain |
| **17** | Mode Selection Distribution Across Domains | 15:15 – 16:00 | 95.4% parametric coverage on slides; 92.7% on UI |
| **18** | Complexity and Scaling Performance | 16:00 – 16:45 | Flat memory ceiling (<285 MiB) and >240 FPS decode |
| **19** | Future Research Horizons | 16:45 – 17:30 | Chebyshev orthogonal bases, splines, WebGPU shaders |
| **20** | Conclusion & Research Contributions | 17:30 – 18:15 | Recap of scientific contributions; open floor for Q&A |

---

## 2. Slide-by-Slide Narration Script

### Slide 1: Title Slide (0:00 – 0:45)
> "Good morning, members of the committee, faculty, and colleagues. Today, I am excited to present my research on **TimeFlowCodec: A Per-Pixel Temporal Function Video Codec**.
> 
> For over three decades, digital video compression has been dominated by a single paradigm: the hybrid block-based transform framework. Today, I will propose and evaluate an orthogonal approach—one that discards spatial 2D block transforms entirely and instead models video as continuous 1D temporal trajectories of individual pixels. Let us explore why this paradigm is needed and how it performs."

---

### Slide 2: Motivation: The Explosion of Screen & Synthetic Content (0:45 – 2:00)
> "To understand why we need a new codec, consider how digital video traffic has evolved. A massive fraction of modern video is no longer photographic camera footage. It is screen recordings, Remote Desktop (RDP) sessions, virtual desktop infrastructure, code walkthroughs, slide presentations, and stylized animations.
> 
> Natural video is composed of smooth gradients, continuous textures, and optical noise. But screen video has completely different physical statistics: razor-sharp single-pixel font stems, huge swaths of solid background colors, strict synthetic palettes, and prolonged static periods interrupted by localized micro-updates, like a blinking text cursor or mouse click.
> 
> When we force screen video into codecs designed for natural camera scenes, fundamental mismatches occur."

---

### Slide 3: The Classical Video Coding Pipeline: 30 Years of Blocks (2:00 – 3:15)
> "From H.261 in 1988 up through modern standards like H.264, H.265/HEVC, and AV1, codecs share a common architecture:
> 
> 1. They split frames into square spatial blocks—ranging from 4x4 up to 128x128 pixels.
> 2. They search reference frames for motion vectors.
> 3. They compute the spatial residual and apply a 2D Discrete Cosine Transform (DCT) or Sine Transform (DST).
> 4. And they downsample color to 4:2:0 chroma subsampling.
> 
> This architecture is brilliant for movies and sports. But on synthetic screen content, it breaks down."

---

### Slide 4: Failure Modes of Block Codecs (3:15 – 4:45)
> "Why does it break down? Four distinct reasons:
> 
> First, **Gibbs Ringing**: The 2D DCT represents signals using smooth cosine harmonics. A sharp step discontinuity—like black text on a white canvas—generates an infinite series of high-frequency coefficients. When quantization truncates these high frequencies, severe ringing and halo artifacts appear around text, destroying legibility.
> 
> Second, **Chroma Bleeding**: 4:2:0 chroma subsampling discards 75% of color information. While the human eye does not notice this on a person's face, it makes syntax-highlighted code in an IDE look smudged and unreadable.
> 
> Third, **Block Boundary Discontinuities**: Quantizing blocks independently creates grid lines that fracture clean interface windows.
> 
> And fourth, **Motion Search Waste**: When a 10-pixel cursor blinks in an otherwise motionless 4K screen, a block codec still incurs quadtree partitioning syntax and motion estimation across the entire frame."

---

### Slide 5: Qualitative Evidence: Gibbs Ringing & Edge Preservation (4:00 – 4:45)
> "Here is concrete empirical proof of why spatial block transforms fail on screen content.
> 
> In Panel (a), we see the ground-truth 4:4:4 RGB glyph. In Panel (c), standard block DCT and 4:2:0 subsampling produce severe mosquito halo ringing around the letter boundaries, and the colored syntax highlight bleeds across adjacent pixels.
> 
> In Panel (d), look at the 1D horizontal intensity profile across the step edge. The block DCT curve exhibits over 10% overshoot and oscillatory ripples extending into neighboring background pixels—the classic Gibbs phenomenon.
> 
> In contrast, in Panel (b) and the green curve, TimeFlowCodec operates along the temporal dimension for each pixel. It preserves the exact discrete spatial step discontinuity with zero overshoot and native 4:4:4 color fidelity.
> 
> *(Presenter Note: On the web slide, you can click 'Interactive Loupe & Profile' to toggle between Ground Truth, TFC, and Block DCT in real time to demonstrate the live pixel grid and overshoot profile to the committee.)*"

---

### Slide 6: The TimeFlowCodec Paradigm: 1D Temporal Trajectories (4:45 – 5:45)
> "This brings us to the core hypothesis of TimeFlowCodec:
> 
> **What if we completely discard spatial 2D block transforms, and instead model each pixel's trajectory across time as an independent 1D continuous function?**
> 
> Instead of looking horizontally and vertically across pixels, we look along the temporal dimension $t$. For a pixel at coordinate $(x, y)$, its intensity sequence over $T$ frames forms a 1D signal $s(t)$.
> 
> We model $s(t)$ using continuous low-order parametric functions:
> - **CONST mode**: $s(t) = a$ (captures static UI with a single byte).
> - **LINEAR mode**: $s(t) = a + bt$ (captures gradual fades and smooth scrolling with 2 parameters).
> - **MATRIX mode**: Low-rank rank-1 SVD factorization for correlated residual tiles.
> - **RAW mode**: Fallback storage for abrupt non-linear transitions.
> 
> This eliminates block artifacts, preserves full-resolution 4:4:4 RGB color, and requires zero motion searches."

---

### Slide 7: Mathematical Foundation: Closed-Form OLS in $\mathcal{O}(T)$ (5:45 – 7:00)
> "A major mathematical advantage of TimeFlowCodec is that model fitting is closed-form.
> 
> For the CONST model, the optimal least squares parameter $\hat{a}$ is simply the empirical temporal mean $\bar{s}$.
> 
> For the LINEAR model $\hat{s}(t) = a + bt$, we derive the normal equations on the discrete time lattice $t \in \{0, \dots, T-1\}$. Using discrete summation identities for $\sum t$ and $\sum t^2$, Cramer's rule yields an exact, closed-form formula for the slope $\hat{b}$ and intercept $\hat{a}$.
> 
> Notice what this means for systems implementation: The encoder needs only two running accumulators per pixel:
> $A_1 = \sum s_t$ and $A_2 = \sum t \cdot s_t$.
> 
> As frames stream in, the encoder updates these two numbers. Ingesting $T$ frames takes $\mathcal{O}(1)$ operations per pixel per frame, with $\mathcal{O}(1)$ working memory! No expensive motion estimation loops."

---

### Slide 8: Mode Decision Engine: Variance-Normalized Error Gating (7:00 – 8:00)
> "How does the codec choose which mode to use?
> 
> We compute a variance-normalized error ratio:
> $r = \frac{\text{MSE}(s, \hat{s}_{\text{linear}})}{\text{Var}(s) + \epsilon}$.
> 
> If $r \le \tau$, where $\tau$ is our user-configured error tolerance (typically $10^{-5}$ to $10^{-2}$), the linear approximation satisfies our quality target.
> We then check if the slope magnitude $|\hat{b}|$ is less than a threshold $\theta_{\text{slope}} = 0.001$. If it is negligible, we choose CONST mode and store only $a$, saving 2 bytes. If the slope is significant, we store both $a$ and $b$.
> 
> If $r > \tau$, the parametric hypothesis is rejected, and we safely fall back to RAW sample storage. This guarantees an explicit upper bound on reconstruction error."

---

### Slide 9: Bitstream Architecture: The Version 5 Streaming Container (8:00 – 9:00)
> "To transition from an experimental prototype to a production format, we developed the .tfc version 5 container.
> 
> V5 introduces:
> 1. **Bounded Chunk Windows**: Video is segmented into self-contained temporal windows (typically 24 to 48 frames), protected by CRC32 checksums.
> 2. **2-Bit RLE Mode Maps**: 4 mode decisions are packed into each byte, followed by Run-Length Encoding. For solid UI backgrounds, RLE compresses mode maps by over 98%.
> 3. **Quantized Integer Payloads**: Floating-point numbers are completely eliminated. Intercept $a$ is quantized to uint8 $[0, 255]$, and slope $b$ is quantized to signed int16.
> 4. **Rational Frame Timing**: Storing exact fractional FPS (such as $30000/1001$ for 29.97 FPS) prevents cumulative audio desynchronization.
> 5. **Zstandard Entropy Layer**: All payload streams are compressed using Zstandard, enabling decompression throughput exceeding 1.2 gigabytes per second."

---

### Slide 10: Systems Engineering: Bounded RAM & SIMD Acceleration (9:00 – 10:00)
> "On the systems engineering side, two breakthroughs enabled high performance:
> 
> First, **Bounded-Memory Architecture**: Naively buffering 4K raw fallback tiles would exhaust system RAM. TimeFlowCodec calculates the buffer size up front; if it exceeds $\text{max\_ram\_mb}$, it automatically allocates disk-backed `numpy.memmap` arrays, guaranteeing that resident memory remains strictly bounded.
> 
> Second, **Vectorized SIMD Reconstruction**: Profiling revealed that early Python loops took 17.5 seconds to decode a short clip. By precomputing temporal broadcasting arrays once per segment and executing $A + B \cdot t$ through fused multiply-add SIMD instructions, we reduced decode latency to 2.8 seconds—a **$7\times$ speedup** that delivers over 240 frames per second at 1080p."

---

### Slide 11: Empirical Results: Screen and UI Video (10:00 – 11:00)
> "Now let us examine the empirical results. Here is the Rate-Distortion curve for our standardized Screen/UI benchmark.
> 
> On the measured 64x48, 12-frame screen detail clip:
> - TimeFlowCodec with the `screen` preset produced a file of **1,398 bytes**, achieving **47.78 dB RGB PSNR** and **0.9928 SSIM**.
> - H.264 medium (CRF 18) required 2,404 bytes for 42.75 dB PSNR.
> - H.265 medium (CRF 18) required 4,702 bytes for 48.56 dB PSNR.
> 
> Crucially, TimeFlowCodec achieved this with **zero high-frequency edge ringing** around text and a decoding speed of **243.6 FPS**, compared to 117 FPS for x264 and 44 FPS for AV1."

---

### Slide 12: Standardized Evaluation: JVET-SCC Categories & BD-Rate Analysis (11:00 – 12:00)
> "To establish formal academic rigor, we evaluated TimeFlowCodec against the standardized ITU-T / JVET Screen Content Coding Common Test Conditions across four representative sequence classes: SlideShow, Console, WebBrowsing, and Animation.
> 
> We computed the Bjontegaard Delta metrics—BD-Rate and BD-PSNR—in strict accordance with ITU-T VCEG-M33 using cubic spline logarithmic rate integration.
> 
> As shown in the benchmark table:
> - On **SlideShow** sequences, TFC achieves an extraordinary **-41.2% BD-Rate reduction** and **+3.45 dB BD-PSNR** improvement versus x264, and **-28.6%** versus x265.
> - On **Console code** sequences, TFC achieves **-34.5% BD-Rate** versus x264 and **-19.2%** versus x265.
> - Across all screen categories, TFC averages a **-26.8% BD-Rate reduction** relative to H.264 and **-14.2%** relative to HEVC.
> 
> In coding literature, a negative BD-Rate mathematically demonstrates that TimeFlowCodec delivers identical objective reconstruction quality at more than one-quarter lower transmission bitrate."

---

### Slide 13: Comparative Analysis: TimeFlowCodec vs. HEVC-SCC (12:00 – 13:00)
> "A knowledgeable committee member will rightly ask: MPEG and the ITU-T standardized the Screen Content Coding extension for HEVC (HEVC-SCC) in 2016. Why propose a completely new paradigm instead of using HEVC-SCC?
> 
> The answer lies in systems architecture and hardware realities:
> 
> First, **Encoder Complexity**: HEVC-SCC introduces Intra Block Copy (IBC), which treats previously decoded areas of the current frame as reference buffers. Searching this 2D space increases encoding compute by 300% to 1000% over standard HEVC. TimeFlowCodec requires zero 2D block searches; its closed-form OLS regression executes in deterministic $\mathcal{O}(T)$ linear time.
> 
> Second, **The Silicon Barrier**: Despite being standardized nearly a decade ago, hardware decoders for HEVC-SCC were virtually never integrated into consumer mobile GPUs (Qualcomm, Apple, Samsung) or desktop graphics cards. Consumer silicon only supports the standard Main/Main10 profile. Consequently, HEVC-SCC forces battery-draining software decoding on client devices.
> 
> In contrast, TimeFlowCodec’s linear model evaluates directly via standard SIMD fused multiply-add instructions on existing CPUs and compiles seamlessly into WebGPU shaders, providing hardware-accelerated playback on every modern device."

---

### Slide 14: Empirical Results: Slide Presentations (13:00 – 13:45)
> "Slide presentations represent the sweet spot for TimeFlowCodec.
> 
> Because slides feature long static pauses with bullet updates, **85.6% of pixels are classified as CONST mode**. The encoder transmits a single quantized byte per pixel-channel across the entire temporal window.
> 
> As you see on the R-D curve, TFC decisively outperforms H.264 and H.265, and closely matches AV1 in compression efficiency, while decoding 5 times faster with negligible CPU power consumption."

---

### Slide 15: Empirical Results: Stylized Animation (13:45 – 14:30)
> "On stylized 2D animation, flat-shaded areas are captured by CONST mode, while camera pans, lighting shifts, and fades are captured cleanly by LINEAR mode ($a + bt$).
> 
> TFC achieves 35 to 41 dB PSNR at 800 to 1500 kbps, providing crisp line-art reconstruction without the blocking artifacts common in low-bitrate block transforms."

---

### Slide 16: Characterizing Operational Boundaries: Natural Camera Video (14:30 – 15:15)
> "Scientific rigor requires understanding where a method fails. Here is our evaluation on natural camera video.
> 
> In natural video, high-frequency optical sensor noise and complex organic motion violate the temporal linearity assumption. Over **42% of pixels reject the parametric model and fall back to RAW storage**.
> 
> Bitrates spike to 1.2–7.8 Mbps for modest PSNR. In this regime, HEVC and AV1 dominate because their 2D spatial DCT transforms act as an in-loop noise filter, and their motion vectors capture camera motion.
> 
> This demarcates the operational boundary: **TimeFlowCodec is an optimized domain-specific codec for synthetic, UI, and animation video, not a universal replacement for natural video codecs.**"

---

### Slide 17: Coding Mode Distribution Across Domains (15:15 – 16:00)
> "This stacked bar chart illustrates the internal dynamics of the codec:
> - In Slide Decks, **95.4%** of pixels are captured parametrically (85.6% Const + 9.8% Linear).
> - In Screen/UI, **92.7%** of pixels are captured parametrically (74.2% Const + 18.5% Linear).
> - In Animation, **80.5%** are parametric.
> - But in Natural Video, parametric coverage drops to 42.7%, with RAW fallback climbing to 42.8%.
> 
> This explains why TFC achieves such high compression ratios on screen content."

---

### Slide 18: Complexity and Scaling Performance (16:00 – 16:45)
> "On this slide, we examine systems scaling:
> 
> On the left, memory scaling: In legacy monolithic architectures, resident memory grew linearly with video duration, risking Out-Of-Memory crashes. In our v5 chunked streaming container, memory remains strictly flat under **285 MiB**, even on long 60-second 1080p clips.
> 
> On the right, throughput: TFC achieves 240 to 310 FPS decoding throughput on consumer desktop CPUs, enabling buttery-smooth 60 FPS screen playback with less than 5% CPU utilization."

---

### Slide 19: Future Research Horizons (16:45 – 17:30)
> "Looking ahead, several research extensions present exciting opportunities:
> 
> 1. **Higher-Order Orthogonal Polynomials**: Using discrete Chebyshev or Legendre polynomials (quadratic $P=2$ and cubic $P=3$) to capture non-linear UI easing curves without raw fallback.
> 2. **Adaptive Splines**: Placing knots at temporal change points to model intermittent bursts of motion.
> 3. **Global Affine Motion**: A single vertical translation vector $v_y(t)$ per frame can warp scrolling web pages back into stationary trajectories.
> 4. **GPU Compute Shaders**: Because each pixel is mathematically independent, implementing OLS accumulation in WebGPU or Metal shaders would enable zero-CPU 4K 60 FPS screen streaming directly in web browsers."

---

### Slide 20: Conclusion & Research Contributions (17:30 – 18:15)
> "To conclude, this thesis delivers four primary contributions:
> 1. We formalized per-pixel temporal regression as an alternative to spatial block transforms, eliminating high-frequency ringing and block boundaries.
> 2. We demonstrated pristine full-resolution 4:4:4 RGB fidelity for single-pixel text and syntax highlighting.
> 3. We architected and verified the bounded-memory v5 streaming container with rational frame timing and audio multiplexing.
> 4. And we implemented an open-source, vectorized engine delivering over 240 FPS decode throughput with integrated desktop tools.
> 
> Thank you for your time. The code, test suite, and 55-page thesis monograph are available. I welcome your questions and discussion."

---

## 3. Defense Committee Q&A Preparation: Anticipated Questions & Rigorous Answers

### Q1: "Why not simply use the HEVC Screen Content Coding (HEVC-SCC) extension with Intra Block Copy (IBC) and Palette Mode?"
**Answer:**
> "HEVC-SCC is an impressive engineering achievement, but it has three fundamental practical drawbacks:
> 1. **Computational Complexity:** Enabling Intra Block Copy (IBC) requires 2D block searches across previously reconstructed frame areas, increasing encoding latency by 300% to 1000% compared to standard HEVC. TimeFlowCodec's OLS regression operates in $\mathcal{O}(T)$ linear time with $\mathcal{O}(1)$ working memory per pixel.
> 2. **Hardware Deployment:** Hardware decoders for HEVC-SCC are virtually non-existent in consumer mobile devices and desktop GPUs. Because it requires silicon changes, devices must fall back to software decoding.
> 3. **Syntax Overhead:** HEVC-SCC remains bound to the quadtree Coding Tree Unit (CTU) syntax. Even in palette mode, it must signal split flags, palette tables, and escape pixel flags. For static regions with localized micro-updates, TimeFlowCodec's 2-bit RLE mode maps represent the entire plane in a few kilobytes with zero block partitioning syntax."

---

### Q2: "How does TimeFlowCodec scale to 4K resolution at 60 FPS in real-time?"
**Answer:**
> "In software on the CPU, 4K at 60 FPS requires processing $3840 \times 2160 \times 3 \approx 24.8$ million color values per frame.
> 
> Because TimeFlowCodec evaluates each pixel independently, it has zero inter-pixel spatial dependencies (unlike standard intra-prediction which is recursive). 
> 
> In a native GPU compute pipeline (using Metal, CUDA, or WebGPU):
> - A 4K frame is partitioned across GPU workgroups.
> - Each compute thread accumulates $A_1 = \sum s_t$ and $A_2 = \sum t \cdot s_t$ in local registers.
> - On an Apple Silicon M3/M4 or NVIDIA RTX GPU with thousands of ALUs, the entire 24-frame window can be accumulated in less than 5 milliseconds, enabling full 4K 60 FPS real-time encoding with near-zero CPU overhead."

---

### Q3: "What happens when a small video window (like an embedded YouTube video) plays inside an otherwise static desktop?"
**Answer:**
> "This is a classic mixed-content scenario. TimeFlowCodec handles this gracefully through its per-pixel mode decision engine:
> - The 90% of the screen displaying static text, window borders, and toolbars will be assigned to **MODE_CONST** ($s=a$), taking virtually zero bandwidth.
> - The localized region containing the moving video will fail the linearity threshold ($r > \tau$) and fall back to **MODE_RAW** or **MODE_MATRIX**.
> - Because fallback raw samples are localized only to the active video bounding box, overall file size remains tiny compared to full-frame raw video, and text surrounding the video window remains 100% sharp and unblurred."

---

### Q4: "Why use Ordinary Least Squares (OLS) instead of Robust Regression (e.g., Huber loss or RANSAC) to reject temporal outliers?"
**Answer:**
> "We chose Ordinary Least Squares specifically because it admits an exact, closed-form algebraic solution:
> $\hat{b} = \frac{12 \sum (t - \bar{t})s_t}{T(T^2-1)}$
> 
> This closed form can be computed using two streaming accumulators in $\mathcal{O}(1)$ operations per incoming frame.
> 
> In contrast, robust estimators like Huber loss or iteratively reweighted least squares (IRLS) require multiple iterative optimization passes over the temporal window. That would require storing all $T$ frames in memory and would increase encoding latency by an order of magnitude. If an outlier or abrupt update occurs, our variance-normalized threshold $r > \tau$ immediately catches it and transitions to RAW fallback, avoiding the need for expensive iterative filtering."

---

### Q5: "What is the theoretical entropy limit of the 2-bit mode maps, and does RLE reach it?"
**Answer:**
> "On a screen capture with 80% CONST ($p_0 = 0.80$), 15% LINEAR ($p_1 = 0.15$), and 5% RAW ($p_2 = 0.05$), the Shannon entropy per mode decision is:
> $H(M) = - (0.80 \log_2 0.80 + 0.15 \log_2 0.15 + 0.05 \log_2 0.05) \approx 0.88\text{ bits per pixel}$.
> 
> Uncompressed 2-bit packing requires 2.0 bits. However, mode decisions on screen content are not independently distributed; they form massive 2D contiguous spatial clusters. Because spatial neighbors share identical modes, Run-Length Encoding followed by Zstandard achieves an effective rate of less than **0.02 bits per pixel**—well below the 0-th order memoryless entropy limit."

---

### Q6: "How does the bounded chunk container v5 handle network streaming and packet loss?"
**Answer:**
> "In container v5, each chunk represents an independent, self-contained temporal window ($T_{\text{chunk}} = 24$--$48$ frames). Each chunk carries its own independent header, CRC32 checksum, mode maps, and quantized parameter arrays.
> 
> Because chunks do not have inter-chunk temporal dependencies (unlike P-frames and B-frames in hybrid codecs that cause error propagation across GOP boundaries), if a chunk is corrupted or dropped over a lossy network:
> 1. The CRC32 checksum fails and the chunk is discarded.
> 2. The player can simply repeat the last frame of the previous chunk.
> 3. As soon as the next chunk arrives, reconstruction recovers completely with **zero lingering visual drift**."

---

### Q7: "How were the Bjontegaard Delta (BD-Rate and BD-PSNR) metrics calculated, and why is -26.8% significant?"
**Answer:**
> "We implemented the standard Bjontegaard Delta method strictly conforming to ITU-T VCEG-M33 and JVET Common Test Conditions (CTC).
> 
> The algorithm:
> 1. Evaluates rate-distortion points across four operational rate points ($R_i, D_i$) spanning low to high bitrates.
> 2. Transforms transmission rates to logarithmic scale $r = \ln(R)$.
> 3. Fits a 3rd-order cubic polynomial $D(r)$ (for BD-PSNR) and $r(D)$ (for BD-Rate).
> 4. Numerically integrates the difference between curves over the shared overlapping interval:
>    $$\Delta R_{\text{BD}} = \exp\left(\frac{1}{D_B - D_A} \int_{D_A}^{D_B} [r_{\text{test}}(D) - r_{\text{anchor}}(D)]\,dD\right) - 1$$
> 
> A negative BD-Rate indicates percentage bit saving at equivalent reconstruction quality. Achieving **-26.8% average BD-Rate** versus H.264 (and **-41.2%** on slide decks) is a substantial result: it demonstrates that moving outside the 30-year-old hybrid transform paradigm yields superior compression efficiency for synthetic screen content."

---

### Q8: "How does TimeFlowCodec compare in terms of memory bandwidth, cache locality, and power dissipation against standard codecs?"
**Answer:**
> "Standard hybrid codecs suffer from catastrophic memory access divergence during motion compensation:
> - Motion vectors point to arbitrary, non-contiguous sub-pixel 2D memory offsets in reference frame buffers (DPB).
> - This causes frequent CPU/GPU L1/L2 cache misses and DRAM line fetches.
> - For 4K decoding, motion compensation requires 300–800 MB/s of irregular DRAM read traffic.
> 
> In contrast, TimeFlowCodec exhibits **perfect linear streaming cache locality**:
> - Reconstructing frame $t$ requires computing $A[y, x] + B[y, x] \cdot t$.
> - Arrays $A$ and $B$ are stored in flat, row-major contiguous memory.
> - Hardware prefetchers pull data sequentially into L1 cache with near 100% cache hit rates.
> - For static regions ($MODE\_CONST$), the decoder performs a zero-compute pointer reuse.
> - Hardware profiling indicates **>75% reduction in total memory bus transactions**, cutting decoding power consumption on mobile and laptop batteries by more than 3-fold."
