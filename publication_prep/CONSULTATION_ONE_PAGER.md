# TimeFlowCodec: Intellectual Property & Patent Consultation One-Pager

**Inventor:** Rich Chau | Senior, Applied & Computational Mathematics, UC Irvine | richc3@uci.edu  
**Status:** CONFIDENTIAL — Research complete; zero public disclosure to date.  
**Objective:** Evaluate U.S. Provisional Patent filing & copyright registration prior to academic preprint publication.

---

### 1. What is the Invention? (The Core Technical Advance)
**TimeFlowCodec (TFC)** is a novel video compression paradigm designed specifically for computer screens, software GUIs, remote desktops, and synthetic graphics.

* **Conventional Video Codecs (H.264, HEVC, AV1, VVC):** Rely on 2D spatial block transforms (DCT/DST). When compressing sharp synthetic text and window borders (which are mathematical step discontinuities), 2D Fourier transforms produce severe **Gibbs ringing** (mosquito halos) and blur. They also default to **4:2:0 chroma subsampling**, which destroys 1-pixel colored text stems (syntax highlighting).
* **TimeFlowCodec Solution:** **Completely eliminates 2D spatial block transforms.** Instead, it treats the video volume as independent 1D temporal pixel trajectories $s_{x,y}(t)$ and fits them using closed-form Ordinary Least Squares (OLS) regression $s(t) = a + b \cdot t$ over short temporal windows, paired with adaptive mode classification (Static, Linear, Fallback) and a lightweight streaming container.

### 2. Key Advantages & Experimental Proof
1. **Zero Gibbs Ringing:** Text and UI borders retain mathematically perfect step edges (no frequency truncation).
2. **Native 4:4:4 RGB Fidelity:** Preserves full chrominance without color bleeding or subpixel decimation.
3. **$O(1)$ Memory Architecture:** Requires zero line-buffer hardware and zero Decoded Picture Buffer (DPB).
4. **Extreme Throughput:** C99 reference decoder reaches **>970 FPS** on 640×368 and **>150 FPS** on 720p HD on Apple Silicon (M4).
5. **Lossless at Half the Bitrate:** On terminal consoles (`scc_console_code`), achieves lossless ($\infty$ dB PSNR, 1.0000 SSIM) at **23.3 kbps**—a **48.6% bitrate reduction** vs. H.264 (45.3 kbps at 40.2 dB).

### 3. Commercial Market & Licensing Strategy
* **Target Markets:** Virtual Desktop Infrastructure (VMware, Citrix, AWS WorkSpaces), Remote Collaboration (Figma, Canva, Miro, Zoom/Teams screen sharing), Cloud Gaming, and Thin-Client Embedded Displays.
* **Dual-Licensing Structure:** Core reference decoder (`libtfc/` ANSI C) intended for open standards/MIT to foster adoption; high-performance encoder, live streaming capture, and desktop app intended for commercial/proprietary licensing.

### 4. Disclosure Timeline & Novelty Preservation
* **Current Status:** The source code and repository are **strictly Private**. No preprints (Zenodo/arXiv) or packages (PyPI) have been published.
* **The Absolute Novelty Urgency:** European (EPO), Japanese (JPO), and Chinese (CNIPA) patent laws require **absolute novelty**—any public preprint or public GitHub release forfeits foreign patent rights instantly.
* **Proposed Roadmap:**
  1. File a **U.S. Provisional Patent Application** covering the 1D temporal regression pipeline and streaming container.
  2. Receive official USPTO filing receipt (securing the 12-month international PCT priority window).
  3. Immediately publish the 6-page academic preprint on Zenodo/arXiv and open-source the repository.

---

### 5. Specific Questions for IP Counsel / Technology Transfer Officer
1. **Patentability & Claim Scope:** Does replacing 2D block DCTs with 1D temporal closed-form regression satisfy 35 U.S.C. § 101 / § 103 (patentable subject matter & non-obviousness)? Can claims cover both the mathematical streaming method and the hardware decoder architecture?
2. **University Ownership (UCI):** Does UC Irvine claim ownership under the UC Patent Policy for an undergraduate independent project created entirely on personal hardware outside of funded lab research?
3. **Provisional Filing Timeline:** Can we file an informal or formal provisional application this week to enable immediate preprint publication?
4. **Copyright Registration:** Should we file Form TX with the U.S. Copyright Office for the reference C/Python codebase simultaneously?
