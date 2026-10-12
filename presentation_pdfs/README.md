# TimeFlowCodec Presentation & Academic Deliverables

This directory contains the complete suite of compiled presentation materials, academic papers, monograph, and author credentials for the **TimeFlowCodec (TFC)** research project.

---

## Document Index

| Filename | Type | Pages / Slides | Aspect / Format | Target Audience & Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **[`TimeFlowCodec_Presentation_Slides.pdf`](./TimeFlowCodec_Presentation_Slides.pdf)** | Defense / Conference Slides | 20 slides | 16:9 Widescreen | Live projector presentations, defense talks, and keynote addresses. |
| **[`TimeFlowCodec_Presentation_Handout.pdf`](./TimeFlowCodec_Presentation_Handout.pdf)** | Print Handout | 10 pages | 2-on-1 Letter | Physical distribution to committee members, attendees, and reviewers. |
| **[`TimeFlowCodec_Thesis_Monograph.pdf`](./TimeFlowCodec_Thesis_Monograph.pdf)** | Doctoral Thesis / Monograph | 55 pages | Single-column Report | Comprehensive mathematical foundations, Sobolev bounds, container specs, and benchmarks. |
| **[`TimeFlowCodec_IEEE_Paper.pdf`](./TimeFlowCodec_IEEE_Paper.pdf)** | Conference Paper | 5 pages | Two-column IEEEtran | Peer-reviewed publication manuscript adhering to IEEE conference standards. |

---

## Detailed Summaries

### 1. Presentation Slides (`TimeFlowCodec_Presentation_Slides.pdf`)
* **Aspect Ratio:** 16:9 Widescreen (10 pt Madrid theme with UCI official Anteater Blue `#0064A4` and Gold `#FFD200`).
* **Structure:**
  * **Part I — Motivation & Foundational Failure:** The 30-year legacy of 2D DCT hybrid codecs (H.264, HEVC, AV1) and why spatial transforms catastrophically fail on synthetic screen content (Gibbs ringing, 4:2:0 chroma blur).
  * **Part II — Mathematical Architecture:** Per-pixel 1D temporal regression, closed-form OLS parameter derivation ($a_{x,y}, b_{x,y}$), integer motion compensation, and residual dead-zone quantization.
  * **Part III — Streaming Container & SIMD Engine:** Container v5 streaming architecture, chunked seek tables, ARM NEON / x86 AVX2 vectorization, zero-copy `numpy.memmap`, and the WebAssembly in-browser player.
  * **Part IV — Empirical Validation & Results:** ITU-T VCEG-M33 SCC benchmarks, Bjontegaard Delta rate reductions ($-26.8\%$ vs H.264, $-12.4\%$ vs HEVC), visual crop comparison, and $>1,700$ FPS decode throughput.
  * **Part V — Impact & Roadmap:** Production deployment in remote desktops/VDI, WebAssembly browser streaming, and future 2D affine motion trajectory extensions.

### 2. Presentation Handouts (`TimeFlowCodec_Presentation_Handout.pdf`)
* **Layout:** 2 slides per page in landscape orientation on US Letter paper with a 5 mm shrink margin.
* **Optimized For:** Dual-side physical printing, committee annotations, and compact offline reading.

### 3. Thesis Monograph (`TimeFlowCodec_Thesis_Monograph.pdf`)
* **Comprehensive Chapters:**
  1. Introduction & Motivation
  2. The Gibbs Phenomenon and Theoretical Limits of 2D Block Transforms
  3. Mathematical Framework: Per-Pixel 1D Temporal Functional Coding
  4. Spatial Compaction and Rate-Distortion Optimization
  5. Systems Architecture: Streaming Container v5 & High-Throughput Decoders
  6. Empirical Evaluation on JVET Screen Content Coding Benchmarks
  7. Computational Complexity and SIMD Acceleration
  8. Future Directions & Affine Motion Extensions
  9. Conclusion
  * **Appendices:** A (Analytical Proofs), B (CLI Reference & Streaming Protocol), C (Binary Container Specification).

### 4. IEEE Conference Paper (`TimeFlowCodec_IEEE_Paper.pdf`)
* **Title:** *TimeFlowCodec: An Orthogonal Video Coding Paradigm for Screen and Synthetic Content via Per-Pixel Temporal Regression*
* **Format:** IEEE conference publication format with figures, rate-distortion curves, benchmark tables, and citations.

---

## Source Files & Recompilation Instructions

All PDFs can be rebuilt from their respective LaTeX sources using `pdflatex`:

```bash
# 1. Presentation Slides & Handouts
cd presentation
pdflatex -interaction=nonstopmode presentation.tex
pdflatex -interaction=nonstopmode presentation_handout.tex

# 2. Thesis Monograph
cd ..
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# 3. IEEE Conference Paper
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
```
