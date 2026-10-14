# TimeFlowCodec Publication & IP Consultation Bundle

This directory contains the self-contained publication package, archive materials, and confidential intellectual property disclosure documentation for **TimeFlowCodec (TFC)**.

> [!CAUTION]
> **CONFIDENTIALITY & PATENT NOVELTY NOTICE:**  
> Do **NOT** publish, upload, or publicly release any materials in this directory until intellectual property (IP) and patent consultations have concluded. Public preprint deposit (Zenodo, arXiv) or making the GitHub repository public constitutes an irreversible public disclosure that forfeits international patent rights under European (EPO), Japanese (JPO), and Chinese (CNIPA) patent law unless an initial priority application (e.g., U.S. Provisional Patent) is filed first.

---

## 1. Directory Contents

| Artifact | Format | Description |
| :--- | :--- | :--- |
| **[`TimeFlowCodec_Preprint_Paper.pdf`](TimeFlowCodec_Preprint_Paper.pdf)** | PDF (6 pages) | Final academic preprint manuscript compiled with `pdflatex` using standard IEEE conference formatting (`IEEEtran`). Includes empirical rate-distortion benchmarks, Apple M4 throughput measurements, and formal mathematical proofs. |
| **[`TimeFlowCodec_LaTeX_Source.tar.gz`](TimeFlowCodec_LaTeX_Source.tar.gz)** | Compressed Tarball | Complete, self-contained LaTeX distribution bundle (`main.tex`, `references.bib`, `IEEEtran.cls`, `IEEEtran.bst`, figures). Ready for direct upload to arXiv or publisher portals. |
| **[`CONSULTATION_ONE_PAGER.md`](CONSULTATION_ONE_PAGER.md)** | Markdown | Executive 1-page meeting cheat sheet for attorneys and UCI technology transfer officers. Distills the problem, invention, empirical advantages, commercial markets, and critical decision questions. |
| **[`PATENT_AND_IP_DISCLOSURE_BRIEF.md`](PATENT_AND_IP_DISCLOSURE_BRIEF.md)** | Markdown | Confidential invention disclosure brief prepared for discussions with patent attorneys and UCI Beall Applied Innovation. Summarizes the technical problem, core inventive steps, prior art distinctions, and suggested claim scope. |
| **[`zenodo_metadata.json`](zenodo_metadata.json)** | JSON | Machine-readable deposition metadata schema conforming to the Zenodo REST API specification. Pre-populated with author details, abstract, keywords, and CC-BY-4.0 licensing. |
| **[`latex_source/`](latex_source/)** | Directory | Unpacked source tree of the LaTeX manuscript for direct inspection, local editing, or incremental rebuilds. |

---

## 2. Pre-Publication IP Checklist

Prior to public release, execute the following steps with your legal counsel or university advisor:

- [ ] **UCI Beall Applied Innovation Review:**  
  Verify UC Irvine's Patent Policy regarding undergraduate independent research developed on personal hardware. Confirm whether an institutional waiver or technology disclosure form is required.
- [ ] **Provisional Patent Application Filing:**  
  If patent protection is pursued, file a U.S. Provisional Patent Application with the USPTO containing the technical descriptions from `PATENT_AND_IP_DISCLOSURE_BRIEF.md` and `TimeFlowCodec_Preprint_Paper.pdf`. This establishes an official priority date and initiates the 12-month PCT window.
- [ ] **U.S. Copyright Registration:**  
  Submit Form TX to the U.S. Copyright Office for the software source code (`libtfc/`, `timeflowcodec/`) and technical documentation to secure statutory damage remedies.

---

## 3. Post-Clearance Publication Workflow

Once legal/patent clearance is confirmed:

### A. Deposit Preprint on Zenodo
1. Log in to [Zenodo.org](https://zenodo.org).
2. Click **New Upload**.
3. Upload `TimeFlowCodec_Preprint_Paper.pdf` and `TimeFlowCodec_LaTeX_Source.tar.gz`.
4. Copy the field values from `zenodo_metadata.json`:
   - **Upload type:** Publication -> Preprint
   - **Title:** *TimeFlowCodec: An Orthogonal Video Coding Architecture for Screen and Synthetic Content via Per-Pixel Temporal Regression*
   - **Authors:** Rich Chau (University of California, Irvine)
   - **Abstract:** Paste the HTML-formatted abstract from `zenodo_metadata.json`.
   - **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0).
5. Click **Publish** to receive a persistent Digital Object Identifier (DOI).

### B. Make GitHub Repository Public & Enable Pages
1. Navigate to the GitHub repository: `https://github.com/rchau8502/TimeFlowCodec/settings`.
2. Scroll to the **Danger Zone** -> Click **Change repository visibility** -> Select **Make public**.
3. Go to **Pages** in the sidebar:
   - **Source:** Deploy from a branch.
   - **Branch:** `main` / `docs` folder (or root `/web_player/`).
4. Update repository README with the assigned Zenodo DOI badge.

### C. Distribution & Package Indexing
1. Tag release `v2.0.0` on GitHub.
2. Publish `dist/timeflowcodec-2.0.0-py3-none-any.whl` to PyPI:
   ```bash
   python -m twine upload dist/timeflowcodec-2.0.0-py3-none-any.whl
   ```
