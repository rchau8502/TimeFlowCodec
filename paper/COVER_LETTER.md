# IEEE Journal Submission Cover Letter

**Date:** October 21, 2026  
**To:** Editor-in-Chief, *IEEE Internet of Things Journal* / *IEEE Sensors Journal*  
**Subject:** Submission of Original Research Manuscript  

**Manuscript Title:**  
*A Flash-Endurance-Aware Two-Tier Compression Engine for High-Frequency Edge Telemetry*

**Author:**  
Rich Chau  
Department of Mathematics  
University of California, Irvine  
Irvine, CA 92697, USA  
Email: `richc3@uci.edu`

---

### Dear Editor-in-Chief and Editorial Board,

I am pleased to submit our original research manuscript entitled **"A Flash-Endurance-Aware Two-Tier Compression Engine for High-Frequency Edge Telemetry"** for consideration as a regular research paper in the *IEEE Internet of Things Journal* (or *IEEE Sensors Journal*).

#### Background and Core Engineering Problem:
In contemporary Internet of Things (IoT), autonomous vehicle testing fleets, and industrial cyber-physical systems, edge devices generate tens of gigabytes of high-frequency floating-point telemetry per unit daily (e.g., 10--50 kHz accelerometer vibration, CAN-FD, and IMU streams). When logged directly to solid-state storage (NAND flash, NVMe, eMMC), continuous unbuffered writes induce severe **Write Amplification Factors (WAF $\approx 10\times\text{--}50\times$)**, rapidly exhausting physical flash write endurance (Terabytes Written, TBW) and causing premature hardware burnout within months. Concurrently, exorbitant commercial 4G/5G data rates prevent continuous cloud offloading.

Existing compression solutions fail in edge environments: general-purpose byte-dictionary compressors (Zstandard, Snappy) and differential floating-point algorithms (Gorilla, FPC) collapse to minimal compression ratios ($1.10\times\text{--}1.15\times$) because microscopic physical sensor noise scrambles IEEE-754 mantissa bits, creating pseudo-random binary entropy that defeats dictionary and XOR matching.

#### Key Contributions of This Work:
1. **Two-Tier Mathematical Cascade:** We propose a unified streaming engine that executes an $O(1)$ scalar-moment Auto Linear Regression (ALR) stage for smooth steady-state signals ($>85\%$ of operation), cascading on-demand to a closed-form, fixed-basis Multilayer Perceptron (Tiny MLP) with INT8 quantized weights ($<256$ bytes footprint) to parameterize non-linear transient dynamics (braking, mechanical shocks).
2. **Flash-Endurance-Aware Hardware Subsystem:** Variable-length compressed payloads are staged in memory and flushed exclusively in strict 4KB/64KB page-aligned sectors using direct I/O (`O_DIRECT`), mathematically eliminating FTL write amplification and extending physical SSD operating lifespans by an estimated $5\times\text{--}10\times$.
3. **Empirical Superiority:** Evaluated on realistic telemetry traces with analog sensor noise, the engine achieves a **22.39$\times$ compression ratio (95.5\% data reduction)**---outperforming Meta's Zstandard and Facebook's Gorilla by over $20\times$---with continuous throughput exceeding **250,000 samples/sec** and sub-4 microsecond latency.

#### Compliance & Originality Statements:
- This manuscript represents original work that has not been published previously and is not under consideration for publication elsewhere.
- The underlying open-source implementation and reproduction suite have been made permanently available.
- All correspondence regarding this paper should be directed to the corresponding author at `richc3@uci.edu`.

Thank you for your consideration of our work. We look forward to hearing from you regarding the peer-review process.

Sincerely,  
**Rich Chau**  
University of California, Irvine  
Department of Mathematics  
