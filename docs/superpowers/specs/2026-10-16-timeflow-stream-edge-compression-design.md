# TimeFlow Stream: Two-Tier Edge Sensor & Telemetry Compression System
## Design Specification

**Date:** 2026-10-16  
**Status:** Approved for Implementation Planning  
**Target Systems:** Embedded Linux, RTOS, Automotive CAN/Ethernet, Industrial Gateways, Drone Blackbox Recorders  
**Primary Media:** NVMe SSDs, eMMC, Industrial SD Cards, USB Flash  

---

### 1. Executive Summary & Value Proposition

TimeFlow Stream is an ultra-lightweight, deterministic edge-compression engine and daemon designed to compress high-frequency numerical time-series and sensor streams in real-time before writing them to physical flash storage (SSD, eMMC, USB) or transmitting them over cellular/satellite uplinks.

By deploying a **Two-Tier Cascade Architecture**—combining streaming **Auto Linear Regression (Tier 1)** for predictable steady-state signals with a **Tiny Quantized Multilayer Perceptron (Tier 2)** for non-linear dynamic transients—TimeFlow Stream delivers:
- **10x–30x Data Volume Reduction** across multi-channel numerical sensor streams.
- **5x–10x Flash Endurance Extension (TBW)** by eliminating SSD write amplification via 4KB/64KB page-aligned direct block I/O.
- **$<2\%$ CPU Overhead** on embedded ARM cores (Cortex-A/Cortex-M) with zero runtime heap allocation (`malloc`-free during active streaming).
- **80%–90% Cellular Offload Cost Reduction** for connected vehicle fleets, remote industrial sites, and drone operations.

---

### 2. Goals & Non-Goals

#### Goals
- Provide a deterministic C99 / Rust core library (`libtfc_stream`) compatible with POSIX Linux, QNX, and RTOS (FreeRTOS, VxWorks).
- Execute Tier 1 Auto Linear Regression with $O(1)$ time complexity per sample ($<50$ nanoseconds per channel).
- Execute Tier 2 Quantized MLP inference using INT8 weights to parameterize non-linear residual waveforms when linear error exceeds tolerance $\epsilon$.
- Provide clean protocol adapters for Automotive (SocketCAN, DBC), Industrial IoT (Modbus, OPC-UA, 50kHz ADC), and Aerospace (MAVLink, PX4 uORB).
- Support direct-to-disk write buffering that flushes only in 4KB/64KB sector-aligned blocks (`O_DIRECT`).
- Provide an open client outreach playbook and structured discovery interview script to monetize the technology through paid POCs, OEM licensing, and fleet subscriptions.

#### Non-Goals
- Video/Image coding: This specification explicitly targets numerical time-series and sensor streams; screen/video pixel compression remains isolated in the legacy TimeFlowCodec video pipeline.
- Heavy deep learning training on edge devices: The Tier 2 MLP uses pre-quantized small models or tiny online ridge/basis fitting; it does not perform backpropagation on edge microcontrollers.
- Cloud-only analytical databases: While an edge-to-cloud sync socket is provided, this project does not replace Parquet/ClickHouse; it acts as the high-efficiency edge ingestion and storage pre-processor.

---

### 3. System Architecture & Component Breakdown

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: PROTOCOL ADAPTERS                      │
│                                                                        │
│   [ Automotive ]           [ Industrial IoT ]         [ Aerospace/UAV ]│
│   • SocketCAN / CAN-FD     • Modbus TCP/RTU           • MAVLink v1/v2  │
│   • DBC Signal Decoder     • OPC-UA Subscriptions     • PX4 uORB       │
│   • ROS2 / MCAP Bridge     • High-Speed ADC (50kHz)   • ArduPilot Data │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Unifies to: {channel_id, timestamp_ns, float64_val}
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 LAYER 2: UNIFIED TWO-TIER CORE ENGINE                  │
│                                                                        │
│   • Multi-Channel Demuxer (tracks each sensor channel independently)   │
│   • Tier 1: Auto Linear Regression (Piecewise linear fitter, O(1))     │
│   • Tier 2: Tiny Quantized MLP (INT8 weights, non-linear transients)   │
│   • Fallback: Lossless Delta-Bitpacking                                │
│   • Static Memory Arena (Zero malloc during active streaming)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Emits compressed byte blocks
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               LAYER 3: SSD & FLASH OPTIMIZED STORAGE BACKEND           │
│                                                                        │
│   • 4KB / 64KB Page-Aligned Direct I/O (`O_DIRECT` bypasses OS cache)   │
│   • Flash Wear-Leveling Mitigation (eliminates write amplification)    │
│   • Configurable Targets:                                              │
│       ├── Append-Only Log File (.tfc-stream) on NVMe/SSD/eMMC/USB      │
│       ├── Circular Blackbox Ring Buffer (auto-rolls old data)          │
│       └── Live Edge-to-Cloud Sync Socket (cellular / WiFi offload)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 4. Technical Engine Specifications

#### 4.1 Tier 1: Streaming Auto Linear Regression (ALR)
- **Input**: Real-time stream of tuples $(t_i, y_i)$ for channel $c$.
- **State per Channel**: Accumulated running statistics in registers:
  $$\sum 1 = n, \quad \sum t, \quad \sum y, \quad \sum t^2, \quad \sum ty$$
- **Linear Fit**:
  $$b = \frac{n \sum ty - (\sum t)(\sum y)}{n \sum t^2 - (\sum t)^2}, \quad a = \frac{\sum y - b \sum t}{n}$$
- **Online Decision Rule**:
  For each incoming sample $(t_k, y_k)$, test residual:
  $$|y_k - (a + b \cdot t_k)| \le \epsilon_c$$
  - If true: Update accumulated statistics in $O(1)$ operations.
  - If false (and current window length $L \ge L_{\min}$): Close current segment, emit tuple `(t_start, L, a, b)`, and start new window.
  - If false (and $L < L_{\min}$): Signal a dynamic transient and pass the window buffer to Tier 2.

#### 4.2 Tier 2: Tiny Quantized Multilayer Perceptron (MLP)
- **Architecture**: Feed-forward MLP with 1 input (normalized time offset $\tau \in [0, 1]$), 2 hidden layers (8 to 16 neurons with ReLU/GELU activations), and 1 output (reconstructed value).
- **Weight Quantization**: INT8 weights and INT32 accumulators; total parameter footprint $< 256$ bytes per model.
- **Role**: When transient phenomena occur (sudden braking, mechanical impact, sudden pressure drop), the non-linear curve is parameterized into a small latent vector or codebook index, yielding $5\times$ to $10\times$ compression compared to storing raw float arrays.
- **Evaluation**: Executed via SIMD (ARM NEON or x86 AVX2) vector dot products.

#### 4.3 Fallback: Lossless Delta-Bitpacking
- If a sensor channel exhibits high-entropy white noise or non-converging chaotic readings where Tier 2 error $> \epsilon_c$, the engine automatically falls back to XOR/Delta bitpacking (Gorilla-style floating-point compression) to ensure strict mathematical error bounds with zero catastrophic failure.

---

### 5. Storage Backend & SSD Flash Longevity Layer

#### 5.1 Flash Wear Mitigation
- **The Problem**: Flash memory (NAND) is physically read/written in pages (typically 4KB) and erased in blocks (typically 2MB–8MB). Standard unbuffered logging calls (`fprintf`, small `write()`) force the SSD controller's Flash Translation Layer (FTL) to repeatedly rewrite entire 4KB pages for a few bytes of new log data, causing Write Amplification Factors (WAF) of $20\times$ to $100\times$.
- **The TFC Solution**:
  1. Compressed tuples are written into an in-memory double-buffer.
  2. When the buffer reaches 4096 bytes (or 64KB for high-throughput NVMe), the engine issues a single page-aligned direct write using Linux `O_DIRECT` or POSIX `posix_memalign`.
  3. Bypasses the OS page cache, eliminating CPU dirty-page flush spikes and reducing NAND wear by up to $95\%$.

#### 5.2 Storage Modes
1. **Append-Only Compressed Log (`.tfc-stream`)**: Standard logging to external USB/SSD.
2. **Circular Blackbox Ring Buffer**: Pre-allocated fixed-size file on eMMC/NVMe (e.g., 4GB). Once full, new blocks overwrite the oldest blocks. Freezes upon digital trigger (crash accelerometer event, CAN emergency stop, or operator disarm).

---

### 6. Industry Adapters

#### 6.1 Automotive Profile
- **Protocols**: SocketCAN (`can0`, `can1`), CAN-FD, automotive Ethernet (SOME/IP, DOIP).
- **DBC Integration**: Parses standard CAN DBC files at startup to map message IDs and bit offsets into labeled, physical engineering channels (e.g. `VehicleSpeed_kmh`, `SteeringAngle_deg`).
- **Target OS**: Automotive Grade Linux (AGL), QNX Neutrino, Ubuntu Core.

#### 6.2 Industrial IoT Profile
- **Protocols**: Modbus TCP/RTU, OPC-UA client subscription, raw DAQ/ADC streams.
- **High-Speed Vibration**: Ingests continuous 16-bit / 24-bit accelerometer streams at 10 kHz–50 kHz.
- **Target OS**: Yocto Linux on ARM Cortex-A7/A53 edge gateways (Advantech, Siemens, Moxa).

#### 6.3 Aerospace & Drone Profile
- **Protocols**: MAVLink v1/v2 over serial UART or UDP, PX4 uORB messaging.
- **Flight Blackbox**: Captures IMU, attitude quaternion, GPS coordinates, motor PWMs, and battery telemetry.
- **Target OS**: FreeRTOS, PX4 RTOS, Linux on Companion Computers (Raspberry Pi CM4, NVIDIA Jetson).

---

### 7. Monetization Strategy & Customer Discovery

#### 7.1 Commercial Revenue Models
1. **Paid 4-Week Proof of Concept (POC)**: \$15,000 – \$30,000 fixed price. The customer provides 10GB–50GB of raw sample logs; we deliver an empirical benchmark report demonstrating exact compression ratio, flash wear reduction, and CPU profile on their hardware.
2. **Annual Commercial OEM License**: \$30,000 – \$80,000/year for hardware data logger OEMs and Tier-1 automotive/industrial suppliers to embed `libtfc_stream` into production hardware.
3. **Fleet Device Subscription**: \$5 – \$15/month per active edge logger for AV test fleets and connected commercial assets.

#### 7.2 Target Buyer Personas
- **Automotive**: `Director of Vehicle Telematics`, `Head of ADAS Data Infrastructure`, `Lead Embedded Systems Engineer`.
- **Industrial IoT**: `VP of Predictive Maintenance`, `Chief Reliability Officer`, `Industrial IoT Product Manager`.
- **Aerospace/Drones**: `Lead Avionics Software Engineer`, `Director of Flight Safety`, `UAV Systems Architect`.

#### 7.3 Discovery Interview Script
1. *"How many channels and at what frequencies are your edge systems logging today, and what is your daily gigabyte volume per asset?"*
2. *"How often do field SSDs or eMMC flash memory fail due to TBW write endurance limits, and what does drive replacement cost your team?"*
3. *"What is your monthly cellular/satellite offload spend per device, and how much data do you leave stranded on local drives?"*
4. *"Do you currently downsample or drop high-frequency channels (like 20kHz vibration or raw CAN) because storage or bandwidth cannot handle them?"*
5. *"If an embedded C daemon could compress your sensor streams by 15x–25x with $<2\%$ CPU usage before touching the disk, what would that save you annually in hardware and cloud costs?"*

---

### 8. Verification & Success Metrics

| Metric | Target Specification |
| :--- | :--- |
| **Compression Ratio (Steady-State)** | $\ge 15\times$ (relative to raw 64-bit IEEE-754 time-series) |
| **Compression Ratio (Dynamic/Transients)** | $\ge 5\times$ (via Tier 2 MLP parameterization) |
| **CPU Utilization** | $< 2.0\%$ single-core usage on ARM Cortex-A53 @ 1.2 GHz |
| **Per-Sample Latency (Tier 1)** | $< 50$ nanoseconds |
| **Memory Footprint** | $< 512$ KB static RAM arena (Zero runtime `malloc`) |
| **Write Amplification Mitigation** | $100\%$ page-aligned 4KB direct writes (`O_DIRECT`) |

---
*End of Design Specification.*
