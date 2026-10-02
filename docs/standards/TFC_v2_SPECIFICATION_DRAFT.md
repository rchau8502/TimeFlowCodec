# TimeFlowCodec Bitstream Specification (Version 2.0 Draft)

**Standard Track:** Working Group Draft Specification  
**Document ID:** AOM-TFC-SPEC-2.0-DRAFT  
**Target Bodies:** Alliance for Open Media (AOMedia) Screen Content Working Group / IETF  
**Subject:** High-Throughput Coordinate-Wise Temporal Function Video Coding  

---

## 1. Scope & Conformance

This specification defines the bitstream syntax, semantics, and normative decoding process for **TimeFlowCodec Version 2 (TFCv2)**. A compliant decoder SHALL be capable of parsing bitstreams conforming to the Profiles and Levels defined herein and synthesizing reconstructed video frames matching the normative mathematical process within specified numeric tolerances.

### 1.1 Normative References
- **ITU-T Recommendation BT.601 / BT.709:** Parameter values for the HDTV standard for the production and international exchange of programmes.
- **ITU-T Recommendation BT.2020:** Parameter values for ultra-high definition television systems.
- **IEEE Std 754-2008:** Standard for Floating-Point Arithmetic.

---

## 2. Architecture & Bitstream Syntax

### 2.1 Bitstream Hierarchy
A TFCv2 bitstream consists of an ordered sequence of bytes structured into:
1. **Container Header (`TFC5`):** Carries global stream metadata, spatial dimensions $(W, H)$, frame rate, profiles, and color space flags.
2. **Chunk Records (`VIDS`):** Independent, self-contained temporal segments encompassing $T_{\text{chunk}}$ frames.
3. **Plane Streams:** Sequentially encoded color planes ($P_0, P_1, P_2$). For RGB coding, these correspond to R, G, and B. For YUV coding, these correspond to Y, U, and V.
4. **Segment Payloads:** Sub-temporal intervals of length $L \le T_{\text{chunk}}$ storing mode bitstreams, quantized temporal parameters, and fallback residuals.

### 2.2 Container Header Syntax (`TFC5`)

```
TFC5_Header {
    magic_bytes                4 bytes   /* ASCII: 'TFC5' */
    metadata_length            32 bits   /* Little-endian unsigned int */
    metadata_json_payload      variable  /* UTF-8 JSON stream */
}
```

The `metadata_json_payload` MUST contain the following normative fields:
* `"version"`: `integer` (5 for TFC container v5, codec version 2).
* `"width"`: `integer` (horizontal resolution $W \in [1, 65535]$).
* `"height"`: `integer` (vertical resolution $H \in [1, 65535]$).
* `"num_frames"`: `integer` (total frames in sequence).
* `"fps_num"`: `integer` (frame rate numerator).
* `"fps_den"`: `integer` (frame rate denominator).
* `"profile"`: `string` (`"main_screen"`, `"professional_hdr"`, or `"ultra_low_latency"`).
* `"color_format"`: `integer` (1: RGB 4:4:4, 2: YUV 4:4:4, 3: YUV 4:2:0).

### 2.3 Chunk Record Syntax (`VIDS`)

```
Chunk_Record {
    record_tag                 4 bytes   /* ASCII: 'VIDS' */
    payload_size               64 bits   /* Little-endian uint64 */
    crc32_checksum             32 bits   /* IEEE 802.3 CRC32 of payload */
    inner_chunk_data           variable  /* Length: payload_size */
}
```

### 2.4 Plane Payload Syntax (Version 4/5)

Each plane stream is structured into $M$ sequential temporal segments:

```
Plane_Payload {
    segment_count              32 bits   /* Number of segments M */
    for (s = 0; s < segment_count; s++) {
        segment_length         32 bits   /* Segment duration L */
        logical_count          32 bits   /* Number of spatial coordinates N */
        mode_stream_length     32 bits   /* Byte length of RLE mode stream */
        b_slope_scale          32 bits   /* IEEE 754 float32 linear slope multiplier */
        raw_scale              32 bits   /* IEEE 754 float32 raw residual multiplier */
        const_count            32 bits   /* Number of CONST mode coordinates */
        linear_count           32 bits   /* Number of LINEAR mode coordinates */
        raw_count              32 bits   /* Number of RAW mode coordinates */
        has_mvs                32 bits   /* 0: no motion comp, 1: block motion comp */
        
        mode_rle_bytes         variable  /* Length: mode_stream_length */
        
        if (has_mvs == 1) {
            motion_vector_bytes variable /* Length: (L - 1) * logical_count bytes */
        }
        
        const_intercepts       variable  /* Length: const_count bytes (uint8) */
        linear_intercepts      variable  /* Length: linear_count bytes (uint8) */
        linear_slopes          variable  /* Length: linear_count * 2 bytes (int16) */
        
        if (raw_count > 0) {
            raw_intercepts     variable  /* Length: raw_count bytes (uint8) */
            raw_slopes         variable  /* Length: raw_count * 2 bytes (int16) */
            raw_residuals      variable  /* Length: raw_count * L * 2 bytes (int16) */
        }
    }
}
```

---

## 3. Profiles, Levels, and Operating Points

### 3.1 Profiles

| Profile Name | Bit Depth | Chroma Subsampling | Supported Modes | Target Application |
| :--- | :---: | :---: | :---: | :--- |
| **Main Screen Profile** | 8-bit | 4:4:4 | `CONST`, `LINEAR`, `RAW` | Desktop UI, presentation decks, screen recording |
| **Professional Master Profile** | 8, 10, 12-bit | 4:4:4 / 4:2:2 | `CONST`, `LINEAR`, `QUADRATIC`, `RAW` | Studio animation, digital signage, HDR mastering |
| **Ultra-Low-Latency Profile** | 8-bit | 4:4:4 | `CONST`, `LINEAR` | Cloud gaming, remote desktop (WebRTC), VDI |

### 3.2 Levels

| Level | Max Luma Samples / sec | Max Resolution @ Frame Rate | Max Bitrate |
| :---: | :---: | :---: | :---: |
| **1.0** | 30,000,000 | 1280 × 720 @ 30 fps | 5 Mbps |
| **2.0** | 125,000,000 | 1920 × 1080 @ 60 fps | 20 Mbps |
| **3.0** | 500,000,000 | 3840 × 2160 @ 60 fps | 60 Mbps |
| **4.0** | 2,000,000,000 | 7680 × 4320 @ 60 fps | 200 Mbps |

---

## 4. Normative Decoding & Pixel Synthesis Process

For coordinate index $i \in \{0, \dots, N-1\}$ and temporal frame index $t \in \{0, \dots, L-1\}$:

### 4.1 Mode Unpacking & Parameter Reconstruction
1. **Mode Decode:** Unpack RLE byte stream to retrieve mode $m_i \in \{0, 1, 2\}$:
   * $m_i = 0$ (`MODE_TFC_LINEAR`): Coordinate modeled by linear trajectory.
   * $m_i = 1$ (`MODE_TFC_CONST`): Coordinate modeled by constant luminance.
   * $m_i = 2$ (`MODE_FB_RAW`): Coordinate modeled by linear predictor + quantized residual sequence.
2. **Parameter Dequantization:**
   * Constant: $\hat{a}_i = \text{const\_intercepts}[k]$.
   * Linear: $\hat{a}_i = \text{linear\_intercepts}[k], \quad \hat{b}_i = \text{linear\_slopes}[k] \times \text{b\_slope\_scale}$.
   * Raw: $\hat{a}_i = \text{raw\_intercepts}[k], \quad \hat{b}_i = \text{raw\_slopes}[k] \times \text{b\_slope\_scale}, \quad \hat{r}_i(t) = \text{raw\_residuals}[k \cdot L + t] \times \text{raw\_scale}$.

### 4.2 Linear / Quadratic Synthesis Formula
The reconstructed intensity sample $V_i(t)$ prior to spatial expansion SHALL be computed as:

$$V_i(t) = \text{clip}\left( \text{round}\left( \hat{a}_i + \hat{b}_i \cdot t + \hat{c}_i \cdot t^2 + \delta_i(t) \right), \, 0, \, 255 \right)$$

where:
* $\hat{c}_i = 0$ for Profile 0 (`Main Screen Profile`).
* $\delta_i(t) = \hat{r}_i(t)$ if $m_i = \text{MODE\_FB\_RAW}$, else $\delta_i(t) = 0$.

### 4.3 Spatial Tile Expansion
When tile dimension $K > 1$, coordinate index $i$ corresponds to tile grid coordinate $(ty, tx)$. For all spatial pixels $(y, x)$ satisfying:
$$y \in [ty \cdot K, \, \min((ty+1) \cdot K, H) - 1], \quad x \in [tx \cdot K, \, \min((tx+1) \cdot K, W) - 1]$$
the reconstructed plane pixel is assigned:
$$I(y, x, t) = V_{ty \cdot \text{tiles\_x} + tx}(t)$$

---

## 5. Decoder Working Memory & Hardware Bounds

### 5.1 $\mathcal{O}(1)$ Working Line Memory Conformance
A compliant hardware or software implementation SHALL require **at most one active frame line buffer** for temporal synthesis. Decoders SHALL NOT require:
1. Spatial 2D intra-block search buffers.
2. Recursive spatial filtering windows (e.g. deblocking or SAO filters).
3. Frame reordering buffers (B-frame buffers).

### 5.2 Arithmetic Tolerances
Due to single-precision floating-point vs. fixed-point integer rounding in SIMD multiply-accumulate units, compliant decoders SHALL produce reconstructed pixel outputs differing by at most $\pm 1$ Least Significant Bit (LSB) from the ANSI C reference implementation (`libtfc`).
