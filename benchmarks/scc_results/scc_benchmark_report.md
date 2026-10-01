# JVET Screen Content Coding (SCC) Benchmark Report
**Date:** 2026-10-01 10:47:13

## 1. Bjontegaard Delta Summary (TFC vs H.264/libx264)
| Sequence Name | BD-Rate (%) | BD-PSNR (dB) |
|---|---:|---:|

> *Note: Negative BD-Rate indicates bitstream savings conforming to ITU-T VCEG-M33.*

## 2. Full Rate-Distortion Sampling
| Codec | Clip | Parameter | Rate (kbps) | PSNR (dB) | SSIM | Enc Time (s) | Dec Time (s) |
|---|---|---|---:|---:|---:|---:|---:|
| x264 | `scc_slideshow` | CRF=23 | 22.4 | 39.58 | 0.9960 | 0.17 | 0.13 |
| x264 | `scc_slideshow` | CRF=28 | 20.4 | 39.56 | 0.9961 | 0.21 | 0.15 |
| tfc | `scc_slideshow` | tau=0.001 | 83.5 | 67.06 | 1.0000 | 20.94 | 2.43 |
| tfc | `scc_slideshow` | tau=0.005 | 76.8 | 47.78 | 0.9997 | 22.08 | 2.44 |
| x264 | `scc_console_code` | CRF=23 | 45.3 | 40.20 | 0.9925 | 0.16 | 0.21 |
| x264 | `scc_console_code` | CRF=28 | 39.0 | 39.01 | 0.9880 | 0.17 | 0.22 |
| tfc | `scc_console_code` | tau=0.001 | 29.3 | inf | 1.0000 | 19.42 | 2.50 |
| tfc | `scc_console_code` | tau=0.005 | 29.3 | inf | 1.0000 | 19.80 | 2.30 |
| x264 | `scc_web_browsing` | CRF=23 | 40.7 | 40.10 | 0.9966 | 0.19 | 0.17 |
| x264 | `scc_web_browsing` | CRF=28 | 43.3 | 39.40 | 0.9947 | 0.24 | 0.18 |
| tfc | `scc_web_browsing` | tau=0.001 | 691.1 | 45.49 | 0.9971 | 13.05 | 3.31 |
| tfc | `scc_web_browsing` | tau=0.005 | 691.1 | 45.49 | 0.9971 | 12.79 | 3.59 |
| x264 | `scc_animation_ui` | CRF=23 | 50.9 | 41.12 | 0.9930 | 0.18 | 0.16 |
| x264 | `scc_animation_ui` | CRF=28 | 46.6 | 41.02 | 0.9925 | 0.17 | 0.17 |
| tfc | `scc_animation_ui` | tau=0.001 | 252.0 | 91.86 | 1.0000 | 17.11 | 2.62 |
| tfc | `scc_animation_ui` | tau=0.005 | 252.0 | 91.86 | 1.0000 | 16.83 | 2.52 |