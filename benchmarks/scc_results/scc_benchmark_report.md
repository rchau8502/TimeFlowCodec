# JVET Screen Content Coding (SCC) Benchmark Report
**Date:** 2026-10-13 12:54:45

## 1. Bjontegaard Delta Summary (TFC vs H.264/libx264)
| Sequence Name | BD-Rate (%) | BD-PSNR (dB) |
|---|---:|---:|
| `scc_animation_ui` | **+0.00%** | **+53.04 dB** |
| `scc_console_code` | **+0.00%** | **+0.00 dB** |
| `scc_slideshow` | **+0.00%** | **+0.00 dB** |
| `scc_web_browsing` | **+472.85%** | **+0.00 dB** |
| **Overall Average** | **+118.21%** | **+13.26 dB** |

> *Note: Negative BD-Rate indicates bitstream savings conforming to ITU-T VCEG-M33.*

## 2. Full Rate-Distortion Sampling
| Codec | Clip | Parameter | Rate (kbps) | PSNR (dB) | SSIM | Enc Time (s) | Dec Time (s) |
|---|---|---|---:|---:|---:|---:|---:|
| x264 | `scc_slideshow` | CRF=18 | 25.6 | 39.99 | 0.9960 | 0.35 | 0.32 |
| x264 | `scc_slideshow` | CRF=23 | 22.4 | 39.58 | 0.9960 | 0.30 | 0.21 |
| x264 | `scc_slideshow` | CRF=28 | 20.4 | 39.56 | 0.9961 | 0.29 | 0.19 |
| x264 | `scc_slideshow` | CRF=33 | 22.3 | 38.26 | 0.9928 | 0.27 | 0.19 |
| tfc | `scc_slideshow` | tau=0.0001 | 41.6 | inf | 1.0000 | 61.23 | 2.66 |
| tfc | `scc_slideshow` | tau=0.001 | 41.4 | 67.08 | 1.0000 | 42.54 | 3.23 |
| tfc | `scc_slideshow` | tau=0.005 | 38.8 | 47.78 | 0.9997 | 40.85 | 2.62 |
| tfc | `scc_slideshow` | tau=0.02 | 31.3 | 40.38 | 0.9988 | 36.04 | 1.89 |
| x264 | `scc_console_code` | CRF=18 | 52.2 | 40.80 | 0.9945 | 0.35 | 0.30 |
| x264 | `scc_console_code` | CRF=23 | 45.3 | 40.20 | 0.9925 | 0.47 | 0.32 |
| x264 | `scc_console_code` | CRF=28 | 39.0 | 39.01 | 0.9880 | 0.40 | 0.27 |
| x264 | `scc_console_code` | CRF=33 | 35.6 | 37.06 | 0.9820 | 0.33 | 0.28 |
| tfc | `scc_console_code` | tau=0.0001 | 23.3 | inf | 1.0000 | 39.79 | 2.65 |
| tfc | `scc_console_code` | tau=0.001 | 23.3 | inf | 1.0000 | 35.10 | 1.97 |
| tfc | `scc_console_code` | tau=0.005 | 23.3 | inf | 1.0000 | 33.27 | 1.81 |
| tfc | `scc_console_code` | tau=0.02 | 23.3 | inf | 1.0000 | 36.52 | 2.13 |
| x264 | `scc_web_browsing` | CRF=18 | 43.1 | 40.55 | 0.9973 | 0.30 | 0.16 |
| x264 | `scc_web_browsing` | CRF=23 | 40.7 | 40.10 | 0.9966 | 0.36 | 0.20 |
| x264 | `scc_web_browsing` | CRF=28 | 43.3 | 39.40 | 0.9947 | 0.33 | 0.17 |
| x264 | `scc_web_browsing` | CRF=33 | 47.5 | 38.16 | 0.9897 | 0.47 | 0.24 |
| tfc | `scc_web_browsing` | tau=0.0001 | 347.1 | inf | 1.0000 | 26.97 | 3.40 |
| tfc | `scc_web_browsing` | tau=0.001 | 257.1 | 45.49 | 0.9971 | 20.74 | 2.99 |
| tfc | `scc_web_browsing` | tau=0.005 | 257.1 | 45.49 | 0.9971 | 21.33 | 2.96 |
| tfc | `scc_web_browsing` | tau=0.02 | 246.2 | 28.87 | 0.9763 | 20.33 | 3.74 |
| x264 | `scc_animation_ui` | CRF=18 | 57.1 | 41.21 | 0.9933 | 0.26 | 0.17 |
| x264 | `scc_animation_ui` | CRF=23 | 50.9 | 41.12 | 0.9930 | 0.31 | 0.20 |
| x264 | `scc_animation_ui` | CRF=28 | 46.6 | 41.02 | 0.9925 | 0.28 | 0.20 |
| x264 | `scc_animation_ui` | CRF=33 | 45.5 | 40.86 | 0.9922 | 0.25 | 0.23 |
| tfc | `scc_animation_ui` | tau=0.0001 | 50.4 | inf | 1.0000 | 60.22 | 3.22 |
| tfc | `scc_animation_ui` | tau=0.001 | 50.4 | inf | 1.0000 | 35.94 | 2.74 |
| tfc | `scc_animation_ui` | tau=0.005 | 50.4 | inf | 1.0000 | 42.10 | 3.40 |
| tfc | `scc_animation_ui` | tau=0.02 | 80.4 | 51.41 | 0.9995 | 33.28 | 2.49 |