# TimeFlowCodec Benchmark Report

Env: {'python': '3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 11:23:37) [Clang 14.0.6 ]', 'platform': 'darwin', 'git_commit': '7b635d288e0899c3f25274fc9eb20167ad3808cf', 'git_dirty': True, 'source_sha256': {'__init__.py': '82ac3e40135541624fc1af65575ff67e8b4f97996f09a46b267926b0ab39a175', '__main__.py': '3129adff95e0f38b87ab34cddfc27082c7892bcaec267ff5d40f160fdf55b44a', 'bench.py': '4bb3b5dfaa054e6ed7cd46850eaf69052153b0c255e10bc82ff9002c8a4ebecb', 'cli.py': '64e74393dc3a2ad2a8db853a4cb8cc515d006e66ee02081783c9625f1c2d2d13', 'constants.py': '5db91c2965918773d2fc3907fbeabc941885f3e77e80e70115b5224c227de461', 'decoder.py': '1891b522924895dcacc9f1609f2be09cb04f1e754c656ec9e997f33e85cc881a', 'encoder.py': '1d7a8c539353996594d474b4c2fd47d3a7c9606246af3898c0aec3e1789dbca1', 'format.py': '874c77888e2f383f2facabda104deb3bf2b726ec5ec61b0a8bc8cea754e723b7', 'models.py': '84fc22cd20dc95be79bffdb257cac8cc110bfff64177dba00a3ed0650b8660c2', 'profile.py': '503e854d4c1171a8c575155dc0fa3d695be6ddc8796b47388f7a7daba1f913a0', 'utils.py': '0b3b33ccd5fc8668c814ddab649ee60a886a43d736fdb0a66a3849c7c553956f', 'version.py': 'a7faa85bfb331e6afa5f7768c3a30b5c78d9f8c501fa672eb1ab0940783b4b31'}, 'ffmpeg': 'ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers', 'settings': {'clips': '/tmp/tfc-screen-clips', 'out': 'docs/benchmarks/screen', 'codecs': 'tfc,x264,x265,av1', 'presets': 'medium', 'crf': '18,23', 'tau': None, 'slope_threshold': 0.001, 'payload_comp_type': 3, 'tfc_preset': 'screen'}}

Metrics use decoded RGB frames directly: RGB PSNR and mean per-frame windowed RGB SSIM. TFC decode timing excludes video export. All metrics buffer short clips in RAM. Compare file sizes only at similar measured quality; equal CRF is not equal quality.

## Scoreboard

| codec | preset | crf | clip | size (bytes) | enc time (s) | dec time (s) | psnr | ssim |
|---|---|---|---|---|---|---|---|---|
| tfc | screen | - | screen_detail.mp4 | 1398 | 0.78 | 0.05 | 47.77666258566855 | 0.9927782567489487 |
| x264 | medium | 18 | screen_detail.mp4 | 2404 | 0.12 | 0.10 | 42.748328084681745 | 0.9980333352692717 |
| x264 | medium | 23 | screen_detail.mp4 | 2317 | 0.10 | 0.11 | 36.060219030265 | 0.9953354049413704 |
| x265 | medium | 18 | screen_detail.mp4 | 4702 | 0.18 | 0.11 | 48.56426401058897 | 0.9993873143183215 |
| x265 | medium | 23 | screen_detail.mp4 | 4551 | 0.16 | 0.14 | 44.37732675044355 | 0.9978809468299367 |
| av1 | medium | 18 | screen_detail.mp4 | 1468 | 0.43 | 0.11 | 77.63382387832438 | 0.9999992091849493 |
| av1 | medium | 23 | screen_detail.mp4 | 1516 | 0.36 | 0.08 | 76.34087601847088 | 0.9999980986765634 |