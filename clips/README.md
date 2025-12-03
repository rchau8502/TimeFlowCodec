# Benchmark Clips Guidance

No large binaries are checked in. Use your own clips or fetch public-domain samples.

Recommended clip types:
- UI/screencast (e.g., slide decks, terminal demos)
- Animation/anime
- Low-noise static cam
- High-motion/noisy (failure case)

Optional: use Git LFS if you add clips (`git lfs install` then `git lfs track "clips/*.mp4"`).

You can also generate synthetic clips with ffmpeg lavfi (used in CI smoke):
```bash
# Scrolling gradient (easy case)
ffmpeg -y -f lavfi -i testsrc=size=320x240:rate=10 -t 2 clips/gradient.mp4
# Noisy motion (hard case)
ffmpeg -y -f lavfi -i noise=alls=20:allf=t+u:size=320x240:rate=10 -t 2 clips/noise.mp4
```
