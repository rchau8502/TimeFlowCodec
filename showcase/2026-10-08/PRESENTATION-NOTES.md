# Suggested showcase wording

"On this original 8-second, 640x360 color animation, native TimeFlowCodec preserved every RGB pixel in 49,948 bytes. That is 99.55% smaller than H.264 RGB with libx264rgb / veryslow / CRF 0, and 94.54% smaller than AV1 RGB with libaom-av1 / cpu-used 2 / lossless 1. The animation deliberately uses linear per-pixel color changes, which match TFC's temporal model."

Show the scrolling control next: lossless H.264 uses 39,718 bytes; TFC uses 240,365 bytes. Motion prediction wins there.

The supported claim is that TFC can outperform these named encoder settings on this specific synthetic content at exact RGB quality. These results do not support a universal 60% saving, a general codec ranking, or a patent claim. Encoding times are not controlled speed comparisons.

Share SHOWCASE.mp4 for a quick presentation; share the full folder for original masters, native TFC files, comparison encodes, decoded stills, raw measurements and reproduction source. Media is original and dedicated under CC0 1.0. Presentation playback uses H.264 for compatibility and is not a live TFC decoder.
