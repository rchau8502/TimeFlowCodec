"""Generate canonical Screen Content Coding (SCC) benchmark sequences.

Simulates the standard JVET / ITU-T Screen Content Coding test categories:
1. scc_slideshow.mp4: Slide presentation with text bullets, graphs, and transitions.
2. scc_console_code.mp4: Developer terminal with syntax-highlighted code and blinking cursor.
3. scc_web_browsing.mp4: Web UI layout with cards, typography, and vertical viewport scrolling.
4. scc_animation_ui.mp4: Vector motion graphics with geometric transforms and easing curves.

Encoded losslessly in full 24-bit RGB using libx264rgb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def generate_slideshow(path: Path, frames: int = 36, fps: int = 12) -> None:
    """Generate 1280x720 slide presentation sequence."""
    h, w = 720, 1280
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]
    ) as writer:
        for t in range(frames):
            # Clean light slide canvas
            frame = np.full((h, w, 3), 248, dtype=np.uint8)
            # Header banner (UCI Anteater Blue)
            frame[0:90, :] = [0, 100, 164]
            # Gold accent line
            frame[90:96, :] = [255, 210, 0]
            # Slide Title simulated text block
            frame[30:60, 60:480] = 255
            # Subtitle
            frame[65:75, 60:320] = 220
            # Left Column: bullet list items (sharp single-pixel synthetic text lines)
            for b in range(5):
                y_base = 160 + b * 90
                # Bullet dot
                frame[y_base:y_base+14, 80:94] = [0, 100, 164]
                # Text line 1
                frame[y_base:y_base+12, 110:550] = [40, 44, 52]
                # Text line 2
                frame[y_base+20:y_base+30, 110:480] = [90, 96, 108]
                # Progressive bullet reveal
                if b > 2 and t < 18:
                    frame[y_base-10:y_base+50, 70:580] = 248  # Hidden until frame 18
            # Right Column: Synthetic chart
            frame[160:640, 680:1200] = 255  # Chart card
            # Axes
            frame[600:602, 720:1160] = [60, 60, 60]
            frame[220:602, 720:722] = [60, 60, 60]
            # Bars with progressive growth
            colors = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40]]
            for i, c in enumerate(colors):
                bx = 760 + i * 100
                target_bh = 120 + i * 70
                bh = int(target_bh * min(1.0, (t + 1) / 24.0))
                if bh > 0:
                    frame[600-bh:600, bx:bx+60] = c
            writer.append_data(frame)
    print(f"Generated {path.name} ({w}x{h}, {frames} frames)")


def generate_console_code(path: Path, frames: int = 48, fps: int = 24) -> None:
    """Generate 1280x720 code editor terminal sequence."""
    h, w = 720, 1280
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]
    ) as writer:
        for t in range(frames):
            # Dark IDE background (#1E1E1E)
            frame = np.full((h, w, 3), 30, dtype=np.uint8)
            # Window titlebar (#323233)
            frame[0:36, :] = [50, 50, 51]
            # Window control dots (Red, Yellow, Green)
            frame[12:24, 18:30] = [237, 101, 90]
            frame[12:24, 38:50] = [229, 192, 123]
            frame[12:24, 58:70] = [97, 174, 238]
            # Sidebar gutter (#252526)
            frame[36:, 0:60] = [37, 37, 38]
            # Code line numbers & syntax rows
            for line_idx in range(22):
                y = 56 + line_idx * 28
                # Line number
                frame[y:y+12, 20:45] = [100, 100, 100]
                # Indentation
                indent = 80 + (line_idx % 4) * 35
                # Keyword (Blue #61AFEF)
                frame[y:y+14, indent:indent+55] = [97, 174, 238]
                # Identifier (White #ABB2BF)
                frame[y:y+14, indent+65:indent+160] = [171, 178, 191]
                # Operator / String (Green #98C379)
                frame[y:y+14, indent+170:indent+290] = [152, 195, 121]
                # Comment (Grey #5C6370)
                if line_idx % 5 == 0:
                    frame[y:y+12, indent+310:indent+490] = [92, 99, 112]
            # Blinking text cursor at line 8
            cursor_y = 56 + 8 * 28
            cursor_x = 80 + 310 + (t % 12) * 4
            if (t // 8) % 2 == 0:  # Blink toggle
                frame[cursor_y:cursor_y+18, cursor_x:cursor_x+3] = [255, 255, 255]
            writer.append_data(frame)
    print(f"Generated {path.name} ({w}x{h}, {frames} frames)")


def generate_web_browsing(path: Path, frames: int = 36, fps: int = 12) -> None:
    """Generate 1280x720 web browsing sequence with vertical scrolling."""
    h, w = 720, 1280
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]
    ) as writer:
        for t in range(frames):
            frame = np.full((h, w, 3), 245, dtype=np.uint8)
            # Browser navigation bar
            frame[0:60, :] = [255, 255, 255]
            frame[59:60, :] = [220, 220, 220]
            # URL search bar
            frame[14:46, 120:960] = [238, 240, 242]
            # Scroll offset (scroll down after frame 10)
            scroll_y = int(max(0, t - 10) * 8)
            # Web page card layout
            for card_i in range(8):
                cy = 90 + card_i * 180 - scroll_y
                if -150 < cy < h:
                    card_top = max(60, cy)
                    card_bot = min(h, cy + 150)
                    if card_bot > card_top:
                        frame[card_top:card_bot, 120:1160] = [255, 255, 255]
                        # Card thumbnail
                        img_top = max(card_top, cy + 15)
                        img_bot = min(card_bot, cy + 135)
                        if img_bot > img_top:
                            frame[img_top:img_bot, 140:300] = [31, 119, 180]
                        # Card headline
                        txt_top = max(card_top, cy + 25)
                        txt_bot = min(card_bot, cy + 45)
                        if txt_bot > txt_top:
                            frame[txt_top:txt_bot, 330:850] = [33, 37, 41]
                        # Card summary text
                        txt_top2 = max(card_top, cy + 60)
                        txt_bot2 = min(card_bot, cy + 75)
                        if txt_bot2 > txt_top2:
                            frame[txt_top2:txt_bot2, 330:980] = [108, 117, 125]
            writer.append_data(frame)
    print(f"Generated {path.name} ({w}x{h}, {frames} frames)")


def generate_animation_ui(path: Path, frames: int = 36, fps: int = 24) -> None:
    """Generate 1280x720 UI animation with vector geometry and easing."""
    h, w = 720, 1280
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]
    ) as writer:
        for t in range(frames):
            frame = np.full((h, w, 3), 20, dtype=np.uint8)
            # Animated modal dialog box with cubic easing scale
            progress = min(1.0, (t + 1) / 20.0)
            ease = 1.0 - (1.0 - progress) ** 3  # Cubic ease-out
            bw = int(480 * ease)
            bh = int(320 * ease)
            if bw > 10 and bh > 10:
                bx0 = (w - bw) // 2
                by0 = (h - bh) // 2
                frame[by0:by0+bh, bx0:bx0+bw] = [38, 42, 54]
                # Header
                frame[by0:by0+40, bx0:bx0+bw] = [0, 100, 164]
                # Pulsing confirmation button
                btn_w = int(140 * ease)
                btn_h = int(44 * ease)
                btn_x = bx0 + (bw - btn_w) // 2
                btn_y = by0 + bh - 60
                pulse_color = int(210 + 40 * np.sin(t / 3.0))
                frame[btn_y:btn_y+btn_h, btn_x:btn_x+btn_w] = [0, pulse_color, 120]
            writer.append_data(frame)
    print(f"Generated {path.name} ({w}x{h}, {frames} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("clips/scc_benchmarks"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    generate_slideshow(args.out_dir / "scc_slideshow.mp4")
    generate_console_code(args.out_dir / "scc_console_code.mp4")
    generate_web_browsing(args.out_dir / "scc_web_browsing.mp4")
    generate_animation_ui(args.out_dir / "scc_animation_ui.mp4")
    print("\nAll canonical SCC benchmark sequences generated successfully!")


if __name__ == "__main__":
    main()
