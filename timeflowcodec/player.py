"""TimeFlowCodec Native Player Application.

Copyright (c) 2025–2026 Rich Chau. All rights reserved.
Proprietary Desktop Application License (see LICENSE Section 2).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class FileOpenApplication(QApplication):
    """Dispatch Finder file-open events, including those arriving during startup."""

    def __init__(self, args):
        self.open_handler = None
        self.pending_files = []
        super().__init__(args)

    def event(self, event):
        if event.type() == QEvent.Type.FileOpen and event.file():
            if self.open_handler is None:
                self.pending_files.append(event.file())
            else:
                self.open_handler(event.file())
            return True
        return super().event(event)

    def set_open_handler(self, handler):
        self.open_handler = handler
        for path in self.pending_files:
            handler(path)
        self.pending_files.clear()


class PlayerWindow(QMainWindow):
    def __init__(self, parent=None, demo_path=None):
        super().__init__(parent)
        self.setWindowTitle("TimeFlowCodec Player")
        self.resize(900, 640)
        self.setAcceptDrops(True)
        self._source_fps = None
        self.audio_output = None
        self.audio = None
        self.frames = None
        self.process = None
        self.cache = None
        self.playing = False
        self.frame_index = 0
        self._error_text = ""
        self._pixmap = QPixmap()
        self._start_time = 0.0
        self._start_frame = 0

        root = QWidget()
        layout = QVBoxLayout(root)
        controls = QHBoxLayout()
        self.open_button = QPushButton("Open .tfc…")
        self.open_button.clicked.connect(self.choose_file)
        self.cancel_button = QPushButton("Cancel loading")
        self.cancel_button.clicked.connect(self.cancel_load)
        self.cancel_button.setEnabled(False)
        self.demo_button = QPushButton("Play demo")
        self.demo_button.setVisible(bool(demo_path))
        self.demo_button.clicked.connect(lambda: self._play_demo(demo_path))
        controls.addWidget(self.open_button)
        controls.addWidget(self.demo_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch()
        controls.addWidget(QLabel("Playback FPS"))
        self.fps = QDoubleSpinBox()
        self.fps.setRange(1, 240)
        self.fps.setDecimals(3)
        self.fps.setValue(30)
        self.fps.valueChanged.connect(self._fps_changed)
        controls.addWidget(self.fps)
        layout.addLayout(controls)
        note = QLabel("v5 uses source timing and audio. For legacy files, set FPS to match the source.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.display = QLabel("Open or drop a .tfc file to play")
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display.setMinimumSize(160, 120)
        self.display.setStyleSheet("background: #101419; color: #dce3eb;")
        layout.addWidget(self.display, 1)
        transport = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.seek)
        self.position = QLabel("0 / 0")
        transport.addWidget(self.play_button)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.position)
        layout.addLayout(transport)
        self.status = QLabel("v5 playback uses a temporary disk cache. Legacy preview limits still apply.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.setCentralWidget(root)
        self.timer = QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self._tick)

    def _play_demo(self, path):
        if path:
            self.fps.setValue(12)
            self.open_file(path)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open TimeFlowCodec video", "", "TimeFlowCodec (*.tfc)")
        if path:
            self.open_file(path)

    def open_file(self, path: str):
        self.pause()
        if not self._stop_child():
            return
        self._discard_cache()
        self._pixmap = QPixmap()
        self.display.setText("Decoding frames…")
        self.play_button.setEnabled(False)
        self.slider.setEnabled(False)
        self.position.setText("0 / 0")
        self.cache = tempfile.TemporaryDirectory(prefix="timeflow-player-")
        self._error_text = ""
        self.status.setText(f"Loading {Path(path).name}…")
        self.setWindowTitle(f"{Path(path).name} — TimeFlowCodec Player")
        process = QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(Path(__file__).resolve().parent.parent))
        process.readyReadStandardError.connect(self._read_error)
        process.finished.connect(self._loaded)
        process.errorOccurred.connect(self._process_error)
        output = str(Path(self.cache.name) / "frames.npy")
        if getattr(sys, "frozen", False):
            args = ["--decode-for-player", str(Path(path).resolve()), output]
        else:
            args = ["-m", "timeflowcodec.player_decode", str(Path(path).resolve()), output]
        self.cancel_button.setEnabled(True)
        process.start(sys.executable, args)

    def _read_error(self):
        if self.process is not None:
            text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
            self._error_text = (self._error_text + text)[-4000:]

    def _process_error(self, error):
        if self.process is not None and error == QProcess.ProcessError.FailedToStart:
            message = self.process.errorString()
            self._stop_child()
            self._discard_cache()
            self.status.setText(f"Could not open video: {message}")

    def _loaded(self, code, exit_status):
        process = self.sender()
        if process is not self.process:
            return
        self._read_error()
        self.process = None
        process.deleteLater()
        self.cancel_button.setEnabled(False)
        try:
            if code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
                raise ValueError(self._error_text.strip() or "Decoder process stopped unexpectedly.")
            cache_path = Path(self.cache.name) / "frames.npy"
            metadata_path = cache_path.with_suffix(".json")
            if metadata_path.exists():
                from .player_decode import DiskFrames

                metadata = json.loads(metadata_path.read_text())
                self.frames = DiskFrames(cache_path)
                self._source_fps = metadata["fps_num"] / metadata["fps_den"]
                self.fps.setValue(self._source_fps)
                self.fps.setEnabled(False)
                if metadata["has_audio"]:
                    self._ensure_audio()
                    self.audio.setSource(QUrl.fromLocalFile(str(cache_path.with_suffix(".mka"))))
            else:
                self.frames = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            if self.frames.ndim != 4 or self.frames.shape[-1] != 3 or not len(self.frames):
                raise ValueError("Invalid frame cache.")
            self.slider.setRange(0, len(self.frames) - 1)
            self.slider.setEnabled(True)
            self.play_button.setEnabled(True)
            self.status.setText(f"{self.frames.shape[2]} × {self.frames.shape[1]} · {len(self.frames)} frames")
            self.seek(0)
            self.play()
        except Exception as exc:
            self._discard_cache()
            self.display.setText("Unable to load video")
            self.status.setText(f"Could not open video: {exc}")

    def _stop_child(self):
        process = self.process
        self.process = None
        if process is not None:
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
                if not process.waitForFinished(3000):
                    if process.state() != QProcess.ProcessState.NotRunning:
                        self.process = process
                        self.status.setText("Waiting for the decoder to stop…")
                        return False
            process.deleteLater()
        self.cancel_button.setEnabled(False)
        return True

    def _ensure_audio(self):
        if self.audio is None:
            self.audio_output = QAudioOutput(self)
            self.audio = QMediaPlayer(self)
            self.audio.setAudioOutput(self.audio_output)
            self.audio.errorOccurred.connect(self._audio_error)

    def _audio_error(self, _error, message):
        self.pause()
        self.status.setText(f"Audio playback failed: {message}")

    def _discard_cache(self):
        if self.audio is not None:
            self.audio.stop()
            self.audio.setSource(QUrl())
        self._source_fps = None
        self.fps.setEnabled(True)
        if self.frames is not None:
            mmap = getattr(self.frames, "_mmap", None)
            self.frames = None
            if mmap is not None:
                mmap.close()
        if self.cache is not None:
            self.cache.cleanup()
            self.cache = None

    def cancel_load(self):
        if self.process is not None and self._stop_child():
            self._discard_cache()
            self.display.setText("Open or drop a .tfc file to play")
            self.status.setText("Loading cancelled.")

    def seek(self, index):
        if self.frames is None:
            return
        self.frame_index = max(0, min(int(index), len(self.frames) - 1))
        self._start_frame = self.frame_index
        self._start_time = time.monotonic()
        if self.audio is not None and not self.audio.source().isEmpty():
            self.audio.setPosition(round(self.frame_index * 1000 / self._playback_fps()))
        self._render()

    def _render(self):
        frame = self.frames[self.frame_index]
        height, width = frame.shape[:2]
        # Detach Qt image storage before the memory-mapped cache can be closed.
        image = QImage(frame.data, width, height, int(frame.strides[0]), QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._scale_image()
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_index)
        self.slider.blockSignals(False)
        seconds = self.frame_index / self._playback_fps()
        self.position.setText(f"{self.frame_index + 1} / {len(self.frames)} · {seconds:.2f}s")

    def _scale_image(self):
        if not self._pixmap.isNull():
            self.display.setPixmap(
                self._pixmap.scaled(
                    self.display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )

    def play(self):
        if self.frames is None:
            return
        if self.frame_index == len(self.frames) - 1:
            self.seek(0)
        self.playing = True
        self._start_frame = self.frame_index
        self._start_time = time.monotonic()
        self.play_button.setText("Pause")
        self.timer.start()
        if self.audio is not None and not self.audio.source().isEmpty():
            self.audio.play()

    def pause(self):
        self.playing = False
        if self.audio is not None:
            self.audio.pause()
        self.timer.stop()
        self.play_button.setText("Play")

    def toggle_play(self):
        self.pause() if self.playing else self.play()

    def _tick(self):
        if self.frames is None or not self.playing:
            return
        index = self._start_frame + int((time.monotonic() - self._start_time) * self._playback_fps())
        if (
            self.audio is not None
            and not self.audio.source().isEmpty()
            and self.audio.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        ):
            index = int(self.audio.position() * self._playback_fps() / 1000)
        next_index = min(index, len(self.frames) - 1)
        if next_index != self.frame_index:
            self.frame_index = next_index
            self._render()
        if index >= len(self.frames):
            self.pause()

    def _playback_fps(self):
        return self._source_fps or self.fps.value()

    def _fps_changed(self):
        if self.frames is not None:
            self.seek(self.frame_index)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._scale_image()

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_file(url.toLocalFile())
                event.acceptProposedAction()
                break

    def closeEvent(self, event):  # noqa: N802
        self.pause()
        if not self._stop_child():
            event.ignore()
            return
        self._discard_cache()
        event.accept()


def main(argv: list[str] | None = None):
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--decode-for-player"]:
        from .player_decode import main as decode_main

        return decode_main(args[1:])
    parser = argparse.ArgumentParser(description="Play a TimeFlowCodec video without converting it.")
    parser.add_argument("input", nargs="?", help=".tfc file to open")
    parser.add_argument("--fps", type=float, default=30, help="Source FPS (legacy files do not store timing)")
    parsed = parser.parse_args(args)
    if not 1 <= parsed.fps <= 240:
        parser.error("FPS must be between 1 and 240")
    app = FileOpenApplication([sys.argv[0]])
    app.setApplicationName("TimeFlowCodec Player")
    window = PlayerWindow()
    window.fps.setValue(parsed.fps)
    app.set_open_handler(window.open_file)
    window.show()
    if parsed.input:
        window.open_file(parsed.input)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
