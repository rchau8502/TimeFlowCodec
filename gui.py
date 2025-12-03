"""PySide6 GUI for RGB TimeFlowCodec."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from timeflowcodec import decode_tfc_to_video, encode_video_to_tfc


class CodecWorker(QThread):
    progress = Signal(int)
    message = Signal(str)
    finished_success = Signal(str)
    failed = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.progress.emit(0)
            self.func(*self.args, **self.kwargs)
            self.progress.emit(100)
            self.finished_success.emit("Done")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TimeFlowCodec RGB (Per-Pixel)")

        tabs = QTabWidget()
        tabs.addTab(self._build_compress_tab(), "Compress")
        tabs.addTab(self._build_decompress_tab(), "Decompress")
        self.setCentralWidget(tabs)

        self.encode_worker: CodecWorker | None = None
        self.decode_worker: CodecWorker | None = None

    def _build_compress_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)

        self.in_video_edit = QLineEdit()
        browse_in = QPushButton("Browse…")
        browse_in.clicked.connect(lambda: self._browse_file(self.in_video_edit, "Video Files (*.mp4 *.avi *.mov *.*)"))

        self.out_tfc_edit = QLineEdit()
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(lambda: self._browse_save(self.out_tfc_edit, "TFC Files (*.tfc)"))

        self.tau_spin = QDoubleSpinBox()
        self.tau_spin.setDecimals(4)
        self.tau_spin.setRange(0.0, 10.0)
        self.tau_spin.setValue(0.1)
        self.tau_spin.setSingleStep(0.01)

        self.slope_spin = QDoubleSpinBox()
        self.slope_spin.setDecimals(6)
        self.slope_spin.setRange(0.0, 1.0)
        self.slope_spin.setValue(1e-3)
        self.slope_spin.setSingleStep(1e-3)

        self.comp_combo = QComboBox()
        self.comp_combo.addItems(["None", "zlib", "LZMA"])
        self.comp_combo.setCurrentIndex(1)

        grid.addWidget(QLabel("Input video"), 0, 0)
        grid.addWidget(self.in_video_edit, 0, 1)
        grid.addWidget(browse_in, 0, 2)

        grid.addWidget(QLabel("Output .tfc"), 1, 0)
        grid.addWidget(self.out_tfc_edit, 1, 1)
        grid.addWidget(browse_out, 1, 2)

        grid.addWidget(QLabel("Tau"), 2, 0)
        grid.addWidget(self.tau_spin, 2, 1)

        grid.addWidget(QLabel("Slope threshold"), 3, 0)
        grid.addWidget(self.slope_spin, 3, 1)

        grid.addWidget(QLabel("Payload compression"), 4, 0)
        grid.addWidget(self.comp_combo, 4, 1)

        layout.addLayout(grid)

        self.compress_status = QLabel("")
        self.compress_progress = QProgressBar()
        self.compress_progress.setRange(0, 100)

        start_btn = QPushButton("Start Compression")
        start_btn.clicked.connect(self._start_compress)
        self.compress_start_btn = start_btn

        layout.addWidget(start_btn)
        layout.addWidget(self.compress_progress)
        layout.addWidget(self.compress_status)
        widget.setLayout(layout)
        return widget

    def _start_compress(self):
        input_path = self.in_video_edit.text().strip()
        output_path = self.out_tfc_edit.text().strip()
        if not input_path or not output_path:
            self.compress_status.setText("Select input and output paths")
            return
        comp_idx = self.comp_combo.currentIndex()
        self.compress_start_btn.setEnabled(False)
        self.compress_status.setText("Compressing…")
        self.compress_progress.setValue(0)
        self.encode_worker = CodecWorker(
            encode_video_to_tfc,
            input_path,
            output_path,
            self.tau_spin.value(),
            self.slope_spin.value(),
            comp_idx,
        )
        self.encode_worker.progress.connect(self.compress_progress.setValue)
        self.encode_worker.finished_success.connect(self._on_compress_done)
        self.encode_worker.failed.connect(self._on_compress_error)
        self.encode_worker.start()

    def _on_compress_done(self, msg: str):
        self.compress_status.setText("Compression complete")
        self.compress_progress.setValue(100)
        self.compress_start_btn.setEnabled(True)
        self.encode_worker = None

    def _on_compress_error(self, err: str):
        self.compress_status.setText(f"Error: {err}")
        self.compress_start_btn.setEnabled(True)
        self.encode_worker = None

    def _build_decompress_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)

        self.in_tfc_edit = QLineEdit()
        browse_in = QPushButton("Browse…")
        browse_in.clicked.connect(lambda: self._browse_file(self.in_tfc_edit, "TFC Files (*.tfc)"))

        self.out_video_edit = QLineEdit()
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(lambda: self._browse_save(self.out_video_edit, "Video Files (*.mp4 *.avi *.mov *.*)"))

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(30)

        grid.addWidget(QLabel("Input .tfc"), 0, 0)
        grid.addWidget(self.in_tfc_edit, 0, 1)
        grid.addWidget(browse_in, 0, 2)

        grid.addWidget(QLabel("Output video"), 1, 0)
        grid.addWidget(self.out_video_edit, 1, 1)
        grid.addWidget(browse_out, 1, 2)

        grid.addWidget(QLabel("FPS"), 2, 0)
        grid.addWidget(self.fps_spin, 2, 1)

        layout.addLayout(grid)

        self.decompress_status = QLabel("")
        self.decompress_progress = QProgressBar()
        self.decompress_progress.setRange(0, 100)

        start_btn = QPushButton("Start Decompression")
        start_btn.clicked.connect(self._start_decompress)
        self.decompress_start_btn = start_btn

        layout.addWidget(start_btn)
        layout.addWidget(self.decompress_progress)
        layout.addWidget(self.decompress_status)
        widget.setLayout(layout)
        return widget

    def _start_decompress(self):
        input_path = self.in_tfc_edit.text().strip()
        output_path = self.out_video_edit.text().strip()
        if not input_path or not output_path:
            self.decompress_status.setText("Select input and output paths")
            return
        self.decompress_start_btn.setEnabled(False)
        self.decompress_status.setText("Decompressing…")
        self.decompress_progress.setValue(0)
        self.decode_worker = CodecWorker(
            decode_tfc_to_video,
            input_path,
            output_path,
            self.fps_spin.value(),
        )
        self.decode_worker.progress.connect(self.decompress_progress.setValue)
        self.decode_worker.finished_success.connect(self._on_decompress_done)
        self.decode_worker.failed.connect(self._on_decompress_error)
        self.decode_worker.start()

    def _on_decompress_done(self, msg: str):
        self.decompress_status.setText("Decompression complete")
        self.decompress_progress.setValue(100)
        self.decompress_start_btn.setEnabled(True)
        self.decode_worker = None

    def _on_decompress_error(self, err: str):
        self.decompress_status.setText(f"Error: {err}")
        self.decompress_start_btn.setEnabled(True)
        self.decode_worker = None

    def _browse_file(self, line_edit: QLineEdit, filter_str: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", str(Path.home()), filter_str)
        if path:
            line_edit.setText(path)

    def _browse_save(self, line_edit: QLineEdit, filter_str: str):
        path, _ = QFileDialog.getSaveFileName(self, "Save file", str(Path.home()), filter_str)
        if path:
            line_edit.setText(path)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(600, 300)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
