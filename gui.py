"""Desktop GUI for TimeFlowCodec with Apple Silicon-friendly defaults."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from timeflowcodec import decode_tfc_to_video, encode_video_to_tfc, get_version_string


def _is_apple_silicon(
    system_name: str | None = None, machine_name: str | None = None
) -> bool:
    system_name = system_name or platform.system()
    machine_name = (machine_name or platform.machine()).lower()
    return system_name == "Darwin" and machine_name in {"arm64", "aarch64"}


def _suggest_output_path(input_path: str, suffix: str) -> str:
    if not input_path:
        return ""
    path = Path(input_path)
    return str(path.with_suffix(suffix))


def _preset_ui_defaults(preset: str) -> dict[str, float | int | bool | str]:
    if preset == "anime":
        return {
            "tau": 0.20,
            "slope_threshold": 0.003,
            "scene_cut": "auto",
            "scene_threshold": 0.27,
            "matrix_mode": True,
            "compression_index": 3,
        }
    if preset == "lownoise":
        return {
            "tau": 0.14,
            "slope_threshold": 0.0015,
            "scene_cut": "auto",
            "scene_threshold": 0.32,
            "matrix_mode": True,
            "compression_index": 3,
        }
    return {
        "tau": 0.10,
        "slope_threshold": 0.001,
        "scene_cut": "off",
        "scene_threshold": 0.35,
        "matrix_mode": False,
        "compression_index": 3,
    }


class FileDropLineEdit(QLineEdit):
    fileDropped = Signal(str)

    def __init__(self, placeholder: str):
        super().__init__()
        self.setAcceptDrops(True)
        self.setPlaceholderText(placeholder)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.setText(path)
                self.fileDropped.emit(path)
                event.acceptProposedAction()
                return
        super().dropEvent(event)


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
            self.message.emit("Running codec job")
            self.func(*self.args, **self.kwargs)
            self.progress.emit(100)
            self.finished_success.emit("Done")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.apple_silicon = _is_apple_silicon()
        self.encode_worker: CodecWorker | None = None
        self.decode_worker: CodecWorker | None = None

        self.setWindowTitle("TimeFlowCodec")
        self.resize(1100, 760)
        self.setMinimumSize(960, 680)
        self._apply_style()

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(18)

        root_layout.addWidget(self._build_header())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_compress_tab(), "Compress")
        self.tabs.addTab(self._build_decompress_tab(), "Decompress")
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(root)

    def _apply_style(self) -> None:
        QApplication.setStyle("Fusion")
        self.setStyleSheet(
            """
            QWidget {
                background: #11161d;
                color: #eef3f7;
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
                font-size: 14px;
            }
            QMainWindow {
                background: #0f141b;
            }
            QTabWidget::pane {
                border: 1px solid #26303d;
                border-radius: 18px;
                background: #121922;
                padding: 12px;
            }
            QTabBar::tab {
                background: #192330;
                color: #c7d4df;
                border: 0;
                padding: 12px 22px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                margin-right: 6px;
            }
            QTabBar::tab:selected {
                background: #0ea5a4;
                color: #061517;
                font-weight: 700;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #324153;
                border-radius: 12px;
                padding: 10px 12px;
                background: #0e141b;
                selection-background-color: #0ea5a4;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #3ad0cf;
            }
            QPushButton {
                border: 0;
                border-radius: 14px;
                padding: 11px 18px;
                background: #1f2b38;
                color: #eef3f7;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #28384a;
            }
            QPushButton#primaryButton {
                background: #12b8b1;
                color: #062021;
                font-weight: 800;
                padding: 14px 18px;
            }
            QPushButton#primaryButton:hover {
                background: #3ad0cf;
            }
            QCheckBox {
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 6px;
                border: 1px solid #4d6077;
                background: #0d141b;
            }
            QCheckBox::indicator:checked {
                background: #12b8b1;
                border: 1px solid #12b8b1;
            }
            QProgressBar {
                border: 0;
                border-radius: 8px;
                background: #0b1016;
                min-height: 12px;
                max-height: 12px;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 8px;
                background: #12b8b1;
            }
            QLabel#heroTitle {
                font-size: 30px;
                font-weight: 800;
                color: #f5fbff;
            }
            QLabel#heroSub {
                color: #9fb0c1;
                font-size: 15px;
            }
            QLabel#badge {
                background: #16222d;
                border: 1px solid #243446;
                border-radius: 999px;
                padding: 6px 10px;
                color: #b9cad8;
                font-size: 12px;
                font-weight: 700;
            }
            QFrame#card {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #121b25, stop:1 #0e141c);
                border: 1px solid #273241;
                border-radius: 20px;
            }
            QLabel#cardTitle {
                font-size: 16px;
                font-weight: 800;
                color: #f0f7fb;
            }
            QLabel#cardCopy {
                color: #95a8b8;
                line-height: 1.4em;
            }
            QLabel#statusGood {
                color: #73e0bc;
                font-weight: 700;
            }
            QLabel#statusWarn {
                color: #f5cb71;
                font-weight: 700;
            }
            QLabel#statusBad {
                color: #ff9d9d;
                font-weight: 700;
            }
            """
        )

    def _build_header(self) -> QWidget:
        card = self._make_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("TimeFlowCodec")
        title.setObjectName("heroTitle")
        subtitle = QLabel(
            "Per-pixel temporal compression with presets for anime, UI capture, and low-noise footage."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("heroSub")

        badges = QHBoxLayout()
        badges.setSpacing(10)
        badges.addWidget(self._make_badge(get_version_string()))
        if self.apple_silicon:
            badges.addWidget(self._make_badge("Apple Silicon detected"))
            badges.addWidget(self._make_badge("Mac profile recommended"))
        else:
            badges.addWidget(self._make_badge("Cross-platform desktop app"))
        badges.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(badges)
        return card

    def _build_compress_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(16)

        layout.addWidget(
            self._make_info_strip(
                "Compression",
                "Drop an `.mp4`, `.mov`, or `.avi` file. The app defaults to "
                "the anime preset and Apple Silicon-safe settings.",
            )
        )

        io_card = self._make_card()
        io_layout = QFormLayout(io_card)
        io_layout.setContentsMargins(20, 18, 20, 18)
        io_layout.setSpacing(14)
        io_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.in_video_edit = FileDropLineEdit("Drop a source video here")
        self.out_tfc_edit = FileDropLineEdit("Output .tfc path")
        self.in_video_edit.textChanged.connect(self._autofill_compress_output)
        self.in_video_edit.fileDropped.connect(self._autofill_compress_output)

        io_layout.addRow(
            "Input video", self._build_path_row(self.in_video_edit, self._browse_input_video)
        )
        io_layout.addRow(
            "Output .tfc", self._build_path_row(self.out_tfc_edit, self._browse_output_tfc)
        )
        layout.addWidget(io_card)

        controls = QHBoxLayout()
        controls.setSpacing(16)

        profile_card = self._make_card()
        profile_layout = QFormLayout(profile_card)
        profile_layout.setContentsMargins(20, 18, 20, 18)
        profile_layout.setSpacing(14)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["anime", "lownoise", "custom"])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)

        self.comp_combo = QComboBox()
        self.comp_combo.addItems(["None", "zlib", "LZMA", "zstd"])
        self.comp_combo.setCurrentIndex(3)

        self.macbook_profile_checkbox = QCheckBox("Use Apple Silicon / MacBook safe runtime")
        self.macbook_profile_checkbox.setChecked(self.apple_silicon)
        self.macbook_profile_checkbox.toggled.connect(self._refresh_platform_status)

        self.matrix_mode_checkbox = QCheckBox("Enable matrix low-rank fallback")
        self.matrix_mode_checkbox.setChecked(True)

        profile_layout.addRow("Preset", self.preset_combo)
        profile_layout.addRow("Payload compression", self.comp_combo)
        profile_layout.addRow("", self.macbook_profile_checkbox)
        profile_layout.addRow("", self.matrix_mode_checkbox)

        self.platform_status = QLabel("")
        self.platform_status.setWordWrap(True)
        self.platform_status.setObjectName("cardCopy")
        profile_layout.addRow("Platform", self.platform_status)

        model_card = self._make_card()
        model_layout = QFormLayout(model_card)
        model_layout.setContentsMargins(20, 18, 20, 18)
        model_layout.setSpacing(14)

        self.tau_spin = QDoubleSpinBox()
        self.tau_spin.setDecimals(4)
        self.tau_spin.setRange(0.0, 10.0)
        self.tau_spin.setSingleStep(0.01)

        self.slope_spin = QDoubleSpinBox()
        self.slope_spin.setDecimals(6)
        self.slope_spin.setRange(0.0, 1.0)
        self.slope_spin.setSingleStep(0.0005)

        self.scene_cut_combo = QComboBox()
        self.scene_cut_combo.addItems(["off", "auto"])

        self.scene_threshold_spin = QDoubleSpinBox()
        self.scene_threshold_spin.setDecimals(3)
        self.scene_threshold_spin.setRange(0.01, 2.0)
        self.scene_threshold_spin.setSingleStep(0.05)

        model_layout.addRow("Tau", self.tau_spin)
        model_layout.addRow("Slope threshold", self.slope_spin)
        model_layout.addRow("Scene cut", self.scene_cut_combo)
        model_layout.addRow("Scene threshold", self.scene_threshold_spin)

        self.preset_note = QLabel("")
        self.preset_note.setWordWrap(True)
        self.preset_note.setObjectName("cardCopy")
        model_layout.addRow("Note", self.preset_note)

        controls.addWidget(profile_card, 1)
        controls.addWidget(model_card, 1)
        layout.addLayout(controls)

        action_card = self._make_card()
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(20, 18, 20, 18)
        action_layout.setSpacing(12)

        self.compress_status = QLabel("Ready")
        self.compress_status.setObjectName("statusGood")
        self.compress_status.setWordWrap(True)

        self.compress_progress = QProgressBar()
        self.compress_progress.setRange(0, 100)
        self.compress_progress.setValue(0)

        self.compress_start_btn = QPushButton("Start Compression")
        self.compress_start_btn.setObjectName("primaryButton")
        self.compress_start_btn.clicked.connect(self._start_compress)

        action_layout.addWidget(self.compress_start_btn)
        action_layout.addWidget(self.compress_progress)
        action_layout.addWidget(self.compress_status)
        layout.addWidget(action_card)

        layout.addStretch(1)
        self._on_preset_changed(self.preset_combo.currentText())
        self._refresh_platform_status()
        return page

    def _build_decompress_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(16)

        layout.addWidget(
            self._make_info_strip(
                "Decompression",
                "Decompression streams frames directly to the output video by "
                "default so large files stay reasonable on laptops.",
            )
        )

        io_card = self._make_card()
        io_layout = QFormLayout(io_card)
        io_layout.setContentsMargins(20, 18, 20, 18)
        io_layout.setSpacing(14)

        self.in_tfc_edit = FileDropLineEdit("Drop a .tfc file here")
        self.out_video_edit = FileDropLineEdit("Output video path")
        self.in_tfc_edit.textChanged.connect(self._autofill_decompress_output)
        self.in_tfc_edit.fileDropped.connect(self._autofill_decompress_output)

        io_layout.addRow(
            "Input .tfc", self._build_path_row(self.in_tfc_edit, self._browse_input_tfc)
        )
        io_layout.addRow(
            "Output video",
            self._build_path_row(self.out_video_edit, self._browse_output_video),
        )
        layout.addWidget(io_card)

        settings_card = self._make_card()
        settings_layout = QFormLayout(settings_card)
        settings_layout.setContentsMargins(20, 18, 20, 18)
        settings_layout.setSpacing(14)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(30)

        self.stream_output_checkbox = QCheckBox("Stream decoded frames to disk")
        self.stream_output_checkbox.setChecked(True)

        self.decode_hint = QLabel(
            "Recommended on Apple Silicon: keep streaming enabled to avoid buffering the full video in RAM."
        )
        self.decode_hint.setObjectName("cardCopy")
        self.decode_hint.setWordWrap(True)

        settings_layout.addRow("FPS", self.fps_spin)
        settings_layout.addRow("", self.stream_output_checkbox)
        settings_layout.addRow("Hint", self.decode_hint)
        layout.addWidget(settings_card)

        action_card = self._make_card()
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(20, 18, 20, 18)
        action_layout.setSpacing(12)

        self.decompress_status = QLabel("Ready")
        self.decompress_status.setObjectName("statusGood")
        self.decompress_status.setWordWrap(True)

        self.decompress_progress = QProgressBar()
        self.decompress_progress.setRange(0, 100)
        self.decompress_progress.setValue(0)

        self.decompress_start_btn = QPushButton("Start Decompression")
        self.decompress_start_btn.setObjectName("primaryButton")
        self.decompress_start_btn.clicked.connect(self._start_decompress)

        action_layout.addWidget(self.decompress_start_btn)
        action_layout.addWidget(self.decompress_progress)
        action_layout.addWidget(self.decompress_status)
        layout.addWidget(action_card)

        layout.addStretch(1)
        return page

    def _make_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        return card

    def _make_info_strip(self, title_text: str, body_text: str) -> QWidget:
        card = self._make_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        title = QLabel(title_text)
        title.setObjectName("cardTitle")
        body = QLabel(body_text)
        body.setObjectName("cardCopy")
        body.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(body)
        return card

    def _make_badge(self, text: str) -> QLabel:
        badge = QLabel(text)
        badge.setObjectName("badge")
        return badge

    def _build_path_row(self, line_edit: QLineEdit, callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        button = QPushButton("Browse…")
        button.clicked.connect(callback)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return row

    def _browse_input_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input video",
            str(Path.home()),
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm *.*)",
        )
        if path:
            self.in_video_edit.setText(path)

    def _browse_output_tfc(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save .tfc file", self.out_tfc_edit.text() or str(Path.home()), "TFC Files (*.tfc)"
        )
        if path:
            self.out_tfc_edit.setText(path)

    def _browse_input_tfc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select .tfc file", str(Path.home()), "TFC Files (*.tfc)"
        )
        if path:
            self.in_tfc_edit.setText(path)

    def _browse_output_video(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save reconstructed video",
            self.out_video_edit.text() or str(Path.home()),
            "Video Files (*.mp4 *.mov *.avi *.mkv *.*)",
        )
        if path:
            self.out_video_edit.setText(path)

    def _autofill_compress_output(self, input_path: str) -> None:
        if not input_path:
            return
        if not self.out_tfc_edit.text().strip():
            self.out_tfc_edit.setText(_suggest_output_path(input_path, ".tfc"))

    def _autofill_decompress_output(self, input_path: str) -> None:
        if not input_path:
            return
        if not self.out_video_edit.text().strip():
            self.out_video_edit.setText(_suggest_output_path(input_path, ".mp4"))

    def _refresh_platform_status(self) -> None:
        if self.apple_silicon and self.macbook_profile_checkbox.isChecked():
            self.platform_status.setText(
                "Apple Silicon mode active. The encoder will prefer bounded RAM "
                "behavior, tile sharing, uint8 internals, and safer scene handling."
            )
        elif self.apple_silicon:
            self.platform_status.setText(
                "Apple Silicon detected, but the Mac safe profile is off. "
                "This can increase RAM pressure on large clips."
            )
        else:
            self.platform_status.setText(
                "Cross-platform mode. The same UI works on macOS and Linux, "
                "but the Mac safe profile is still available."
            )

    def _on_preset_changed(self, preset: str) -> None:
        defaults = _preset_ui_defaults(preset)
        self.tau_spin.setValue(float(defaults["tau"]))
        self.slope_spin.setValue(float(defaults["slope_threshold"]))
        self.scene_cut_combo.setCurrentText(str(defaults["scene_cut"]))
        self.scene_threshold_spin.setValue(float(defaults["scene_threshold"]))
        self.matrix_mode_checkbox.setChecked(bool(defaults["matrix_mode"]))
        self.comp_combo.setCurrentIndex(int(defaults["compression_index"]))

        is_custom = preset == "custom"
        for widget in (
            self.tau_spin,
            self.slope_spin,
            self.scene_cut_combo,
            self.scene_threshold_spin,
        ):
            widget.setEnabled(is_custom)

        if preset == "anime":
            self.preset_note.setText(
                "Anime preset raises model tolerance, keeps matrix mode on, "
                "and targets stronger compression on flat or temporally coherent content."
            )
        elif preset == "lownoise":
            self.preset_note.setText(
                "Low-noise preset is more conservative than anime but still "
                "optimized for clean footage with stable backgrounds."
            )
        else:
            self.preset_note.setText(
                "Custom preset exposes the raw knobs. Use this only if you know "
                "you need different rate/distortion behavior."
            )

    def _set_busy_state(self, compress: bool, running: bool, status: str) -> None:
        progress = self.compress_progress if compress else self.decompress_progress
        button = self.compress_start_btn if compress else self.decompress_start_btn
        label = self.compress_status if compress else self.decompress_status

        label.setText(status)
        label.setObjectName("statusWarn" if running else "statusGood")
        label.style().unpolish(label)
        label.style().polish(label)
        button.setEnabled(not running)
        if running:
            progress.setRange(0, 0)
        else:
            progress.setRange(0, 100)
            progress.setValue(100)

    def _start_compress(self) -> None:
        input_path = self.in_video_edit.text().strip()
        output_path = self.out_tfc_edit.text().strip()
        if not input_path or not output_path:
            self.compress_status.setObjectName("statusBad")
            self.compress_status.setText("Select both input and output paths.")
            return

        self._set_busy_state(True, True, "Compressing with bounded-memory settings…")
        self.encode_worker = CodecWorker(
            encode_video_to_tfc,
            input_path,
            output_path,
            tau=self.tau_spin.value(),
            slope_threshold=self.slope_spin.value(),
            payload_comp_type=self.comp_combo.currentIndex(),
            container_version=2,
            dtype="uint8",
            macbook_profile=self.macbook_profile_checkbox.isChecked(),
            scene_cut=self.scene_cut_combo.currentText(),
            scene_threshold=self.scene_threshold_spin.value(),
            matrix_mode=self.matrix_mode_checkbox.isChecked(),
            preset=self.preset_combo.currentText(),
        )
        self.encode_worker.finished_success.connect(self._on_compress_done)
        self.encode_worker.failed.connect(self._on_compress_error)
        self.encode_worker.start()

    def _on_compress_done(self, _msg: str) -> None:
        self._set_busy_state(True, False, "Compression complete.")
        self.encode_worker = None

    def _on_compress_error(self, err: str) -> None:
        self.compress_progress.setRange(0, 100)
        self.compress_progress.setValue(0)
        self.compress_start_btn.setEnabled(True)
        self.compress_status.setObjectName("statusBad")
        self.compress_status.setText(f"Compression failed: {err}")
        self.compress_status.style().unpolish(self.compress_status)
        self.compress_status.style().polish(self.compress_status)
        self.encode_worker = None

    def _start_decompress(self) -> None:
        input_path = self.in_tfc_edit.text().strip()
        output_path = self.out_video_edit.text().strip()
        if not input_path or not output_path:
            self.decompress_status.setObjectName("statusBad")
            self.decompress_status.setText("Select both input and output paths.")
            return

        self._set_busy_state(False, True, "Decompressing with streaming output…")
        self.decode_worker = CodecWorker(
            decode_tfc_to_video,
            input_path,
            output_path,
            self.fps_spin.value(),
            self.stream_output_checkbox.isChecked(),
        )
        self.decode_worker.finished_success.connect(self._on_decompress_done)
        self.decode_worker.failed.connect(self._on_decompress_error)
        self.decode_worker.start()

    def _on_decompress_done(self, _msg: str) -> None:
        self._set_busy_state(False, False, "Decompression complete.")
        self.decode_worker = None

    def _on_decompress_error(self, err: str) -> None:
        self.decompress_progress.setRange(0, 100)
        self.decompress_progress.setValue(0)
        self.decompress_start_btn.setEnabled(True)
        self.decompress_status.setObjectName("statusBad")
        self.decompress_status.setText(f"Decompression failed: {err}")
        self.decompress_status.style().unpolish(self.decompress_status)
        self.decompress_status.style().polish(self.decompress_status)
        self.decode_worker = None


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("TimeFlowCodec")
    app.setFont(QFont("SF Pro Text", 13))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
