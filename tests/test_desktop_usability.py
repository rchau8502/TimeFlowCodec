"""Regressions for controls users must be able to see and reach."""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtWidgets import QApplication, QScrollArea

import gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('size', [(1100, 760), (960, 680)])
def test_compress_controls_keep_usable_height_and_can_be_scrolled_to(app, size):
    window = gui.MainWindow()
    window.resize(*size)
    window.show()
    try:
        index = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == 'Compress')
        window.tabs.setCurrentIndex(index)
        app.processEvents()
        for name in ['in_video_edit', 'out_tfc_edit', 'preset_combo', 'compress_start_btn']:
            control = getattr(window, name)
            assert control.height() >= control.minimumSizeHint().height(), name
        page = window.tabs.widget(index)
        assert isinstance(page, QScrollArea)
        page.ensureWidgetVisible(window.compress_start_btn)
        app.processEvents()
        assert not window.compress_start_btn.visibleRegion().isEmpty()
        window.advanced_toggle.setChecked(True)
        app.processEvents()
        page.ensureWidgetVisible(window.tau_spin)
        app.processEvents()
        assert window.tau_spin.height() >= window.tau_spin.minimumSizeHint().height()
        assert not window.tau_spin.visibleRegion().isEmpty()
    finally:
        window.close()


def test_playback_is_visible_on_launch_and_uses_same_window(app, monkeypatch):
    window = gui.MainWindow()
    window.show()
    try:
        assert window.tabs.tabText(window.tabs.currentIndex()) == 'Play'
        assert window.player_window.open_button.isVisible()
        called = []
        monkeypatch.setattr(window.player_window, 'open_file', called.append)
        window.open_player('/tmp/example.tfc')
        assert called == ['/tmp/example.tfc']
        assert window.player_window.window() is window
    finally:
        window.close()


@pytest.mark.parametrize('mode', ['compress', 'decompress'])
def test_job_object_lives_until_thread_finishes_and_close_waits(app, monkeypatch, mode):
    import time
    from threading import Event

    finished_function = Event()
    release_thread = Event()
    base_worker = gui.CodecWorker

    class LingeringWorker(base_worker):
        def run(self):
            super().run()
            finished_function.set()
            release_thread.wait(5)

    monkeypatch.setattr(gui, 'CodecWorker', LingeringWorker)
    monkeypatch.setattr(gui, 'encode_video_to_tfc', lambda *a, **kw: None)
    monkeypatch.setattr(gui, 'decode_tfc_to_video', lambda *a, **kw: None)
    window = gui.MainWindow()
    window.show()
    window.in_video_edit.setText('/tmp/input.mp4')
    window.in_tfc_edit.setText('/tmp/input.tfc')
    getattr(window, '_start_' + mode)()
    worker_name = 'encode_worker' if mode == 'compress' else 'decode_worker'
    worker = getattr(window, worker_name)  # Keep a test reference to prevent an abort in the old code.
    try:
        assert finished_function.wait(3)
        app.processEvents()
        assert getattr(window, worker_name) is worker
        assert not window.compress_start_btn.isEnabled()
        assert not window.decompress_start_btn.isEnabled()
        window.close()
        assert window.isVisible()
    finally:
        release_thread.set()
        worker.wait(3000)
        deadline = time.monotonic() + 3
        while getattr(window, worker_name) is not None and time.monotonic() < deadline:
            app.processEvents()
        window.close()


def test_demo_has_play_action_and_writable_output_outside_app_assets(app):
    from pathlib import Path

    window = gui.MainWindow()
    window.show()
    try:
        assert window.player_window.demo_button.isVisible()
        window._load_demo_video()
        output = Path(window.out_tfc_edit.text())
        assert output.parent != Path(gui._demo_asset('demo_input.mp4')).parent
        output.write_bytes(b'test output')
        window._load_demo_tfc()
        assert window.fps_spin.value() == 12
        assert Path(window.out_video_edit.text()).parent == output.parent
    finally:
        window.close()
    assert not output.exists()


def test_play_result_uses_completed_job_path_not_edited_field(app, monkeypatch, tmp_path):
    import time

    monkeypatch.setattr(gui, 'encode_video_to_tfc', lambda *a, **kw: None)
    window = gui.MainWindow()
    window.show()
    window.in_video_edit.setText(str(tmp_path / 'input.mp4'))
    completed_path = str(tmp_path / 'completed.tfc')
    window.out_tfc_edit.setText(completed_path)
    try:
        assert not window.play_result_btn.isEnabled()
        window._start_compress()
        deadline = time.monotonic() + 3
        while window.encode_worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert window.play_result_btn.isEnabled()
        window.out_tfc_edit.setText(str(tmp_path / 'another.tfc'))
        opened = []
        monkeypatch.setattr(window, 'open_player', opened.append)
        window.play_result_btn.click()
        assert opened == [completed_path]
    finally:
        window.close()
