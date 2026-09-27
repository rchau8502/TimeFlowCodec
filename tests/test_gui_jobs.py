import os
import time
from threading import Event

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtWidgets import QApplication

import gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def wait_for(app, predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert predicate()


def test_cancel_and_close_wait_for_thread_cleanup(app, monkeypatch):
    entered = Event()
    released = Event()
    def encode(*args, **kwargs):
        entered.set()
        while not kwargs['cancel_check']():
            time.sleep(.005)
        released.wait(3)
    monkeypatch.setattr(gui, 'encode_video_to_tfc', encode)
    window = gui.MainWindow()
    window.show()
    window.in_video_edit.setText('input.mp4')
    window._start_compress()
    wait_for(app, entered.is_set)
    try:
        assert not window.decompress_start_btn.isEnabled()
        window.close()
        assert window.isVisible()
        assert window.encode_worker is not None
    finally:
        released.set()
        wait_for(app, lambda: window.encode_worker is None)
        window.close()
    assert not window.isVisible()


def test_decode_metadata_fps_progress_and_cleanup(app, monkeypatch):
    values = {}
    def decode(*args, **kwargs):
        values.update(kwargs)
        kwargs['progress_callback']('Decoding', 2, 4)
    monkeypatch.setattr(gui, 'decode_tfc_to_video', decode)
    window = gui.MainWindow()
    window.in_tfc_edit.setText('input.tfc')
    window._start_decompress()
    wait_for(app, lambda: window.decode_worker is None)
    assert values['fps'] is None
    assert window.decompress_progress.value() == 100
    assert window.compress_start_btn.isEnabled()
    window.close()


def test_inspection_reads_only_first_frame_and_closes_reader(tmp_path, monkeypatch):
    import numpy as np
    class Reader:
        def get_meta_data(self):
            return {'fps': 24, 'duration': 2}
        def get_data(self, index):
            assert index == 0
            return np.zeros((12, 16, 3), dtype=np.uint8)
        def close(self):
            self.closed = True
    reader = Reader()
    monkeypatch.setattr('imageio.v2.get_reader', lambda path: reader)
    path = tmp_path / 'sample.mp4'
    path.write_bytes(b'123')
    result = gui.inspect_video(str(path))
    assert result['width'] == 16
    assert result['height'] == 12
    assert result['bytes'] == 3
    assert reader.closed
