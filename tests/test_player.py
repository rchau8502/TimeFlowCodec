"""Native playback and cancellable decode integration, without a media backend."""

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QProcess
from PySide6.QtGui import QFileOpenEvent
from PySide6.QtWidgets import QApplication

from timeflowcodec import player, player_decode
from timeflowcodec.encoder import encode_video_to_tfc
from timeflowcodec.format import write_header


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or player.FileOpenApplication([])


@pytest.fixture(params=[1, 2, 3, 4])
def clip(monkeypatch, tmp_path, request):
    frames = np.random.default_rng(6).integers(0, 256, (5, 4, 6, 3), dtype=np.uint8)
    monkeypatch.setattr("timeflowcodec.encoder.imageio.get_reader", lambda _: iter(frames))
    path = tmp_path / "clip with spaces.tfc"
    encode_video_to_tfc(
        str(tmp_path / "source"), str(path), tiling=1, tau=0, payload_comp_type=0, container_version=request.param
    )
    return path, frames


def wait_for(app, condition):
    deadline = time.monotonic() + 10
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert condition()


def test_decode_cache_and_size_preflight(clip, tmp_path, monkeypatch):
    path, frames = clip
    cache = tmp_path / "frames.npy"
    player_decode.decode_cache(path, cache)
    actual = np.load(cache, allow_pickle=False)
    assert np.max(np.abs(actual.astype(int) - frames.astype(int))) <= 1
    monkeypatch.setattr(player_decode, "MAX_RGB_BYTES", 1)
    with pytest.raises(ValueError, match="256 MiB|limit"):
        player_decode.decode_cache(path, tmp_path / "rejected.npy")
    assert not (tmp_path / "rejected.npy").exists()


@pytest.mark.parametrize("width,height,count", [(0, 2, 2), (2, 0, 2), (2, 2, 0), (100000, 100000, 1)])
def test_bad_dimensions_rejected_before_decode(tmp_path, monkeypatch, width, height, count):
    path = tmp_path / "invalid.tfc"
    with path.open("wb") as output:
        write_header(output, dict(version=3, width=width, height=height, num_frames=count))
    monkeypatch.setattr(player_decode, "decode_tfc_to_video", lambda *a: pytest.fail("decoder called"))
    with pytest.raises(ValueError):
        player_decode.decode_cache(path, tmp_path / "cache.npy")


def test_real_process_playback_seek_replay_and_cleanup(app, clip, monkeypatch):
    path, frames = clip
    window = player.PlayerWindow()
    window.show()
    try:
        window.open_file(str(path))
        wait_for(app, lambda: window.frames is not None or window.process is None)
        assert window.frames is not None, window.status.text()
        assert np.max(np.abs(window.frames.astype(int) - frames.astype(int))) <= 1
        assert window.playing
        window.pause()
        assert not window.playing
        window.seek(3)
        assert window.frame_index == 3
        assert window.slider.value() == 3
        window.fps.setValue(24)
        assert window.frame_index == 3
        now = [100.0]
        monkeypatch.setattr(player.time, "monotonic", lambda: now[0])
        window.play()
        now[0] += 2 / 24 + 0.001
        window._tick()
        assert window.frame_index == 4
        assert not window.playing
        window.play()
        assert window.frame_index == 0
        assert window.playing
        cache = Path(window.cache.name)
    finally:
        window.close()
    assert not cache.exists()


def test_invalid_file_recovers_then_cancel_releases_child(app, clip, tmp_path):
    path, _ = clip
    broken = tmp_path / "bad.tfc"
    broken.write_bytes(b"invalid")
    window = player.PlayerWindow()
    try:
        window.open_file(str(broken))
        wait_for(app, lambda: window.process is None)
        assert "Could not open" in window.status.text()
        window.open_file(str(path))
        child = window.process
        cache = Path(window.cache.name)
        window.cancel_load()
        assert window.process is None
        assert child.state() == QProcess.ProcessState.NotRunning
        assert not cache.exists()
        window.open_file(str(path))
        wait_for(app, lambda: window.frames is not None or window.process is None)
        assert window.frames is not None, window.status.text()
    finally:
        window.close()


def test_close_during_decode(app, clip):
    window = player.PlayerWindow()
    window.open_file(str(clip[0]))
    child = window.process
    cache = Path(window.cache.name)
    window.close()
    assert child.state() == QProcess.ProcessState.NotRunning
    assert not cache.exists()


def test_file_open_event_queues_until_window_ready(app):
    # Call the event implementation on a harness even if another Qt test created
    # a plain QApplication first: Qt allows only one application per process.
    class Harness:
        open_handler = None
        pending_files = []

    harness = Harness()
    event = QFileOpenEvent("/tmp/open me.tfc")
    assert player.FileOpenApplication.event(harness, event)
    opened = []
    player.FileOpenApplication.set_open_handler(harness, opened.append)
    assert opened == ["/tmp/open me.tfc"]


def test_existing_gui_can_open_player_and_close_its_child(app, clip):
    import gui

    window = gui.MainWindow()
    try:
        window.open_player(str(clip[0]))
        playback = window.player_window
        assert playback.isVisible()
        child = playback.process
        window.close()
        assert child.state() == QProcess.ProcessState.NotRunning
        assert playback.cache is None
    finally:
        window.close()


def test_reopening_while_loading_stops_previous_child(app, clip):
    window = player.PlayerWindow()
    try:
        window.open_file(str(clip[0]))
        previous = window.process
        previous_cache = Path(window.cache.name)
        window.open_file(str(clip[0]))
        assert previous.state() == QProcess.ProcessState.NotRunning
        assert not previous_cache.exists()
        wait_for(app, lambda: window.frames is not None or window.process is None)
        assert window.frames is not None, window.status.text()
    finally:
        window.close()


def test_failed_process_start_can_be_closed(app, clip, monkeypatch):
    window = player.PlayerWindow()
    monkeypatch.setattr(player.sys, "executable", "/nonexistent/timeflow-python")
    try:
        window.open_file(str(clip[0]))
        wait_for(app, lambda: window.process is None)
        assert "Could not open" in window.status.text()
        assert window.cache is None
        assert not window.cancel_button.isEnabled()
    finally:
        window.close()


def test_v5_player_uses_exact_rate_and_disk_frames(app, tmp_path):
    from test_media import make_video

    from timeflowcodec.player_decode import DiskFrames

    source, encoded = tmp_path / "source.mp4", tmp_path / "video.tfc"
    make_video(source, "30000/1001")
    encode_video_to_tfc(str(source), str(encoded), container_version=5, window=3)
    window = player.PlayerWindow()
    try:
        window.open_file(str(encoded))
        wait_for(app, lambda: window.frames is not None or window.process is None)
        assert isinstance(window.frames, DiskFrames), window.status.text()
        window.pause()
        assert window._playback_fps() == 30000 / 1001
        assert not window.fps.isEnabled()
        window.seek(14)
        assert window.frame_index == 14
    finally:
        window.close()
