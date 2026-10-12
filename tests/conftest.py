import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Ensure a single QApplication exists for the test session and is cleanly shut down."""
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(["--platform", "offscreen"])
        yield app
        app.closeAllWindows()
        app.processEvents()
    except ImportError:
        yield None


@pytest.fixture(autouse=True)
def cleanup_qt():
    """Process pending Qt events and collect garbage after each test."""
    yield
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except ImportError:
        pass
    gc.collect()

