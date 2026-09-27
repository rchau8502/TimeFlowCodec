"""Write complete outputs beside their destination before replacing it."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_destination(destination):
    """Yield a temporary path with the destination suffix; publish on success."""
    destination = Path(destination)
    fd, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=destination.suffix, dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield str(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
