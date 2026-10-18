import os
import pytest
from timeflowstream.storage.block_writer import AlignedBlockWriter, CircularRingBufferWriter

def test_aligned_block_writer(tmp_path):
    log_file = tmp_path / "test_stream.bin"
    writer = AlignedBlockWriter(str(log_file), block_size=4096)

    # Push 10,000 bytes of data (spans multiple 4KB pages)
    payload = b"X" * 10000
    writer.write(payload)
    writer.close()

    # File size on disk MUST be an exact multiple of 4096 bytes (zero write amplification)
    actual_size = os.path.getsize(str(log_file))
    assert actual_size > 0
    assert actual_size % 4096 == 0

def test_circular_ring_buffer(tmp_path):
    ring_file = tmp_path / "ring_buffer.bin"
    # 3 blocks of 4096 = 12,288 bytes maximum
    ring = CircularRingBufferWriter(str(ring_file), block_size=4096, max_blocks=3)

    # Write 5 blocks of data -> should wrap around and keep exactly 3 blocks
    for i in range(5):
        block = bytes([i] * 4096)
        ring.write_block(block)
    ring.close()

    actual_size = os.path.getsize(str(ring_file))
    assert actual_size == 3 * 4096
