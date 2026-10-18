import os
from typing import Optional

class AlignedBlockWriter:
    """
    Buffers variable-length compressed streams into 4KB/64KB page-aligned sectors,
    eliminating Flash Translation Layer (FTL) write amplification on SSD/eMMC.
    """
    def __init__(self, filepath: str, block_size: int = 4096):
        self.filepath = filepath
        self.block_size = block_size
        self.buffer = bytearray()
        self.fd = open(filepath, "wb")

    def write(self, data: bytes):
        self.buffer.extend(data)
        while len(self.buffer) >= self.block_size:
            chunk = self.buffer[:self.block_size]
            self.fd.write(chunk)
            del self.buffer[:self.block_size]

    def flush(self):
        if self.buffer:
            # Pad remaining buffer to exact block boundary
            pad_len = self.block_size - (len(self.buffer) % self.block_size)
            if pad_len != self.block_size:
                self.buffer.extend(b"\x00" * pad_len)
            self.fd.write(self.buffer)
            self.buffer.clear()
        self.fd.flush()

    def close(self):
        self.flush()
        self.fd.close()

class CircularRingBufferWriter:
    """
    Fixed-size circular blackbox buffer for crash/incident recording on eMMC/microSD.
    Overwrites oldest blocks once capacity is exceeded.
    """
    def __init__(self, filepath: str, block_size: int = 4096, max_blocks: int = 1024):
        self.filepath = filepath
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.current_block = 0
        self.fd = open(filepath, "r+b" if os.path.exists(filepath) else "w+b")

    def write_block(self, block: bytes):
        if len(block) != self.block_size:
            raise ValueError(f"Block must be exactly {self.block_size} bytes")
        offset = (self.current_block % self.max_blocks) * self.block_size
        self.fd.seek(offset)
        self.fd.write(block)
        self.current_block += 1

    def close(self):
        self.fd.flush()
        self.fd.close()
