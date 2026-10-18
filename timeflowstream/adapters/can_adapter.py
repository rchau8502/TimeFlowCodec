import struct
from dataclasses import dataclass

@dataclass
class ChannelSample:
    channel_id: int
    timestamp_ns: int
    value: float

class CANFrameAdapter:
    """Decodes standard/CAN-FD frames into physical floating-point channel samples."""
    def parse_frame(self, can_id: int, payload: bytes, timestamp_ns: int) -> ChannelSample:
        # Extracts uint16 from offset 4 as scaled signal
        raw_val = struct.unpack_from("<H", payload, 4)[0]
        return ChannelSample(channel_id=can_id, timestamp_ns=timestamp_ns, value=float(raw_val))
