import struct
from typing import List
from timeflowstream.adapters.can_adapter import ChannelSample

class IndustrialDAQAdapter:
    """Parses multi-channel high-frequency ADC vibration buffers."""
    def __init__(self, sampling_rate_hz: int = 10000):
        self.sampling_rate_hz = sampling_rate_hz

    def parse_adc_packet(self, payload: bytes, timestamp_ns: int) -> List[ChannelSample]:
        num_channels = len(payload) // 2
        samples = []
        for ch in range(num_channels):
            val = struct.unpack_from("<h", payload, ch * 2)[0]
            samples.append(ChannelSample(channel_id=ch, timestamp_ns=timestamp_ns, value=float(val)))
        return samples
