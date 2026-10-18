import pytest
from timeflowstream.adapters.can_adapter import CANFrameAdapter
from timeflowstream.adapters.industrial_adapter import IndustrialDAQAdapter
from timeflowstream.adapters.mavlink_adapter import MAVLinkTelemetryAdapter

def test_can_frame_adapter():
    adapter = CANFrameAdapter()
    # Mock CAN frame: ID 0x120 (Vehicle Speed), 8-byte payload
    raw_frame = bytes([0x00, 0x12, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00])
    sample = adapter.parse_frame(can_id=0x120, payload=raw_frame, timestamp_ns=1000000)
    assert sample.channel_id == 0x120
    assert sample.value == 42.0

def test_industrial_daq_adapter():
    adapter = IndustrialDAQAdapter(sampling_rate_hz=10000)
    # 4 channels of 16-bit ADC samples
    raw_adc = bytes([0x00, 0x08, 0x00, 0x10])
    samples = adapter.parse_adc_packet(raw_adc, timestamp_ns=2000000)
    assert len(samples) == 2

def test_mavlink_adapter():
    adapter = MAVLinkTelemetryAdapter()
    # Mock attitude message: roll=0.1, pitch=-0.2, yaw=1.57
    samples = adapter.parse_attitude(roll=0.1, pitch=-0.2, yaw=1.57, timestamp_ns=3000000)
    assert len(samples) == 3
    assert samples[0].value == 0.1
