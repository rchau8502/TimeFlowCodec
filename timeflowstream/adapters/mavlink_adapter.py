from typing import List
from timeflowstream.adapters.can_adapter import ChannelSample

class MAVLinkTelemetryAdapter:
    """Parses UAV attitude and flight dynamics telemetry packets."""
    ATTITUDE_ROLL_CH = 101
    ATTITUDE_PITCH_CH = 102
    ATTITUDE_YAW_CH = 103

    def parse_attitude(self, roll: float, pitch: float, yaw: float, timestamp_ns: int) -> List[ChannelSample]:
        return [
            ChannelSample(channel_id=self.ATTITUDE_ROLL_CH, timestamp_ns=timestamp_ns, value=roll),
            ChannelSample(channel_id=self.ATTITUDE_PITCH_CH, timestamp_ns=timestamp_ns, value=pitch),
            ChannelSample(channel_id=self.ATTITUDE_YAW_CH, timestamp_ns=timestamp_ns, value=yaw),
        ]
