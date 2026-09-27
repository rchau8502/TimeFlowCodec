import contextlib
import json
import resource
import sys
import time
from pathlib import Path

from timeflowcodec.decoder import decode_tfc_to_video
from timeflowcodec.encoder import encode_video_to_tfc
from timeflowcodec.player_decode import decode_cache

root = Path(sys.argv[1])
mode = sys.argv[2]
source = root / (sys.argv[3] + ".mp4")
encoded = source.with_suffix(".tfc")
started = time.monotonic()
with open(root / "codec.log", "a") as log, contextlib.redirect_stdout(log):
    if mode == "encode":
        encode_video_to_tfc(str(source), str(encoded), container_version=5, preset="screen")
    elif mode == "decode":
        decode_tfc_to_video(str(encoded), str(source.with_suffix(".restored.mp4")), stream_output=True)
    else:
        decode_cache(encoded, source.with_suffix(".npy"))
print(
    json.dumps(
        {
            "mode": mode,
            "source": source.name,
            "seconds": time.monotonic() - started,
            "python_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "child_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "tfc_bytes": encoded.stat().st_size,
        }
    )
)
