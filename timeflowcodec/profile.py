from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from .encoder import encode_video_to_tfc
from .decoder import decode_tfc_to_video


def current_rss_mb() -> float:
    if psutil is None:
        return 0.0
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def run_profile(input_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_tfc = out_dir / "_tmp.tfc"
    result = {}

    tracemalloc.start()
    rss_start = current_rss_mb()
    t0 = time.perf_counter()
    encode_video_to_tfc(str(input_path), str(temp_tfc))
    t1 = time.perf_counter()
    enc_time = t1 - t0
    rss_enc = current_rss_mb()
    peak_traces, peak_size = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t2 = time.perf_counter()
    decode_tfc_to_video(str(temp_tfc), str(out_dir / "recon.mp4"))
    t3 = time.perf_counter()
    dec_time = t3 - t2
    rss_dec = current_rss_mb()
    peak_traces_dec, peak_size_dec = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result["encode_time_sec"] = enc_time
    result["decode_time_sec"] = dec_time
    result["rss_start_mb"] = rss_start
    result["rss_encode_mb"] = rss_enc
    result["rss_decode_mb"] = rss_dec
    result["tracemalloc_peak_bytes_encode"] = peak_size
    result["tracemalloc_peak_bytes_decode"] = peak_size_dec
    result["psutil_available"] = psutil is not None

    result["env"] = {
        "python": sys.version,
        "platform": sys.platform,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    }

    with open(out_dir / "profile.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Profile TimeFlowCodec encode/decode")
    parser.add_argument("--input", type=Path, required=True, help="Input mp4")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for profile")
    args = parser.parse_args(argv)

    res = run_profile(args.input, args.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
