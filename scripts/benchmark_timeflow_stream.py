import math
import os
import random
import struct
import sys
import time
import zlib
import zstandard as zstd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, List
from timeflowstream.core.cascade_engine import CascadeStreamEngine

def generate_telemetry_trace(num_samples: int = 10000, noise_std: float = 0.02) -> Tuple[List[float], List[float]]:
    random.seed(42)
    t_vals = [float(i) for i in range(num_samples)]
    y_vals = []
    for t in t_vals:
        noise = random.gauss(0, noise_std) if noise_std > 0 else 0.0
        # 75% Cruising (Tier 1 Linear) + 25% Intermittent Transient Shocks (Tier 2 MLP)
        if t % 200 < 150:
            y_vals.append(65.0 + 0.05 * (t % 150) + noise)
        else:
            y_vals.append(65.0 + 15.0 * math.sin((t % 50) * 0.3) + noise)
    return t_vals, y_vals

def run_gorilla_compression(y_vals: List[float]) -> int:
    xor_bits = 0
    prev_int = struct.unpack('<Q', struct.pack('<d', y_vals[0]))[0]
    for y in y_vals[1:]:
        cur_int = struct.unpack('<Q', struct.pack('<d', y))[0]
        xor_val = cur_int ^ prev_int
        if xor_val == 0:
            xor_bits += 1
        else:
            binary = f'{xor_val:064b}'
            leading = len(binary) - len(binary.lstrip('0'))
            trailing = len(binary) - len(binary.rstrip('0'))
            meaningful = 64 - leading - trailing
            xor_bits += 1 + 5 + 6 + meaningful
        prev_int = cur_int
    return (xor_bits + 7) // 8

def run_full_benchmark(num_samples: int = 10000, noise_std: float = 0.02) -> Dict[str, Any]:
    t_vals, y_vals = generate_telemetry_trace(num_samples, noise_std=noise_std)
    raw_bytes = struct.pack(f"<{len(y_vals)}d", *y_vals)
    raw_size = len(raw_bytes)

    # 1. Zlib
    t0 = time.perf_counter()
    zlib_bytes = zlib.compress(raw_bytes, level=6)
    t_zlib = time.perf_counter() - t0

    # 2. Zstandard Level 3 (Default)
    cctx3 = zstd.ZstdCompressor(level=3)
    t0 = time.perf_counter()
    zstd3_bytes = cctx3.compress(raw_bytes)
    t_zstd3 = time.perf_counter() - t0

    # 3. Zstandard Level 19 (Ultra)
    cctx19 = zstd.ZstdCompressor(level=19)
    t0 = time.perf_counter()
    zstd19_bytes = cctx19.compress(raw_bytes)
    t_zstd19 = time.perf_counter() - t0

    # 4. Gorilla XOR
    t0 = time.perf_counter()
    gorilla_len = run_gorilla_compression(y_vals)
    t_gorilla = time.perf_counter() - t0

    # 5. TimeFlow Stream
    engine = CascadeStreamEngine(tolerance=0.08)
    t0 = time.perf_counter()
    compressed_bytes = engine.compress_stream(channel_id=0x120, t_vals=t_vals, y_vals=y_vals)
    t_tfc = time.perf_counter() - t0

    # Decompress and calculate MAE
    recon_t, recon_y = engine.decompress_stream(compressed_bytes)
    avg_error = sum(abs(y - r) for y, r in zip(y_vals, recon_y)) / max(len(y_vals), 1)

    return {
        "num_samples": num_samples,
        "raw_bytes": raw_size,
        "tfc_bytes": len(compressed_bytes),
        "tfc_ratio": raw_size / len(compressed_bytes),
        "tfc_speed": num_samples / max(t_tfc, 1e-6),
        "tfc_mae": avg_error,
        "zlib_bytes": len(zlib_bytes),
        "zlib_ratio": raw_size / len(zlib_bytes),
        "zstd3_bytes": len(zstd3_bytes),
        "zstd3_ratio": raw_size / len(zstd3_bytes),
        "zstd19_bytes": len(zstd19_bytes),
        "zstd19_ratio": raw_size / len(zstd19_bytes),
        "gorilla_bytes": gorilla_len,
        "gorilla_ratio": raw_size / gorilla_len,
        "compression_ratio": raw_size / len(compressed_bytes),
        "samples_per_second": num_samples / max(t_tfc, 1e-6)
    }

if __name__ == "__main__":
    res = run_full_benchmark(50000, noise_std=0.02)
    print("\n=========================================================================================")
    print("                 TimeFlow Stream: Comparative Industry Benchmark Report                  ")
    print("=========================================================================================")
    print(f"Total Telemetry Samples:  {res['num_samples']:,} (64-bit IEEE-754 Floats)")
    print(f"Raw Input Size:           {res['raw_bytes']:,} bytes ({res['raw_bytes']/1024:.1f} KB)")
    print("-----------------------------------------------------------------------------------------")
    print(f"{'Method / Codec':<25} | {'Compressed Size':<16} | {'Ratio':<10} | {'Space Reduction':<15}")
    print("-----------------------------------------------------------------------------------------")
    print(f"{'Uncompressed IEEE-754':<25} | {res['raw_bytes']:>10,} bytes | {'1.00x':<10} | {'0.0%':<15}")
    print(f"{'Zlib / Gzip (Level 6)':<25} | {res['zlib_bytes']:>10,} bytes | {res['zlib_ratio']:<9.2f}x | {100*(1-1/res['zlib_ratio']):<14.1f}%")
    print(f"{'Zstandard (Level 3)':<25} | {res['zstd3_bytes']:>10,} bytes | {res['zstd3_ratio']:<9.2f}x | {100*(1-1/res['zstd3_ratio']):<14.1f}%")
    print(f"{'Zstandard (Level 19)':<25} | {res['zstd19_bytes']:>10,} bytes | {res['zstd19_ratio']:<9.2f}x | {100*(1-1/res['zstd19_ratio']):<14.1f}%")
    print(f"{'Facebook Gorilla (VLDB)':<25} | {res['gorilla_bytes']:>10,} bytes | {res['gorilla_ratio']:<9.2f}x | {100*(1-1/res['gorilla_ratio']):<14.1f}%")
    print(f"{'TimeFlow Stream (Ours)':<25} | {res['tfc_bytes']:>10,} bytes | {res['tfc_ratio']:<9.2f}x | {100*(1-1/res['tfc_ratio']):<14.1f}%")
    print("-----------------------------------------------------------------------------------------")
    print(f"TimeFlow Stream Processing Throughput: {res['tfc_speed']:,.0f} samples/sec")
    print(f"Reconstruction Mean Absolute Error:    {res['tfc_mae']:.4f}")
    print("=========================================================================================\n")
