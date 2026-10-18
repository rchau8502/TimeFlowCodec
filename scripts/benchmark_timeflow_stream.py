import math
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, List
from timeflowstream.core.cascade_engine import CascadeStreamEngine

def generate_telemetry_trace(num_samples: int = 10000) -> Tuple[List[float], List[float]]:
    t_vals = [float(i) for i in range(num_samples)]
    y_vals = []
    for t in t_vals:
        # 75% Cruising (Tier 1 Linear) + 25% Intermittent Transient Shocks (Tier 2 MLP)
        if t % 200 < 150:
            y_vals.append(65.0 + 0.05 * (t % 150))
        else:
            y_vals.append(65.0 + 15.0 * math.sin((t % 50) * 0.3))
    return t_vals, y_vals

def run_full_benchmark(num_samples: int = 10000) -> Dict[str, Any]:
    t_vals, y_vals = generate_telemetry_trace(num_samples)
    engine = CascadeStreamEngine(tolerance=0.08)

    start_time = time.perf_counter()
    compressed_bytes = engine.compress_stream(channel_id=0x120, t_vals=t_vals, y_vals=y_vals)
    elapsed = time.perf_counter() - start_time

    raw_bytes = len(y_vals) * 8
    compressed_len = len(compressed_bytes)
    ratio = raw_bytes / max(compressed_len, 1)
    samples_per_sec = num_samples / max(elapsed, 1e-6)

    # Verification: decompress and check round-trip integrity
    recon_t, recon_y = engine.decompress_stream(compressed_bytes)
    avg_error = sum(abs(y - r) for y, r in zip(y_vals, recon_y)) / max(len(y_vals), 1)

    return {
        "num_samples": num_samples,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_len,
        "compression_ratio": ratio,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples_per_sec,
        "reconstructed_samples": len(recon_y),
        "mean_absolute_error": avg_error
    }

if __name__ == "__main__":
    results = run_full_benchmark(50000)
    print("\n=======================================================")
    print("      TimeFlow Stream: Edge Telemetry Benchmark Report  ")
    print("=======================================================")
    print(f"Total Samples:        {results['num_samples']:,}")
    print(f"Raw IEEE-754 Data:    {results['raw_bytes']:,} bytes ({results['raw_bytes'] / 1024:.1f} KB)")
    print(f"Compressed Payload:   {results['compressed_bytes']:,} bytes ({results['compressed_bytes'] / 1024:.1f} KB)")
    print(f"Compression Ratio:    {results['compression_ratio']:.2f}x ({(1.0 - 1.0/results['compression_ratio'])*100:.1f}% savings)")
    print(f"Processing Speed:     {results['samples_per_second']:,.0f} samples/sec")
    print(f"Latency per Sample:   {(results['elapsed_seconds'] / results['num_samples']) * 1e9:.1f} nanoseconds")
    print(f"Reconstruction MAE:   {results['mean_absolute_error']:.4f}")
    print("=======================================================\n")
