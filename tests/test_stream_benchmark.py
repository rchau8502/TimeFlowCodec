from scripts.benchmark_timeflow_stream import run_full_benchmark

def test_benchmark_runs_and_reports_metrics():
    report = run_full_benchmark(num_samples=500)
    assert "compression_ratio" in report
    assert report["compression_ratio"] >= 5.0
    assert report["samples_per_second"] > 10000.0
