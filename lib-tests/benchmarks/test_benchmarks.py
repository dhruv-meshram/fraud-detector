"""Benchmark suite calculating latency distribution (p50, p95, p99 percentiles)."""

import pytest
import time
import numpy as np
from fraud_detector import FraudDetector
from fraud_detector.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

def test_latency_distribution_benchmarks():
    """Calculates p50, p95, and p99 latency percentiles over a series of events."""
    detector = FraudDetector(
        profile_store=InMemoryProfileStore(),
        cache_store=InMemoryCacheStore(),
        db_store=InMemoryDBStore(),
        alert_producer=ConsoleAlertProducer()
    )
    
    user_id = "benchmark_user"
    detector.pipeline.db_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0, "dev", True)
    
    latencies = []
    
    # Run 1,000 requests and track individual latencies
    for i in range(1000):
        event = {
            "user_id": user_id,
            "latitude": 37.7749 + (i % 5) * 0.001,
            "longitude": -122.4194 - (i % 5) * 0.001,
            "timestamp": 1600000000.0 + i * 10,
            "device_hash": "dev"
        }
        
        start = time.perf_counter()
        detector.analyze(event)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg_latency = np.mean(latencies)
    throughput = 1000 / (sum(latencies) / 1000)
    
    print("\n--- LATENCY BENCHMARK RESULTS ---")
    print(f"Total Requests: 1,000")
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"p50 (Median):    {p50:.4f} ms")
    print(f"p95:             {p95:.4f} ms")
    print(f"p99:             {p99:.4f} ms")
    print(f"Throughput:      {throughput:.2f} rps")
    print("---------------------------------")
    
    # Save the distribution details for the report runner to discover
    import json
    from pathlib import Path
    reports_dir = Path("/home/dhruv/Documents/fraud-detector/lib-tests/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "latency_distribution.json", "w") as f:
        json.dump({
            "p50": round(p50, 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
            "avg": round(avg_latency, 4),
            "throughput": round(throughput, 2)
        }, f, indent=2)
        
    assert p95 < 10.0  # p95 latency must be under 10ms in local in-memory setup
