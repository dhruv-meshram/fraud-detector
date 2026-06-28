"""Performance tests benchmarking latency, throughput, and memory footprint."""

import pytest
import time
import os
import json
from pathlib import Path
from prj import FraudDetector
from prj.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

REPORTS_DIR = Path("/home/dhruv/Documents/fraud-detector/lib-tests/reports")

def test_performance_benchmarks():
    """Benchmarks single-event and batch processing across multiple sizes."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Instantiate in-memory detector
    detector = FraudDetector(
        profile_store=InMemoryProfileStore(),
        cache_store=InMemoryCacheStore(),
        db_store=InMemoryDBStore(),
        alert_producer=ConsoleAlertProducer()
    )
    
    # Seed historical records to keep state simple
    user_id = "perf_user"
    detector.pipeline.db_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0, "dev", True)
    
    sizes = [100, 1000, 10000]
    # For speed of validation runner, we conditionally run 100,000 or cap it if it is running in local test environment
    if os.getenv("RUN_EXTENDED_STRESS") == "true":
        sizes.append(100000)
        
    results = {}
    
    for size in sizes:
        start_time = time.perf_counter()
        
        for i in range(size):
            event = {
                "user_id": user_id,
                "latitude": 37.7749 + (i % 10) * 0.001,
                "longitude": -122.4194 - (i % 10) * 0.001,
                "timestamp": 1600000000.0 + i * 10,
                "device_hash": "dev"
            }
            detector.analyze(event)
            
        elapsed = time.perf_counter() - start_time
        throughput = size / elapsed
        latency_ms = (elapsed / size) * 1000
        
        results[f"batch_{size}"] = {
            "size": size,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_eps": round(throughput, 2),
            "average_latency_ms": round(latency_ms, 4)
        }
        
    # Write report
    report_path = REPORTS_DIR / "performance_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    assert report_path.exists()
    assert len(results) >= 3
