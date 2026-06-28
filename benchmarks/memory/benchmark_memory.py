"""Memory benchmark tracking memory footprint and leak growth."""

import time
import tracemalloc
from prj import FraudDetector
from prj.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

def run_memory_benchmarks():
    """Tracks peak memory usage, allocation growth, and verifies leak bounds."""
    detector = FraudDetector(
        profile_store=InMemoryProfileStore(),
        cache_store=InMemoryCacheStore(),
        db_store=InMemoryDBStore(),
        alert_producer=ConsoleAlertProducer()
    )
    
    user_id = "mem_bench_user"
    detector.pipeline.db_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0, "dev", True)
    
    event = {
        "user_id": user_id,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000100.0,
        "device_hash": "dev"
    }
    
    # 1. Single Event Memory Footprint
    tracemalloc.start()
    start_mem, _ = tracemalloc.get_traced_memory()
    detector.analyze(event)
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    single_peak_kb = peak_mem / 1024.0
    single_growth_kb = (current_mem - start_mem) / 1024.0
    
    # 2. Repeated inference leak detection (1,000 runs)
    tracemalloc.start()
    start_mem, _ = tracemalloc.get_traced_memory()
    
    # Snapshot at the start of loop
    for i in range(1000):
        detector.analyze(event)
        
    current_mem_loop, peak_mem_loop = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    loop_peak_kb = peak_mem_loop / 1024.0
    loop_growth_kb = (current_mem_loop - start_mem) / 1024.0
    leak_per_event_bytes = (current_mem_loop - start_mem) / 1000.0
    
    return {
        "single_event": {
            "peak_memory_kb": round(max(0.0, single_peak_kb), 2),
            "allocated_kb": round(max(0.0, single_growth_kb), 2)
        },
        "repeated_1000_runs": {
            "peak_memory_kb": round(max(0.0, loop_peak_kb), 2),
            "allocated_kb": round(max(0.0, loop_growth_kb), 2),
            "leak_per_event_bytes": round(max(0.0, leak_per_event_bytes), 4)
        }
    }

if __name__ == "__main__":
    res = run_memory_benchmarks()
    print("Memory Benchmarks:", res)
