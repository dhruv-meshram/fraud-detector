"""Stress benchmark verifying SDK stability under continuous load and input floods."""

import time
import concurrent.futures
from prj import FraudDetector
from prj.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

def run_stress_benchmarks():
    """Executes intensive loads and malformed payloads to detect crashes or slowdowns."""
    p_store = InMemoryProfileStore()
    c_store = InMemoryCacheStore()
    d_store = InMemoryDBStore()
    a_producer = ConsoleAlertProducer()
    
    detector = FraudDetector(
        profile_store=p_store,
        cache_store=c_store,
        db_store=d_store,
        alert_producer=a_producer
    )
    
    user_id = "stress_bench_user"
    p_store.save_profile(user_id, {
        "user_id": user_id,
        "clusters": [{"cluster_id": 0, "centroid_lat": 37.7749, "centroid_lon": -122.4194, "dynamic_radius_km": 15.0}]
    })
    d_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0, "dev", True)
    c_store.set_last_node(user_id, {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1600000000.0, "device_hash": "dev"})
    
    event = {
        "user_id": user_id,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000100.0,
        "device_hash": "dev"
    }
    
    results = {}
    
    # 1. Continuous rapid-fire inference
    t0 = time.perf_counter()
    latencies_early = []
    latencies_late = []
    
    for i in range(1000):
        q0 = time.perf_counter()
        detector.analyze(event)
        lat = (time.perf_counter() - q0) * 1000
        if i < 100:
            latencies_early.append(lat)
        elif i >= 900:
            latencies_late.append(lat)
            
    elapsed_continuous = time.perf_counter() - t0
    
    # Verify no massive slowdown (late latencies shouldn't be significantly higher than early)
    slowdown_ratio = sum(latencies_late) / max(0.001, sum(latencies_early))
    
    results["continuous_inference"] = {
        "count": 1000,
        "elapsed_seconds": round(elapsed_continuous, 4),
        "early_avg_ms": round(float(sum(latencies_early)/100), 4),
        "late_avg_ms": round(float(sum(latencies_late)/100), 4),
        "slowdown_ratio": round(slowdown_ratio, 2)
    }
    
    # 2. Malformed input flood
    malformed_event = {
        "user_id": "stress_malformed",
        "latitude": "not-a-float",
        "longitude": -122.4194,
        "timestamp": 1600000000.0,
        "device_hash": "dev"
    }
    
    errors_caught = 0
    t0 = time.perf_counter()
    for _ in range(500):
        try:
            detector.analyze(malformed_event)
        except Exception:
            errors_caught += 1
            
    elapsed_flood = time.perf_counter() - t0
    
    results["malformed_input_flood"] = {
        "count": 500,
        "errors_caught": errors_caught,
        "elapsed_seconds": round(elapsed_flood, 4),
        "robustness_pct": round((errors_caught / 500.0) * 100.0, 2)
    }
    
    return results

if __name__ == "__main__":
    res = run_stress_benchmarks()
    print("Stress Benchmarks:", res)
