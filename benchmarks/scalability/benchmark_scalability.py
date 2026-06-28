"""Scalability benchmark evaluating throughput and latency under large user counts."""

import time
import tracemalloc
import numpy as np
from prj import FraudDetector
from prj.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer
from benchmarks.utils import fit_complexity

def run_scalability_benchmarks():
    """Measures performance scalability under rising user registries."""
    sizes = [1, 10, 100, 1000, 10000]
    
    results = {}
    
    for count in sizes:
        p_store = InMemoryProfileStore()
        c_store = InMemoryCacheStore()
        d_store = InMemoryDBStore()
        a_producer = ConsoleAlertProducer()
        
        # Populate count users in store
        for i in range(count):
            user_id = f"user_{i}"
            p_store.save_profile(user_id, {
                "user_id": user_id,
                "clusters": [{"cluster_id": 0, "centroid_lat": 37.7749, "centroid_lon": -122.4194, "dynamic_radius_km": 15.0}]
            })
            d_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0, "dev", True)
            c_store.set_last_node(user_id, {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1600000000.0, "device_hash": "dev"})
            
        detector = FraudDetector(
            profile_store=p_store,
            cache_store=c_store,
            db_store=d_store,
            alert_producer=a_producer
        )
        
        # Measure latency & throughput over 1000 random requests
        latencies = []
        
        tracemalloc.start()
        start_mem, _ = tracemalloc.get_traced_memory()
        
        t0 = time.perf_counter()
        for r in range(1000):
            # Query random user from the populated ones
            u_idx = r % count
            user_id = f"user_{u_idx}"
            event = {
                "user_id": user_id,
                "latitude": 37.7749,
                "longitude": -122.4194,
                "timestamp": 1600000100.0,
                "device_hash": "dev"
            }
            
            q0 = time.perf_counter()
            detector.analyze(event)
            latencies.append((time.perf_counter() - q0) * 1000)
            
        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        throughput = 1000.0 / elapsed
        avg_latency = np.mean(latencies)
        peak_kb = peak_mem / 1024.0
        
        results[f"users_{count}"] = {
            "user_count": count,
            "throughput_eps": round(throughput, 2),
            "average_latency_ms": round(float(avg_latency), 4),
            "peak_memory_kb": round(peak_kb, 2)
        }
        
    # Fit empirical complexity of average latency vs user size
    user_sizes = [results[f"users_{c}"]["user_count"] for c in sizes]
    avg_lats = [results[f"users_{c}"]["average_latency_ms"] for c in sizes]
    results["complexity"] = fit_complexity(user_sizes, avg_lats)
    
    return results

if __name__ == "__main__":
    res = run_scalability_benchmarks()
    print("Scalability Benchmarks:", res)
