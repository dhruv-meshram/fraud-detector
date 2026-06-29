"""Batch processing benchmark evaluating throughput, latency, and memory."""

import time
import tracemalloc
import concurrent.futures
from fraud_detector import FraudDetector
from fraud_detector.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer
from benchmarks.datasets.generator import generate_events

def analyze_batch_sequential(detector, events):
    """Processes a batch of events sequentially."""
    results = []
    for event in events:
        results.append(detector.analyze(event))
    return results

def analyze_batch_parallel(detector, events, max_workers=8):
    """Processes a batch of events concurrently using a ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(detector.analyze, events))
    return results

def run_batch_benchmarks():
    """Benchmarks throughput, latency, and memory of batch processing."""
    sizes = [100, 1000, 10000]
    
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
    
    # Seed user profile
    user_id = "batch_bench_user"
    p_store.save_profile(user_id, {
        "user_id": user_id,
        "clusters": [{"cluster_id": 0, "centroid_lat": 37.7749, "centroid_lon": -122.4194, "dynamic_radius_km": 15.0}]
    })
    d_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0, "dev", True)
    c_store.set_last_node(user_id, {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1600000000.0, "device_hash": "dev"})
    
    results = {}
    
    for size in sizes:
        events = generate_events(size, user_id=user_id)
        
        # 1. Sequential batch processing
        tracemalloc.start()
        start_mem, _ = tracemalloc.get_traced_memory()
        
        t0 = time.perf_counter()
        analyze_batch_sequential(detector, events)
        elapsed_seq = time.perf_counter() - t0
        
        _, peak_mem_seq = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        throughput_seq = size / elapsed_seq
        latency_seq_ms = (elapsed_seq / size) * 1000
        
        # 2. Parallel batch processing
        tracemalloc.start()
        start_mem_p, _ = tracemalloc.get_traced_memory()
        
        t0 = time.perf_counter()
        analyze_batch_parallel(detector, events)
        elapsed_par = time.perf_counter() - t0
        
        _, peak_mem_par = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        throughput_par = size / elapsed_par
        latency_par_ms = (elapsed_par / size) * 1000
        
        results[f"batch_{size}"] = {
            "size": size,
            "sequential": {
                "elapsed_seconds": round(elapsed_seq, 4),
                "throughput_eps": round(throughput_seq, 2),
                "average_latency_ms": round(latency_seq_ms, 4),
                "peak_memory_kb": round(peak_mem_seq / 1024.0, 2)
            },
            "parallel": {
                "elapsed_seconds": round(elapsed_par, 4),
                "throughput_eps": round(throughput_par, 2),
                "average_latency_ms": round(latency_par_ms, 4),
                "peak_memory_kb": round(peak_mem_par / 1024.0, 2)
            }
        }
        
    return results

if __name__ == "__main__":
    res = run_batch_benchmarks()
    print("Batch Benchmarks:", res)
