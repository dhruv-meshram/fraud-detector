"""ML inference benchmark evaluating profile loading, inference latency, and percentiles."""

import time
import numpy as np
from benchmarks.utils import measure_memory
from prj.ml.inference import InferenceService
from prj.adapters import InMemoryProfileStore
from prj.models.event import LoginEvent

def run_ml_benchmarks():
    """Benchmarks spatial centroid matching, profile loading, and percentiles."""
    profile_store = InMemoryProfileStore()
    service = InferenceService(profile_store=profile_store)
    
    # 1. Seed small, medium, large user profiles
    # Small user: 1 cluster
    profile_store.save_profile("small_user", {
        "user_id": "small_user",
        "clusters": [{"cluster_id": 0, "centroid_lat": 37.7749, "centroid_lon": -122.4194, "dynamic_radius_km": 15.0}]
    })
    
    # Medium user: 5 clusters
    profile_store.save_profile("medium_user", {
        "user_id": "medium_user",
        "clusters": [
            {"cluster_id": i, "centroid_lat": 37.7749 + i*0.1, "centroid_lon": -122.4194 - i*0.1, "dynamic_radius_km": 15.0}
            for i in range(5)
        ]
    })
    
    # Large user: 20 clusters
    profile_store.save_profile("large_user", {
        "user_id": "large_user",
        "clusters": [
            {"cluster_id": i, "centroid_lat": 37.7749 + i*0.1, "centroid_lon": -122.4194 - i*0.1, "dynamic_radius_km": 15.0}
            for i in range(20)
        ]
    })
    
    users = ["small_user", "medium_user", "large_user"]
    results = {}
    
    for user in users:
        # Measure profile load latency
        load_times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            service.load_profile(user)
            load_times.append((time.perf_counter() - t0) * 1000) # to ms
            
        # Create standard event
        event = LoginEvent(
            user_id=user,
            latitude=37.7749,
            longitude=-122.4194,
            timestamp=1600000000.0,
            device_hash="dev_test"
        )
        
        # Measure inference latency
        inference_times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            service.predict(event)
            inference_times.append((time.perf_counter() - t0) * 1000) # to ms
            
        # Measure memory growth
        _, peak_kb = measure_memory(service.predict, event)
        
        results[user] = {
            "profile_load": {
                "p50": round(np.percentile(load_times, 50), 4),
                "p95": round(np.percentile(load_times, 95), 4),
                "p99": round(np.percentile(load_times, 99), 4)
            },
            "inference": {
                "p50": round(np.percentile(inference_times, 50), 4),
                "p95": round(np.percentile(inference_times, 95), 4),
                "p99": round(np.percentile(inference_times, 99), 4)
            },
            "peak_memory_kb": round(peak_kb, 2)
        }
        
    return results

if __name__ == "__main__":
    res = run_ml_benchmarks()
    print("ML Benchmarks:", res)
