"""Storage adapter benchmark measuring read/write times and serialization cost."""

import time
import json
import tempfile
from pathlib import Path
from benchmarks.utils import measure_memory
from fraud_detector.adapters import FileProfileStore, InMemoryProfileStore

def run_adapter_benchmarks():
    """Measures load times, lookup times, and serialization costs for adapters."""
    results = {}
    
    # 1. Profile Serialization Benchmark
    # Define a standard profile payload
    profile_data = {
        "user_id": "test_adapter_user",
        "clusters": [
            {"cluster_id": i, "centroid_lat": 37.7749 + i*0.01, "centroid_lon": -122.4194 - i*0.01, "dynamic_radius_km": 15.0}
            for i in range(10)
        ]
    }
    
    # Measure serialization cost (JSON stringification)
    ser_times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        json.dumps(profile_data)
        ser_times.append((time.perf_counter() - t0) * 1000) # ms
        
    # Measure deserialization cost
    json_str = json.dumps(profile_data)
    deser_times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        json.loads(json_str)
        deser_times.append((time.perf_counter() - t0) * 1000) # ms
        
    results["serialization"] = {
        "json_dump_ms_avg": round(float(sum(ser_times)/1000), 5),
        "json_load_ms_avg": round(float(sum(deser_times)/1000), 5)
    }
    
    # 2. FileProfileStore Benchmark
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        store = FileProfileStore(profiles_dir=tmpdir_path)
        
        # Write times
        write_times = []
        for i in range(100):
            user_id = f"user_{i}"
            t0 = time.perf_counter()
            store.save_profile(user_id, profile_data)
            write_times.append((time.perf_counter() - t0) * 1000)
            
        # Read times
        read_times = []
        for i in range(100):
            user_id = f"user_{i}"
            t0 = time.perf_counter()
            store.get_profile(user_id)
            read_times.append((time.perf_counter() - t0) * 1000)
            
        results["file_profile_store"] = {
            "write_ms_avg": round(float(sum(write_times)/100), 4),
            "read_ms_avg": round(float(sum(read_times)/100), 4)
        }
        
    # 3. InMemoryProfileStore Benchmark
    mem_store = InMemoryProfileStore()
    mem_write_times = []
    for i in range(100):
        user_id = f"user_{i}"
        t0 = time.perf_counter()
        mem_store.save_profile(user_id, profile_data)
        mem_write_times.append((time.perf_counter() - t0) * 1000)
        
    mem_read_times = []
    for i in range(100):
        user_id = f"user_{i}"
        t0 = time.perf_counter()
        mem_store.get_profile(user_id)
        mem_read_times.append((time.perf_counter() - t0) * 1000)
        
    results["in_memory_profile_store"] = {
        "write_ms_avg": round(float(sum(mem_write_times)/100), 4),
        "read_ms_avg": round(float(sum(mem_read_times)/100), 4)
    }
    
    return results

if __name__ == "__main__":
    res = run_adapter_benchmarks()
    print("Adapter Benchmarks:", res)
