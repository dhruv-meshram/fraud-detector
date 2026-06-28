"""Algorithmic benchmark evaluating spatial functions and BallTree operations."""

import time
import numpy as np
from benchmarks.datasets.generator import generate_coordinates
from benchmarks.utils import fit_complexity, measure_memory
from prj.algorithms import haversine_distance, validate_velocity, calculate_risk_score
from prj.algorithms.spatial.bounding_box import in_bounding_box
from prj.algorithms.trees.ball_tree import BallTree

def run_algo_benchmarks():
    """Runs timing and memory benchmarks for all algorithms and estimates empirical complexity."""
    sizes = [10, 100, 1000, 10000, 100000]
    
    results = {
        "haversine": {"times": [], "complexity": "", "memory": []},
        "bounding_box": {"times": [], "complexity": "", "memory": []},
        "velocity_validator": {"times": [], "complexity": "", "memory": []},
        "risk_scoring": {"times": [], "complexity": "", "memory": []},
        "balltree_build": {"times": [], "complexity": "", "memory": []},
        "balltree_query": {"times": [], "complexity": "", "memory": []}
    }
    
    # 1. Haversine Distance (Run N times)
    for n in sizes:
        coords = generate_coordinates(n)
        start = time.perf_counter()
        # Compute N haversine computations
        for i in range(n - 1):
            haversine_distance(coords[i, 0], coords[i, 1], coords[i+1, 0], coords[i+1, 1])
        elapsed = time.perf_counter() - start
        results["haversine"]["times"].append(elapsed)
        
        # Measure peak memory
        _, peak_kb = measure_memory(
            lambda c: [haversine_distance(c[i, 0], c[i, 1], c[i+1, 0], c[i+1, 1]) for i in range(len(c) - 1)],
            coords
        )
        results["haversine"]["memory"].append(peak_kb)
        
    results["haversine"]["complexity"] = fit_complexity(sizes, results["haversine"]["times"])
    
    # 2. Bounding Box (Run N times)
    for n in sizes:
        coords = generate_coordinates(n)
        start = time.perf_counter()
        for i in range(n - 1):
            in_bounding_box(coords[i, 0], coords[i, 1], coords[i+1, 0], coords[i+1, 1], 15.0)
        elapsed = time.perf_counter() - start
        results["bounding_box"]["times"].append(elapsed)
        
        _, peak_kb = measure_memory(
            lambda c: [in_bounding_box(c[i, 0], c[i, 1], c[i+1, 0], c[i+1, 1], 15.0) for i in range(len(c) - 1)],
            coords
        )
        results["bounding_box"]["memory"].append(peak_kb)
        
    results["bounding_box"]["complexity"] = fit_complexity(sizes, results["bounding_box"]["times"])

    # 3. Velocity Validator (Run N times)
    for n in sizes:
        start = time.perf_counter()
        for i in range(n):
            validate_velocity(
                {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1600000000.0},
                {"latitude": 37.7800, "longitude": -122.4200, "timestamp": 1600000100.0}
            )
        elapsed = time.perf_counter() - start
        results["velocity_validator"]["times"].append(elapsed)
        
        _, peak_kb = measure_memory(
            lambda count: [validate_velocity(
                {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1600000000.0},
                {"latitude": 37.7800, "longitude": -122.4200, "timestamp": 1600000100.0}
            ) for _ in range(count)],
            n
        )
        results["velocity_validator"]["memory"].append(peak_kb)
        
    results["velocity_validator"]["complexity"] = fit_complexity(sizes, results["velocity_validator"]["times"])

    # 4. Risk Scoring (Run N times)
    for n in sizes:
        start = time.perf_counter()
        for i in range(n):
            calculate_risk_score(450.0, 85.0, True)
        elapsed = time.perf_counter() - start
        results["risk_scoring"]["times"].append(elapsed)
        
        _, peak_kb = measure_memory(
            lambda count: [calculate_risk_score(450.0, 85.0, True) for _ in range(count)],
            n
        )
        results["risk_scoring"]["memory"].append(peak_kb)
        
    results["risk_scoring"]["complexity"] = fit_complexity(sizes, results["risk_scoring"]["times"])

    # 5. BallTree Build (Build tree of size N)
    for n in sizes:
        coords = generate_coordinates(n)
        start = time.perf_counter()
        tree = BallTree(coords)
        elapsed = time.perf_counter() - start
        results["balltree_build"]["times"].append(elapsed)
        
        _, peak_kb = measure_memory(BallTree, coords)
        results["balltree_build"]["memory"].append(peak_kb)
        
    results["balltree_build"]["complexity"] = fit_complexity(sizes, results["balltree_build"]["times"])

    # 6. BallTree Query (Single nearest neighbor query on tree of size N)
    for n in sizes:
        coords = generate_coordinates(n)
        tree = BallTree(coords)
        query_pt = np.array([37.7749, -122.4194])
        
        start = time.perf_counter()
        tree.query(query_pt, k=1)
        elapsed = time.perf_counter() - start
        results["balltree_query"]["times"].append(elapsed)
        
        _, peak_kb = measure_memory(tree.query, query_pt, k=1)
        results["balltree_query"]["memory"].append(peak_kb)
        
    results["balltree_query"]["complexity"] = fit_complexity(sizes, results["balltree_query"]["times"])

    return results

if __name__ == "__main__":
    res = run_algo_benchmarks()
    print("Benchmark complete:", res)
