"""Utility functions for benchmarking, complexity fitting, and profiling."""

import time
import tracemalloc
import cProfile
import pstats
import numpy as np

def fit_complexity(sizes, times):
    """Fits execution times to various theoretical models and returns the best fit.
    
    Models evaluated: O(1), O(log n), O(n), O(n log n), O(n^2).
    """
    x = np.array(sizes, dtype=float)
    y = np.array(times, dtype=float)
    
    # Avoid zero or negative values in transformations
    x_safe = np.maximum(x, 1.0)
    
    models = {
        "O(1)": lambda v: np.ones_like(v),
        "O(log n)": lambda v: np.log2(v),
        "O(n)": lambda v: v,
        "O(n log n)": lambda v: v * np.log2(v),
        "O(n^2)": lambda v: v**2
    }
    
    best_model = "O(1)"
    min_rss = float("inf")
    
    for name, transform in models.items():
        feat = transform(x_safe)
        
        # Construct feature matrix X
        if name == "O(1)":
            X = np.ones((len(x), 1))
        else:
            X = np.vstack([feat, np.ones_like(feat)]).T
            
        try:
            # Solve linear least squares
            coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            
            # Compute Residual Sum of Squares (RSS)
            if len(residuals) > 0:
                rss = residuals[0]
            else:
                pred = X.dot(coef)
                rss = np.sum((y - pred)**2)
                
            if rss < min_rss:
                min_rss = rss
                best_model = name
        except Exception:
            continue
            
    return best_model

def measure_memory(func, *args, **kwargs):
    """Measures peak memory allocation during the execution of func.
    
    Returns (result, peak_memory_kb).
    """
    tracemalloc.start()
    start_mem, _ = tracemalloc.get_traced_memory()
    
    result = func(*args, **kwargs)
    
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Convert bytes to KB
    allocated_kb = (peak_mem - start_mem) / 1024.0
    return result, max(0.0, allocated_kb)

def run_profiler(func, output_path, *args, **kwargs):
    """Runs cProfile on the given function and writes statistics to output_path."""
    prof = cProfile.Profile()
    result = prof.runcall(func, *args, **kwargs)
    
    prof.dump_stats(output_path)
    
    # Format and save human readable call tree / stats as text
    txt_path = output_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        stats = pstats.Stats(prof, stream=f)
        stats.strip_dirs().sort_stats("cumulative").print_stats(40)
        
    return result
