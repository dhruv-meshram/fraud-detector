"""Synthetic dataset generator for benchmark runs."""

import numpy as np

def generate_coordinates(n: int) -> np.ndarray:
    """Generates an array of shape (n, 2) containing latitude and longitude coordinates."""
    # San Francisco as base center
    lat_center = 37.7749
    lon_center = -122.4194
    
    # Random offsets around the center (up to ~500km)
    lat_offset = np.random.uniform(-4.5, 4.5, size=n)
    lon_offset = np.random.uniform(-4.5, 4.5, size=n)
    
    coords = np.column_stack((lat_center + lat_offset, lon_center + lon_offset))
    return coords

def generate_events(n: int, user_id: str = "perf_user"):
    """Generates a list of n serialized login event payloads."""
    events = []
    base_ts = 1600000000.0
    coords = generate_coordinates(n)
    
    for i in range(n):
        events.append({
            "user_id": user_id,
            "latitude": float(coords[i, 0]),
            "longitude": float(coords[i, 1]),
            "timestamp": base_ts + i * 300, # 5 mins apart
            "device_hash": f"dev_{i % 3}"
        })
    return events
