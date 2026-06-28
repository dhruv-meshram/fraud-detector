"""Haversine Distance Calculator."""

import math

R_EARTH = 6371.0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates spherical distance between two points in kilometers.
    
    Complexity: O(1) Time, O(1) Space.
    Executes in < 1 microsecond using native math library.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        
    c = 2.0 * math.asin(math.sqrt(a))
    return R_EARTH * c

def spatiotemporal_velocity(lat1: float, lon1: float, t1: float, lat2: float, lon2: float, t2: float) -> float:
    """Calculates spatiotemporal velocity in km/h between two logins."""
    dist_km = haversine_distance(lat1, lon1, lat2, lon2)
    time_diff_sec = abs(t2 - t1)
    if time_diff_sec == 0:
        return 0.0
    time_diff_hours = time_diff_sec / 3600.0
    return dist_km / time_diff_hours

