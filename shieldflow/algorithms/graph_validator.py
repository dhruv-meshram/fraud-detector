from shieldflow.algorithms.haversine import haversine_distance
from shieldflow.config import MAX_VELOCITY_KMH

def validate_velocity(
    last_login: dict,
    current_login: dict,
    speed_limit_kmh: float = MAX_VELOCITY_KMH
) -> tuple[bool, float]:
    """Calculates temporal delta dt and compares velocity against max threshold.
    
    If dt == 0 and distance > 0, or velocity > max speed limit, it flags immediately.
    
    Returns:
        (is_possible, velocity_kmh)
    """
    lat1 = float(last_login["latitude"])
    lon1 = float(last_login["longitude"])
    ts1 = float(last_login["timestamp"])
    
    lat2 = float(current_login["latitude"])
    lon2 = float(current_login["longitude"])
    ts2 = float(current_login["timestamp"])
    
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    dt = abs(ts2 - ts1)
    
    if dt == 0.0:
        if dist > 0.0:
            return False, float('inf')
        return True, 0.0
        
    hours = dt / 3600.0
    velocity = dist / hours
    
    if velocity > speed_limit_kmh:
        return False, velocity
        
    return True, velocity
