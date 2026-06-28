"""Velocity Validator: Evaluates spatiotemporal continuity (speed limit checks)."""

from algorithms.spatial.haversine import haversine_distance

DEFAULT_SPEED_LIMIT_KMH = 900.0

def validate_velocity(
    last_login: dict, 
    current_login: dict, 
    speed_limit_kmh: float = DEFAULT_SPEED_LIMIT_KMH
) -> bool:
    """Checks if the travel velocity between two logins is physically possible.
    
    Args:
        last_login: dict containing keys 'latitude', 'longitude', and 'timestamp' (UNIX).
        current_login: dict containing keys 'latitude', 'longitude', and 'timestamp' (UNIX).
        speed_limit_kmh: Speed threshold in km/h (default: 900.0).
        
    Returns:
        Boolean indicating if the speed is physically possible (<= speed_limit_kmh).
    """
    lat1 = float(last_login["latitude"])
    lon1 = float(last_login["longitude"])
    ts1 = float(last_login["timestamp"])
    
    lat2 = float(current_login["latitude"])
    lon2 = float(current_login["longitude"])
    ts2 = float(current_login["timestamp"])
    
    delta_t = abs(ts2 - ts1)
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    
    if delta_t < 1.0:
        # Avoid division by zero for sub-second rapid-fire logins
        # If coordinates are identical or virtually same location, it's possible.
        return dist < 0.1
        
    # Convert delta_t from seconds to hours
    hours = delta_t / 3600.0
    velocity = dist / hours
    
    return velocity <= speed_limit_kmh
