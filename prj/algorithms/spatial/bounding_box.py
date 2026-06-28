"""Bounding Box Pre-filter for fast spatial checks."""

import math

def in_bounding_box(lat: float, lon: float, center_lat: float, center_lon: float, radius_km: float) -> bool:
    """Arithmetic pre-filter to bypass expensive trigonometry if coordinates are far.
    
    Complexity: O(1) Time, O(1) Space.
    Handles dateline crossing (longitude wrap-around) and polar safety.
    """
    # 1 degree of latitude is approximately 111.1 km
    lat_delta = radius_km / 111.1
    
    if lat < center_lat - lat_delta or lat > center_lat + lat_delta:
        return False
        
    # 1 degree of longitude is approximately 111.1 * cos(latitude) km
    cos_lat = math.cos(math.radians(center_lat))
    
    # Polar safety check: at the poles, longitude boundaries span all degrees
    if abs(cos_lat) < 1e-4:
        return True
        
    lon_delta = radius_km / (111.1 * cos_lat)
    
    # Calculate difference and wrap around the 180-meridian
    diff_lon = lon - center_lon
    diff_lon = (diff_lon + 180) % 360 - 180
    
    return abs(diff_lon) <= lon_delta
