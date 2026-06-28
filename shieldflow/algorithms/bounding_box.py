import math

def inside_bounding_box(
    lat: float,
    lon: float,
    centroid_lat: float,
    centroid_lon: float,
    radius_km: float
) -> bool:
    """Performs a fast rectangular bounding box check around a centroid."""
    # Approximate degree translation
    delta_lat = radius_km / 111.0
    
    cos_lat = math.cos(math.radians(centroid_lat))
    if abs(cos_lat) < 1e-6:
        delta_lon = 180.0
    else:
        delta_lon = radius_km / (111.0 * cos_lat)
        
    lat_min = centroid_lat - delta_lat
    lat_max = centroid_lat + delta_lat
    lon_min = centroid_lon - delta_lon
    lon_max = centroid_lon + delta_lon
    
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)

def passes_bounding_box_filter(
    lat: float,
    lon: float,
    clusters: list
) -> bool:
    """Checks if coordinate falls inside the bounding box of any known user cluster profile."""
    if not clusters:
        return False
        
    for cluster in clusters:
        c_lat = float(cluster["centroid_lat"])
        c_lon = float(cluster["centroid_lon"])
        r_km = float(cluster["dynamic_radius"])
        
        if inside_bounding_box(lat, lon, c_lat, c_lon, r_km):
            return True
            
    return False
