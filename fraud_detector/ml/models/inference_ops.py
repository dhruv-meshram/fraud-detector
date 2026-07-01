"""ShieldFlow Production Inference Operations.

Performs sub-millisecond O(K) geographic boundary evaluation against a user's 
historical DBSCAN centroids, avoiding full DBSCAN recalculation at inference time.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from fraud_detector.ml.features.geo_features import haversine_distance_numpy

DEFAULT_PROFILES_DIR = Path("/home/dhruv/Documents/fraud-detector/fraud_detector/data/processed/profiles")

def load_user_profile(user_id: str, profiles_dir: Path = DEFAULT_PROFILES_DIR) -> Optional[Dict[str, Any]]:
    """Loads the precomputed spatial profile JSON for a specific user.
    
    Args:
        user_id: The UUID/identifier of the user.
        profiles_dir: The directory containing the user profile JSON files.
        
    Returns:
        Dictionary profile if found, otherwise None.
    """
    profile_path = Path(profiles_dir) / f"{user_id}.json"
    if not profile_path.exists():
        return None
        
    try:
        with open(profile_path, "r") as f:
            return json.load(f)
    except Exception as e:
        # Log error or handle corruption
        print(f"Error loading profile for {user_id}: {e}")
        return None

def evaluate_login_location(
    user_id: str, 
    lat: float, 
    lon: float, 
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
    buffer_km: float = 0.0
) -> Dict[str, Any]:
    """Evaluates whether an incoming login coordinate is within the user's known spatial zones.
    
    Performs an O(K) comparison against all precomputed cluster centroids.
    
    Args:
        user_id: The UUID/identifier of the user.
        lat: Latitude of the incoming login (degrees).
        lon: Longitude of the incoming login (degrees).
        profiles_dir: Path to directory containing precomputed profiles.
        buffer_km: Additional runtime distance cushion to append to cluster radii.
        
    Returns:
        A dictionary containing:
            - 'status': 'KNOWN_ZONE', 'OUTLIER', or 'NO_PROFILE'
            - 'closest_cluster_id': ID of the closest cluster (None if NO_PROFILE or no clusters)
            - 'distance_km': Distance to the closest cluster centroid (None if NO_PROFILE)
            - 'dynamic_radius_km': Radius of the closest cluster (None if NO_PROFILE)
    """
    profile = load_user_profile(user_id, profiles_dir)
    
    if not profile or not profile.get("clusters"):
        return {
            "status": "NO_PROFILE",
            "closest_cluster_id": None,
            "distance_km": None,
            "dynamic_radius_km": None
        }
        
    clusters = profile["clusters"]
    
    min_dist = float("inf")
    closest_cluster = None
    
    # Calculate distance to each cluster centroid
    for cluster in clusters:
        c_lat = cluster["centroid_lat"]
        c_lon = cluster["centroid_lon"]
        
        dist = haversine_distance_numpy(lat, lon, c_lat, c_lon)
        if dist < min_dist:
            min_dist = dist
            closest_cluster = cluster
            
    if closest_cluster is None:
        return {
            "status": "OUTLIER",
            "closest_cluster_id": None,
            "distance_km": None,
            "dynamic_radius_km": None
        }
        
    # Check if inside closest cluster boundary
    allowed_radius = closest_cluster["dynamic_radius_km"] + buffer_km
    is_known = min_dist <= allowed_radius
    
    return {
        "status": "KNOWN_ZONE" if is_known else "OUTLIER",
        "closest_cluster_id": closest_cluster["cluster_id"],
        "distance_km": round(float(min_dist), 3),
        "dynamic_radius_km": closest_cluster["dynamic_radius_km"]
    }
