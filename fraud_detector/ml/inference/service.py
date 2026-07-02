import json
from pathlib import Path
from typing import Dict, Any, Optional
from fraud_detector.models.event import LoginEvent
from fraud_detector.algorithms import haversine_distance

DEFAULT_PROFILES_DIR = Path("/home/dhruv/Documents/fraud-detector/fraud_detector/data/processed/profiles")

from fraud_detector.adapters.profile import PostgreSQLProfileStore

class InferenceService:
    """Encapsulates spatial profile loading and centroid evaluation (predict)."""
    
    def __init__(self, profiles_dir: Optional[Path] = None, profile_store = None, db_engine_or_conn = None):
        self.profiles_dir = Path(profiles_dir) if profiles_dir else DEFAULT_PROFILES_DIR
        self.profile_store = profile_store or PostgreSQLProfileStore(db_engine_or_conn=db_engine_or_conn)

    def load_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Loads precomputed spatial profile for a user."""
        return self.profile_store.get_profile(user_id)

    def predict(self, event: LoginEvent) -> Dict[str, Any]:
        """Evaluates whether an incoming login event coordinate matches the user's spatial baseline.
        
        Returns a dictionary containing:
            - 'status': 'KNOWN_ZONE', 'OUTLIER', or 'NO_PROFILE'
            - 'closest_cluster_id': ID of the closest cluster
            - 'distance_km': Distance in km to closest cluster centroid
            - 'dynamic_radius_km': Allowed cluster boundary radius
            - 'profile': The complete profile dictionary
        """
        user_id = event.user_id
        lat = event.latitude
        lon = event.longitude
        
        profile = self.load_profile(user_id)
        if not profile or not profile.get("clusters"):
            return {
                "status": "NO_PROFILE",
                "closest_cluster_id": None,
                "distance_km": None,
                "dynamic_radius_km": None,
                "profile": None
            }
            
        clusters = profile["clusters"]
        min_dist = float("inf")
        closest_cluster = None
        
        for cluster in clusters:
            c_lat = cluster["centroid_lat"]
            c_lon = cluster["centroid_lon"]
            dist = haversine_distance(lat, lon, c_lat, c_lon)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster
                
        if closest_cluster is None:
            return {
                "status": "OUTLIER",
                "closest_cluster_id": None,
                "distance_km": None,
                "dynamic_radius_km": None,
                "profile": profile
            }
            
        allowed_radius = closest_cluster["dynamic_radius_km"]
        is_known = min_dist <= allowed_radius
        
        return {
            "status": "KNOWN_ZONE" if is_known else "OUTLIER",
            "closest_cluster_id": closest_cluster["cluster_id"],
            "distance_km": round(float(min_dist), 3),
            "dynamic_radius_km": closest_cluster["dynamic_radius_km"],
            "profile": profile
        }
