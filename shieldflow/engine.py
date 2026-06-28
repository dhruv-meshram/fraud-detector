import logging
from shieldflow.config import (
    MAX_VELOCITY_KMH,
    COLD_START_THRESHOLD,
    RETRAIN_BATCH_INTERVAL
)
from shieldflow.algorithms.haversine import haversine_distance
from shieldflow.algorithms.graph_validator import validate_velocity
from shieldflow.algorithms.bounding_box import passes_bounding_box_filter
from shieldflow.ml.dbscan_gpu import run_dbscan_gpu
from shieldflow.ml.profile_store import ProfileStore

logger = logging.getLogger("ShieldFlowEngine")
logging.basicConfig(level=logging.INFO)

class ShieldFlowEngine:
    """Standalone, programmatic ShieldFlow security engine microservice plugin."""

    def __init__(self) -> None:
        self.store = ProfileStore()
        # Track training iterations per user
        self.retrain_counts = {}

    def evaluate_risk(self, event: dict) -> dict:
        """Evaluates the risk level of an incoming login event log.
        
        Args:
            event: dict containing 'user_id', 'timestamp', 'latitude', 'longitude', 'device_hash'.
            
        Returns:
            dict containing verdict and telemetry parameters.
        """
        user_id = event["user_id"]
        lat = float(event["latitude"])
        lon = float(event["longitude"])
        ts = float(event["timestamp"])
        device = event["device_hash"]

        history = self.store.get_history(user_id)
        clusters = self.store.get_profile(user_id)
        
        # 1. Spatiotemporal Velocity check (Deterministic Graph path)
        velocity_kmh = 0.0
        if history:
            last_login = history[-1]
            is_possible, velocity_kmh = validate_velocity(last_login, event)
            if not is_possible:
                return {
                    "user_id": user_id,
                    "timestamp": ts,
                    "verdict": "HIGH_RISK",
                    "is_flagged": True,
                    "telemetry": {
                        "engine_state": self._get_engine_state(user_id),
                        "calculated_velocity_kmh": round(velocity_kmh, 2),
                        "spatial_status": "UNKNOWN"
                    }
                }

        # 2. Determine Current Engine State
        engine_state = self._get_engine_state(user_id)

        # 3. Handle Cold Start Bootstrapping
        if len(history) < COLD_START_THRESHOLD:
            # Under bootstrapping, auto-verify logins with valid velocity
            self.verify_login(event)
            return {
                "user_id": user_id,
                "timestamp": ts,
                "verdict": "LOW_RISK",
                "is_flagged": False,
                "telemetry": {
                    "engine_state": "BOOTSTRAPPING",
                    "calculated_velocity_kmh": round(velocity_kmh, 2),
                    "spatial_status": "UNKNOWN"
                }
            }

        # 4. Handle Active Profile Inference
        device_mismatch = not self.store.is_known_device(user_id, device)
        spatial_status = "NOISE_OUTLIER"
        
        # Optimization check: Bounding box pass first
        bbox_match = passes_bounding_box_filter(lat, lon, clusters)
        
        closest_dist = 99999.0
        if clusters:
            for cluster in clusters:
                c_lat = float(cluster["centroid_lat"])
                c_lon = float(cluster["centroid_lon"])
                r_km = float(cluster["dynamic_radius"])
                dist = haversine_distance(lat, lon, c_lat, c_lon)
                
                if dist < closest_dist:
                    closest_dist = dist
                
                # Check actual spatial boundary if bounding box matched
                if bbox_match and dist <= r_km:
                    spatial_status = "INSIDE_CLUSTER"
        else:
            closest_dist = 0.0

        # 5. Determine Verdict and Flags
        if spatial_status == "NOISE_OUTLIER":
            if closest_dist > 150.0:
                verdict = "HIGH_RISK"
            else:
                verdict = "MEDIUM_RISK"
            is_flagged = True
        elif device_mismatch:
            verdict = "MEDIUM_RISK"
            is_flagged = True
        else:
            verdict = "LOW_RISK"
            is_flagged = False
            # Auto-verify safe logins
            self.verify_login(event)

        return {
            "user_id": user_id,
            "timestamp": ts,
            "verdict": verdict,
            "is_flagged": is_flagged,
            "telemetry": {
                "engine_state": engine_state,
                "calculated_velocity_kmh": round(velocity_kmh, 2),
                "spatial_status": spatial_status
            }
        }

    def verify_login(self, event: dict) -> None:
        """Confirms a login event (either via cold start or post-MFA user challenge resolution)."""
        user_id = event["user_id"]
        self.store.add_history(user_id, event)
        history = self.store.get_history(user_id)
        
        h_len = len(history)
        # Check model creation and update intervals
        if h_len == COLD_START_THRESHOLD:
            self._retrain_profile(user_id)
        elif h_len > COLD_START_THRESHOLD and (h_len - COLD_START_THRESHOLD) % RETRAIN_BATCH_INTERVAL == 0:
            self._retrain_profile(user_id)

    def _retrain_profile(self, user_id: str) -> None:
        """Executes DBSCAN clustering model calculation on user's verified coordinates."""
        history = self.store.get_history(user_id)
        clusters = run_dbscan_gpu(history)
        self.store.save_profile(user_id, clusters)
        
        # Track training rounds
        if user_id not in self.retrain_counts:
            self.retrain_counts[user_id] = 0
        self.retrain_counts[user_id] += 1
        logger.info(f"Retrained profile for user {user_id}. Found {len(clusters)} clusters. Round: {self.retrain_counts[user_id]}")

    def _get_engine_state(self, user_id: str) -> str:
        history = self.store.get_history(user_id)
        if len(history) < COLD_START_THRESHOLD:
            return "BOOTSTRAPPING"
        rounds = self.retrain_counts.get(user_id, 0)
        if rounds > 1:
            return "ADAPTIVE_STABILIZED"
        return "ACTIVE_INFERENCE"
