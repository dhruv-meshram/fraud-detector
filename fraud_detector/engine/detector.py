from typing import Dict, Any, Union
from fraud_detector.models.event import LoginEvent
from fraud_detector.models.result import FraudResult
from fraud_detector.engine.pipeline import DetectionPipeline

class FraudDetector:
    """The main external interface for orchestrating login anomaly and fraud checks."""
    
    def __init__(self, profile_store=None, cache_store=None, db_store=None, alert_producer=None):
        self.pipeline = DetectionPipeline(
            profile_store=profile_store,
            cache_store=cache_store,
            db_store=db_store,
            alert_producer=alert_producer
        )

    def analyze(self, login_data: Union[Dict[str, Any], LoginEvent]) -> FraudResult:
        """Analyzes a login event for potential anomalies and impossible travel."""
        if isinstance(login_data, dict):
            event = LoginEvent(**login_data)
        else:
            event = login_data
            
        return self.pipeline.process(event)

    def verify_anomaly(self, user_id: str, latitude: float, longitude: float, timestamp: float, device_hash: str, is_verified: bool):
        """Processes MFA outcome to update baselines and record logs."""
        self.pipeline.db_store.record_login(user_id, latitude, longitude, timestamp, device_hash, is_verified)
        if is_verified:
            self.pipeline.cache_store.set_last_node(user_id, {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": timestamp,
                "device_hash": device_hash
            })

