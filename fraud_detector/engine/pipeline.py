from fraud_detector.models.event import LoginEvent
from fraud_detector.models.result import FraudResult, RiskBreakdown
from fraud_detector.algorithms import validate_velocity, calculate_risk_score, haversine_distance, spatiotemporal_velocity
from fraud_detector.ml.inference import InferenceService
from fraud_detector.adapters import FileProfileStore, RedisCacheStore, PostgresDBStore, KafkaAlertProducer

class DetectionPipeline:
    """The orchestrating pipeline that sequentially applies validations, checks, ML inferences, and scores."""
    
    def __init__(self, profile_store=None, cache_store=None, db_store=None, alert_producer=None):
        self.profile_store = profile_store or FileProfileStore()
        self.cache_store = cache_store or RedisCacheStore()
        self.db_store = db_store or PostgresDBStore()
        self.alert_producer = alert_producer or KafkaAlertProducer()
        
        # Instantiate inference service with profile_store abstraction
        self.inference_service = InferenceService(profile_store=self.profile_store)


    def process(self, event: LoginEvent) -> FraudResult:
        user_id = event.user_id
        lat = event.latitude
        lon = event.longitude
        ts = event.timestamp
        device = event.device_hash

        # 1. Fetch last verified login node
        last_node = self.cache_store.get_last_node(user_id)
        if not last_node:
            last_node = self.db_store.get_last_verified_login(user_id)
            if last_node:
                # Auto-hydrate cache
                self.cache_store.set_last_node(user_id, last_node)
        velocity_kmh = 0.0

        # 2. Graph Validator: Spatiotemporal Velocity Check
        if last_node:
            velocity_kmh = spatiotemporal_velocity(
                last_node["latitude"], last_node["longitude"], last_node["timestamp"],
                lat, lon, ts
            )
            is_possible = validate_velocity(last_node, {"latitude": lat, "longitude": lon, "timestamp": ts})
            
            if not is_possible:
                verdict = "HIGH_RISK"
                status = "IMPOSSIBLE_VELOCITY"
                reason = f"Impossible velocity of {velocity_kmh:.2f} km/h detected relative to last login."
                
                # Emit Kafka Alert
                self.alert_producer.emit_event(
                    topic="shieldflow.anomalies",
                    key=user_id,
                    value={"user_id": user_id, "reason": reason, "velocity_kmh": velocity_kmh, "verdict": verdict}
                )
                
                return FraudResult(
                    risk_score=100.0,
                    is_fraudulent=True,
                    reasons=[reason],
                    status=status,
                    details=RiskBreakdown(
                        velocity_kmh=round(velocity_kmh, 2),
                        distance_km=0.0,
                        device_mismatch=False
                    )
                )

        # 3. ML Inference layer evaluation
        evaluation = self.inference_service.predict(event)
        
        # 3a. Cold Start Fallback
        if evaluation["status"] == "NO_PROFILE":
            verdict = "LOW_RISK"
            status = "COLD_START_BYPASS"
            reason = "User is in Cold Start state (< 10 logins). Passed velocity checks."
            
            # In cold start, if it is low risk, we auto-verify
            self.db_store.record_login(user_id, lat, lon, ts, device, is_verified=True)
            self.cache_store.set_last_node(user_id, {"latitude": lat, "longitude": lon, "timestamp": ts, "device_hash": device})
            
            return FraudResult(
                risk_score=0.0,
                is_fraudulent=False,
                reasons=[reason],
                status=status,
                details=RiskBreakdown(
                    velocity_kmh=round(velocity_kmh, 2),
                    distance_km=0.0,
                    device_mismatch=False
                )
            )

        # 3b. ML Inference boundary check
        if evaluation["status"] == "KNOWN_ZONE":
            verdict = "LOW_RISK"
            status = "KNOWN_ZONE"
            reason = f"Login matches spatial cluster {evaluation['closest_cluster_id']}."
            
            # Save low risk as verified login
            self.db_store.record_login(user_id, lat, lon, ts, device, is_verified=True)
            self.cache_store.set_last_node(user_id, {"latitude": lat, "longitude": lon, "timestamp": ts, "device_hash": device})
            
            return FraudResult(
                risk_score=0.0,
                is_fraudulent=False,
                reasons=[reason],
                status=status,
                details=RiskBreakdown(
                    velocity_kmh=round(velocity_kmh, 2),
                    distance_km=round(evaluation.get("distance_km", 0.0), 2),
                    device_mismatch=False
                )
            )

        # 3c. Spatial Outlier: Fallback to Multi-Factor Scoring
        profile = evaluation["profile"]
        closest_dist = evaluation["distance_km"]
        
        # Check device mismatch
        historic_devices = self.db_store.get_all_device_hashes(user_id)
        device_mismatch = device not in historic_devices if historic_devices else True
        
        # Calculate Multi-Factor score
        mfa_score = calculate_risk_score(velocity_kmh, closest_dist, device_mismatch)
        
        # Verdict threshold check
        if mfa_score >= 0.70:
            verdict = "HIGH_RISK"
            is_fraudulent = True
            reason = f"Spatial outlier with high multi-factor risk score of {mfa_score:.2f}."
        else:
            verdict = "MEDIUM_RISK"
            is_fraudulent = False
            reason = f"Spatial outlier with moderate multi-factor risk score of {mfa_score:.2f}."
            
        status = "OUTLIER"
        
        # Emit Kafka event for verification pipeline
        self.alert_producer.emit_event(
            topic="shieldflow.anomalies",
            key=user_id,
            value={
                "user_id": user_id,
                "reason": reason,
                "score": mfa_score,
                "velocity_kmh": velocity_kmh,
                "distance_km": closest_dist,
                "device_mismatch": device_mismatch,
                "verdict": verdict
            }
        )
        
        return FraudResult(
            risk_score=round(mfa_score * 100.0, 2),
            is_fraudulent=is_fraudulent,
            reasons=[reason],
            status=status,
            details=RiskBreakdown(
                velocity_kmh=round(velocity_kmh, 2),
                distance_km=round(closest_dist, 2),
                device_mismatch=device_mismatch
            )
        )
