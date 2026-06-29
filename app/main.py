"""ShieldFlow FastAPI Gatekeeper API.

Exposes endpoints to evaluate real-time login risks and verify anomaly MFA challenges.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import math

from training.retrain import check_and_trigger_retrain

from fraud_detector import FraudDetector
from fraud_detector.adapters import PostgreSQLProfileStore, RedisCacheStore, PostgresDBStore, KafkaAlertProducer

# Instantiate detector with production-grade storage adapters
profile_store = PostgreSQLProfileStore()
cache_store = RedisCacheStore()
db_store = PostgresDBStore()
alert_producer = KafkaAlertProducer()

detector = FraudDetector(
    profile_store=profile_store,
    cache_store=cache_store,
    db_store=db_store,
    alert_producer=alert_producer
)

app = FastAPI(
    title="ShieldFlow Risk Evaluation Gateway",
    description="Deterministic Graph + Spatial DBSCAN Risk Evaluator API",
    version="1.0.0"
)

# --- Pydantic Data Models ---
class LoginEvent(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", example="user_123")
    latitude: float = Field(..., description="Login latitude coordinate", example=37.7749)
    longitude: float = Field(..., description="Login longitude coordinate", example=-122.4194)
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp(), description="UNIX timestamp")
    device_hash: str = Field(..., description="Client device unique hash value", example="abc123device")

class AnomalyVerification(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    latitude: float = Field(..., description="Login latitude coordinate")
    longitude: float = Field(..., description="Login longitude coordinate")
    timestamp: float = Field(..., description="UNIX timestamp")
    device_hash: str = Field(..., description="Client device unique hash value")
    is_verified: bool = Field(..., description="Did the user successfully pass MFA challenge?")

# --- API Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/evaluate_risk")
def evaluate_risk(event: LoginEvent):
    """Evaluates the risk level of an incoming login event using Graph + ML."""
    res = detector.analyze({
        "user_id": event.user_id,
        "latitude": event.latitude,
        "longitude": event.longitude,
        "timestamp": event.timestamp,
        "device_hash": event.device_hash
    })
    
    return {
        "verdict": res.verdict,
        "status": res.status,
        "score": res.risk_score / 100.0,
        "reason": res.reasons[0] if res.reasons else "",
        "details": {
            "velocity_kmh": round(res.details.velocity_kmh, 2),
            "distance_km": round(res.details.distance_km, 2) if res.status not in ["IMPOSSIBLE_VELOCITY", "COLD_START_BYPASS"] else None,
            "device_mismatch": res.details.device_mismatch if res.status == "OUTLIER" else None
        }
    }

@app.post("/analyze")
def analyze(event: LoginEvent):
    """SDK-native analysis endpoint."""
    return detector.analyze({
        "user_id": event.user_id,
        "latitude": event.latitude,
        "longitude": event.longitude,
        "timestamp": event.timestamp,
        "device_hash": event.device_hash
    })

@app.post("/verify_anomaly")
def verify_anomaly(payload: AnomalyVerification, background_tasks: BackgroundTasks):
    """Processes MFA outcome to update baselines and trigger retraining checks."""
    user_id = payload.user_id
    lat = payload.latitude
    lon = payload.longitude
    ts = payload.timestamp
    device = payload.device_hash
    is_verified = payload.is_verified
    
    # 1. Update database and cache via detector encapsulating method
    detector.verify_anomaly(user_id, lat, lon, ts, device, is_verified)
        
    # 2. Trigger Retraining Engine check as a background task
    retrain_verdict = {}
    try:
        retrain_verdict = check_and_trigger_retrain(
            user_id=user_id,
            new_login_lat=lat,
            new_login_lon=lon,
            new_login_ts=ts,
            new_login_device=device
        )
    except Exception as e:
        print(f"[RETRAIN FAILED] Retrain logic check failed: {e}")
        
    return {
        "status": "success",
        "recorded_verified": is_verified,
        "retrain_status": retrain_verdict
    }
