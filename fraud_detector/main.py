import time
import logging
from typing import Dict, Any, Optional

from fraud_detector.adapters.db_sql import SQLDatabaseManager
from fraud_detector.adapters.db import PostgresDBStore
from fraud_detector.adapters.profile import PostgreSQLProfileStore
from fraud_detector.engine.detector import FraudDetector
from fraud_detector.models.event import LoginEvent
from fraud_detector.ml.training.retrain import check_and_trigger_retrain

logger = logging.getLogger("fraud_detector")

def check_fraud(
    db_conn = None,
    user_id: str = None,
    device_fingerprint: str = None,
    ip_address: str = "unknown",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    event: Optional[Dict[str, Any]] = None
) -> float:
    """The primary public entry point for ShieldFlow Fraud Detection SDK.
    
    Accepts:
        db_conn: SQLAlchemy engine/connection, raw DBAPI connection, or None (defaults to sqlite in-memory).
        user_id: Unique user identifier.
        device_fingerprint: Client device unique fingerprint hash.
        ip_address: IP address of the login attempt.
        latitude: Latitude of the login attempt (-90.0 to 90.0).
        longitude: Longitude of the login attempt (-180.0 to 180.0).
        event: Optional dict containing metadata and optional 'timestamp' (UNIX epoch).
        
    Returns:
        fraud_score: float between 0.0 and 1.0 (calculated as risk_score / 100.0).
    """
    # 1. Input Validation
    if not user_id:
        raise ValueError("user_id is required and cannot be empty.")
    if not device_fingerprint:
        raise ValueError("device_fingerprint is required and cannot be empty.")
        
    if event is None:
        event = {}
        
    lat = latitude
    lon = longitude
    
    # Fallback to checking inside event dict for backwards compatibility
    if lat is None:
        lat = event.get("latitude")
    if lon is None:
        lon = event.get("longitude")
        
    if lat is None:
        raise ValueError("latitude is required and cannot be None.")
    if lon is None:
        raise ValueError("longitude is required and cannot be None.")
        
    # Verify coordinate types
    if not isinstance(lat, (int, float)):
        raise ValueError(f"latitude must be a numeric value, got {type(lat)}.")
    if not isinstance(lon, (int, float)):
        raise ValueError(f"longitude must be a numeric value, got {type(lon)}.")
        
    # Verify coordinate ranges
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"latitude must be between -90.0 and 90.0, got {lat}.")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"longitude must be between -180.0 and 180.0, got {lon}.")
        
    timestamp = event.get("timestamp")
    if timestamp is None:
        timestamp = time.time()
        
    # 2. Initialize Database Manager (Auto-initializes tables if not present)
    sql_manager = SQLDatabaseManager(db_engine_or_conn=db_conn)
    db_store = PostgresDBStore(sql_manager=sql_manager)
    profile_store = PostgreSQLProfileStore(sql_manager=sql_manager)
    
    # Construct internal LoginEvent
    login_event = LoginEvent(
        user_id=user_id,
        latitude=float(lat),
        longitude=float(lon),
        timestamp=float(timestamp),
        device_hash=device_fingerprint
    )
    
    # 3. Store incoming event (recorded as unverified initially)
    db_store.record_login(
        user_id=user_id,
        lat=login_event.latitude,
        lon=login_event.longitude,
        ts=login_event.timestamp,
        device_hash=login_event.device_hash,
        is_verified=False,
        ip_address=ip_address
    )
    
    # 4. Run Fraud Algorithms
    detector = FraudDetector(profile_store=profile_store, db_store=db_store)
    result = detector.analyze(login_event)
    
    # 5. If not flagged as fraud, update event to is_verified = True and update behavioral profile
    if not result.is_fraudulent:
        db_store.verify_last_login(user_id, login_event.timestamp)
        # Update last node in local cache if applicable
        detector.pipeline.cache_store.set_last_node(user_id, {
            "latitude": login_event.latitude,
            "longitude": login_event.longitude,
            "timestamp": login_event.timestamp,
            "device_hash": login_event.device_hash
        })
        # Check and trigger retraining / mini-model update
        check_and_trigger_retrain(
            user_id=user_id,
            new_login_lat=login_event.latitude,
            new_login_lon=login_event.longitude,
            new_login_ts=login_event.timestamp,
            new_login_device=login_event.device_hash,
            db_store=db_store,
            profile_store=profile_store
        )
        
    # Returns score normalized between 0 and 1
    return float(result.risk_score / 100.0)
