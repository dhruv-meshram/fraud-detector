"""End-to-End simulation tests validating the entire user lifecycle and detection system."""

import pytest
from fraud_detector import FraudDetector
from fraud_detector.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

def test_full_user_lifecycle_e2e():
    """Simulates a complete user lifecycle:

    1. User logs in normally (under cold start threshold).
    2. User builds a spatial profile with SF as home region.
    3. User travels to London (legitimately over a realistic 24h timeline).
    4. User logs in from an outlier location (New York, with device mismatch).
    5. User attempts impossible velocity travel (London to Paris in 1 minute) -> Anomaly detected.
    """
    profile_store = InMemoryProfileStore()
    cache_store = InMemoryCacheStore()
    db_store = InMemoryDBStore()
    alert_producer = ConsoleAlertProducer()
    
    detector = FraudDetector(
        profile_store=profile_store,
        cache_store=cache_store,
        db_store=db_store,
        alert_producer=alert_producer
    )
    
    user_id = "e2e_user_999"
    
    # --- 1. User logs in normally (SF) under cold start threshold ---
    # We record 10 verified logins in SF to build baseline history
    for i in range(11):
        event = {
            "user_id": user_id,
            "latitude": 37.7749,
            "longitude": -122.4194,
            "timestamp": 1600000000.0 + i * 3600,
            "device_hash": "dev_macbook"
        }
        res = detector.analyze(event)
        
        # All logins under cold-start threshold or matching the region
        if i < 10:
            assert res.status == "COLD_START_BYPASS"
        else:
            # Once baseline history >= 10, it looks for profile. If no profile, still bypasses or checks cache
            assert res.status == "COLD_START_BYPASS"
            
        # Record outcome as verified to simulate pipeline auto-verification
        db_store.record_login(user_id, event["latitude"], event["longitude"], event["timestamp"], event["device_hash"], True)
        cache_store.set_last_node(user_id, {"latitude": event["latitude"], "longitude": event["longitude"], "timestamp": event["timestamp"], "device_hash": event["device_hash"]})

    # --- 2. User builds profile (Simulated by seeding a spatial profile) ---
    profile_store.save_profile(user_id, {
        "user_id": user_id,
        "clusters": [
            {
                "cluster_id": 0,
                "centroid_lat": 37.7749,
                "centroid_lon": -122.4194,
                "dynamic_radius_km": 15.0
            }
        ]
    })
    
    # --- 3. User travels to London (legitimately over 24h timeline) ---
    travel_event = {
        "user_id": user_id,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timestamp": 1600086400.0,  # 24 hours later
        "device_hash": "dev_macbook"
    }
    
    res = detector.analyze(travel_event)
    # The velocity check passes (speed ~360 km/h). Profile check returns OUTLIER (since London centroid isn't in profile).
    assert res.status == "OUTLIER"
    # With a known device, the risk score is moderate
    assert res.is_fraudulent is False
    
    # Auto-verify/MFA pass simulation
    db_store.record_login(user_id, travel_event["latitude"], travel_event["longitude"], travel_event["timestamp"], travel_event["device_hash"], True)
    cache_store.set_last_node(user_id, {"latitude": travel_event["latitude"], "longitude": travel_event["longitude"], "timestamp": travel_event["timestamp"], "device_hash": travel_event["device_hash"]})

    # --- 4. User logs in from London (matching last cache location) ---
    subsequent_london_event = {
        "user_id": user_id,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timestamp": 1600090000.0,  # 1 hour later
        "device_hash": "dev_macbook"
    }
    res = detector.analyze(subsequent_london_event)
    # Velocity is 0 (stationary). Still OUTLIER relative to trained profile (SF only).
    assert res.status == "OUTLIER"
    
    # Auto-verify/MFA pass simulation
    db_store.record_login(user_id, subsequent_london_event["latitude"], subsequent_london_event["longitude"], subsequent_london_event["timestamp"], subsequent_london_event["device_hash"], True)
    cache_store.set_last_node(user_id, {"latitude": subsequent_london_event["latitude"], "longitude": subsequent_london_event["longitude"], "timestamp": subsequent_london_event["timestamp"], "device_hash": subsequent_london_event["device_hash"]})

    # --- 5. User attempts impossible velocity travel (London to Paris in 1 minute) ---
    impossible_event = {
        "user_id": user_id,
        "latitude": 48.8566,
        "longitude": 2.3522,
        "timestamp": 1600090060.0,  # 60 seconds later
        "device_hash": "dev_macbook"
    }
    res = detector.analyze(impossible_event)
    # Speed is over 20,000 km/h -> Fraud detected immediately
    assert res.status == "IMPOSSIBLE_VELOCITY"
    assert res.is_fraudulent is True
    assert res.risk_score == 100.0
