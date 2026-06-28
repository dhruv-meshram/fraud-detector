"""ML Inference tests validating user spatial profile matching and classification."""

import pytest
import time
from prj import FraudDetector
from prj.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

@pytest.fixture
def ml_detector():
    """Returns a detector seeded with a multi-cluster spatial profile for ml_user."""
    profile_store = InMemoryProfileStore()
    cache_store = InMemoryCacheStore()
    db_store = InMemoryDBStore()
    alert_producer = ConsoleAlertProducer()
    
    # Seed profile with two clusters: SF (radius 15km) and London (radius 10km)
    user_id = "ml_user"
    profile_store.save_profile(user_id, {
        "user_id": user_id,
        "clusters": [
            {
                "cluster_id": 0,
                "centroid_lat": 37.7749,
                "centroid_lon": -122.4194,
                "dynamic_radius_km": 15.0
            },
            {
                "cluster_id": 1,
                "centroid_lat": 51.5074,
                "centroid_lon": -0.1278,
                "dynamic_radius_km": 10.0
            }
        ]
    })
    
    # Seed database with enough historical logins to bypass cold start (>10 logins)
    for i in range(15):
        db_store.record_login(
            user_id=user_id,
            lat=37.7749,
            lon=-122.4194,
            ts=1600000000.0 + i * 3600,
            device_hash="dev_known",
            is_verified=True
        )
        
    return FraudDetector(
        profile_store=profile_store,
        cache_store=cache_store,
        db_store=db_store,
        alert_producer=alert_producer
    )

def test_ml_known_zone(ml_detector):
    """Scenario 1 & 2: Login coordinate is directly within the known cluster (radius 15km)."""
    user_id = "ml_user"
    
    # Coordinate in SF, ~5km from centroid
    eval_event = {
        "user_id": user_id,
        "latitude": 37.7400,
        "longitude": -122.4300,
        "timestamp": 1600060000.0,
        "device_hash": "dev_known"
    }
    
    result = ml_detector.analyze(eval_event)
    assert result.status == "KNOWN_ZONE"
    assert result.is_fraudulent is False
    assert result.risk_score == 0.0
    assert result.details.distance_km < 15.0

def test_ml_outlier_zone(ml_detector):
    """Scenario 3 & 4: Login coordinate is outside all clusters (outlier)."""
    user_id = "ml_user"
    
    # Coordinate in New York, which is an outlier. Use a timestamp 24+ hours later to avoid velocity flag.
    eval_event = {
        "user_id": user_id,
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timestamp": 1600150000.0,
        "device_hash": "dev_known"
    }
    
    result = ml_detector.analyze(eval_event)
    assert result.status == "OUTLIER"
    # An outlier with known device should result in medium/high risk based on multi-factor score
    assert result.risk_score > 0.0
    assert "Spatial outlier" in result.reasons[0]

def test_ml_sparse_history_cold_start():
    """Scenario 5: User with sparse history (< 10 logins) defaults to cold start bypass."""
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
    
    user_id = "sparse_user"
    # Seed only 3 logins
    for i in range(3):
        db_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0 + i*100, "dev", True)
        
    # Use a timestamp 24+ hours later to avoid velocity flags
    eval_event = {
        "user_id": user_id,
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timestamp": 1600100000.0,
        "device_hash": "dev"
    }
    
    result = detector.analyze(eval_event)
    assert result.status == "COLD_START_BYPASS"
    assert result.risk_score == 0.0

def test_ml_inference_latency_bounds(ml_detector):
    """Verifies that ML inference execution is fast and completed under tight bounds."""
    user_id = "ml_user"
    eval_event = {
        "user_id": user_id,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600060000.0,
        "device_hash": "dev_known"
    }
    
    start_time = time.perf_counter()
    ml_detector.analyze(eval_event)
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    # Headless local analysis must be well under 10ms
    assert duration_ms < 10.0
