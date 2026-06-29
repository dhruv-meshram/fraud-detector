"""Algorithmic tests verifying rules-based fraud detection via the public API."""

import pytest
from fraud_detector import FraudDetector
from fraud_detector.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

@pytest.fixture
def clean_detector():
    """Returns a fresh FraudDetector running with clean in-memory mock adapters."""
    return FraudDetector(
        profile_store=InMemoryProfileStore(),
        cache_store=InMemoryCacheStore(),
        db_store=InMemoryDBStore(),
        alert_producer=ConsoleAlertProducer()
    )

def test_impossible_travel_mumbai_to_london(clean_detector):
    """Scenario: Mumbai to London in 30 minutes (impossible travel)."""
    user_id = "algo_mumbai_london"
    
    # 1. Seed Mumbai verified login
    clean_detector.pipeline.db_store.record_login(
        user_id=user_id,
        lat=19.0760,
        lon=72.8777,
        ts=1600000000.0,
        device_hash="dev_seed",
        is_verified=True
    )
    
    # 2. Analyze London login 30 minutes later
    eval_event = {
        "user_id": user_id,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timestamp": 1600001800.0,
        "device_hash": "dev_seed"
    }
    
    result = clean_detector.analyze(eval_event)
    assert result.status == "IMPOSSIBLE_VELOCITY"
    assert result.is_fraudulent is True
    assert result.risk_score == 100.0
    assert "Impossible velocity" in result.reasons[0]

def test_normal_legitimate_travel(clean_detector):
    """Scenario: Travel between cities with reasonable timeline."""
    user_id = "algo_normal_travel"
    
    # 1. Seed Mumbai verified login
    clean_detector.pipeline.db_store.record_login(
        user_id=user_id,
        lat=19.0760,
        lon=72.8777,
        ts=1600000000.0,
        device_hash="dev_seed",
        is_verified=True
    )
    
    # 2. Analyze London login 24 hours later (86400 seconds)
    eval_event = {
        "user_id": user_id,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timestamp": 1600086400.0,
        "device_hash": "dev_seed"
    }
    
    result = clean_detector.analyze(eval_event)
    # The velocity should be normal, so it won't hit impossible travel rules.
    # Because there are no profiles loaded, it will default to COLD_START_BYPASS.
    assert result.status == "COLD_START_BYPASS"
    assert result.is_fraudulent is False

def test_stationary_users(clean_detector):
    """Scenario: User logging in from exact same location multiple times."""
    user_id = "algo_stationary"
    
    # Seed
    clean_detector.pipeline.db_store.record_login(
        user_id=user_id,
        lat=37.7749,
        lon=-122.4194,
        ts=1600000000.0,
        device_hash="dev_seed",
        is_verified=True
    )
    
    # Second login
    eval_event = {
        "user_id": user_id,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000600.0,
        "device_hash": "dev_seed"
    }
    
    result = clean_detector.analyze(eval_event)
    assert result.details.velocity_kmh == 0.0
    assert result.status == "COLD_START_BYPASS"

def test_dateline_crossing(clean_detector):
    """Scenario: User crosses the 180-degree longitude dateline (e.g. Fiji to Samoa)."""
    user_id = "algo_dateline"
    
    # Fiji (178.0 longitude)
    clean_detector.pipeline.db_store.record_login(
        user_id=user_id,
        lat=-18.0,
        lon=178.0,
        ts=1600000000.0,
        device_hash="dev_seed",
        is_verified=True
    )
    
    # Samoa (-172.0 longitude) in 5 minutes (300 seconds).
    # Actual distance is about 1000km, which requires speed > 12000 km/h (impossible travel).
    eval_event = {
        "user_id": user_id,
        "latitude": -14.0,
        "longitude": -172.0,
        "timestamp": 1600000300.0,
        "device_hash": "dev_seed"
    }
    
    result = clean_detector.analyze(eval_event)
    assert result.status == "IMPOSSIBLE_VELOCITY"
    assert result.is_fraudulent is True

def test_polar_coordinates(clean_detector):
    """Scenario: User logging in near the north pole."""
    user_id = "algo_polar"
    
    # Near pole, 0 degrees lon
    clean_detector.pipeline.db_store.record_login(
        user_id=user_id,
        lat=89.9,
        lon=0.0,
        ts=1600000000.0,
        device_hash="dev_seed",
        is_verified=True
    )
    
    # Near pole, 180 degrees lon.
    # In polar coordinates, a crossing of 180 lon near the pole is a very short physical distance.
    # Distance is about 22km. Should be fully possible in 30 minutes (1800 seconds).
    eval_event = {
        "user_id": user_id,
        "latitude": 89.9,
        "longitude": 180.0,
        "timestamp": 1600001800.0,
        "device_hash": "dev_seed"
    }
    
    result = clean_detector.analyze(eval_event)
    # The physical distance is small (~22km), so speed should be very low (~44 km/h), i.e., valid.
    assert result.status == "COLD_START_BYPASS"
    assert result.is_fraudulent is False
    assert result.details.velocity_kmh < 100.0
