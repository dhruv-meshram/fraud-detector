"""Regression tests verifying stability of outputs and preventing scoring/verdict drift."""

import pytest
from fraud_detector import FraudDetector
from fraud_detector.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

def test_fixed_impossible_travel_regression():
    """Asserts that a fixed, known impossible travel event produces the exact same risk outputs."""
    detector = FraudDetector(
        profile_store=InMemoryProfileStore(),
        cache_store=InMemoryCacheStore(),
        db_store=InMemoryDBStore(),
        alert_producer=ConsoleAlertProducer()
    )
    
    user_id = "regression_user"
    # Seed SF verified login at timestamp 1600000000.0
    detector.pipeline.db_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0, "dev", True)
    
    # London 1 hour later (impossible velocity)
    eval_event = {
        "user_id": user_id,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timestamp": 1600003600.0,
        "device_hash": "dev"
    }
    
    result = detector.analyze(eval_event)
    
    # Assert exact outputs matching the stable schema
    assert result.status == "IMPOSSIBLE_VELOCITY"
    assert result.is_fraudulent is True
    assert result.risk_score == 100.0
    assert result.details.velocity_kmh > 8000.0

def test_fixed_cold_start_regression():
    """Asserts that a cold-start event always resolves to low-risk with zero score."""
    detector = FraudDetector(
        profile_store=InMemoryProfileStore(),
        cache_store=InMemoryCacheStore(),
        db_store=InMemoryDBStore(),
        alert_producer=ConsoleAlertProducer()
    )
    
    eval_event = {
        "user_id": "new_regression_user",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000000.0,
        "device_hash": "dev"
    }
    
    result = detector.analyze(eval_event)
    
    assert result.status == "COLD_START_BYPASS"
    assert result.is_fraudulent is False
    assert result.risk_score == 0.0
