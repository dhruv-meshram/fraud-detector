"""Smoke tests verifying public module import and minimal detector operations."""

import pytest
from fraud_detector import FraudDetector
from fraud_detector.models import LoginEvent, FraudResult
from fraud_detector.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

def test_imports_and_instantiation():
    """Asserts that all public SDK modules, classes, and models can be imported and instantiated."""
    detector = FraudDetector()
    assert isinstance(detector, FraudDetector)
    assert detector.pipeline is not None

def test_basic_headless_evaluation():
    """Performs a basic headless evaluation to ensure the pipeline runs without errors."""
    detector = FraudDetector(
        profile_store=InMemoryProfileStore(),
        cache_store=InMemoryCacheStore(),
        db_store=InMemoryDBStore(),
        alert_producer=ConsoleAlertProducer()
    )
    
    event_data = {
        "user_id": "smoke_user",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000000.0,
        "device_hash": "dev_smoke"
    }
    
    result = detector.analyze(event_data)
    assert isinstance(result, FraudResult)
    assert hasattr(result, "risk_score")
    assert hasattr(result, "is_fraudulent")
    assert hasattr(result, "reasons")
    assert hasattr(result, "status")
    assert hasattr(result, "details")
    
    # Since it's the first event for this user, it should fall back to cold start bypass
    assert result.status == "COLD_START_BYPASS"
    assert result.risk_score == 0.0
