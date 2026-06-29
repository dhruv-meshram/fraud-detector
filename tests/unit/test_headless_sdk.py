"""Unit tests demonstrating and verifying headless SDK usage using mock storage adapters."""

import pytest
from fraud_detector import FraudDetector
from fraud_detector.adapters import (
    InMemoryProfileStore,
    InMemoryCacheStore,
    InMemoryDBStore,
    ConsoleAlertProducer
)

def test_headless_sdk_impossible_travel():
    """Verifies that the SDK can run completely headless with mock adapters."""
    # 1. Instantiate the mock/in-memory adapters
    profile_store = InMemoryProfileStore()
    cache_store = InMemoryCacheStore()
    db_store = InMemoryDBStore()
    alert_producer = ConsoleAlertProducer()

    # 2. Instantiate FraudDetector with the mock adapters
    detector = FraudDetector(
        profile_store=profile_store,
        cache_store=cache_store,
        db_store=db_store,
        alert_producer=alert_producer
    )

    user_id = "headless_test_user"

    # 3. Seed a verified starting coordinate in the mock DB store
    db_store.record_login(
        user_id=user_id,
        lat=37.7749,
        lon=-122.4194,
        ts=1600000000.0,
        device_hash="dev_test_headless",
        is_verified=True
    )

    # 4. Analyze an event representing impossible travel (London, 1 hour later)
    eval_event = {
        "user_id": user_id,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timestamp": 1600003600.0,
        "device_hash": "dev_test_headless"
    }

    result = detector.analyze(eval_event)

    # 5. Assert the results match expectations
    assert result.status == "IMPOSSIBLE_VELOCITY"
    assert result.risk_score == 100.0
    assert "Impossible velocity" in result.reasons[0]
    assert result.details.velocity_kmh > 5000.0

def test_headless_sdk_cold_start():
    """Verifies cold start behavior for a new user in headless mode."""
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

    user_id = "new_headless_user"

    # Evaluate the first login of a new user (no prior logins)
    eval_event = {
        "user_id": user_id,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000000.0,
        "device_hash": "dev_new"
    }

    result = detector.analyze(eval_event)

    # Since the user has no history and profile is not loaded, should hit COLD_START_BYPASS
    assert result.status == "COLD_START_BYPASS"
    assert result.risk_score == 0.0
    assert len(result.reasons) == 1
    assert "Cold Start" in result.reasons[0]
