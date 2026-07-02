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

def test_check_fraud_end_to_end():
    """Verifies the new check_fraud API function using an in-memory SQLite database."""
    from sqlalchemy import create_engine
    from fraud_detector import check_fraud
    import warnings
    
    # Create a fresh in-memory SQLite database for testing
    engine = create_engine("sqlite://")
    
    # Verify that check_fraud creates tables and runs correctly
    user_id = "sql_test_user"
    device_hash = "sql_device"
    
    # 1. Run first check (Cold Start Bypass) by passing coordinates directly
    score1 = check_fraud(
        db_conn=engine,
        user_id=user_id,
        device_fingerprint=device_hash,
        ip_address="192.168.1.1",
        latitude=37.7749,
        longitude=-122.4194,
        event={"timestamp": 1600000000.0}
    )
    assert score1 == 0.0
    
    # Verify table existence and check warn trigger on subsequent calls
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        score2 = check_fraud(
            db_conn=engine,
            user_id=user_id,
            device_fingerprint=device_hash,
            ip_address="192.168.1.1",
            latitude=51.5074,
            longitude=-0.1278,
            event={"timestamp": 1600003600.0}
        )
        assert len(w) >= 1
        assert "SDK tables already exist" in str(w[-1].message)
        
    # Since the first login was verified, second login will check velocity (impossible travel)
    # Velocity between SF and London is impossible, so score should be 1.0 (normalized from 100)
    assert score2 == 1.0

def test_check_fraud_coordinate_validation():
    """Verifies coordinate validation for check_fraud (bounds and type checks)."""
    from sqlalchemy import create_engine
    from fraud_detector import check_fraud
    
    engine = create_engine("sqlite://")
    user_id = "validation_user"
    device_hash = "validation_device"
    
    # 1. Missing coordinates should raise ValueError
    with pytest.raises(ValueError, match="latitude is required"):
        check_fraud(db_conn=engine, user_id=user_id, device_fingerprint=device_hash)
        
    with pytest.raises(ValueError, match="longitude is required"):
        check_fraud(db_conn=engine, user_id=user_id, device_fingerprint=device_hash, latitude=45.0)
        
    # 2. Invalid latitude bounds
    with pytest.raises(ValueError, match="latitude must be between -90.0 and 90.0"):
        check_fraud(db_conn=engine, user_id=user_id, device_fingerprint=device_hash, latitude=-90.1, longitude=0.0)
        
    with pytest.raises(ValueError, match="latitude must be between -90.0 and 90.0"):
        check_fraud(db_conn=engine, user_id=user_id, device_fingerprint=device_hash, latitude=90.1, longitude=0.0)
        
    # 3. Invalid longitude bounds
    with pytest.raises(ValueError, match="longitude must be between -180.0 and 180.0"):
        check_fraud(db_conn=engine, user_id=user_id, device_fingerprint=device_hash, latitude=0.0, longitude=-180.1)
        
    with pytest.raises(ValueError, match="longitude must be between -180.0 and 180.0"):
        check_fraud(db_conn=engine, user_id=user_id, device_fingerprint=device_hash, latitude=0.0, longitude=180.1)
        
    # 4. Invalid types
    with pytest.raises(ValueError, match="latitude must be a numeric value"):
        check_fraud(db_conn=engine, user_id=user_id, device_fingerprint=device_hash, latitude="invalid", longitude=0.0)
        
    with pytest.raises(ValueError, match="longitude must be a numeric value"):
        check_fraud(db_conn=engine, user_id=user_id, device_fingerprint=device_hash, latitude=0.0, longitude="invalid")
