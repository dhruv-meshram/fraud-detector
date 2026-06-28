"""API Contract tests verifying input constraints, validation rules, and schema stability."""

import pytest
from pydantic import ValidationError
from prj.models import LoginEvent, FraudResult
from prj import FraudDetector
from prj.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

def test_login_event_validation_constraints():
    """Asserts that LoginEvent strictly validates field constraints and types."""
    # 1. Missing required field
    with pytest.raises(ValidationError):
        LoginEvent(
            latitude=37.7749,
            longitude=-122.4194,
            timestamp=1600000000.0,
            device_hash="dev_test"
        )  # user_id is missing

    # 2. Malformed coordinates (invalid string representation)
    with pytest.raises(ValidationError):
        LoginEvent(
            user_id="user_123",
            latitude="not-a-number",
            longitude=-122.4194,
            timestamp=1600000000.0,
            device_hash="dev_test"
        )

    # 3. Valid event creation
    event = LoginEvent(
        user_id="user_123",
        latitude=37.7749,
        longitude=-122.4194,
        timestamp=1600000000.0,
        device_hash="dev_test"
    )
    assert event.user_id == "user_123"
    assert event.latitude == 37.7749

def test_login_event_extra_fields():
    """Asserts that extra payload parameters do not break instantiation (ignored or handled gracefully)."""
    payload = {
        "user_id": "user_123",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000000.0,
        "device_hash": "dev_test",
        "extra_parameter_key": "some-value"
    }
    event = LoginEvent(**payload)
    assert event.user_id == "user_123"

def test_detector_accepts_raw_dict():
    """Asserts that the FraudDetector can accept raw dictionary payloads and validate them internally."""
    detector = FraudDetector(
        profile_store=InMemoryProfileStore(),
        cache_store=InMemoryCacheStore(),
        db_store=InMemoryDBStore(),
        alert_producer=ConsoleAlertProducer()
    )
    
    # Passing an invalid dict should raise ValidationError
    invalid_payload = {
        "latitude": 37.7749
    }
    with pytest.raises(ValidationError):
        detector.analyze(invalid_payload)

def test_serialization_stability():
    """Asserts that response model schema fields serialize and deserialize correctly."""
    result_data = {
        "risk_score": 85.5,
        "is_fraudulent": True,
        "reasons": ["Impossible travel"],
        "status": "IMPOSSIBLE_VELOCITY",
        "details": {
            "velocity_kmh": 1200.0,
            "distance_km": 500.0,
            "device_mismatch": False
        }
    }
    
    res = FraudResult(**result_data)
    serialized = res.model_dump()
    assert serialized["risk_score"] == 85.5
    assert serialized["is_fraudulent"] is True
    assert serialized["details"]["velocity_kmh"] == 1200.0
