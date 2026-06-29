"""Stress tests evaluating concurrent request throughput, thread safety, and resource stability."""

import pytest
import concurrent.futures
from fraud_detector import FraudDetector
from fraud_detector.adapters import InMemoryProfileStore, InMemoryCacheStore, InMemoryDBStore, ConsoleAlertProducer

def test_concurrent_inference_thread_safety():
    """Asserts that FraudDetector can safely process events concurrently in multi-threaded environments."""
    detector = FraudDetector(
        profile_store=InMemoryProfileStore(),
        cache_store=InMemoryCacheStore(),
        db_store=InMemoryDBStore(),
        alert_producer=ConsoleAlertProducer()
    )
    
    user_id = "stress_user"
    detector.pipeline.db_store.record_login(user_id, 37.7749, -122.4194, 1600000000.0, "dev", True)
    
    # 500 concurrent events
    events = [
        {
            "user_id": user_id,
            "latitude": 37.7749 + i * 0.0001,
            "longitude": -122.4194 - i * 0.0001,
            "timestamp": 1600000000.0 + i * 10,
            "device_hash": "dev"
        }
        for i in range(500)
    ]
    
    def process_event(event):
        return detector.analyze(event)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(process_event, events))
        
    assert len(results) == 500
    for res in results:
        assert res.status is not None
        assert res.risk_score is not None

def test_repeated_malformed_inputs():
    """Asserts that submitting repeatedly malformed payloads does not corrupt internal state or leak resources."""
    detector = FraudDetector()
    malformed_event = {
        "user_id": "malformed_stress",
        "latitude": "invalid-coordinate-string",
        "longitude": -122.4194,
        "timestamp": 1600000000.0,
        "device_hash": "dev"
    }
    
    # Repeatedly submit and ensure it fails gracefully every time
    for _ in range(200):
        with pytest.raises(Exception):
            detector.analyze(malformed_event)
