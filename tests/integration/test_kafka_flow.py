"""Integration tests validating Kafka message publishing from FastAPI to Celery."""

import json
from pathlib import Path
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_kafka_anomaly_event_publishing():
    """Asserts that evaluating an impossible speed login dispatches a Kafka event."""
    log_path = Path("data/kafka_events.log")
    
    # 1. Clear previous log entries to avoid pollution
    if log_path.exists():
        log_path.unlink()
        
    user_id = "kafka_test_user"
    
    # 2. Seed a verified starting coordinate
    seed_payload = {
        "user_id": user_id,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000000.0,
        "device_hash": "dev_seed",
        "is_verified": True
    }
    verify_res = client.post("/verify_anomaly", json=seed_payload)
    assert verify_res.status_code == 200
    
    # 3. Trigger impossible travel event 1 hour later (+8000km away)
    eval_payload = {
        "user_id": user_id,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timestamp": 1600003600.0,
        "device_hash": "dev_seed"
    }
    eval_res = client.post("/evaluate_risk", json=eval_payload)
    assert eval_res.status_code == 200
    assert eval_res.json()["verdict"] == "HIGH_RISK"
    
    # 4. Verify that the Kafka log file was written with the anomaly event
    assert log_path.exists()
    
    with open(log_path, "r") as f:
        events = [json.loads(line.strip()) for line in f if line.strip()]
        
    # Find our user event
    user_events = [e for e in events if e.get("key") == user_id]
    assert len(user_events) >= 1
    
    anomaly_event = user_events[0]
    assert anomaly_event["topic"] == "shieldflow.anomalies"
    assert anomaly_event["payload"]["user_id"] == user_id
    assert "Impossible velocity" in anomaly_event["payload"]["reason"]
    assert anomaly_event["payload"]["verdict"] == "HIGH_RISK"
