"""Unit tests for API routing and input validation."""

import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    """Asserts health check behaves correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_evaluate_risk_cold_start():
    """Asserts that a brand new user defaults to COLD_START_BYPASS low risk."""
    payload = {
        "user_id": "api_test_cold_user",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "device_hash": "dev_test_iphone"
    }
    response = client.post("/evaluate_risk", json=payload)
    assert response.status_code == 200
    
    res_data = response.json()
    assert res_data["verdict"] == "LOW_RISK"
    assert res_data["status"] == "COLD_START_BYPASS"

def test_evaluate_risk_impossible_travel():
    """Asserts that impossible velocity triggers immediate HIGH_RISK."""
    user_id = "api_test_impossible_user"
    
    # 1. Record a verified seed login coordinate (SF at t=1600000000)
    verify_payload = {
        "user_id": user_id,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000000.0,
        "device_hash": "dev_test_iphone",
        "is_verified": True
    }
    verify_res = client.post("/verify_anomaly", json=verify_payload)
    assert verify_res.status_code == 200
    
    # 2. Evaluate login from London (+8000km away) only 1 hour later (t=1600003600)
    eval_payload = {
        "user_id": user_id,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timestamp": 1600003600.0,
        "device_hash": "dev_test_iphone"
    }
    eval_res = client.post("/evaluate_risk", json=eval_payload)
    assert eval_res.status_code == 200
    
    res_data = eval_res.json()
    assert res_data["verdict"] == "HIGH_RISK"
    assert res_data["status"] == "IMPOSSIBLE_VELOCITY"
    assert res_data["details"]["velocity_kmh"] > 5000.0

def test_verify_anomaly_endpoint():
    """Asserts that verifying an anomaly records the outcome successfully."""
    payload = {
        "user_id": "api_test_verify_user",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "timestamp": 1600000000.0,
        "device_hash": "dev_test_iphone",
        "is_verified": True
    }
    response = client.post("/verify_anomaly", json=payload)
    assert response.status_code == 200
    
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["recorded_verified"] is True
