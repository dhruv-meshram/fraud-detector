"""Dataset generation tests generating synthetic login simulation files."""

import os
import json
import csv
from pathlib import Path
import pytest

DATASETS_DIR = Path("/home/dhruv/Documents/fraud-detector/lib-tests/datasets")

def test_generate_synthetic_datasets():
    """Generates thousands of synthetic login records simulating different behaviors."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = DATASETS_DIR / "generated_events.csv"
    json_path = DATASETS_DIR / "generated_events.json"
    
    events = []
    
    # 1. Normal users (1,000 events) - SF cluster area
    for i in range(1000):
        events.append({
            "user_id": f"normal_user_{i % 10}",
            "latitude": 37.7749 + (i % 5) * 0.01,
            "longitude": -122.4194 - (i % 5) * 0.01,
            "timestamp": 1600000000.0 + i * 60,
            "device_hash": f"dev_normal_{i % 10}",
            "label": "normal"
        })
        
    # 2. Legitimate travelers (500 events) - Travel over realistic times (SF -> London 12h)
    for i in range(500):
        events.append({
            "user_id": f"traveler_{i % 5}",
            "latitude": 37.7749 if i % 2 == 0 else 51.5074,
            "longitude": -122.4194 if i % 2 == 0 else -0.1278,
            "timestamp": 1600000000.0 + i * 43200,  # 12 hours apart
            "device_hash": f"dev_travel_{i % 5}",
            "label": "traveler"
        })
        
    # 3. Fraudsters/Impossible travel (500 events) - Mumbai -> London in 5 minutes
    for i in range(500):
        events.append({
            "user_id": f"fraudster_{i % 5}",
            "latitude": 19.0760 if i % 2 == 0 else 51.5074,
            "longitude": 72.8777 if i % 2 == 0 else -0.1278,
            "timestamp": 1600000000.0 + i * 300,  # 5 minutes apart
            "device_hash": f"dev_fraud_{i % 5}",
            "label": "impossible_travel"
        })
        
    # 4. Account Takeovers (200 events) - Sudden device changes with outliers
    for i in range(200):
        events.append({
            "user_id": f"ato_user_{i % 2}",
            "latitude": 37.7749 if i % 2 == 0 else 48.8566, # SF to Paris
            "longitude": -122.4194 if i % 2 == 0 else 2.3522,
            "timestamp": 1600000000.0 + i * 1800,  # 30 mins apart
            "device_hash": "dev_legit" if i % 2 == 0 else "dev_hacker",
            "label": "account_takeover"
        })

    # Save to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "latitude", "longitude", "timestamp", "device_hash", "label"])
        writer.writeheader()
        writer.writerows(events)
        
    # Save to JSON
    with open(json_path, "w") as f:
        json.dump(events, f, indent=2)
        
    assert csv_path.exists()
    assert json_path.exists()
    assert csv_path.stat().st_size > 0
    assert json_path.stat().st_size > 0
    
    # Verify number of lines/records
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        assert len(rows) == 2200
