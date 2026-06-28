"""Common mock payloads and test fixtures for external library validation."""

# Standard valid event payload
VALID_EVENT = {
    "user_id": "val_user_123",
    "latitude": 37.7749,
    "longitude": -122.4194,
    "timestamp": 1600000000.0,
    "device_hash": "dev_val_1"
}

# Payload representing London coordinate
LONDON_EVENT = {
    "user_id": "val_user_123",
    "latitude": 51.5074,
    "longitude": -0.1278,
    "timestamp": 1600003600.0,  # 1 hour later
    "device_hash": "dev_val_1"
}

# Payload with normal nearby coordinate (Oakland, 15km away)
OAKLAND_EVENT = {
    "user_id": "val_user_123",
    "latitude": 37.8044,
    "longitude": -122.2712,
    "timestamp": 1600001000.0,  # 1000 seconds later
    "device_hash": "dev_val_1"
}
