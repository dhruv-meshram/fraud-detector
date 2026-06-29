"""Integration tests checking atomic Redis profile state swaps."""

from unittest.mock import patch
import json
import pytest
# pyrefly: ignore [missing-import]
import fakeredis

from fraud_detector.ml.models.registry import ModelRegistry
from app.services import RedisClient

@pytest.fixture
def mock_redis():
    """Provides a fresh fake Redis client for each test."""
    fake_client = fakeredis.FakeRedis()
    # Test connection ping simulation
    fake_client.ping = lambda: True
    return fake_client

def test_redis_client_caching_and_fallback(mock_redis):
    """Verifies that RedisClient correctly stores and retrieves last login nodes."""
    with patch('redis.from_url', return_value=mock_redis):
        client = RedisClient()
        assert client.client is mock_redis
        
        user_id = "integration_test_user"
        node = {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "timestamp": 1600000000.0,
            "device_hash": "dev_test"
        }
        
        # Set node
        success = client.set_last_node(user_id, node)
        assert success is True
        
        # Get node
        fetched = client.get_last_node(user_id)
        assert fetched == node

def test_atomic_profile_overwrite(mock_redis):
    """Verifies that registration performs atomic profile swaps in Redis cache."""
    with patch('redis.from_url', return_value=mock_redis):
        registry = ModelRegistry()
        assert registry.redis_client is mock_redis
        
        user_id = "test_user_atomic_swap"
        initial_profile = {
            "user_id": user_id,
            "clusters": [{"centroid_lat": 37.7749, "centroid_lon": -122.4194}]
        }
        
        # Register initial profile
        assert registry.register_profile(user_id, initial_profile) is True
        
        # Verify cached value
        cached_initial = json.loads(mock_redis.get(f"user:profile:{user_id}"))
        assert cached_initial == initial_profile
        
        # Overwrite with updated spatial centroids (e.g. after retraining)
        updated_profile = {
            "user_id": user_id,
            "clusters": [
                {"centroid_lat": 37.7749, "centroid_lon": -122.4194},
                {"centroid_lat": 34.0522, "centroid_lon": -118.2437}
            ]
        }
        assert registry.register_profile(user_id, updated_profile) is True
        
        # Retrieve and verify it returned the new profile instantly (zero downtime)
        cached_updated = json.loads(mock_redis.get(f"user:profile:{user_id}"))
        assert cached_updated == updated_profile
        assert registry.get_profile(user_id) == updated_profile

def test_cache_miss_hydration(mock_redis, tmp_path):
    """Verifies cache-miss fallback reads from disk and hydrates Redis cache."""
    with patch('redis.from_url', return_value=mock_redis):
        registry = ModelRegistry(profiles_dir=tmp_path)
        
        user_id = "cache_miss_user"
        profile_data = {
            "user_id": user_id,
            "clusters": [{"centroid_lat": 51.5074, "centroid_lon": -0.1278}]
        }
        
        # Save directly to disk to simulate cache eviction or server restart
        local_file = tmp_path / f"{user_id}.json"
        with open(local_file, "w") as f:
            f.write(json.dumps(profile_data))
            
        # Ensure Redis cache has nothing
        assert mock_redis.get(f"user:profile:{user_id}") is None
        
        # Get profile (should trigger fallback and hydration)
        loaded = registry.get_profile(user_id)
        assert loaded == profile_data
        
        # Verify that Redis is now hydrated
        cached_val = json.loads(mock_redis.get(f"user:profile:{user_id}"))
        assert cached_val == profile_data
