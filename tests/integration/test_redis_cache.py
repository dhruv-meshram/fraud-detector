"""Integration tests checking atomic Redis profile state swaps."""

from unittest.mock import patch
import json
import pytest
# pyrefly: ignore [missing-import]
import fakeredis

from fraud_detector.ml.models.registry import ModelRegistry
from fraud_detector.adapters.cache import RedisCacheStore

@pytest.fixture
def mock_redis():
    """Provides a fresh fake Redis client for each test."""
    fake_client = fakeredis.FakeRedis()
    # Test connection ping simulation
    fake_client.ping = lambda: True
    return fake_client

def test_redis_cache_store_caching(mock_redis):
    """Verifies that RedisCacheStore correctly stores and retrieves last login nodes."""
    with patch('redis.from_url', return_value=mock_redis):
        client = RedisCacheStore()
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
            "last_updated": "2026-07-02T09:00:00Z",
            "total_logins_trained": 10,
            "clusters": [{
                "cluster_id": 0,
                "centroid_lat": 37.7749,
                "centroid_lon": -122.4194,
                "dynamic_radius_km": 0.0,
                "num_points": 1
            }]
        }
        
        # Register initial profile
        assert registry.register_profile(user_id, initial_profile) is True
        
        # Verify cached value
        cached_initial = json.loads(mock_redis.get(f"user:profile:{user_id}"))
        assert cached_initial == initial_profile
        
        # Overwrite with updated spatial centroids (e.g. after retraining)
        updated_profile = {
            "user_id": user_id,
            "last_updated": "2026-07-02T09:05:00Z",
            "total_logins_trained": 11,
            "clusters": [
                {
                    "cluster_id": 0,
                    "centroid_lat": 37.7749,
                    "centroid_lon": -122.4194,
                    "dynamic_radius_km": 0.0,
                    "num_points": 1
                },
                {
                    "cluster_id": 1,
                    "centroid_lat": 34.0522,
                    "centroid_lon": -118.2437,
                    "dynamic_radius_km": 0.0,
                    "num_points": 1
                }
            ]
        }
        assert registry.register_profile(user_id, updated_profile) is True
        
        # Retrieve and verify it returned the new profile instantly (zero downtime)
        cached_updated = json.loads(mock_redis.get(f"user:profile:{user_id}"))
        assert cached_updated == updated_profile
        assert registry.get_profile(user_id) == updated_profile

def test_cache_miss_hydration(mock_redis):
    """Verifies cache-miss fallback reads from database and hydrates Redis cache."""
    with patch('redis.from_url', return_value=mock_redis):
        registry = ModelRegistry()
        
        user_id = "cache_miss_user"
        profile_data = {
            "user_id": user_id,
            "last_updated": "2026-07-02T09:00:00Z",
            "total_logins_trained": 10,
            "clusters": [{
                "cluster_id": 0,
                "centroid_lat": 51.5074,
                "centroid_lon": -0.1278,
                "dynamic_radius_km": 0.0,
                "num_points": 1
            }]
        }
        
        # Save directly to database to simulate cache eviction or server restart
        registry.profile_store.save_profile(user_id, profile_data)
            
        # Ensure Redis cache has nothing
        assert mock_redis.get(f"user:profile:{user_id}") is None
        
        # Get profile (should trigger fallback and hydration)
        loaded = registry.get_profile(user_id)
        assert loaded == profile_data
        
        # Verify that Redis is now hydrated
        cached_val = json.loads(mock_redis.get(f"user:profile:{user_id}"))
        assert cached_val == profile_data
