"""Integration tests verifying adapter bindings, configuration, and custom stores."""

import pytest
from pathlib import Path
from prj import FraudDetector
from prj.adapters import (
    InMemoryProfileStore,
    InMemoryCacheStore,
    InMemoryDBStore,
    ConsoleAlertProducer,
    FileProfileStore
)

def test_custom_adapter_binding_via_constructor():
    """Asserts that custom storage adapters are correctly assigned to the detector's pipeline."""
    p_store = InMemoryProfileStore()
    c_store = InMemoryCacheStore()
    d_store = InMemoryDBStore()
    a_producer = ConsoleAlertProducer()
    
    detector = FraudDetector(
        profile_store=p_store,
        cache_store=c_store,
        db_store=d_store,
        alert_producer=a_producer
    )
    
    assert detector.pipeline.profile_store is p_store
    assert detector.pipeline.cache_store is c_store
    assert detector.pipeline.db_store is d_store
    assert detector.pipeline.alert_producer is a_producer

def test_file_profile_store_lifecycle(tmp_path):
    """Verifies that the FileProfileStore correctly saves and loads spatial profiles from the disk."""
    store = FileProfileStore(profiles_dir=tmp_path)
    user_id = "integration_file_user"
    profile_data = {
        "user_id": user_id,
        "clusters": [{"centroid_lat": 37.7749, "centroid_lon": -122.4194, "dynamic_radius_km": 15.0}]
    }
    
    # Get non-existent
    assert store.get_profile(user_id) is None
    
    # Save profile
    save_ok = store.save_profile(user_id, profile_data)
    assert save_ok is True
    
    # Load profile and verify
    loaded = store.get_profile(user_id)
    assert loaded == profile_data
