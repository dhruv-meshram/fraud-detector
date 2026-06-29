"""Unit tests for spatial and graph mathematical logic."""

import pytest
from fraud_detector.algorithms.spatial.haversine import haversine_distance
from fraud_detector.algorithms.spatial.bounding_box import in_bounding_box

def test_haversine():
    # Distance from NY to SF (~4130 km)
    ny_lat, ny_lon = 40.7128, -74.0060
    sf_lat, sf_lon = 37.7749, -122.4194
    dist = haversine_distance(ny_lat, ny_lon, sf_lat, sf_lon)
    assert abs(dist - 4130.0) < 15.0

    # Distance to self should be 0
    assert haversine_distance(ny_lat, ny_lon, ny_lat, ny_lon) == 0.0
    
    # Dateline crossing (179 to -179 should be ~222.4 km)
    assert abs(haversine_distance(0.0, 179.0, 0.0, -179.0) - 222.4) < 1.0

def test_bounding_box_filter():
    center_lat, center_lon = 40.7128, -74.0060
    radius = 50.0 # 50 km
    
    # 1. Point within bounding box and radius (approx 10km away)
    near_lat, near_lon = 40.8, -74.0
    assert in_bounding_box(near_lat, near_lon, center_lat, center_lon, radius) is True
    
    # 2. Point far outside bounding box (SF is > 4000km away)
    far_lat, far_lon = 37.7749, -122.4194
    assert in_bounding_box(far_lat, far_lon, center_lat, center_lon, radius) is False

def test_bounding_box_meridian_wrap():
    # Near 180/-180 meridian (dateline)
    center_lat, center_lon = 0.0, 179.99
    radius = 50.0 # ~0.45 degrees at equator
    
    # Across the line (negative longitude)
    test_lat, test_lon = 0.0, -179.99
    # Distance between 179.99 and -179.99 is 0.02 degrees (~2.2 km), so it should be in the box
    assert in_bounding_box(test_lat, test_lon, center_lat, center_lon, radius) is True

def test_bounding_box_polar_safety():
    # Near North Pole (89.9 degrees Lat)
    center_lat, center_lon = 89.9, 0.0
    radius = 50.0
    
    # At high latitude, the bounding box check should return True (pole safety)
    # to avoid division by zero or extreme scaling issues
    assert in_bounding_box(89.9, 179.0, center_lat, center_lon, radius) is True

def test_velocity_validator():
    from fraud_detector.algorithms.graph.velocity_validator import validate_velocity
    
    # 1. Physically possible speed (NY to Philadelphia, ~150km, in 2 hours = 75 km/h)
    login1 = {"latitude": 40.7128, "longitude": -74.0060, "timestamp": 1600000000.0}
    login2 = {"latitude": 39.9526, "longitude": -75.1652, "timestamp": 1600007200.0} # + 2 hours
    assert validate_velocity(login1, login2) is True
    
    # 2. Physically impossible speed (NY to SF, ~4130km, in 1 hour = 4130 km/h)
    login3 = {"latitude": 37.7749, "longitude": -122.4194, "timestamp": 1600003600.0} # + 1 hour from login1
    assert validate_velocity(login1, login3) is False
    
    # 3. Sub-second rapid logins
    # Same location in < 1 second -> Allowed
    login4 = {"latitude": 40.7128, "longitude": -74.0060, "timestamp": 1600000000.5}
    assert validate_velocity(login1, login4) is True
    
    # Different location in < 1 second -> Impossible
    login5 = {"latitude": 39.9526, "longitude": -75.1652, "timestamp": 1600000000.5}
    assert validate_velocity(login1, login5) is False

def test_multi_factor_score():
    from fraud_detector.algorithms.ranking.multi_factor_score import calculate_risk_score
    
    # 1. Minimum risk scenario (low speed, close to centroid, matching device)
    assert calculate_risk_score(50.0, 5.0, False) == 0.0
    
    # 2. Maximum risk scenario (high speed, far from centroid, mismatched device)
    assert calculate_risk_score(950.0, 200.0, True) == 1.0
    
    # 3. Borderline scenario calculation:
    # velocity = 500 km/h -> (500 - 100) / 800 = 0.5 factor (weight 0.5 -> 0.25 contribution)
    # distance = 80 km -> (80 - 10) / 140 = 0.5 factor (weight 0.3 -> 0.15 contribution)
    # device_hash_mismatch = True -> 1.0 factor (weight 0.2 -> 0.20 contribution)
    # Expected score: 0.25 + 0.15 + 0.20 = 0.60
    assert calculate_risk_score(500.0, 80.0, True) == 0.60

def test_ball_tree():
    import numpy as np
    from fraud_detector.algorithms.trees.ball_tree import BallTree
    
    # Coordinates of SF, NY, Philadelphia
    coords = np.array([
        [37.7749, -122.4194],  # SF
        [40.7128, -74.0060],   # NY
        [39.9526, -75.1652]    # Philly (~150km from NY)
    ])
    
    tree = BallTree(coords)
    
    # 1. Query radius of 200km around NY -> Should find NY (index 1) and Philly (index 2)
    indices = tree.query_radius([40.7128, -74.0060], radius_km=200.0)
    assert set(indices) == {1, 2}
    
    # 2. Query nearest neighbor of SF -> Should find SF (index 0) at distance ~0
    dists, nearest_indices = tree.query([37.7749, -122.4194], k=1)
    assert nearest_indices == 0
    assert dists < 1e-4
