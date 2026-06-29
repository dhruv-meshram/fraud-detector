"""Unit tests for spatial clustering and geodesic distance computations."""

import math
import numpy as np
# pyrefly: ignore [missing-import]
import pytest
# pyrefly: ignore [missing-import]
import torch

from fraud_detector.ml.features.geo_features import torch_haversine_matrix, haversine_distance_numpy, to_radians, calculate_geographic_centroid

def haversine_cpu_basic(lat1, lon1, lat2, lon2):
    """Basic CPU Haversine calculation for parity checks."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def test_haversine_matrix_parity():
    """Validates that PyTorch pairwise matrix math matches CPU reference calculation."""
    # Define a set of coordinates (NY, SF, London)
    coords_deg = np.array([
        [40.7128, -74.0060],  # NY
        [37.7749, -122.4194], # SF
        [51.5074, -0.1278]    # London
    ])
    
    # Calculate CPU expected pairwise matrix
    n = len(coords_deg)
    expected_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            expected_matrix[i, j] = haversine_cpu_basic(
                coords_deg[i, 0], coords_deg[i, 1],
                coords_deg[j, 0], coords_deg[j, 1]
            )
            
    # Calculate using PyTorch matrix function
    coords_rad = np.radians(coords_deg)
    coords_tensor = torch.tensor(coords_rad, dtype=torch.float32)
    computed_tensor = torch_haversine_matrix(coords_tensor)
    computed_matrix = computed_tensor.numpy()
    
    # Verify parity (within 1e-3 km tolerance due to float32 precision)
    np.testing.assert_allclose(computed_matrix, expected_matrix, rtol=1e-3, atol=1e-2)

def test_geographic_centroid_calculation():
    """Verifies that geographic centroid averaging works correctly."""
    # Test midpoint between NY and SF
    lats = [40.7128, 37.7749]
    lons = [-74.0060, -122.4194]
    
    centroid_lat, centroid_lon = calculate_geographic_centroid(lats, lons)
    
    # Midpoint should be approximately 41.8 degrees latitude and -98.2 degrees longitude
    assert 40.0 <= centroid_lat <= 43.0
    assert -99.0 <= centroid_lon <= -96.0

def test_numpy_haversine_distance():
    """Verifies numpy vectorized haversine function works for single and array inputs."""
    lat1, lon1 = 40.7128, -74.0060 # NY
    lat2, lon2 = 37.7749, -122.4194 # SF
    
    dist_single = haversine_distance_numpy(lat1, lon1, lat2, lon2)
    cpu_ref = haversine_cpu_basic(lat1, lon1, lat2, lon2)
    
    assert abs(dist_single - cpu_ref) < 1e-3
    
    # Test vectorized array inputs
    lats1 = np.array([40.7128, 51.5074])
    lons1 = np.array([-74.0060, -0.1278])
    lats2 = np.array([37.7749, 19.0760])
    lons2 = np.array([-122.4194, 72.8777])
    
    dists = haversine_distance_numpy(lats1, lons1, lats2, lons2)
    assert len(dists) == 2
    assert abs(dists[0] - haversine_cpu_basic(40.7128, -74.0060, 37.7749, -122.4194)) < 1e-3
    assert abs(dists[1] - haversine_cpu_basic(51.5074, -0.1278, 19.0760, 72.8777)) < 1e-3
