"""ShieldFlow Geo-Spatial Feature Engineering.

Contains mathematical engines for Haversine distance, spatiotemporal velocity,
and PyTorch tensor operations for O(N^2) pairwise distance matrices.
"""

import math
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch

# Earth's radius in kilometers
R_EARTH = 6371.0

def to_radians(degrees):
    """Converts degrees to radians (handles float, numpy arrays, or pandas series)."""
    if isinstance(degrees, (pd.Series, np.ndarray)):
        return np.radians(degrees)
    return math.radians(degrees)

def haversine_distance_numpy(lat1, lon1, lat2, lon2):
    """Computes the Haversine distance between two coordinates in degrees.
    
    Returns:
        Distance in kilometers.
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    a = np.clip(a, 0.0, 1.0)
    
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R_EARTH * c

def spatiotemporal_velocity(lat1, lon1, t1, lat2, lon2, t2):
    """Calculates spatiotemporal velocity in km/h between two logins.
    
    Args:
        lat1, lon1: Coordinates of first login (degrees).
        t1: Timestamp of first login (UNIX epoch seconds).
        lat2, lon2: Coordinates of second login (degrees).
        t2: Timestamp of second login (UNIX epoch seconds).
        
    Returns:
        Velocity in km/h.
    """
    dist_km = haversine_distance_numpy(lat1, lon1, lat2, lon2)
    time_diff_sec = abs(t2 - t1)
    
    if time_diff_sec == 0:
        return 0.0
        
    time_diff_hours = time_diff_sec / 3600.0
    return dist_km / time_diff_hours

def torch_haversine_matrix(coords):
    """Generates an N x N pairwise Haversine distance matrix from coordinate tensors.
    
    Args:
        coords: PyTorch Tensor of shape (N, 2) where coords[:, 0] is latitude (rad)
                and coords[:, 1] is longitude (rad).
                
    Returns:
        PyTorch Tensor of shape (N, N) containing pairwise distances in kilometers.
    """
    import torch
    
    lat = coords[:, 0]  # Latitudes in radians
    lon = coords[:, 1]  # Longitudes in radians
    
    # Expand coordinates to N x N shape
    lat1 = lat.unsqueeze(1)  # (N, 1)
    lat2 = lat.unsqueeze(0)  # (1, N)
    
    lon1 = lon.unsqueeze(1)  # (N, 1)
    lon2 = lon.unsqueeze(0)  # (1, N)
    
    # Compute diffs
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = torch.sin(dlat / 2.0)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0)**2
    a = torch.clamp(a, 0.0, 1.0)
    
    c = 2.0 * torch.asin(torch.sqrt(a))
    return R_EARTH * c

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Performs feature extraction on a logins DataFrame.
    
    Extracts latitude/longitude in radians for DBSCAN clustering.
    
    Args:
        df: Pandas DataFrame containing 'latitude' and 'longitude' columns in degrees.
        
    Returns:
        Modified DataFrame containing 'latitude_rad' and 'longitude_rad'.
    """
    df = df.copy()
    df['latitude_rad'] = to_radians(df['latitude'])
    df['longitude_rad'] = to_radians(df['longitude'])
    return df

def calculate_geographic_centroid(latitudes_deg, longitudes_deg):
    """Calculates the geographic center (centroid) of a set of coordinates.
    
    Uses 3D Cartesian coordinates projection to avoid issues with poles 
    and wrap-around at the 180-meridian.
    """
    if len(latitudes_deg) == 0:
        return 0.0, 0.0
        
    lats_rad = np.radians(latitudes_deg)
    lons_rad = np.radians(longitudes_deg)
    
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_z = np.mean(z)
    
    lon_centroid_rad = np.arctan2(mean_y, mean_x)
    hyp = np.sqrt(mean_x**2 + mean_y**2)
    lat_centroid_rad = np.arctan2(mean_z, hyp)
    
    return np.degrees(lat_centroid_rad), np.degrees(lon_centroid_rad)

