"""Metric Ball Tree implementation for efficient neighborhood searches."""

import numpy as np
# pyrefly: ignore [missing-import]
from sklearn.neighbors import BallTree as SklearnBallTree

class BallTree:
    """A spatial index tree that optimizes neighborhood queries to O(N log N) using the Haversine metric.
    
    Coordinates are passed in degrees and queried in degrees, while the underlying tree
    operates on radians.
    """
    
    R_EARTH = 6371.0  # Earth's radius in kilometers
    
    def __init__(self, coordinates: np.ndarray):
        """Builds a Ball Tree from an array of coordinates.
        
        Args:
            coordinates: A 2D numpy array of shape (N, 2) representing [[latitude, longitude], ...].
        """
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("Coordinates must be a 2D array of shape (N, 2)")
            
        # Convert lat/lon in degrees to radians for the Haversine metric
        # Note: Sklearn's haversine expects [latitude, longitude] in radians
        self.coords_rad = np.radians(self.coordinates)
        
        # Build the sklearn BallTree
        self._tree = SklearnBallTree(self.coords_rad, metric='haversine')
        
    def query_radius(self, point: np.ndarray, radius_km: float) -> np.ndarray:
        """Finds all indices within a given radius in kilometers.
        
        Args:
            point: A 1D array/list [latitude, longitude] in degrees.
            radius_km: Query radius in kilometers.
            
        Returns:
            A 1D numpy array containing the indices of coordinates within the radius.
        """
        # Format input point
        pt = np.asarray(point, dtype=np.float64).reshape(1, 2)
        pt_rad = np.radians(pt)
        
        # Convert radius in km to radians
        radius_rad = radius_km / self.R_EARTH
        
        # Query the tree
        indices = self._tree.query_radius(pt_rad, r=radius_rad)
        return indices[0]

    def query(self, point: np.ndarray, k: int = 1) -> tuple:
        """Finds the k nearest neighbors.
        
        Args:
            point: A 1D array/list [latitude, longitude] in degrees.
            k: Number of nearest neighbors to query (default: 1).
            
        Returns:
            A tuple of (distances_km, indices).
        """
        pt = np.asarray(point, dtype=np.float64).reshape(1, 2)
        pt_rad = np.radians(pt)
        
        # Query the tree
        dists_rad, indices = self._tree.query(pt_rad, k=k)
        
        # Convert distances back to kilometers
        dists_km = dists_rad * self.R_EARTH
        return dists_km[0], indices[0]
