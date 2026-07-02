"""ShieldFlow Offline Batch Training Pipeline.

Ingests verified logins, executes GPU/CPU-accelerated PyTorch Haversine matrix computations,
runs scikit-learn DBSCAN clustering, extracts geographic centroids & dynamic radii,
and persists user spatial profile checkpoints as JSON.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from sklearn.cluster import DBSCAN

from fraud_detector.ml.datasets.splitting import get_rolling_train_window
from fraud_detector.ml.features.geo_features import torch_haversine_matrix, to_radians, haversine_distance_numpy

# Configuration
DEFAULT_CLEAN_LOGINS_PATH = Path("/home/dhruv/Documents/fraud-detector/fraud_detector/data/processed/clean_logins.csv")
DEFAULT_PROFILES_DIR = Path("/home/dhruv/Documents/fraud-detector/fraud_detector/data/processed/profiles")

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

def train_user_model(
    user_id: str, 
    user_logins: pd.DataFrame, 
    eps_km: float = 2.5, 
    min_samples: int = 3,
    train_window_days: int = 90,
    max_train_samples: int = 1000
) -> dict:
    """Trains a DBSCAN micro-model on a single user's historical coordinates.
    
    Returns:
        A dictionary containing the user's spatial profile.
    """
    # 1. Get rolling training window of verified logins
    train_df = get_rolling_train_window(
        user_logins, 
        train_window_days=train_window_days, 
        max_train_samples=max_train_samples
    )
    
    profile = {
        "user_id": user_id,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "total_logins_trained": len(train_df),
        "clusters": []
    }
    
    if len(train_df) < min_samples:
        # Not enough data points to form any cluster
        return profile
        
    # 2. Extract latitude/longitude in radians
    lat_rad = to_radians(train_df['latitude']).values
    lon_rad = to_radians(train_df['longitude']).values
    
    # 3. Create PyTorch tensor matrix and compute pairwise distance
    coords_tensor = torch.tensor(np.column_stack([lat_rad, lon_rad]), dtype=torch.float32)
    dist_matrix_tensor = torch_haversine_matrix(coords_tensor)
    dist_matrix = dist_matrix_tensor.cpu().numpy()
    
    # 4. Train DBSCAN on precomputed distance matrix
    db = DBSCAN(eps=eps_km, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dist_matrix)
    
    # 5. Extract clusters
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1) # Remove noise label
        
    for label in sorted(unique_labels):
        cluster_mask = (labels == label)
        cluster_df = train_df[cluster_mask]
        
        # Calculate geometric center
        centroid_lat, centroid_lon = calculate_geographic_centroid(
            cluster_df['latitude'].values, 
            cluster_df['longitude'].values
        )
        
        # Compute dynamic radius: max distance from centroid to core points + 500m buffer
        distances_to_centroid = haversine_distance_numpy(
            cluster_df['latitude'].values,
            cluster_df['longitude'].values,
            centroid_lat,
            centroid_lon
        )
        max_dist_km = float(np.max(distances_to_centroid))
        dynamic_radius_km = round(max_dist_km + 0.5, 3) # 500m buffer
        
        profile["clusters"].append({
            "cluster_id": int(label),
            "centroid_lat": round(float(centroid_lat), 6),
            "centroid_lon": round(float(centroid_lon), 6),
            "dynamic_radius_km": dynamic_radius_km,
            "num_points": int(len(cluster_df))
        })
        
    return profile

def run_training_pipeline(
    input_file: str = None, 
    profiles_dir: str = None, 
    user_id: str = None,
    eps_km: float = 2.5,
    min_samples: int = 3,
    db_store = None,
    profile_store = None
):
    from fraud_detector.adapters.db import PostgresDBStore
    from fraud_detector.adapters.profile import PostgreSQLProfileStore
    
    if db_store is None:
        db_store = PostgresDBStore()
    if profile_store is None:
        profile_store = PostgreSQLProfileStore()
        
    use_db = False
    try:
        if user_id:
            users = [user_id]
        else:
            users = db_store.get_all_users()
        if users:
            use_db = True
    except Exception:
        use_db = False
        
    if use_db:
        print(f"Loading user list and login history from database...")
    else:
        if not input_file:
            input_file = str(DEFAULT_CLEAN_LOGINS_PATH)
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Clean preprocessed dataset not found at: {input_path}")
            
        print(f"Loading clean logins from: {input_path}...")
        df = pd.read_csv(input_path)
        if user_id:
            users = [user_id]
        else:
            users = df['user_id'].unique()
        
    trained_count = 0
    skipped_count = 0
    
    for i, curr_user in enumerate(users):
        if use_db:
            history_list = db_store.get_user_history(curr_user)
            user_logins = pd.DataFrame(history_list)
        else:
            user_logins = df[df['user_id'] == curr_user]
            
        if user_logins.empty:
            skipped_count += 1
            continue
            
        # Train model
        profile = train_user_model(
            curr_user, 
            user_logins, 
            eps_km=eps_km, 
            min_samples=min_samples
        )
        
        # Persist checkpoint if training occurred
        if profile["total_logins_trained"] >= min_samples:
            profile_store.save_profile(curr_user, profile)
            trained_count += 1
        else:
            skipped_count += 1
            
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(users)} users...")
            
    print(f"Training completed successfully!")
    print(f" --> Profiles Trained & Saved: {trained_count}")
    print(f" --> Users Skipped (insufficient logins): {skipped_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShieldFlow DBSCAN Offline Batch Training Pipeline")
    parser.add_argument("--input", type=str, default=str(DEFAULT_CLEAN_LOGINS_PATH), help="Clean CSV path")
    parser.add_argument("--profiles-dir", type=str, default=str(DEFAULT_PROFILES_DIR), help="Output JSON profiles dir")
    parser.add_argument("--user-id", type=str, default=None, help="Train specifically for a user ID")
    parser.add_argument("--eps", type=float, default=2.5, help="DBSCAN eps radius in kilometers")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples")
    args = parser.parse_args()
    
    run_training_pipeline(
        args.input, 
        args.profiles_dir, 
        user_id=args.user_id,
        eps_km=args.eps,
        min_samples=args.min_samples
    )
