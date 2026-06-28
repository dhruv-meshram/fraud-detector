"""Data Cleaning (Null drops, extreme outlier removal)."""

import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drops records with null coordinates and removes impossible travel outliers."""
    # 1. Missing Value Handling: Drop unresolved Lat/Lon coordinates
    initial_len = len(df)
    df = df.dropna(subset=['latitude', 'longitude'])
    null_dropped = initial_len - len(df)
    if null_dropped > 0:
        print(f"Dropped {null_dropped} records with missing lat/lon coordinates.")
        
    # 2. Outlier Dropping: Remove known "Impossible Travel" logs (velocity > 900 km/h or unverified)
    # This prevents poisoning the clustering model with high-risk events.
    len_before_outliers = len(df)
    
    # We drop records where velocity > 900 km/h OR is_verified == 0 (unverified impossible travel)
    # Keep records where velocity <= 900 and is_verified is 1 (or True)
    df = df[
        (df['velocity_kmh'] <= 900.0) & 
        (df['is_verified'].astype(int) == 1)
    ]
    
    outliers_dropped = len_before_outliers - len(df)
    if outliers_dropped > 0:
        print(f"Dropped {outliers_dropped} impossible travel/unverified outlier logs.")
        
    return df
