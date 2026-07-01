"""Hyperparameter Grid Search for ShieldFlow.

Searches combinations of EPS, min_samples, and velocity thresholds to minimize
FPR, maximize TPR, and optimize Silhouette Scores using Walk-Forward validation.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from sklearn.cluster import DBSCAN

from fraud_detector.ml.datasets.splitting import generate_walk_forward_splits
from fraud_detector.ml.features.geo_features import torch_haversine_matrix, to_radians, haversine_distance_numpy, calculate_geographic_centroid
from fraud_detector.ml.evaluation.metrics import calculate_silhouette_score, calculate_classification_metrics
from fraud_detector.ml.preprocessing.dedup import deduplicate_data

DEFAULT_RAW_INPUT_PATH = Path("/home/dhruv/Documents/fraud-detector/fraud_detector/data/raw/synthetic_logins.csv")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

def evaluate_combination(
    df_raw: pd.DataFrame,
    users: list,
    eps_km: float,
    min_samples: int,
    velocity_th: float,
    min_train: int = 10
):
    """Evaluates a single hyperparameter combination across selected users."""
    tp, fp, tn, fn = 0, 0, 0, 0
    silhouette_scores = []
    
    # Simulate preprocessing per user based on velocity_threshold
    # Filter out null coordinates and impossible travel logs above the threshold
    df_clean = df_raw.dropna(subset=['latitude', 'longitude']).copy()
    
    for user_id in users:
        user_df = df_clean[df_clean['user_id'] == user_id].copy()
        
        # Sort chronologically
        user_df = user_df.sort_values(by='timestamp').reset_index(drop=True)
        
        # Deduplicate same-device logins within 10s
        user_df = deduplicate_data(user_df)
        
        # Generate walk-forward splits
        # Split history up to index N, evaluate N+1
        for i in range(len(user_df)):
            test_log = user_df.iloc[i]
            history = user_df.iloc[:i]
            
            # For training: drop unverified logs and logs exceeding the velocity threshold
            train_df = history[
                (history['is_verified'].astype(int) == 1) & 
                (history['velocity_kmh'] <= velocity_th)
            ]
            
            if len(train_df) < min_train:
                continue
                
            # Train DBSCAN model
            lat_rad = to_radians(train_df['latitude']).values
            lon_rad = to_radians(train_df['longitude']).values
            coords_tensor = torch.tensor(np.column_stack([lat_rad, lon_rad]), dtype=torch.float32)
            dist_matrix_tensor = torch_haversine_matrix(coords_tensor)
            dist_matrix = dist_matrix_tensor.cpu().numpy()
            
            db = DBSCAN(eps=eps_km, min_samples=min_samples, metric='precomputed')
            labels = db.fit_predict(dist_matrix)
            
            sil = calculate_silhouette_score(dist_matrix, labels)
            if sil != -1.0:
                silhouette_scores.append(sil)
                
            # Extract clusters
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
                
            centroids = []
            for label in unique_labels:
                cluster_df = train_df[labels == label]
                centroid_lat, centroid_lon = calculate_geographic_centroid(
                    cluster_df['latitude'].values,
                    cluster_df['longitude'].values
                )
                distances = haversine_distance_numpy(
                    cluster_df['latitude'].values,
                    cluster_df['longitude'].values,
                    centroid_lat,
                    centroid_lon
                )
                dynamic_radius_km = float(np.max(distances)) + 0.5
                centroids.append((centroid_lat, centroid_lon, dynamic_radius_km))
                
            # Evaluate test log
            test_lat = test_log['latitude']
            test_lon = test_log['longitude']
            test_velocity = test_log['velocity_kmh']
            is_verified = int(test_log['is_verified'])
            
            # Predict using bifurcated architecture:
            # 1. Fast Path Spatiotemporal velocity check
            is_velocity_flagged = test_velocity > velocity_th
            
            # 2. Slow Path Spatial centroid boundary check
            is_known_location = False
            for c_lat, c_lon, r_km in centroids:
                dist = haversine_distance_numpy(test_lat, test_lon, c_lat, c_lon)
                if dist <= r_km:
                    is_known_location = True
                    break
                    
            is_flagged = is_velocity_flagged or (not is_known_location)
            
            # Classify
            if is_verified == 1:
                if is_flagged:
                    fp += 1
                else:
                    tn += 1
            else:
                if is_flagged:
                    tp += 1
                else:
                    fn += 1
                    
    metrics = calculate_classification_metrics(tp, fp, tn, fn)
    avg_sil = np.mean(silhouette_scores) if silhouette_scores else 0.0
    
    return {
        "eps_km": eps_km,
        "min_samples": min_samples,
        "velocity_threshold_kmh": velocity_th,
        "tpr": metrics["true_positive_rate_tpr"],
        "fpr": metrics["false_positive_rate_fpr"],
        "precision": metrics["precision"],
        "f1": metrics["f1_score"],
        "avg_silhouette": round(avg_sil, 4)
    }

def run_grid_search(input_path: str, users_limit: int):
    print(f"Loading raw logins from: {input_path}")
    df_raw = pd.read_csv(input_path)
    
    users = df_raw['user_id'].unique()
    if users_limit > 0:
        users = users[:users_limit]
        
    print(f"Grid search optimized across {len(users)} users...")
    
    # Parameter grid definition
    eps_grid = [1.0, 2.5, 5.0, 10.0, 15.0]
    min_samples_grid = [2, 3, 5]
    velocity_grid = [500.0, 800.0, 1000.0, 1200.0]
    
    results = []
    total_combs = len(eps_grid) * len(min_samples_grid) * len(velocity_grid)
    curr_comb = 0
    
    start_time = time.time()
    
    if HAS_MLFLOW:
        mlflow.set_experiment("ShieldFlow_Clustering_Tuning")
        
    for eps in eps_grid:
        for min_s in min_samples_grid:
            for vel in velocity_grid:
                curr_comb += 1
                print(f"[{curr_comb}/{total_combs}] Evaluating eps={eps}km, min_samples={min_s}, velocity_threshold={vel}kmh...")
                
                t_sub0 = time.time()
                res = evaluate_combination(df_raw, users, eps, min_s, vel)
                t_sub_dur = time.time() - t_sub0
                
                print(f"    --> TPR: {res['tpr']*100:.2f}% | FPR: {res['fpr']*100:.2f}% | Sil: {res['avg_silhouette']:.3f} | Time: {t_sub_dur:.2f}s")
                results.append(res)
                
                if HAS_MLFLOW:
                    with mlflow.start_run(run_name=f"eps_{eps}_min_{min_s}_vel_{vel}"):
                        mlflow.log_params({
                            "eps_km": eps,
                            "min_samples": min_s,
                            "velocity_threshold_kmh": vel
                        })
                        mlflow.log_metrics({
                            "tpr": res["tpr"],
                            "fpr": res["fpr"],
                            "precision": res["precision"],
                            "f1": res["f1"],
                            "avg_silhouette": res["avg_silhouette"]
                        })
                
    duration = time.time() - start_time
    print(f"\nGrid search completed in {duration/60:.2f} minutes.")
    
    # Convert to DataFrame for sorting
    res_df = pd.DataFrame(results)
    
    # Sort: Prioritize lowest FPR, then highest F1-Score, then highest Silhouette Score
    res_df['score'] = res_df['tpr'] - res_df['fpr'] + 0.1 * res_df['avg_silhouette']
    res_df = res_df.sort_values(by=['fpr', 'score'], ascending=[True, False]).reset_index(drop=True)
    
    print("\n======================= TOP 5 PARAMETER CONFIGURATIONS =======================")
    for idx, row in res_df.head(5).iterrows():
        print(f"Rank {idx+1}: Score={row['score']:.4f}")
        print(f"  --> EPS: {row['eps_km']} km | Min Samples: {row['min_samples']} | Velocity Threshold: {row['velocity_threshold_kmh']} km/h")
        print(f"  --> FPR: {row['fpr']*100:.2f}% | TPR: {row['tpr']*100:.2f}% | F1: {row['f1']*100:.2f}% | Silhouette: {row['avg_silhouette']:.4f}")
        print("-" * 78)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShieldFlow Grid Search Parameter Tuning")
    parser.add_argument("--input", type=str, default=str(DEFAULT_RAW_INPUT_PATH), help="Raw CSV path")
    parser.add_argument("--users-limit", type=int, default=5, help="Number of users to run tuning on (low for performance)")
    args = parser.parse_args()
    
    run_grid_search(args.input, args.users_limit)
