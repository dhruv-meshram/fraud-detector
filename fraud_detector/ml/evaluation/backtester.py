"""ShieldFlow Walk-Forward Evaluation & Latency Backtester.

Evaluates user micro-models using sequential time-series splits, calculates global
FPR, TPR, and Silhouette scores, and benchmarks model execution and lookup latencies.
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

DEFAULT_INPUT_PATH = Path("/home/dhruv/Documents/fraud-detector/data/processed/clean_logins.csv")

def run_walk_forward_evaluation(df: pd.DataFrame, num_users: int, min_train: int, eps: float, min_samples: int):
    """Performs walk-forward evaluation over a subset of users and calculates metrics."""
    users = df['user_id'].unique()
    if num_users > 0:
        users = users[:num_users]
        
    print(f"Starting Walk-Forward Backtest on {len(users)} users...")
    
    tp, fp, tn, fn = 0, 0, 0, 0
    silhouette_scores = []
    total_evals = 0
    
    start_time = time.time()
    
    for idx, user_id in enumerate(users):
        user_df = df[df['user_id'] == user_id]
        
        # Generate chronological splits
        splits = list(generate_walk_forward_splits(
            user_df, 
            min_train_size=min_train,
            train_window_days=90,
            max_train_samples=1000
        ))
        
        for train_df, test_log in splits:
            total_evals += 1
            
            # 1. Convert train coords to radians & build distance matrix
            lat_rad = to_radians(train_df['latitude']).values
            lon_rad = to_radians(train_df['longitude']).values
            coords_tensor = torch.tensor(np.column_stack([lat_rad, lon_rad]), dtype=torch.float32)
            dist_matrix_tensor = torch_haversine_matrix(coords_tensor)
            dist_matrix = dist_matrix_tensor.cpu().numpy()
            
            # 2. Run DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = db.fit_predict(dist_matrix)
            
            # 3. Calculate Silhouette Score for this training window
            sil = calculate_silhouette_score(dist_matrix, labels)
            if sil != -1.0:
                silhouette_scores.append(sil)
                
            # 4. Extract clusters and centroids
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
                
            # 5. Evaluate the test log
            test_lat = test_log['latitude']
            test_lon = test_log['longitude']
            is_verified = int(test_log['is_verified'])
            
            is_known = False
            for c_lat, c_lon, r_km in centroids:
                dist = haversine_distance_numpy(test_lat, test_lon, c_lat, c_lon)
                if dist <= r_km:
                    is_known = True
                    break
                    
            # 6. Classify
            if is_verified == 1: # Legitimate login
                if not is_known:
                    fp += 1 # Falsely flagged as outlier
                else:
                    tn += 1 # Correctly recognized as known
            else: # Unverified anomalous login
                if not is_known:
                    tp += 1 # Correctly caught outlier
                else:
                    fn += 1 # Anomaly went undetected
                    
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(users)} users... (total evaluations: {total_evals})")
            
    duration = time.time() - start_time
    print(f"\nBacktest completed in {duration:.2f} seconds.")
    
    metrics = calculate_classification_metrics(tp, fp, tn, fn)
    avg_sil = np.mean(silhouette_scores) if silhouette_scores else 0.0
    
    print("\n================== BACKTEST EVALUATION RESULTS ==================")
    print(f"Total Walk-Forward Evaluations Run : {total_evals}")
    print(f"True Positives (Anomalies caught)  : {tp}")
    print(f"False Positives (Routine flagged)  : {fp}")
    print(f"True Negatives (Routine accepted)  : {tn}")
    print(f"False Negatives (Anomalies missed) : {fn}")
    print("-----------------------------------------------------------------")
    print(f"True Positive Rate (TPR / Recall)  : {metrics['true_positive_rate_tpr'] * 100:.2f}%")
    print(f"False Positive Rate (FPR)          : {metrics['false_positive_rate_fpr'] * 100:.2f}%")
    print(f"Precision                          : {metrics['precision'] * 100:.2f}%")
    print(f"F1-Score                           : {metrics['f1_score'] * 100:.2f}%")
    print(f"Average Silhouette Coefficient     : {avg_sil:.4f}")
    print("=================================================================\n")

def run_benchmarks():
    """Benchmarks PyTorch pairwise distance computation and O(K) inference lookup."""
    print("====================== PERFORMANCE BENCHMARKS ======================")
    
    # 1. PyTorch Pairwise Distance benchmark (1,000 logs)
    print("Benchmarking PyTorch Pairwise Distance Matrix (1,000 logs)...")
    coords = torch.rand((1000, 2), dtype=torch.float32) # Random latitude/longitude
    
    # Warmup
    _ = torch_haversine_matrix(coords)
    
    times = []
    for _ in range(20):
        t0 = time.time()
        _ = torch_haversine_matrix(coords)
        times.append(time.time() - t0)
        
    avg_time_ms = np.mean(times) * 1000
    print(f" --> Avg Matrix Execution Time: {avg_time_ms:.2f} ms")
    if avg_time_ms < 50.0:
        print(" --> Matrix Benchmark: PASSED (< 50 ms)")
    else:
        print(" --> Matrix Benchmark: FAILED (>= 50 ms)")
        
    # 2. O(K) lookup inference benchmark
    print("\nBenchmarking O(K) Boundary Lookup Inference Latency...")
    # Mocking evaluation loop
    mock_centroids = [(40.7128, -74.0060, 2.5) for _ in range(5)] # standard K=5 centroids
    lat, lon = 40.7130, -74.0058
    
    # Warmup
    for c_lat, c_lon, r_km in mock_centroids:
        _ = haversine_distance_numpy(lat, lon, c_lat, c_lon)
        
    t0 = time.time()
    num_runs = 10000
    for _ in range(num_runs):
        for c_lat, c_lon, r_km in mock_centroids:
            dist = haversine_distance_numpy(lat, lon, c_lat, c_lon)
            _ = dist <= r_km
            
    total_time_ms = (time.time() - t0) * 1000
    avg_lookup_ms = total_time_ms / num_runs
    print(f" --> Total Time for {num_runs} evaluations: {total_time_ms:.2f} ms")
    print(f" --> Avg Single Lookup Latency: {avg_lookup_ms:.4f} ms")
    if avg_lookup_ms < 1.0:
        print(" --> Inference Lookup Benchmark: PASSED (< 1 ms)")
    else:
        print(" --> Inference Lookup Benchmark: FAILED (>= 1 ms)")
        
    print("=================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShieldFlow Evaluation Pipeline")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Clean CSV path")
    parser.add_argument("--users-limit", type=int, default=50, help="Number of users to run backtest on (0 for all)")
    parser.add_argument("--min-train", type=int, default=10, help="Min training logs required to train DBSCAN")
    parser.add_argument("--eps", type=float, default=2.5, help="DBSCAN eps radius in km")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip the backtesting step and run benchmarks only")
    args = parser.parse_args()
    
    if not args.skip_backtest:
        df = pd.read_csv(args.input)
        run_walk_forward_evaluation(
            df=df,
            num_users=args.users_limit,
            min_train=args.min_train,
            eps=args.eps,
            min_samples=args.min_samples
        )
        
    run_benchmarks()
