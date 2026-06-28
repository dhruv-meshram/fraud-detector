import torch
import math
from shieldflow.config import EPSILON_KM, MIN_SAMPLES, CLUSTER_RADIUS_BUFFER_KM
from shieldflow.algorithms.haversine import haversine_distance

def compute_pairwise_haversine(coords: torch.Tensor) -> torch.Tensor:
    """Computes pairwise haversine distance using PyTorch broadcasting.
    
    Args:
        coords: Tensor of shape (N, 2) where coords[:, 0] is latitude and coords[:, 1] is longitude.
        
    Returns:
        Tensor of shape (N, N) with distances in kilometers.
    """
    R = 6371.0
    rad_coords = torch.deg2rad(coords)
    
    lat = rad_coords[:, 0].unsqueeze(1)  # (N, 1)
    lon = rad_coords[:, 1].unsqueeze(1)  # (N, 1)
    
    dlat = lat - lat.t()  # (N, N)
    dlon = lon - lon.t()  # (N, N)
    
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat) * torch.cos(lat.t()) * torch.sin(dlon / 2.0) ** 2
    a = torch.clamp(a, 0.0, 1.0)
    c = 2.0 * torch.asin(torch.sqrt(a))
    
    return R * c

def run_dbscan_gpu(
    history: list,
    epsilon_km: float = EPSILON_KM,
    min_samples: int = MIN_SAMPLES,
    radius_buffer_km: float = CLUSTER_RADIUS_BUFFER_KM
) -> list:
    """Runs a density-based spatial clustering check on login log histories using PyTorch.
    
    Args:
        history: List of login dicts containing 'latitude' and 'longitude'.
        epsilon_km: DBSCAN clustering neighborhood search radius.
        min_samples: Minimum cluster sample size threshold.
        radius_buffer_km: Geofence buffer offset added to dynamic radius.
        
    Returns:
        List of cluster profiles containing centroid coordinates and dynamic radius.
    """
    if len(history) < min_samples:
        return []
        
    coords_list = [[float(item["latitude"]), float(item["longitude"])] for item in history]
    coords = torch.tensor(coords_list, dtype=torch.float32)
    
    # Compute pairwise distance matrix using PyTorch
    dist_matrix = compute_pairwise_haversine(coords)
    
    # Neighbors mask
    neighbors_mask = dist_matrix <= epsilon_km
    
    # Count of neighbors for each point
    degrees = neighbors_mask.sum(dim=1)
    
    # Core points: points having at least min_samples neighbors
    core_points_mask = degrees >= min_samples
    
    N = len(history)
    labels = [-1] * N
    cluster_idx = 0
    
    # Simple DBSCAN cluster components flood-fill
    for i in range(N):
        if labels[i] != -1:
            continue
        if not core_points_mask[i].item():
            continue
            
        labels[i] = cluster_idx
        queue = [j for j in range(N) if neighbors_mask[i, j].item() and j != i]
        
        idx = 0
        while idx < len(queue):
            neighbor = queue[idx]
            idx += 1
            
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_idx
                if core_points_mask[neighbor].item():
                    for k in range(N):
                        if neighbors_mask[neighbor, k].item() and labels[k] == -1 and k not in queue:
                            queue.append(k)
                            
        cluster_idx += 1
        
    # Group coordinates by cluster label to compute centroids
    clusters = []
    for c_id in range(cluster_idx):
        indices = [i for i, label in enumerate(labels) if label == c_id]
        if not indices:
            continue
            
        cluster_coords = coords[indices]
        mean_lat = float(cluster_coords[:, 0].mean().item())
        mean_lon = float(cluster_coords[:, 1].mean().item())
        
        # Max distance from centroid to compute dynamic radius
        max_dist = 0.0
        for idx in indices:
            dist = haversine_distance(mean_lat, mean_lon, float(history[idx]["latitude"]), float(history[idx]["longitude"]))
            if dist > max_dist:
                max_dist = dist
                
        dynamic_radius = max_dist + radius_buffer_km
        
        clusters.append({
            "centroid_lat": mean_lat,
            "centroid_lon": mean_lon,
            "dynamic_radius": dynamic_radius,
            "num_points": len(indices)
        })
        
    return clusters
