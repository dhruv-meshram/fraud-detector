"""ShieldFlow Evaluation Metrics.

Contains helper functions for computing Silhouette Scores on precomputed distance matrices
and calculating classification metrics (FPR, TPR, Precision, Recall, F1).
"""

from typing import Dict
import numpy as np
# pyrefly: ignore [missing-import]
from sklearn.metrics import silhouette_score

def calculate_silhouette_score(dist_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Calculates the Silhouette Score using a precomputed distance matrix.
    
    Returns -1.0 if not enough distinct clusters exist to calculate the score.
    """
    # Filter out noise points (labeled as -1) and check if we have valid clusters
    mask = labels != -1
    filtered_labels = labels[mask]
    
    if len(filtered_labels) == 0:
        return -1.0
        
    unique_labels = set(filtered_labels)
    if len(unique_labels) < 2:
        # Silhouette requires at least 2 distinct clusters
        return -1.0
        
    filtered_dist = dist_matrix[mask][:, mask]
    
    try:
        score = silhouette_score(filtered_dist, filtered_labels, metric="precomputed")
        return float(score)
    except Exception:
        return -1.0

def calculate_classification_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Computes binary classification performance metrics (TPR, FPR, Precision, Recall, F1)."""
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positive_rate_tpr": round(tpr, 4),
        "false_positive_rate_fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }
