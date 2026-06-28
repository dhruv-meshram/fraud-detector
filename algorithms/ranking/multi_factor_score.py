"""Multi-Factor Heuristic Scorer for edge-case risk evaluation."""

def calculate_risk_score(
    velocity: float, 
    distance_from_centroid: float, 
    device_hash_mismatch: bool
) -> float:
    """Computes a multi-factor risk score between 0.0 (minimum risk) and 1.0 (maximum risk).
    
    Formula:
        Score = (0.5 * Velocity_Factor) + (0.3 * Distance_Factor) + (0.2 * Device_Mismatch_Factor)
        
    Normalizations:
        - Velocity: [100 km/h, 900 km/h] -> [0.0, 1.0]
        - Distance: [10 km, 150 km] -> [0.0, 1.0]
        - Device Mismatch: Boolean -> 1.0 or 0.0
    """
    # 1. Normalize Velocity (50% weight)
    # Under 100 km/h is normal; over 900 km/h is maximum speed limit risk
    if velocity <= 100.0:
        vel_factor = 0.0
    elif velocity >= 900.0:
        vel_factor = 1.0
    else:
        vel_factor = (velocity - 100.0) / 800.0
        
    # 2. Normalize Distance from Centroid (30% weight)
    # Under 10 km is normal commute; over 150 km is a clear spatial anomaly
    if distance_from_centroid <= 10.0:
        dist_factor = 0.0
    elif distance_from_centroid >= 150.0:
        dist_factor = 1.0
    else:
        dist_factor = (distance_from_centroid - 10.0) / 140.0
        
    # 3. Normalize Device Hash Mismatch (20% weight)
    dev_factor = 1.0 if device_hash_mismatch else 0.0
    
    # Calculate weighted score
    score = (0.5 * vel_factor) + (0.3 * dist_factor) + (0.2 * dev_factor)
    return round(score, 4)
