"""ShieldFlow Drift & Performance Monitor.

Tracks GeoIP resolution anomalies (data drift), global False Positive Rates (concept drift),
and endpoint p99 latencies to raise MLOps alerts when thresholds are exceeded.
"""

from typing import List, Dict, Any, Union
import numpy as np

# Default cities coordinates used by the generator to spot default city-center spikes
DEFAULT_CITIES = [
    (37.7749, -122.4194), # SF
    (40.7128, -74.0060),  # NY
    (51.5074, -0.1278),   # London
    (19.0760, 72.8777),   # Mumbai
    (23.2156, 72.6369),   # Gandhinagar
    (-33.8688, 151.2093), # Sydney
]

def check_data_drift(
    logins_batch: List[Dict[str, Any]], 
    min_resolution_rate: float = 0.95,
    max_default_center_rate: float = 0.10
) -> Dict[str, Any]:
    """Monitors GeoIP database health for data drift.
    
    Checks the rate of unresolved IPs (null Lat/Lon) or fallback default coordinates.
    """
    total = len(logins_batch)
    if total == 0:
        return {"status": "OK", "resolution_rate": 1.0, "default_center_rate": 0.0}
        
    null_count = 0
    default_center_count = 0
    
    for log in logins_batch:
        lat = log.get("latitude")
        lon = log.get("longitude")
        
        if lat is None or lon is None or np.isnan(lat) or np.isnan(lon):
            null_count += 1
            continue
            
        # Check if coordinates exactly match default city centers (GeoIP fallback signature)
        coords = (round(float(lat), 4), round(float(lon), 4))
        for city_lat, city_lon in DEFAULT_CITIES:
            if abs(coords[0] - round(city_lat, 4)) < 1e-4 and abs(coords[1] - round(city_lon, 4)) < 1e-4:
                default_center_count += 1
                break
                
    resolution_rate = (total - null_count) / total
    default_center_rate = default_center_count / total
    
    alerts = []
    if resolution_rate < min_resolution_rate:
        alerts.append(
            f"[ALERT] Data Drift: GeoIP resolution rate has dropped to {resolution_rate:.2%}! "
            f"GeoIP database health check required."
        )
    if default_center_rate > max_default_center_rate:
        alerts.append(
            f"[ALERT] Data Drift: Spiked default city-center fallback signature to {default_center_rate:.2%}! "
            f"Check GeoIP routing accuracy."
        )
        
    return {
        "status": "ALERT" if alerts else "OK",
        "resolution_rate": round(resolution_rate, 4),
        "default_center_rate": round(default_center_rate, 4),
        "alerts": alerts
    }

def check_concept_drift(
    evaluations_batch: List[Dict[str, Any]], 
    max_global_fpr: float = 0.05
) -> Dict[str, Any]:
    """Monitors global MFA trigger rates (concept drift).
    
    If the False Positive Rate rises above 5%, the mobile ISP routing behavior
    might have changed (e.g., wider IP scattering due to 5G network changes), 
    requiring a global eps increase.
    """
    total = len(evaluations_batch)
    if total == 0:
        return {"status": "OK", "fpr": 0.0}
        
    fp_count = 0
    tn_count = 0
    
    for ev in evaluations_batch:
        is_verified = ev.get("is_verified")
        predicted_status = ev.get("status") # 'KNOWN_ZONE' or 'OUTLIER'
        
        if is_verified == 1 or is_verified is True:
            # Routine login
            if predicted_status == "OUTLIER":
                fp_count += 1
            else:
                tn_count += 1
                
    total_routine = fp_count + tn_count
    fpr = fp_count / total_routine if total_routine > 0 else 0.0
    
    alerts = []
    if fpr > max_global_fpr:
        alerts.append(
            f"[ALERT] Concept Drift: Global False Positive Rate (FPR) is {fpr:.2%}, "
            f"exceeding the {max_global_fpr:.2%} threshold! Mobile ISP routing profiles "
            f"may have expanded. Recommend increasing the global DBSCAN eps parameter."
        )
        
    return {
        "status": "ALERT" if alerts else "OK",
        "fpr": round(fpr, 4),
        "total_routine_evaluated": total_routine,
        "alerts": alerts
    }

def check_latency_drift(
    latencies_ms: List[float], 
    max_p99_ms: float = 50.0
) -> Dict[str, Any]:
    """Monitors endpoint latency SLAs.
    
    Alerts if the p99 latency exceeds 50 ms.
    """
    if not latencies_ms:
        return {"status": "OK", "p99_ms": 0.0}
        
    p99 = np.percentile(latencies_ms, 99)
    
    alerts = []
    if p99 > max_p99_ms:
        alerts.append(
            f"[ALERT] Latency Drift: p99 latency is {p99:.2f} ms, "
            f"exceeding the SLA limit of {max_p99_ms} ms!"
        )
        
    return {
        "status": "ALERT" if alerts else "OK",
        "p99_ms": round(float(p99), 2),
        "alerts": alerts
    }
