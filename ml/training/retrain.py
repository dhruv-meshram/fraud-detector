"""ShieldFlow Event-Driven Retraining & Cold Start Manager.

Implements the cold-start rule-based velocity fallback for sparse users (<10 logins),
triggers first-profile generation at exactly 10 logins, and schedules retraining
on new verified location anomalies.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from ml.features.geo_features import spatiotemporal_velocity, haversine_distance_numpy
from ml.training.train_pipeline import train_user_model, DEFAULT_PROFILES_DIR, DEFAULT_CLEAN_LOGINS_PATH
from ml.models.inference_ops import evaluate_login_location

def run_time_driven_cron(
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
    clean_logins_path: Path = DEFAULT_CLEAN_LOGINS_PATH,
    force_all: bool = False,
    age_days: int = 30
):
    """Time-driven cron job that triggers a pruning and retraining cycle for obsolete locations."""
    print("Starting time-driven cron retraining cycle...")
    if not profiles_dir.exists():
        print(f"Profiles directory {profiles_dir} does not exist. No profiles to update.")
        return
        
    if not clean_logins_path.exists():
        print(f"Clean logins file {clean_logins_path} does not exist.")
        return

    df = pd.read_csv(clean_logins_path)
    now = datetime.now(timezone.utc)
    
    retrained_count = 0
    skipped_count = 0
    
    for profile_file in profiles_dir.glob("*.json"):
        user_id = profile_file.stem
        
        try:
            with open(profile_file, "r") as f:
                profile = json.load(f)
                
            last_updated_str = profile.get("last_updated")
            if not last_updated_str:
                should_update = True
                age_days_actual = "unknown"
            else:
                last_updated = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
                age = now - last_updated
                should_update = force_all or (age.days >= age_days)
                age_days_actual = age.days
                
            if should_update:
                user_df = df[df['user_id'] == user_id]
                verified_history = user_df[user_df['is_verified'].astype(int) == 1]
                if len(verified_history) >= 10:
                    print(f"Retraining user {user_id} (profile age: {age_days_actual} days)...")
                    new_profile = train_user_model(user_id, user_df)
                    with open(profile_file, "w") as f:
                        json.dump(new_profile, f, indent=2)
                    retrained_count += 1
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error checking profile for user {user_id}: {e}")
            
    print(f"Cron retraining cycle finished. Retrained: {retrained_count}, Skipped: {skipped_count}.")

def check_and_trigger_retrain(
    user_id: str,
    new_login_lat: float,
    new_login_lon: float,
    new_login_ts: float,
    new_login_device: str,
    clean_logins_path: Path = DEFAULT_CLEAN_LOGINS_PATH,
    profiles_dir: Path = DEFAULT_PROFILES_DIR
) -> dict:
    """Orchestrates cold start, boundary evaluation, and retraining triggers for a login.
    
    Args:
        user_id: User ID.
        new_login_lat, new_login_lon: Coordinates.
        new_login_ts: UNIX timestamp.
        new_login_device: Device hash.
        clean_logins_path: Path to database clean_logins.csv.
        profiles_dir: Profiles directory.
        
    Returns:
        Evaluation status dict: {'status': ..., 'risk': ..., 'action': ...}
    """
    # 1. Load user's clean verified history
    if not clean_logins_path.exists():
        # Fallback to cold start if clean logins file doesn't exist
        return {
            "status": "NO_HISTORY_DB",
            "risk": "LOW_RISK",
            "action": "BYPASS_ML_COLD_START"
        }
        
    df = pd.read_csv(clean_logins_path)
    user_history = df[df['user_id'] == user_id].sort_values(by='timestamp')
    verified_history = user_history[user_history['is_verified'].astype(int) == 1]
    
    num_verified = len(verified_history)
    
    # --- COLD START STRATEGY (< 10 logins) ---
    if num_verified < 10:
        if num_verified == 0:
            return {
                "status": "COLD_START_FIRST_LOGIN",
                "risk": "LOW_RISK",
                "action": "BYPASS_ML_RULE_FALLBACK"
            }
            
        # Get last login for spatiotemporal velocity check
        last_login = verified_history.iloc[-1]
        last_lat = last_login['latitude']
        last_lon = last_login['longitude']
        last_ts = last_login['timestamp']
        
        velocity = spatiotemporal_velocity(
            last_lat, last_lon, last_ts,
            new_login_lat, new_login_lon, new_login_ts
        )
        
        is_impossible = velocity > 900.0
        return {
            "status": "COLD_START_SPARSE_DATA",
            "risk": "HIGH_RISK" if is_impossible else "LOW_RISK",
            "velocity_kmh": round(velocity, 2),
            "action": "BYPASS_ML_RULE_FALLBACK"
        }
        
    # --- ML PHASE (>= 10 logins) ---
    profile_path = Path(profiles_dir) / f"{user_id}.json"
    
    # Transition check: if exactly 10 verified logins, trigger retraining
    if num_verified == 10 and not profile_path.exists():
        print(f"User {user_id} reached exactly 10 verified logins. Generating first spatial profile...")
        profile = train_user_model(user_id, user_history)
        profiles_dir.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
            
    # Load profile and evaluate risk
    evaluation = evaluate_login_location(user_id, new_login_lat, new_login_lon, profiles_dir)
    
    # Retraining trigger condition: check if user has accumulated 5 new verified anomalies
    # An anomaly is a verified login (is_verified == 1) that fell outside the known boundary (OUTLIER)
    # when evaluated at inference time.
    if profile_path.exists() and evaluation["status"] == "OUTLIER":
        # Check how many recent verified logins have been outliers relative to the profile's clusters
        with open(profile_path, "r") as f:
            profile = json.load(f)
            
        clusters = profile.get("clusters", [])
        
        # Count verified history events that are outliers (distance > radius for all clusters)
        outlier_anomalies_count = 0
        for _, hist_row in verified_history.tail(20).iterrows():
            h_lat = hist_row['latitude']
            h_lon = hist_row['longitude']
            
            is_in_any = False
            for cluster in clusters:
                c_lat = cluster["centroid_lat"]
                c_lon = cluster["centroid_lon"]
                rad = cluster["dynamic_radius_km"]
                dist = haversine_distance_numpy(h_lat, h_lon, c_lat, c_lon)
                if dist <= rad:
                    is_in_any = True
                    break
            if not is_in_any:
                outlier_anomalies_count += 1
                
        # If user has accumulated 5 or more new verified anomalies, trigger retraining
        if outlier_anomalies_count >= 5:
            print(f"User {user_id} has accumulated {outlier_anomalies_count} verified anomalies. Re-training DBSCAN model...")
            profile = train_user_model(user_id, user_history)
            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=2)
            # Re-evaluate with the updated profile
            evaluation = evaluate_login_location(user_id, new_login_lat, new_login_lon, profiles_dir)
            evaluation["retrained"] = True
            
    return {
        "status": "ML_CLASSIFIED",
        "risk": "LOW_RISK" if evaluation["status"] == "KNOWN_ZONE" else "HIGH_RISK",
        "details": evaluation,
        "action": "EVALUATE_SPATIAL_CLUSTERS"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShieldFlow Cold Start & Retraining Trigger test interface")
    parser.add_argument("--cron", action="store_true", help="Run the time-driven cron retraining cycle")
    parser.add_argument("--force", action="store_true", help="Force retraining for all users in cron cycle")
    parser.add_argument("--age-days", type=int, default=30, help="Profile age threshold in days for retraining (default: 30)")
    parser.add_argument("--user-id", type=str, default=None, help="User ID (required if not running in cron mode)")
    parser.add_argument("--lat", type=float, default=None, help="New login latitude (required if not running in cron mode)")
    parser.add_argument("--lon", type=float, default=None, help="New login longitude (required if not running in cron mode)")
    parser.add_argument("--timestamp", type=float, default=None, help="New login timestamp (defaults to current time)")
    parser.add_argument("--device", type=str, default="test_device", help="New login device hash")
    args = parser.parse_args()
    
    if args.cron:
        run_time_driven_cron(force_all=args.force, age_days=args.age_days)
    else:
        if not args.user_id or args.lat is None or args.lon is None:
            parser.error("--user-id, --lat, and --lon are required unless running with --cron")
            
        ts = args.timestamp if args.timestamp else datetime.now(timezone.utc).timestamp()
        
        res = check_and_trigger_retrain(
            user_id=args.user_id,
            new_login_lat=args.lat,
            new_login_lon=args.lon,
            new_login_ts=ts,
            new_login_device=args.device
        )
        print(json.dumps(res, indent=2))
