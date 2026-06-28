import random
import uuid
import time

# Logical Anchors
HOME_LAT, HOME_LON = 37.7749, -122.4194  # San Francisco
RENO_LAT, RENO_LON = 39.5296, -119.8138  # Reno, NV (~300km away)
LONDON_LAT, LONDON_LON = 51.5074, -0.1278  # London (Different Continent)
NY_LAT, NY_LON = 40.7128, -74.0060  # New York City (New Home Base)

def _random_offset(lat: float, lon: float, max_offset: float = 0.008) -> tuple[float, float]:
    """Applies a small random coordinate offset to represent local mobility."""
    offset_lat = random.uniform(-max_offset, max_offset)
    offset_lon = random.uniform(-max_offset, max_offset)
    return lat + offset_lat, lon + offset_lon

def generate_cold_start() -> list[dict]:
    """Generates 8 sequential logins to show cold start bootstrapping (< 10 logs)."""
    user_id = str(uuid.uuid4())
    events = []
    base_ts = time.time() - 86400 * 5  # 5 days ago
    
    for i in range(8):
        lat, lon = _random_offset(HOME_LAT, HOME_LON)
        events.append({
            "user_id": user_id,
            "timestamp": base_ts + (i * 3600),  # 1 hour intervals
            "latitude": lat,
            "longitude": lon,
            "device_hash": "fingerprint_web_chrome"
        })
    return events

def generate_model_activation() -> list[dict]:
    """Generates 25 logins tightly grouped to trigger and test active DBSCAN profile."""
    user_id = str(uuid.uuid4())
    events = []
    base_ts = time.time() - 86400 * 10
    
    for i in range(25):
        lat, lon = _random_offset(HOME_LAT, HOME_LON)
        events.append({
            "user_id": user_id,
            "timestamp": base_ts + (i * 3600),
            "latitude": lat,
            "longitude": lon,
            "device_hash": "fingerprint_web_chrome"
        })
    return events

def generate_impossible_speed() -> list[dict]:
    """Generates a baseline login and a subsequent one 5 minutes later in London."""
    user_id = str(uuid.uuid4())
    base_ts = time.time() - 1000
    
    lat1, lon1 = _random_offset(HOME_LAT, HOME_LON)
    lat2, lon2 = _random_offset(LONDON_LAT, LONDON_LON)
    
    return [
        {
            "user_id": user_id,
            "timestamp": base_ts,
            "latitude": lat1,
            "longitude": lon1,
            "device_hash": "fingerprint_web_chrome"
        },
        {
            "user_id": user_id,
            "timestamp": base_ts + 300,  # 5 minutes later
            "latitude": lat2,
            "longitude": lon2,
            "device_hash": "fingerprint_web_chrome"
        }
    ]

def generate_device_flag() -> list[dict]:
    """Generates 10 baseline logins, then an 11th one at home but with a mutated device hash."""
    user_id = str(uuid.uuid4())
    events = []
    base_ts = time.time() - 86400 * 5
    
    # 10 baseline logins
    for i in range(10):
        lat, lon = _random_offset(HOME_LAT, HOME_LON)
        events.append({
            "user_id": user_id,
            "timestamp": base_ts + (i * 3600),
            "latitude": lat,
            "longitude": lon,
            "device_hash": "fingerprint_web_chrome"
        })
        
    # 11th mutated device log
    lat, lon = _random_offset(HOME_LAT, HOME_LON)
    events.append({
        "user_id": user_id,
        "timestamp": base_ts + (10 * 3600),
        "latitude": lat,
        "longitude": lon,
        "device_hash": "fingerprint_mutated_firefox"
    })
    return events

def generate_outlier_flag() -> list[dict]:
    """Generates 10 baseline logins, then a 300km outlier login, then a return login."""
    user_id = str(uuid.uuid4())
    events = []
    base_ts = time.time() - 86400 * 5
    
    for i in range(10):
        lat, lon = _random_offset(HOME_LAT, HOME_LON)
        events.append({
            "user_id": user_id,
            "timestamp": base_ts + (i * 3600),
            "latitude": lat,
            "longitude": lon,
            "device_hash": "fingerprint_web_chrome"
        })
        
    # Reno outlier
    lat_out, lon_out = _random_offset(RENO_LAT, RENO_LON)
    events.append({
        "user_id": user_id,
        "timestamp": base_ts + (10 * 3600) + 18000,  # 5 hours delay (valid velocity)
        "latitude": lat_out,
        "longitude": lon_out,
        "device_hash": "fingerprint_web_chrome"
    })
    
    # Return login
    lat_ret, lon_ret = _random_offset(HOME_LAT, HOME_LON)
    events.append({
        "user_id": user_id,
        "timestamp": base_ts + (10 * 3600) + 36000,
        "latitude": lat_ret,
        "longitude": lon_ret,
        "device_hash": "fingerprint_web_chrome"
    })
    return events

def generate_adaptive_cluster() -> list[dict]:
    """Generates 10 baseline logins, then relocates to NYC to generate 25 consecutive logins."""
    user_id = str(uuid.uuid4())
    events = []
    base_ts = time.time() - 86400 * 15
    
    # 10 baseline logins at SF
    for i in range(10):
        lat, lon = _random_offset(HOME_LAT, HOME_LON)
        events.append({
            "user_id": user_id,
            "timestamp": base_ts + (i * 3600),
            "latitude": lat,
            "longitude": lon,
            "device_hash": "fingerprint_web_chrome"
        })
        
    # Relocation to NYC (1 outlier + 25 logins)
    # The first one is a spatial outlier
    lat_nyc, lon_nyc = _random_offset(NY_LAT, NY_LON)
    events.append({
        "user_id": user_id,
        "timestamp": base_ts + (11 * 3600) + 86400,  # 24 hours flight time (valid velocity)
        "latitude": lat_nyc,
        "longitude": lon_nyc,
        "device_hash": "fingerprint_web_chrome"
    })
    
    # 24 subsequent logins at NYC
    for i in range(24):
        lat, lon = _random_offset(NY_LAT, NY_LON)
        events.append({
            "user_id": user_id,
            "timestamp": base_ts + (11 * 3600) + 86400 + ((i + 1) * 3600),
            "latitude": lat,
            "longitude": lon,
            "device_hash": "fingerprint_web_chrome"
        })
    return events
