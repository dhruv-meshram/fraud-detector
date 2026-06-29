import argparse
import csv
import heapq
import math
import os
import random
import uuid
from datetime import datetime
from pathlib import Path

# Target output path
DEFAULT_OUTPUT_PATH = Path("/home/dhruv/Documents/fraud-detector/data/raw/synthetic_logins.csv")

# Configuration
NUM_ROWS = 75000
NUM_USERS = 1000
BATCH_SIZE = 10000
START_DATE = datetime(2026, 1, 1).timestamp()

CITIES = [
    (37.7749, -122.4194),   # San Francisco
    (40.7128, -74.0060),    # New York
    (51.5074, -0.1278),     # London
    (19.0760, 72.8777),     # Mumbai
    (23.2156, 72.6369),     # Gandhinagar
    (-33.8688, 151.2093),   # Sydney
]

def haversine(lat1, lon1, lat2, lon2):
    if lat1 is None or lat2 is None:
        return 0.0
    R = 6371.0 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
    return R * (2 * math.asin(math.sqrt(a)))

def generate_batched_dataset(filename):
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing {NUM_USERS} user profiles...")
    
    # Priority Queue to keep events sorted chronologically without holding the whole stream in memory
    event_queue = []
    
    for _ in range(NUM_USERS):
        base_city = random.choice(CITIES)
        user_state = {
            'user_id': str(uuid.uuid4()),
            'base_lat': base_city[0],
            'base_lon': base_city[1],
            'primary_device': uuid.uuid4().hex[:16],
            'secondary_device': uuid.uuid4().hex[:16] if random.random() > 0.7 else None,
            'mobility_factor': random.uniform(0.01, 0.1),
            'last_lat': None,
            'last_lon': None,
            'last_time': None
        }
        # Schedule their very first login within the first 5 days
        first_login_time = START_DATE + random.uniform(0, 5 * 86400)
        heapq.heappush(event_queue, (first_login_time, user_state['user_id'], user_state))

    # Prepare CSV (write headers)
    columns = [
        'user_id', 'timestamp', 'latitude', 'longitude', 'device_hash', 
        'distance_from_last_km', 'velocity_kmh', 'risk_label', 'is_verified'
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

    print(f"Streaming {NUM_ROWS} events to disk in batches of {BATCH_SIZE}...")
    batch = []
    rows_generated = 0

    while rows_generated < NUM_ROWS:
        # Get the chronologically next event
        current_time, _, user = heapq.heappop(event_queue)
        
        # Determine behavior flags
        is_attack = random.random() > 0.99
        is_travel = random.random() > 0.95

        # Device mapping
        device = user['primary_device']
        if user['secondary_device'] and random.random() > 0.9:
            device = user['secondary_device']
        elif random.random() > 0.98:
            device = uuid.uuid4().hex[:16]

        # Calculate new coordinates
        if is_attack:
            lat, lon = random.uniform(-60, 60), random.uniform(-180, 180)
            # Force small time gap for attacks to trigger velocity constraint
            current_time = user['last_time'] + random.uniform(60, 300) if user['last_time'] else current_time
        elif is_travel:
            new_city = random.choice(CITIES)
            lat = new_city[0] + random.uniform(-0.1, 0.1)
            lon = new_city[1] + random.uniform(-0.1, 0.1)
        else:
            # Shift coordinates asymmetrically to prevent simple isotropic symmetry
            lat = user['base_lat'] + random.uniform(-user['mobility_factor'], user['mobility_factor'])
            lon = user['base_lon'] + random.uniform(-user['mobility_factor'] * 1.5, user['mobility_factor'] * 1.5)

        # Calculate metrics
        dist = haversine(user['last_lat'], user['last_lon'], lat, lon)
        time_diff_hours = (current_time - user['last_time']) / 3600.0 if user['last_time'] else 0.0
        velocity = (dist / time_diff_hours) if time_diff_hours > 0 else 0.0

        # Scoring Logic
        if velocity > 900.0:
            risk, verified = "HIGH_RISK", 0
        elif velocity <= 900.0 and (dist > 100.0 or device != user['primary_device']):
            risk, verified = "MEDIUM_RISK", (0 if is_attack else 1)
        else:
            risk, verified = "LOW_RISK", 1

        # Add to current batch
        batch.append({
            'user_id': user['user_id'], 
            'timestamp': current_time,
            'latitude': round(lat, 6), 
            'longitude': round(lon, 6),
            'device_hash': device, 
            'distance_from_last_km': round(dist, 2),
            'velocity_kmh': round(velocity, 2), 
            'risk_label': risk, 
            'is_verified': verified
        })

        # Update user state and schedule their next login (2 to 24 hours later)
        user['last_lat'], user['last_lon'], user['last_time'] = lat, lon, current_time
        next_login_time = current_time + random.uniform(2 * 3600, 24 * 3600)
        heapq.heappush(event_queue, (next_login_time, user['user_id'], user))

        rows_generated += 1

        # Flush batch to disk when full
        if len(batch) >= BATCH_SIZE:
            with open(out_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerows(batch)
            batch = []
            print(f" --> Wrote {rows_generated} rows...")

    # Flush any remaining rows
    if batch:
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerows(batch)
        print(f" --> Wrote {rows_generated} rows...")

    print(f"Success! Labeled, time-sorted dataset saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShieldFlow Queue-Based Labeled Dataset Generator")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path")
    args = parser.parse_args()
    
    generate_batched_dataset(args.output)
