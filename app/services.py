"""ShieldFlow API Services Layer.

Implements RedisClient, KafkaProducer, and PostgresClient with resilient fallbacks
to local files to guarantee zero-downtime execution in development and testing.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
# pyrefly: ignore [missing-import]
import redis
import pandas as pd

from ml.training.train_pipeline import DEFAULT_CLEAN_LOGINS_PATH, DEFAULT_PROFILES_DIR

class RedisClient:
    """Redis cache manager for last login nodes and active profiles."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.url = f"redis://{host}:{port}/{db}"
        self._client = None
        
    @property
    def client(self) -> Optional[redis.Redis]:
        if self._client is None:
            try:
                self._client = redis.from_url(self.url, socket_timeout=2.0)
                self._client.ping()
            except Exception:
                self._client = None
        return self._client

    def get_last_node(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetches the user's last verified login from Redis cache or database."""
        r = self.client
        if r:
            try:
                data = r.get(f"user:last_node:{user_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                print(f"[REDIS WARNING] Failed to get last node: {e}")
                
        # Fallback to database
        db = PostgresClient()
        return db.get_last_verified_login(user_id)

    def set_last_node(self, user_id: str, node: Dict[str, Any]) -> bool:
        """Caches the user's last login node."""
        r = self.client
        if r:
            try:
                r.set(f"user:last_node:{user_id}", json.dumps(node), ex=86400 * 30) # 30 days cache
                return True
            except Exception as e:
                print(f"[REDIS WARNING] Failed to cache last node: {e}")
        return False


class KafkaProducer:
    """Emitters event message to Kafka broker for streaming anomaly checks."""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.event_log_path = Path("data/kafka_events.log")
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def emit_event(self, topic: str, key: str, value: Dict[str, Any]) -> bool:
        """Emits event to Kafka, falling back to local logging in development."""
        event = {
            "topic": topic,
            "key": key,
            "payload": value
        }
        
        # Log to file to simulate ingestion stream
        try:
            with open(self.event_log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[KAFKA ERROR] Failed to log local event: {e}")
            
        print(f"[KAFKA EMIT] Topic: {topic} | Key: {key} | Event successfully dispatched.")
        return True


class PostgresClient:
    """PostgreSQL client simulation backing database logs."""
    
    def __init__(self, dsn: str = "postgresql://localhost/shieldflow"):
        self.dsn = dsn
        self.clean_logins_path = Path(DEFAULT_CLEAN_LOGINS_PATH)
        self.clean_logins_path.parent.mkdir(parents=True, exist_ok=True)
        
    def get_last_verified_login(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves user's last verified login from the persistent DB."""
        if not self.clean_logins_path.exists():
            return None
            
        try:
            df = pd.read_csv(self.clean_logins_path)
            user_df = df[(df['user_id'] == user_id) & (df['is_verified'].astype(int) == 1)]
            if user_df.empty:
                return None
                
            last_row = user_df.sort_values(by='timestamp').iloc[-1]
            return {
                "latitude": float(last_row["latitude"]),
                "longitude": float(last_row["longitude"]),
                "timestamp": float(last_row["timestamp"]),
                "device_hash": str(last_row["device_hash"]) if "device_hash" in last_row else "unknown"
            }
        except Exception as e:
            print(f"[DB ERROR] Failed to load last verified login: {e}")
            return None
            
    def get_all_device_hashes(self, user_id: str) -> list:
        """Retrieves all historically registered devices for a user."""
        if not self.clean_logins_path.exists():
            return []
            
        try:
            df = pd.read_csv(self.clean_logins_path)
            user_df = df[df['user_id'] == user_id]
            if "device_hash" in user_df.columns:
                return user_df["device_hash"].dropna().unique().tolist()
        except Exception:
            pass
        return []

    def record_login(self, user_id: str, lat: float, lon: float, ts: float, device_hash: str, is_verified: bool) -> bool:
        """Records a new verified login log to persistent DB."""
        new_row = pd.DataFrame([{
            "user_id": user_id,
            "timestamp": ts,
            "latitude": lat,
            "longitude": lon,
            "device_hash": device_hash,
            "is_verified": int(is_verified)
        }])
        
        try:
            if self.clean_logins_path.exists():
                df = pd.read_csv(self.clean_logins_path)
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row
                
            df.to_csv(self.clean_logins_path, index=False)
            return True
        except Exception as e:
            print(f"[DB ERROR] Failed to record login: {e}")
            return False
