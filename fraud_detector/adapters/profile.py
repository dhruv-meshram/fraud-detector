import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
# pyrefly: ignore [missing-import]
import redis
from sqlalchemy import select, delete, insert

from fraud_detector.adapters.base import BaseProfileStore
from fraud_detector.adapters.db_sql import SQLDatabaseManager

DEFAULT_PROFILES_DIR = Path("data/processed/profiles")

class FileProfileStore(BaseProfileStore):
    """File-system based spatial profile storage."""
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = Path(profiles_dir) if profiles_dir else DEFAULT_PROFILES_DIR
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        profile_path = self.profiles_dir / f"{user_id}.json"
        if not profile_path.exists():
            return None
        try:
            with open(profile_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[FileProfileStore ERROR] Failed to load {user_id}: {e}")
            return None

    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        profile_path = self.profiles_dir / f"{user_id}.json"
        try:
            with open(profile_path, "w") as f:
                json.dump(profile, f)
            return True
        except Exception as e:
            print(f"[FileProfileStore ERROR] Failed to save {user_id}: {e}")
            return False


class RedisProfileStore(BaseProfileStore):
    """Redis-backed live registry profile storage."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_url = f"redis://{host}:{port}/{db}"
        self._client = None

    @property
    def client(self) -> Optional[redis.Redis]:
        if self._client is None:
            try:
                self._client = redis.from_url(self.redis_url, socket_timeout=2.0)
                self._client.ping()
            except Exception:
                self._client = None
        return self._client

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        r = self.client
        if r:
            try:
                cached = r.get(f"user:profile:{user_id}")
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"[RedisProfileStore WARNING] Failed to get: {e}")
        return None

    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        r = self.client
        if r:
            try:
                r.set(f"user:profile:{user_id}", json.dumps(profile))
                return True
            except Exception as e:
                print(f"[RedisProfileStore WARNING] Failed to save: {e}")
        return False


class PostgreSQLProfileStore(BaseProfileStore):
    """SQL-backed persistent user profile/model store."""
    
    def __init__(self, db_engine_or_conn=None, dsn: Optional[str] = None, sql_manager: Optional[SQLDatabaseManager] = None):
        if sql_manager is not None:
            self.manager = sql_manager
        else:
            self.manager = SQLDatabaseManager(db_engine_or_conn=db_engine_or_conn, dsn=dsn)
            
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        query_profile = select(self.manager.shieldflow_user_profiles).where(
            self.manager.shieldflow_user_profiles.c.user_id == user_id
        )
        query_clusters = select(self.manager.shieldflow_profile_clusters).where(
            self.manager.shieldflow_profile_clusters.c.user_id == user_id
        ).order_by(self.manager.shieldflow_profile_clusters.c.cluster_id.asc())
        
        try:
            with self.manager.engine.connect() as conn:
                p_row = conn.execute(query_profile).first()
                if not p_row:
                    return None
                
                p_map = p_row._mapping
                c_rows = conn.execute(query_clusters).fetchall()
                
                clusters_list = []
                for crow in c_rows:
                    c_map = crow._mapping
                    clusters_list.append({
                        "cluster_id": int(c_map["cluster_id"]),
                        "centroid_lat": float(c_map["centroid_lat"]),
                        "centroid_lon": float(c_map["centroid_lon"]),
                        "dynamic_radius_km": float(c_map["dynamic_radius_km"]),
                        "num_points": int(c_map["num_points"])
                    })
                    
                return {
                    "user_id": str(p_map["user_id"]),
                    "last_updated": str(p_map["last_updated"]),
                    "total_logins_trained": int(p_map["total_logins_trained"]),
                    "clusters": clusters_list
                }
        except Exception as e:
            print(f"[PostgreSQLProfileStore ERROR] Failed to get profile for {user_id}: {e}")
        return None

    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        last_updated = profile.get("last_updated") or datetime.utcnow().isoformat() + "Z"
        total_logins = profile.get("total_logins_trained") or 0
        
        try:
            with self.manager.engine.connect() as conn:
                # Delete existing profile clusters
                conn.execute(delete(self.manager.shieldflow_profile_clusters).where(
                    self.manager.shieldflow_profile_clusters.c.user_id == user_id
                ))
                
                # Upsert user profile
                conn.execute(delete(self.manager.shieldflow_user_profiles).where(
                    self.manager.shieldflow_user_profiles.c.user_id == user_id
                ))
                
                conn.execute(insert(self.manager.shieldflow_user_profiles).values(
                    user_id=user_id,
                    last_updated=last_updated,
                    total_logins_trained=total_logins
                ))
                
                # Insert new clusters
                for cluster in profile.get("clusters", []):
                    conn.execute(insert(self.manager.shieldflow_profile_clusters).values(
                        user_id=user_id,
                        cluster_id=cluster.get("cluster_id", 0),
                        centroid_lat=cluster.get("centroid_lat", 0.0),
                        centroid_lon=cluster.get("centroid_lon", 0.0),
                        dynamic_radius_km=cluster.get("dynamic_radius_km", 0.0),
                        num_points=cluster.get("num_points", 1)
                    ))
                conn.commit()
                return True
        except Exception as e:
            print(f"[PostgreSQLProfileStore ERROR] Failed to save profile for {user_id}: {e}")
            return False


class InMemoryProfileStore(BaseProfileStore):
    """In-memory profile store for unit testing and local sandbox environments."""
    
    def __init__(self):
        self._store = {}

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(user_id)

    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        self._store[user_id] = profile
        return True
