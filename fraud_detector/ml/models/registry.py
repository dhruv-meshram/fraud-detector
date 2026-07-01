"""ShieldFlow Model Registry.

Provides a unified interface to register, backup, retrieve, and hydrate micro-model 
profiles (user spatial JSON baselines) using Redis and PostgreSQL.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
# pyrefly: ignore [missing-import]
import redis

DEFAULT_PROFILES_DIR = Path("/home/dhruv/Documents/fraud-detector/fraud_detector/data/processed/profiles")


class ModelRegistry:
    """Unified Model Registry for ShieldFlow user micro-models."""
    
    def __init__(
        self, 
        redis_host: str = "localhost", 
        redis_port: int = 6379, 
        redis_db: int = 0,
        profiles_dir: Path = DEFAULT_PROFILES_DIR
    ):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection lazily
        self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
        self._redis_client = None

    @property
    def redis_client(self) -> Optional[redis.Redis]:
        """Lazy-loaded Redis client with safety checks."""
        if self._redis_client is None:
            try:
                self._redis_client = redis.from_url(self.redis_url, socket_timeout=2.0)
                # Test connection
                self._redis_client.ping()
            except Exception:
                # Fallback to local storage if Redis is down or unavailable
                self._redis_client = None
        return self._redis_client

    def register_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        """Registers a user profile to the local disk and hydrates the Redis cache.
        
        Returns:
            Boolean indicating successful caching/registration.
        """
        profile_json = json.dumps(profile)
        
        # 1. Persistent backup (file system fallback / simulating PostgreSQL write)
        local_path = self.profiles_dir / f"{user_id}.json"
        try:
            with open(local_path, "w") as f:
                f.write(profile_json)
        except Exception as e:
            print(f"[REGISTRY ERROR] Failed to save persistent backup for {user_id}: {e}")
            return False
            
        # 2. Redis Live Registry Hydration (zero downtime overwrite)
        r = self.redis_client
        if r:
            try:
                r.set(f"user:profile:{user_id}", profile_json)
                return True
            except Exception as e:
                print(f"[REGISTRY WARNING] Failed to cache profile for {user_id} in Redis: {e}")
                
        return True

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a user profile, check Redis first, falling back to disk and auto-hydrating.
        
        Returns:
            Dictionary profile if found, otherwise None.
        """
        redis_key = f"user:profile:{user_id}"
        r = self.redis_client
        
        # 1. Read from cache (sub-millisecond lookups)
        if r:
            try:
                cached = r.get(redis_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"[REGISTRY WARNING] Failed to fetch {user_id} from Redis: {e}")
                
        # 2. Cache Miss: Fallback to persistent backup
        local_path = self.profiles_dir / f"{user_id}.json"
        if local_path.exists():
            try:
                with open(local_path, "r") as f:
                    profile = json.load(f)
                
                # Auto-hydrate Redis cache on miss
                if r:
                    try:
                        r.set(redis_key, json.dumps(profile))
                    except Exception:
                        pass
                        
                return profile
            except Exception as e:
                print(f"[REGISTRY ERROR] Failed to load backup for {user_id}: {e}")
                
        return None
