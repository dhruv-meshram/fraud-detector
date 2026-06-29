import json
from pathlib import Path
from typing import Dict, Any, Optional
# pyrefly: ignore [missing-import]
import redis

from fraud_detector.adapters.base import BaseProfileStore

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
    """PostgreSQL-backed profile storage. (Currently simulated via file-backup in dev)."""
    
    def __init__(self, dsn: str = "postgresql://localhost/shieldflow", backup_dir: Optional[Path] = None):
        self.dsn = dsn
        self.file_store = FileProfileStore(backup_dir)

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        # In mock mode, delegating to file-system store
        return self.file_store.get_profile(user_id)

    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        # In mock mode, delegating to file-system store
        return self.file_store.save_profile(user_id, profile)


class InMemoryProfileStore(BaseProfileStore):
    """In-memory profile store for unit testing and local sandbox environments."""
    
    def __init__(self):
        self._store = {}

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(user_id)

    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        self._store[user_id] = profile
        return True
