import json
from typing import Dict, Any, Optional
# pyrefly: ignore [missing-import]
import redis

from fraud_detector.adapters.base import BaseCacheStore

class RedisCacheStore(BaseCacheStore):
    """Redis-backed fast-path lookup cache."""
    
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

    def get_last_node(self, user_id: str) -> Optional[Dict[str, Any]]:
        r = self.client
        if r:
            try:
                data = r.get(f"user:last_node:{user_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                print(f"[RedisCacheStore WARNING] Failed to get last node: {e}")
        return None

    def set_last_node(self, user_id: str, node: Dict[str, Any]) -> bool:
        r = self.client
        if r:
            try:
                r.set(f"user:last_node:{user_id}", json.dumps(node), ex=86400 * 30)
                return True
            except Exception as e:
                print(f"[RedisCacheStore WARNING] Failed to cache last node: {e}")
        return False


class InMemoryCacheStore(BaseCacheStore):
    """In-memory cache for sandbox environments and test mocks."""
    
    def __init__(self):
        self._cache = {}

    def get_last_node(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(user_id)

    def set_last_node(self, user_id: str, node: Dict[str, Any]) -> bool:
        self._cache[user_id] = node
        return True
