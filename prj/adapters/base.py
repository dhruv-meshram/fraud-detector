from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseProfileStore(ABC):
    """Abstract interface for user ML spatial profile store."""
    
    @abstractmethod
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        pass

class BaseCacheStore(ABC):
    """Abstract interface for fast-path cache mapping user to last login node."""
    
    @abstractmethod
    def get_last_node(self, user_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def set_last_node(self, user_id: str, node: Dict[str, Any]) -> bool:
        pass

class BaseDBStore(ABC):
    """Abstract interface for persistent storage of logins and devices."""
    
    @abstractmethod
    def get_last_verified_login(self, user_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_all_device_hashes(self, user_id: str) -> List[str]:
        pass

    @abstractmethod
    def record_login(self, user_id: str, lat: float, lon: float, ts: float, device_hash: str, is_verified: bool) -> bool:
        pass

class BaseAlertProducer(ABC):
    """Abstract interface for downstream broker alert publishing."""
    
    @abstractmethod
    def emit_event(self, topic: str, key: str, value: Dict[str, Any]) -> bool:
        pass
