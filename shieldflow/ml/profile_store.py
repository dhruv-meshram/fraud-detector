import json

class ProfileStore:
    """Lightweight in-memory profile and state storage engine (mocking Redis/DB)."""
    
    def __init__(self) -> None:
        # Maps user_id -> list of cluster dictionaries
        self._profiles = {}
        # Maps user_id -> list of raw login logs
        self._history = {}
        # Maps user_id -> set of recognized device hashes
        self._device_hashes = {}

    def get_profile(self, user_id: str) -> list:
        """Retrieves spatial clusters for the user."""
        return self._profiles.get(user_id, [])

    def save_profile(self, user_id: str, clusters: list) -> None:
        """Saves spatial clusters for the user."""
        self._profiles[user_id] = clusters

    def get_history(self, user_id: str) -> list:
        """Retrieves raw login history logs for the user."""
        return self._history.get(user_id, [])

    def add_history(self, user_id: str, login_event: dict) -> None:
        """Appends a new login event log to history."""
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(login_event)
        
        # Track device hash
        if user_id not in self._device_hashes:
            self._device_hashes[user_id] = set()
        self._device_hashes[user_id].add(login_event["device_hash"])

    def is_known_device(self, user_id: str, device_hash: str) -> bool:
        """Returns True if the device hash has been observed historically."""
        return device_hash in self._device_hashes.get(user_id, set())

    def clear(self) -> None:
        """Clears all in-memory store states."""
        self._profiles.clear()
        self._history.clear()
        self._device_hashes.clear()
