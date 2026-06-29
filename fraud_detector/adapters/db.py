import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

from fraud_detector.adapters.base import BaseDBStore

DEFAULT_CLEAN_LOGINS_PATH = Path("data/processed/clean_logins.csv")

class PostgresDBStore(BaseDBStore):
    """PostgreSQL database simulation store backing CSV file logs."""
    
    def __init__(self, dsn: str = "postgresql://localhost/shieldflow", clean_logins_path: Optional[Path] = None):
        self.dsn = dsn
        self.clean_logins_path = Path(clean_logins_path) if clean_logins_path else DEFAULT_CLEAN_LOGINS_PATH
        self.clean_logins_path.parent.mkdir(parents=True, exist_ok=True)

    def get_last_verified_login(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.clean_logins_path.exists() or self.clean_logins_path.stat().st_size == 0:
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
            print(f"[PostgresDBStore ERROR] Failed to load last verified login: {e}")
            return None

    def get_all_device_hashes(self, user_id: str) -> List[str]:
        if not self.clean_logins_path.exists() or self.clean_logins_path.stat().st_size == 0:
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
        new_row = pd.DataFrame([{
            "user_id": user_id,
            "timestamp": ts,
            "latitude": lat,
            "longitude": lon,
            "device_hash": device_hash,
            "is_verified": int(is_verified)
        }])
        try:
            if self.clean_logins_path.exists() and self.clean_logins_path.stat().st_size > 0:
                df = pd.read_csv(self.clean_logins_path)
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row
            df.to_csv(self.clean_logins_path, index=False)
            return True
        except Exception as e:
            print(f"[PostgresDBStore ERROR] Failed to record login: {e}")
            return False


class InMemoryDBStore(BaseDBStore):
    """In-memory database store for headless execution."""
    
    def __init__(self):
        self._logins = []

    def get_last_verified_login(self, user_id: str) -> Optional[Dict[str, Any]]:
        verified = [x for x in self._logins if x["user_id"] == user_id and x["is_verified"]]
        if not verified:
            return None
        last = max(verified, key=lambda x: x["timestamp"])
        return {
            "latitude": last["latitude"],
            "longitude": last["longitude"],
            "timestamp": last["timestamp"],
            "device_hash": last["device_hash"]
        }

    def get_all_device_hashes(self, user_id: str) -> List[str]:
        return list(set(x["device_hash"] for x in self._logins if x["user_id"] == user_id))

    def record_login(self, user_id: str, lat: float, lon: float, ts: float, device_hash: str, is_verified: bool) -> bool:
        self._logins.append({
            "user_id": user_id,
            "latitude": lat,
            "longitude": lon,
            "timestamp": ts,
            "device_hash": device_hash,
            "is_verified": is_verified
        })
        return True
