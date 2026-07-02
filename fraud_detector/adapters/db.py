import logging
from typing import Optional, Dict, Any, List
from sqlalchemy import select, insert, update
from fraud_detector.adapters.base import BaseDBStore
from fraud_detector.adapters.db_sql import SQLDatabaseManager

logger = logging.getLogger("fraud_detector")

class PostgresDBStore(BaseDBStore):
    """SQL-backed persistent database store."""
    
    def __init__(self, db_engine_or_conn=None, dsn: Optional[str] = None, sql_manager: Optional[SQLDatabaseManager] = None):
        if sql_manager is not None:
            self.manager = sql_manager
        else:
            self.manager = SQLDatabaseManager(db_engine_or_conn=db_engine_or_conn, dsn=dsn)

    def get_last_verified_login(self, user_id: str) -> Optional[Dict[str, Any]]:
        query = select(self.manager.shieldflow_events).where(
            self.manager.shieldflow_events.c.user_id == user_id,
            self.manager.shieldflow_events.c.is_verified == True
        ).order_by(self.manager.shieldflow_events.c.timestamp.desc()).limit(1)
        
        try:
            with self.manager.engine.connect() as conn:
                row = conn.execute(query).first()
                if row:
                    r_map = row._mapping
                    return {
                        "latitude": float(r_map["latitude"]),
                        "longitude": float(r_map["longitude"]),
                        "timestamp": float(r_map["timestamp"]),
                        "device_hash": str(r_map["device_fingerprint"])
                    }
        except Exception as e:
            logger.error(f"[PostgresDBStore ERROR] Failed to load last verified login: {e}")
        return None

    def get_all_device_hashes(self, user_id: str) -> List[str]:
        query = select(self.manager.shieldflow_events.c.device_fingerprint).where(
            self.manager.shieldflow_events.c.user_id == user_id
        ).distinct()
        
        try:
            with self.manager.engine.connect() as conn:
                rows = conn.execute(query).fetchall()
                return [str(row[0]) for row in rows]
        except Exception as e:
            logger.error(f"[PostgresDBStore ERROR] Failed to load device hashes: {e}")
        return []

    def record_login(
        self,
        user_id: str,
        lat: float,
        lon: float,
        ts: float,
        device_hash: str,
        is_verified: bool,
        ip_address: str = "unknown",
        event_metadata: Optional[str] = None,
        raw_log: Optional[str] = None,
        derived_features: Optional[str] = None
    ) -> bool:
        stmt = insert(self.manager.shieldflow_events).values(
            user_id=user_id,
            latitude=lat,
            longitude=lon,
            timestamp=ts,
            device_fingerprint=device_hash,
            is_verified=is_verified,
            ip_address=ip_address,
            event_metadata=event_metadata,
            raw_log=raw_log,
            derived_features=derived_features
        )
        try:
            with self.manager.engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"[PostgresDBStore ERROR] Failed to record login: {e}")
            return False

    def verify_last_login(self, user_id: str, timestamp: float) -> bool:
        """Update the verification status of a recorded event to True."""
        stmt = update(self.manager.shieldflow_events).where(
            self.manager.shieldflow_events.c.user_id == user_id,
            self.manager.shieldflow_events.c.timestamp == timestamp
        ).values(is_verified=True)
        try:
            with self.manager.engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"[PostgresDBStore ERROR] Failed to verify last login: {e}")
            return False

    def get_user_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all logins for a user from the database."""
        query = select(self.manager.shieldflow_events).where(
            self.manager.shieldflow_events.c.user_id == user_id
        ).order_by(self.manager.shieldflow_events.c.timestamp.asc())
        
        try:
            with self.manager.engine.connect() as conn:
                rows = conn.execute(query).fetchall()
                results = []
                for row in rows:
                    r_map = row._mapping
                    results.append({
                        "user_id": r_map["user_id"],
                        "timestamp": r_map["timestamp"],
                        "latitude": r_map["latitude"],
                        "longitude": r_map["longitude"],
                        "device_hash": r_map["device_fingerprint"],
                        "is_verified": int(r_map["is_verified"]),
                        "ip_address": r_map["ip_address"],
                        "event_metadata": r_map["event_metadata"],
                        "raw_log": r_map["raw_log"],
                        "derived_features": r_map["derived_features"]
                    })
                return results
        except Exception as e:
            logger.error(f"[PostgresDBStore ERROR] Failed to get user history: {e}")
            return []

    def get_all_users(self) -> List[str]:
        """Retrieve all unique users who have login events."""
        query = select(self.manager.shieldflow_events.c.user_id).distinct()
        try:
            with self.manager.engine.connect() as conn:
                rows = conn.execute(query).fetchall()
                return [str(row[0]) for row in rows]
        except Exception as e:
            logger.error(f"[PostgresDBStore ERROR] Failed to get all users: {e}")
            return []



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

    def verify_last_login(self, user_id: str, timestamp: float) -> bool:
        for login in self._logins:
            if login["user_id"] == user_id and login["timestamp"] == timestamp:
                login["is_verified"] = True
                return True
        return False
