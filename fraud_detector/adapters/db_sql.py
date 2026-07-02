import logging
import warnings
from typing import Dict, Any, Optional, List
from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, Integer, Float, Boolean, Text, ForeignKey,
    inspect, select, delete, insert, update
)
from sqlalchemy.engine import Engine, Connection

logger = logging.getLogger("fraud_detector")

class SQLDatabaseManager:
    """Helper to manage tables creation, engine wrapping, and raw DBAPI wrapping."""
    
    def __init__(self, db_engine_or_conn=None, dsn: Optional[str] = None):
        self.metadata = MetaData()
        self.engine = None
        self._conn = None
        
        # 1. Resolve connection/engine or dsn
        if db_engine_or_conn is not None:
            # Check if it is a SQLAlchemy Engine or Connection
            if isinstance(db_engine_or_conn, (Engine, Connection)):
                self.engine = db_engine_or_conn
            elif hasattr(db_engine_or_conn, "cursor") and callable(getattr(db_engine_or_conn, "cursor")):
                # Raw DBAPI connection (like psycopg2 connection or sqlite3 connection)
                module_name = type(db_engine_or_conn).__module__
                dialect = "sqlite" if "sqlite" in module_name else "postgresql"
                self.engine = create_engine(f"{dialect}://", creator=lambda: db_engine_or_conn)
            else:
                self.engine = db_engine_or_conn
        elif dsn is not None:
            self.engine = create_engine(dsn)
        else:
            # Fallback to an in-memory SQLite database (fully headless / filesystem-free)
            self.engine = create_engine("sqlite://")
            
        # Define tables
        self.shieldflow_user_profiles = Table(
            'shieldflow_user_profiles', self.metadata,
            Column('user_id', String(255), primary_key=True),
            Column('last_updated', String(100), nullable=False),
            Column('total_logins_trained', Integer, nullable=False)
        )
        
        self.shieldflow_profile_clusters = Table(
            'shieldflow_profile_clusters', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('user_id', String(255), ForeignKey('shieldflow_user_profiles.user_id', ondelete='CASCADE'), nullable=False),
            Column('cluster_id', Integer, nullable=False),
            Column('centroid_lat', Float, nullable=False),
            Column('centroid_lon', Float, nullable=False),
            Column('dynamic_radius_km', Float, nullable=False),
            Column('num_points', Integer, nullable=False)
        )
        
        self.shieldflow_events = Table(
            'shieldflow_events', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('user_id', String(255), nullable=False),
            Column('device_fingerprint', String(255), nullable=False),
            Column('ip_address', String(255), nullable=False),
            Column('timestamp', Float, nullable=False),
            Column('latitude', Float, nullable=False),
            Column('longitude', Float, nullable=False),
            Column('is_verified', Boolean, nullable=False, default=False),
            Column('event_metadata', Text, nullable=True),
            Column('raw_log', Text, nullable=True),
            Column('derived_features', Text, nullable=True)
        )
        
        self._initialize_tables()

    def _initialize_tables(self):
        try:
            with self.engine.connect() as conn:
                inspector = inspect(conn)
                existing_tables = inspector.get_table_names()
                
                already_existing = []
                tables_to_create = []
                
                for t in ['shieldflow_user_profiles', 'shieldflow_profile_clusters', 'shieldflow_events']:
                    if t in existing_tables:
                        already_existing.append(t)
                    else:
                        tables_to_create.append(t)
                        
                if already_existing:
                    msg = f"SDK tables already exist in the database: {', '.join(already_existing)}. Skipping recreation."
                    warnings.warn(msg, UserWarning)
                    logger.warning(msg)
                    
                if tables_to_create:
                    tables_objs = [
                        self.shieldflow_user_profiles if 'shieldflow_user_profiles' in tables_to_create else None,
                        self.shieldflow_profile_clusters if 'shieldflow_profile_clusters' in tables_to_create else None,
                        self.shieldflow_events if 'shieldflow_events' in tables_to_create else None
                    ]
                    tables_objs = [x for x in tables_objs if x is not None]
                    self.metadata.create_all(conn, tables=tables_objs, checkfirst=True)
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}")
