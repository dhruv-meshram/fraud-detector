"""ShieldFlow Celery Background Task Orchestrator.

Defines the Celery application and worker tasks for asynchronous micro-model retraining,
ensuring training is offloaded from the real-time API loop.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# pyrefly: ignore [missing-import]
from celery import Celery

# Ensure project root is in the path
sys.path.append(str(Path(__file__).parent.parent))

from fraud_detector.ml.training.train_pipeline import train_user_model, DEFAULT_CLEAN_LOGINS_PATH
from fraud_detector.ml.models.registry import ModelRegistry

# Retrieve broker from environment or default to local Redis container
REDIS_BROKER = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "shieldflow_worker",
    broker=REDIS_BROKER,
    backend=REDIS_BROKER
)

# Celery configurations
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1
)

@app.task(name="tasks.retrain_user_model")
def retrain_user_model_task(user_id: str) -> bool:
    """Asynchronously retrains a user's DBSCAN model and hydrates the registry.
    
    Args:
        user_id: User ID to retrain.
        
    Returns:
        Boolean indicating task success.
    """
    print(f"[WORKER] Starting asynchronous retraining for user {user_id}...")
    from fraud_detector.adapters.db import PostgresDBStore
    
    db_store = PostgresDBStore()
    
    try:
        # 1. Load user logs from DB
        history = db_store.get_user_history(user_id)
        if not history:
            print(f"[WORKER WARNING] User {user_id} has no login history in DB.")
            return False
            
        user_df = pd.DataFrame(history)
        if len(user_df) < 10:
            print(f"[WORKER WARNING] User {user_id} has {len(user_df)} logins. Bypassing training due to sparse data.")
            return False
            
        # 2. Re-train micro-model
        profile = train_user_model(user_id, user_df)
        
        # 3. Save profile to registry (updates SQL DB & Redis)
        registry = ModelRegistry()
        success = registry.register_profile(user_id, profile)
        
        print(f"[WORKER] Successfully retrained and registered model for user {user_id}.")
        return success
        
    except Exception as e:
        print(f"[WORKER ERROR] Retraining failed for user {user_id}: {e}")
        return False

if __name__ == "__main__":
    # Start the Celery worker locally if executed directly
    print("[WORKER] Starting Celery worker loop...")
    app.start(argv=["worker", "--loglevel=info"])
