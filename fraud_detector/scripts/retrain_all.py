"""ShieldFlow Global Retraining Command Line Tool.

Triggers spatial profile updates for all users in the system, either synchronously
or asynchronously by queuing tasks to the Celery worker queue.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in the path
sys.path.append(str(Path(__file__).parent.parent))

from fraud_detector.ml.training.train_pipeline import DEFAULT_CLEAN_LOGINS_PATH
from fraud_detector.ml.training.retrain import check_and_trigger_retrain
from fraud_detector.scripts.run_worker import retrain_user_model_task

def retrain_all_users(async_mode: bool = False):
    """Triggers retraining for all users with sufficient historical logins."""
    from fraud_detector.adapters.db import PostgresDBStore
    
    db_store = PostgresDBStore()
    
    try:
        users = db_store.get_all_users()
        active_users = []
        for u in users:
            history = db_store.get_user_history(u)
            # Filter for users who can be trained (>= 10 logins)
            if len(history) >= 10:
                active_users.append(u)
    except Exception as e:
        print(f"[ERROR] Failed to query users from DB: {e}")
        return
        
    print(f"Found {len(active_users)} active users with sufficient logins (>= 10) for training.")
    
    queued_count = 0
    success_count = 0
    
    for idx, user_id in enumerate(active_users):
        if async_mode:
            # Dispatch to Celery worker queue
            retrain_user_model_task.delay(user_id)
            queued_count += 1
        else:
            # Train synchronously
            try:
                retrain_user_model_task(user_id)
                success_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to train user {user_id}: {e}")
                
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(active_users)} users...")
            
    if async_mode:
        print(f"[SUCCESS] Dispatched {queued_count} retraining jobs to Celery queue.")
    else:
        print(f"[SUCCESS] Completed synchronous retraining for {success_count} users.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShieldFlow Global Retrain CLI runner")
    parser.add_argument(
        "--async", 
        dest="async_mode", 
        action="store_true", 
        help="Queue jobs asynchronously in Celery instead of training synchronously"
    )
    args = parser.parse_args()
    
    retrain_all_users(async_mode=args.async_mode)
