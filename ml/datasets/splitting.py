"""ShieldFlow Per-User Dataset Splitting & Validation Utilities.

Contains utilities for extracting rolling time-window training sets and
generating walk-forward validation splits to support backtesting of micro-models.
"""

from typing import Generator, Tuple
import pandas as pd

def get_rolling_train_window(
    user_df: pd.DataFrame, 
    train_window_days: int = 90, 
    max_train_samples: int = 1000
) -> pd.DataFrame:
    """Extracts a rolling time-window of verified logins for model training.
    
    Filters for logs within the specified day window relative to the most recent login,
    caps the maximum number of training samples, and preserves chronological order.
    
    Args:
        user_df: Sorted DataFrame of logins for a single user.
        train_window_days: Number of days of historical data to retain.
        max_train_samples: Maximum number of recent samples to train on.
        
    Returns:
        DataFrame containing the training set.
    """
    if user_df.empty:
        return user_df
        
    # Ensure sorted order
    user_df = user_df.sort_values(by='timestamp').copy()
    
    # Filter for only verified logs (DBSCAN trains only on habitual legitimate behavior)
    verified_df = user_df[user_df['is_verified'].astype(int) == 1]
    if verified_df.empty:
        return verified_df
        
    # Get the timestamp of the most recent log
    latest_ts = verified_df['timestamp'].max()
    
    # Calculate cutoff timestamp
    cutoff_ts = latest_ts - (train_window_days * 24 * 3600)
    
    # Filter by time window
    train_df = verified_df[verified_df['timestamp'] >= cutoff_ts]
    
    # Cap by max_train_samples (taking the most recent ones)
    if len(train_df) > max_train_samples:
        train_df = train_df.tail(max_train_samples)
        
    return train_df.reset_index(drop=True)

def generate_walk_forward_splits(
    user_df: pd.DataFrame, 
    min_train_size: int = 10,
    train_window_days: int = 90,
    max_train_samples: int = 1000
) -> Generator[Tuple[pd.DataFrame, pd.Series], None, None]:
    """Generates successive (train_df, test_log) splits using Time-Series Walk-Forward validation.
    
    For each split, the train set is constructed using rolling time-windows up to 
    the timestamp of the test log (exclusive), and the test log represents the next 
    consecutive login event.
    
    Args:
        user_df: Sorted DataFrame of logins for a single user.
        min_train_size: Minimum number of historical verified logins required to train.
        train_window_days: Rolling window size (in days) to use for the training set.
        max_train_samples: Maximum number of recent samples to use for the training set.
        
    Yields:
        Tuple of (train_df, test_log_series) for evaluation.
    """
    # Ensure sorting
    user_df = user_df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Walk forward from the index where we have enough historical data
    for i in range(len(user_df)):
        # Current log to test/predict on (N+1)
        test_log = user_df.iloc[i]
        
        # Historical candidates up to index i (1 to N)
        history = user_df.iloc[:i]
        
        # Filter for verified logins in history
        verified_history = history[history['is_verified'].astype(int) == 1]
        
        # Check if we have enough training samples
        if len(verified_history) < min_train_size:
            continue
            
        # Get rolling training set relative to the test log's timestamp
        latest_history_ts = test_log['timestamp']
        cutoff_ts = latest_history_ts - (train_window_days * 24 * 3600)
        
        train_df = verified_history[verified_history['timestamp'] >= cutoff_ts]
        if len(train_df) > max_train_samples:
            train_df = train_df.tail(max_train_samples)
            
        # Ensure we still meet min_train_size after rolling window cuts
        if len(train_df) < min_train_size:
            continue
            
        yield train_df.reset_index(drop=True), test_log
